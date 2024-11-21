import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Tuple
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from tqdm import tqdm
from config import IAMTrainingConfig as cfg
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.time_proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4)
        )
    
    def forward(self, t):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.time_proj(embeddings)

class FiLM(nn.Module):
    def __init__(self, num_features, condition_vector_dim):
        super().__init__()
        self.condition_generator = nn.Sequential(
            nn.Linear(condition_vector_dim, num_features * 2),
            nn.SiLU(),
            nn.Linear(num_features * 2, num_features * 2)
        )
        
    def forward(self, x, condition):
        if condition is None:
            return x
            
        # L2 normalization
        if len(condition.shape) == 3:
            condition = condition.mean(dim=1)
        condition = condition / (torch.norm(condition, dim=-1, keepdim=True) + 1e-8)
        
        params = self.condition_generator(condition)
        gamma, beta = torch.chunk(params, chunks=2, dim=-1)
        
        gamma = gamma.view(*gamma.shape, 1, 1)
        beta = beta.view(*beta.shape, 1, 1)
        
        return gamma * x + beta
    
class FiLM_UNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # FiLM 레이어 추가 (down blocks에만 적용)
        self.film_layers = nn.ModuleList([
            FiLM(
                num_features=block.resnets[0].conv1.out_channels,
                condition_vector_dim=kwargs['cross_attention_dim']
            )
            for block in self.down_blocks
        ])

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # 기존 UNet forward 로직 유지하면서 FiLM만 추가
        sample = self.conv_in(sample)

        # Down blocks with FiLM
        down_block_res_samples = (sample,)
        for down_block_idx, down_block in enumerate(self.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=timestep)

            # FiLM 적용
            sample = self.film_layers[down_block_idx](sample, encoder_hidden_states)
            down_block_res_samples += res_samples

        # Mid block
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = self.mid_block(sample, timestep)

        # Up blocks
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                )

        # Output
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)

class GC_DDPM(nn.Module):
    def __init__(
        self,
        num_writers=252,
        writer_embed_dim=256,
        image_size=64,
        max_width=512,
        in_channels=2,
        out_channels=2,
        n_timesteps=1000,
    ):
        super().__init__()
        
        self.num_writers = num_writers
        self.null_writer_id = num_writers  # null writer ID를 명시적으로 저장
        
        # Writer embedding (+1 for null token)
        self.writer_embedding = nn.Embedding(num_writers + 1, writer_embed_dim)  # +1은 null token을 위한 것
        # print(f"Embedding size: {self.writer_embedding.weight.shape}")
        self.writer_proj = nn.Sequential(
            nn.Linear(writer_embed_dim, writer_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(writer_embed_dim * 4, writer_embed_dim * 4)
        )

        # self.unet = FiLM_UNet2DConditionModel(
        #     sample_size=(image_size, max_width),
        #     in_channels=in_channels,
        #     out_channels=out_channels,
        #     layers_per_block=2,
        #     block_out_channels=(128, 256, 512, 512),
        #     cross_attention_dim=writer_embed_dim * 4,
        #     down_block_types=(
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "DownBlock2D",
        #     ),
        #     up_block_types=(
        #         "UpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #     ), 
        # )

        # UNet 초기화 시 time_embedding_type과 차원을 명시적으로 지정
        self.unet = UNet2DConditionModel(
            sample_size=(image_size, max_width),
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),
            cross_attention_dim=writer_embed_dim * 4,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            time_embedding_type="positional",  # 명시적으로 positional 지정
            time_embedding_dim=512,  # 시간 임베딩 차원 명시적 지정
            projection_class_embeddings_input_dim=None  # 클래스 임베딩 비활성화
        )

        # self.writer_embedding = nn.Embedding(num_writers, writer_embed_dim)
        # self.writer_norm = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        # self.writer_proj = nn.Linear(writer_embed_dim, writer_embed_dim * 4)

        # DDPM 관련 파라미터는 제거 (스케줄러가 처리)
        self.n_timesteps = n_timesteps

    def forward(self, x, glyph, timesteps, writer_ids=None, training=True):
        """학습 시에는 랜덤하게 조건을 null로 설정"""
        if training:
            # 10% 확률로 각각 null로 설정
            use_null_glyph = torch.rand(glyph.shape[0], device=glyph.device) < 0.1
            use_null_writer = torch.rand(glyph.shape[0], device=glyph.device) < 0.1
            
            # Null glyph 처리
            null_glyph = torch.zeros_like(glyph)
            glyph = torch.where(use_null_glyph.view(-1, 1, 1, 1), null_glyph, glyph)
            
            # Null writer 처리
            if writer_ids is not None:
                writer_ids = torch.where(use_null_writer, 
                                    torch.tensor(self.null_writer_id, device=writer_ids.device),
                                    writer_ids)
            
        # 입력 정규화
        x = x * 2.0 - 1.0
        glyph = glyph * 2.0 - 1.0
        
        # Writer embedding 처리
        if writer_ids is not None:
            # 디버깅을 위한 writer_ids 정보 출력
            # print(f"writer_ids min: {writer_ids.min()}, max: {writer_ids.max()}, shape: {writer_ids.shape}")
            # print(f"num_writers: {self.num_writers}")
            
            w_emb = self.writer_embedding(writer_ids)
            w_emb = w_emb / torch.norm(w_emb, dim=-1, keepdim=True)
            w_emb = self.writer_proj(w_emb)
            w_emb = w_emb.unsqueeze(1)  # (batch_size, 1, embed_dim)
        else:
            w_emb = None
        
        # 타입과 디바이스 확인
        x_input = torch.cat([x, glyph], dim=1)
        x_input = x_input.to(dtype=torch.float32)
        timesteps = timesteps.to(dtype=torch.long)
        if w_emb is not None:
            w_emb = w_emb.to(dtype=torch.float32)
        
        # U-Net forward
        pred = self.unet(
            sample=x_input,
            timestep=timesteps,
            encoder_hidden_states=w_emb,
            return_dict=True
        )
        
        return torch.chunk(pred.sample, 2, dim=1)
    @torch.no_grad()
    def sample(
        self, 
        glyph, 
        writer_ids, 
        scheduler,
        use_guidance=True, 
        content_scale=3.0, 
        style_scale=1.0, 
        num_inference_steps=50,
        eta=0.0  # DDIM 샘플링을 위한 eta 파라미터 추가
    ):
        """샘플링 메서드 - DDPM 또는 DDIM"""
        device = cfg.device
        batch_size = glyph.shape[0]
        
        # 모든 입력을 동일한 디바이스로 이동
        glyph = glyph.to(device)
        writer_ids = writer_ids.to(device)
        
        # 초기 노이즈에서 시작
        x = torch.randn((batch_size, 1, glyph.shape[2], glyph.shape[3]), device=device)
        
        # 스케줄러의 타임스텝 설정
        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps.to(device)
        
        # Null 조건 준비
        null_glyph = torch.zeros_like(glyph)
        null_writer_ids = torch.ones_like(writer_ids) * self.num_writers
        
        for i, t in enumerate(tqdm(timesteps, desc="Sampling")):
            t_batch = t.expand(batch_size)
            
            if use_guidance:
                # Classifier-free guidance 계산
                noise_pred_full, _ = self.forward(x, glyph, t_batch, writer_ids, training=False)
                noise_pred_no_writer, _ = self.forward(x, glyph, t_batch, null_writer_ids, training=False)
                noise_pred_no_glyph, _ = self.forward(x, null_glyph, t_batch, writer_ids, training=False)
                noise_pred_no_both, _ = self.forward(x, null_glyph, t_batch, null_writer_ids, training=False)
                
                noise_pred = (
                    noise_pred_full + 
                    content_scale * noise_pred_no_writer +
                    style_scale * noise_pred_no_glyph - 
                    (content_scale + style_scale) * noise_pred_no_both
                )
            else:
                noise_pred, _ = self.forward(x, glyph, t_batch, writer_ids, training=False)
            
            # 스케줄러를 사용하여 다음 샘플 계산
            x = scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=x,
                eta=eta
            ).prev_sample
        
        x = (x + 1) / 2
        return x.clamp(0, 1)
