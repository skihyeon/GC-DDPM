import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Any, Tuple, List
from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from tqdm import tqdm
from config import IAMTrainingConfig as cfg
import math


class FiLMLayer(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        beta = self.beta.view(1, -1, 1, 1)
        return gamma * x + beta
    
class UNet2DConditionModelWithFiLM(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 각 다운/업 블록의 출력에 대한 FiLM 레이어 초기화
        self.down_film_layers = nn.ModuleList([
            FiLMLayer(block.resnets[-1].out_channels)
            for block in self.down_blocks
        ])
        self.up_film_layers = nn.ModuleList([
            FiLMLayer(block.resnets[-1].out_channels)
            for block in self.up_blocks
        ])
        self.mid_film = FiLMLayer(self.mid_block.resnets[-1].out_channels)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs
    ) -> Union[UNet2DConditionOutput, Tuple]:
        # 1. 시간 임베딩
        t_emb = self.get_time_embed(sample=sample, timestep=timestep)
        emb = self.time_embedding(t_emb)

        # 2. 전처리
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0
        
        sample = self.conv_in(sample)

        # 3. 다운샘플링
        down_block_res_samples = (sample,)
        for downsample_block, film_layer in zip(self.down_blocks, self.down_film_layers):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **kwargs
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb
                )
            # FiLM 적용
            sample = film_layer(sample)
            down_block_res_samples += res_samples

        # 4. 중간 블록
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **kwargs
                )
            else:
                sample = self.mid_block(sample, emb)
            sample = self.mid_film(sample)

        # 5. 업샘플링
        for up_block, film_layer in zip(self.up_blocks, self.up_film_layers):
            res_samples = down_block_res_samples[-len(up_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    **kwargs
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                )
            # FiLM 적용
            sample = film_layer(sample)

        # 6. 후처리
        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample)


class GC_DDPM(nn.Module):
    def __init__(
        self,
        num_writers: int = 252,
        writer_embed_dim: int = 256,
        image_size: int = 64,
        max_width: int = 512,
        in_channels: int = 2,
        out_channels: int = 2,
        n_timesteps: int = 1000,
    ) -> None:
        super().__init__()
        
        self.num_writers = num_writers
        self.null_writer_id = num_writers  # null writer ID를 명시적으로 저장
        
        self.writer_embedding = nn.Embedding(num_writers + 1, writer_embed_dim)  # +1은 null token을 위한 것
        self.writer_proj = nn.Sequential(
            nn.Linear(writer_embed_dim, writer_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(writer_embed_dim * 4, writer_embed_dim * 4)
        )

        self.unet = UNet2DConditionModelWithFiLM(
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
            projection_class_embeddings_input_dim=None,  # 클래스 임베딩 비활성화
            norm_num_groups = 2
        )
        
        # self.unet = UNet2DConditionModel(
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
        #     time_embedding_type="positional",  # 명시적으로 positional 지정
        #     time_embedding_dim=512,  # 시간 임베딩 차원 명시적 지정
        #     projection_class_embeddings_input_dim=None  # 클래스 임베딩 비활성화
        # )
        
        self.n_timesteps = n_timesteps

    def forward(
        self, 
        x: torch.Tensor, 
        glyph: torch.Tensor, 
        timesteps: torch.Tensor, 
        writer_ids: Optional[torch.Tensor] = None, 
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """학습 시에는 랜덤하게 조건을 null로 설정"""

        x = x * 2.0 - 1.0
        glyph = glyph * 2.0 - 1.0
        
        if writer_ids is not None:
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
        
        # # U-Net forward
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
        glyph: torch.Tensor, 
        writer_ids: torch.Tensor, 
        scheduler: DDIMScheduler,
        use_guidance: bool = True, 
        content_scale: float = 3.0, 
        style_scale: float = 1.0, 
        num_inference_steps: int = 50,
        eta: float = 0.0  # DDIM 샘플링을 위한 eta 파라미터 추가
    ) -> torch.Tensor:
        """샘플링 메서드 - DDIM"""
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
