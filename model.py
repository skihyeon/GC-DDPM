import torch
import torch.nn as nn
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from typing import Union, Optional, Dict, Any, Tuple
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from tqdm import tqdm
from config import IAMTrainingConfig as cfg


class FiLM(nn.Module):
    def __init__(self, num_features, condition_vector_dim):
        super().__init__()
        self.num_features = num_features
        self.condition_generator = nn.Linear(condition_vector_dim, num_features * 2)

    def forward(self, x, condition):
        # condition이 None인 경우 처리
        if condition is None:
            # identity mapping 반환
            return x
            
        # L2 normalization 적용
        if len(condition.shape) == 3:
            condition = condition.mean(dim=1)
        condition = condition / torch.norm(condition, dim=-1, keepdim=True)
        
        params = self.condition_generator(condition)
        gamma, beta = torch.chunk(params, chunks=2, dim=-1)
        
        gamma = gamma.view(*gamma.shape, 1, 1).expand_as(x)
        beta = beta.view(*beta.shape, 1, 1).expand_as(x)
        
        return gamma * x + beta
    
    
class FiLM_UNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.film_layers_down = nn.ModuleList([
            FiLM(num_features=block.resnets[0].conv1.out_channels, condition_vector_dim=kwargs['cross_attention_dim'])
            for block in self.down_blocks
        ])
        
        self.film_layers_up = nn.ModuleList([
            FiLM(num_features=block.resnets[0].conv1.out_channels, condition_vector_dim=kwargs['cross_attention_dim'])
            for block in self.up_blocks
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
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        if encoder_hidden_states is None:
            batch_size = sample.shape[0]
            # 모델의 cross_attention_dim 크기에 맞는 제로 텐서 생성
            encoder_hidden_states = torch.zeros(
                (batch_size, 1, self.config.cross_attention_dim),  
                device=sample.device,
                dtype=sample.dtype
            )

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for down_block_idx, down_block in enumerate(self.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)

            sample = self.film_layers_down[down_block_idx](sample, encoder_hidden_states)

            down_block_res_samples += res_samples

        if self.mid_block is not None:
            if hasattr(self.mid_block, 'has_cross_attention') and self.mid_block.has_cross_attention:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = self.mid_block(sample, emb) 
        for up_block_idx, up_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(up_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(up_block.resnets)]

            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                )
            else:
                sample = up_block(hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples)

            sample = self.film_layers_up[up_block_idx](sample, encoder_hidden_states)

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
        betas=(1e-4, 0.02),  # β_start와 β_end 값 추가
        n_timesteps=1000,    # timestep 수 추가
    ):
        super().__init__()
        
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
        #     time_embedding_act_fn="swish"
        # )

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
            time_embedding_act_fn="swish"
        )


        # self.writer_embedding = nn.Embedding(num_writers, writer_embed_dim)
        # self.writer_norm = nn.LayerNorm(writer_embed_dim)
        # self.writer_proj = nn.Linear(writer_embed_dim, writer_embed_dim * 4)
            # Writer embedding (논문에 명시된 대로)
        self.writer_embedding = nn.Embedding(num_writers, writer_embed_dim)
        self.writer_norm = lambda x: x / torch.norm(x, dim=-1, keepdim=True)
        self.writer_proj = nn.Linear(writer_embed_dim, writer_embed_dim * 4)
        
        # DDPM 관련 파라미터 초기화
        self.n_timesteps = n_timesteps
        self.register_buffer('betas', torch.linspace(betas[0], betas[1], n_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # α̅_t-1 계산을 위한 shifted alphas_cumprod
        alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # β̃_t 계산
        self.register_buffer('beta_tilde', 
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        
    def compute_mean(self, x_t, pred_noise, t):
        alpha_t = self.alphas[t]
        alpha_t_bar = self.alphas_cumprod[t]
        
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_t_bar) * pred_noise
        )
        return mean
        
    def compute_variance(self, var_pred, timesteps):
        """논문 수식 (6)에 따른 variance 계산"""
        beta_t = self.betas[timesteps]
        beta_tilde_t = self.beta_tilde[timesteps]
        
        # Σθ(xt, g, w) = exp(νθ(xt, g, w) log βt + (1 - νθ(xt, g, w)) log β̃t)
        var = torch.exp(
            var_pred * torch.log(beta_t) + 
            (1 - var_pred) * torch.log(beta_tilde_t)
        )
        return var
        
    def forward(self, x, glyph, timesteps, writer_ids=None, use_guidance=False, content_scale=3.0, style_scale=1.0):
        # 입력 정규화 - clamp 추가
        x = torch.clamp(x * 2.0 - 1.0, -1.0, 1.0)
        glyph = torch.clamp(glyph * 2.0 - 1.0, -1.0, 1.0)
        
        x_input = torch.cat([x, glyph], dim=1)
        
        # 빈 인코더 상태 생성 (재사용)
        empty_encoder_states = torch.zeros(
            (x_input.shape[0], 1, self.unet.config.cross_attention_dim),
            device=x_input.device,
            dtype=x_input.dtype
        )
        
        if use_guidance:
            null_glyph = torch.zeros_like(glyph)
            w_emb = self.writer_embedding(writer_ids)
            
            # writer embedding 정규화에 epsilon 추가
            w_emb = w_emb / (torch.norm(w_emb, dim=-1, keepdim=True) + 1e-8)
            w_emb = self.writer_proj(w_emb)
            
            # NaN 체크 및 처리
            if torch.isnan(w_emb).any():
                w_emb = torch.nan_to_num(w_emb, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 1. Full condition
            pred_full = self.unet(x_input, timesteps, encoder_hidden_states=w_emb.unsqueeze(1)).sample
            
            # 2. Content only
            pred_content = self.unet(x_input, timesteps, encoder_hidden_states=empty_encoder_states).sample
            
            # 3. Style only
            x_input_style = torch.cat([x, null_glyph], dim=1)
            pred_style = self.unet(x_input_style, timesteps, encoder_hidden_states=w_emb.unsqueeze(1)).sample
            
            # 4. No condition
            pred_uncond = self.unet(x_input_style, timesteps, encoder_hidden_states=empty_encoder_states).sample
            
            # Guidance scale 값 제한
            content_scale = torch.clamp(torch.tensor(content_scale), -10.0, 10.0)
            style_scale = torch.clamp(torch.tensor(style_scale), -10.0, 10.0)
            
            # Classifier-free guidance 적용
            noise_pred = (
                pred_full + 
                content_scale * pred_content + 
                style_scale * pred_style - 
                (content_scale + style_scale) * pred_uncond
            )
            
            # NaN 체크 및 처리
            noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)
            
            _, var_pred = torch.chunk(pred_full, 2, dim=1)
            noise_pred, _ = torch.chunk(noise_pred, 2, dim=1)
            
        else:
            if writer_ids is not None:
                w_emb = self.writer_embedding(writer_ids)
                w_emb = w_emb / (torch.norm(w_emb, dim=-1, keepdim=True) + 1e-8)
                w_emb = self.writer_proj(w_emb)
                
                # NaN 체크 및 처리
                if torch.isnan(w_emb).any():
                    w_emb = torch.nan_to_num(w_emb, nan=0.0, posinf=1.0, neginf=-1.0)
                
                encoder_states = w_emb.unsqueeze(1)
            else:
                encoder_states = empty_encoder_states
            
            pred = self.unet(x_input, timesteps, encoder_hidden_states=encoder_states).sample
            noise_pred, var_pred = torch.chunk(pred, 2, dim=1)
        
        # 최종 출력값 clamp 및 NaN 처리
        noise_pred = torch.clamp(noise_pred, -10.0, 10.0)
        var_pred = torch.clamp(var_pred, -10.0, 10.0)
        
        noise_pred = torch.nan_to_num(noise_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        var_pred = torch.nan_to_num(var_pred, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return noise_pred, var_pred

    def predict_start_from_noise(self, x_t, t, noise):
        """노이즈로부터 x_0 예측"""
        alpha_cumprod_t = self.alphas_cumprod[t]
        return (
            torch.sqrt(1. / alpha_cumprod_t) * x_t -
            torch.sqrt(1. / alpha_cumprod_t - 1) * noise
        )

    @torch.no_grad()
    def ddim_sample(
        self,
        glyph,
        writer_ids,
        use_guidance=True,
        content_scale=3.0,
        style_scale=1.0,
        num_inference_steps=50,
        eta=0.0  # η=0이면 DDIM, η=1이면 DDPM
    ):
        # device = next(self.parameters()).device
        device = cfg.device
        batch_size = glyph.shape[0]
        
        # 초기 노이즈에서 시작
        x = torch.randn((batch_size, 1, glyph.shape[2], glyph.shape[3]), device=device)
        
        # DDIM 타임스텝 시퀀스 생성
        timesteps = torch.linspace(self.n_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(tqdm(timesteps, desc="DDIM sampling")):
            # 다음 타임스텝 계산
            next_t = timesteps[i + 1] if i < len(timesteps) - 1 else torch.tensor(-1)
            
            # 현재 타임스텝의 배치 생성
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 노이즈 예측
            noise_pred, _ = self.forward(
                x, glyph, t_batch, writer_ids,
                use_guidance, content_scale, style_scale
            )
            
            # x_0 예측
            pred_original_sample = self.predict_start_from_noise(x, t, noise_pred)
            
            if next_t >= 0:
                alpha_cumprod_t = self.alphas_cumprod[t]
                alpha_cumprod_next_t = self.alphas_cumprod[next_t]
                
                # DDIM 업데이트 공식
                sigma = eta * torch.sqrt(
                    (1 - alpha_cumprod_next_t) / (1 - alpha_cumprod_t) *
                    (1 - alpha_cumprod_t / alpha_cumprod_next_t)
                )
                
                # 노이즈 샘플링
                noise = torch.randn_like(x) if eta > 0 else 0.
                
                # DDIM 업데이트
                x = torch.sqrt(alpha_cumprod_next_t) * pred_original_sample + \
                    torch.sqrt(1 - alpha_cumprod_next_t - sigma ** 2) * noise_pred + \
                    sigma * noise
            else:
                # 마지막 스텝
                x = pred_original_sample
        
        x = (x + 1) / 2
        return x.clamp(0, 1)
    @torch.no_grad()
    def sample(self, glyph, writer_ids, use_guidance=True, content_scale=3.0, style_scale=1.0, num_inference_steps=50):
        """기존 sample 메서드를 DDIM을 사용하도록 수정"""
        return self.ddim_sample(
            glyph=glyph,
            writer_ids=writer_ids,
            use_guidance=use_guidance,
            content_scale=content_scale,
            style_scale=style_scale,
            num_inference_steps=num_inference_steps,
            eta=0.0  # DDIM 사용 (η=0)
        )
