import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from typing import Union, Optional, Dict, Any, Tuple
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput

class FiLM(nn.Module):
    def __init__(self, num_features, condition_vector_dim):
        super(FiLM, self).__init__()
        self.num_features = num_features
        self.condition_generator = nn.Linear(condition_vector_dim, num_features * 2)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.mean(dim=1)
            
        params = self.condition_generator(condition)
        
        if params.shape[1] < 2:
            raise ValueError(f"params shape {params.shape} is too small to chunk into gamma and beta")
            
        gamma, beta = torch.chunk(params, chunks=2, dim=-1)
        
        gamma = gamma.view(*gamma.shape, 1, 1).expand_as(x)
        beta = beta.view(*beta.shape, 1, 1).expand_as(x)
        
        output = gamma * x + beta
        
        return output
    
    
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
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

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
        
        self.unet = FiLM_UNet2DConditionModel(
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

        self.writer_embedding = nn.Embedding(num_writers, writer_embed_dim)
        self.writer_norm = nn.LayerNorm(writer_embed_dim)
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
        
    def compute_variance(self, pred_v, t):
        # v_θ를 사용하여 분산 계산
        beta_t = self.betas[t]
        beta_tilde_t = self.beta_tilde[t]
        
        variance = torch.exp(
            pred_v * torch.log(beta_t) + (1 - pred_v) * torch.log(beta_tilde_t)
        )
        return variance
        
    def forward(self, x, glyph, timesteps, writer_ids=None, use_guidance=False, content_scale=3.0, style_scale=1.0):
        # 입력 정규화
        x = x * 2.0 - 1.0
        glyph = glyph * 2.0 - 1.0
        
        # 기본 입력 생성
        x_input = torch.cat([x, glyph], dim=1)
        uncond_glyph = torch.zeros_like(glyph)
        x_input_style = torch.cat([x, uncond_glyph], dim=1)

        if use_guidance:
            # 1. Unconditional embedding (no writer, no glyph)
            uncond_ids = torch.zeros_like(writer_ids)
            uncond_emb = self.writer_embedding(uncond_ids)
            uncond_emb = self.writer_norm(uncond_emb)
            uncond_emb = uncond_emb / torch.norm(uncond_emb, dim=-1, keepdim=True)
            uncond_emb = self.writer_proj(uncond_emb)
            uncond_hidden = uncond_emb.unsqueeze(1)
            
            # 2. Conditional writer embedding
            cond_emb = self.writer_embedding(writer_ids)
            cond_emb = self.writer_norm(cond_emb)
            cond_emb = cond_emb / torch.norm(cond_emb, dim=-1, keepdim=True)
            cond_emb = self.writer_proj(cond_emb)
            cond_hidden = cond_emb.unsqueeze(1)
            
            # ε_θ(x_t, ∅, ∅) - 완전 unconditional
            x_input_uncond = torch.cat([x, uncond_glyph], dim=1)
            full_uncond = self.unet(x_input_uncond, timesteps, encoder_hidden_states=uncond_hidden).sample
            full_uncond_noise, full_uncond_v = torch.split(full_uncond, 1, dim=1)
            
            # ε_θ(x_t, g, ∅) - content-only
            content_only = self.unet(x_input, timesteps, encoder_hidden_states=uncond_hidden).sample
            content_noise, content_v = torch.split(content_only, 1, dim=1)
            
            # ε_θ(x_t, ∅, w) - style-only
            style_only = self.unet(x_input_style, timesteps, encoder_hidden_states=cond_hidden).sample
            style_noise, style_v = torch.split(style_only, 1, dim=1)
            
            # ε_θ(x_t, g, w) - full conditional
            full_cond = self.unet(x_input, timesteps, encoder_hidden_states=cond_hidden).sample
            full_cond_noise, full_cond_v = torch.split(full_cond, 1, dim=1)
            
            # Classifier-free guidance 적용
            noise_pred = (
                full_cond_noise + 
                content_scale * content_noise + 
                style_scale * style_noise - 
                (content_scale + style_scale) * full_uncond_noise
            )
            
            # Variance prediction도 동일한 방식으로 결합
            var_pred = (
                full_cond_v + 
                content_scale * content_v + 
                style_scale * style_v - 
                (content_scale + style_scale) * full_uncond_v
            )
            
        else:
            # 기본 조건부 생성
            w_emb = self.writer_embedding(writer_ids)
            w_emb = self.writer_norm(w_emb)
            w_emb = w_emb / torch.norm(w_emb, dim=-1, keepdim=True)
            w_emb = self.writer_proj(w_emb)
            encoder_hidden_states = w_emb.unsqueeze(1)
            
            pred = self.unet(x_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred, var_pred = torch.split(pred, 1, dim=1)
        
        return noise_pred, var_pred

    @torch.no_grad()
    def sample(self, glyph, writer_ids, use_guidance=True, content_scale=3.0, style_scale=1.0):
        device = next(self.parameters()).device
        batch_size = glyph.shape[0]
        
        # 초기 노이즈에서 시작
        x = torch.randn((batch_size, 1, glyph.shape[2], glyph.shape[3]), device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.n_timesteps)):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 노이즈와 분산 예측
            noise_pred, var_pred = self.forward(
                x, glyph, timesteps, writer_ids, 
                use_guidance, content_scale, style_scale
            )
            
            # 평균과 분산 계산
            mean = self.compute_mean(x, noise_pred, t)
            variance = self.compute_variance(var_pred, t)
            
            # 노이즈 추가 (t > 0인 경우에만)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(variance) * noise
            else:
                x = mean
                
        # [-1, 1] 범위를 [0, 1] 범위로 변환
        x = (x + 1) / 2
        return x.clamp(0, 1)