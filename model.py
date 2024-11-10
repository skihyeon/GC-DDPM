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
        # condition 차원 확인 및 처리
        if len(condition.shape) == 3:  # [batch, seq_len, hidden_dim]
            condition = condition.mean(dim=1)  # [batch, hidden_dim]
            
        params = self.condition_generator(condition)  # [batch, num_features*2]
        
        # 차원 확인
        if params.shape[1] < 2:
            raise ValueError(f"params shape {params.shape} is too small to chunk into gamma and beta")
            
        gamma, beta = torch.chunk(params, chunks=2, dim=-1)  # dim=-1로 마지막 차원에서 분할
        
        # x와 동일한 shape로 확장
        gamma = gamma.view(*gamma.shape, 1, 1).expand_as(x)
        beta = beta.view(*beta.shape, 1, 1).expand_as(x)
        
        # FiLM 레이어 통과 전후 값 비교
        # print(f"FiLM layer - Input tensor stats:")
        # print(f"  Mean: {x.mean():.4f}, Std: {x.std():.4f}")
        # print(f"  Min: {x.min():.4f}, Max: {x.max():.4f}")
        
        output = gamma * x + beta
        
        # print(f"FiLM layer - Output tensor stats:")
        # print(f"  Mean: {output.mean():.4f}, Std: {output.std():.4f}")
        # print(f"  Min: {output.min():.4f}, Max: {output.max():.4f}")
        
        return output
    
    
class FiLM_UNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # FiLM 레이어 추가
        self.film_layers_down = nn.ModuleList([
            FiLM(num_features=block.resnets[0].conv1.out_channels, condition_vector_dim=kwargs['cross_attention_dim'])
            for block in self.down_blocks
        ])
        
        self.film_layers_up = nn.ModuleList([
            FiLM(num_features=block.resnets[0].conv1.out_channels, condition_vector_dim=kwargs['cross_attention_dim'])
            for block in self.up_blocks
        ])
        
        # print("FiLM layers initialized:")
        # print(f"Down blocks FiLM layers: {len(self.film_layers_down)}")
        # print(f"Up blocks FiLM layers: {len(self.film_layers_up)}")

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
        # 1. 시간 임베딩
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. 사전 처리
        sample = self.conv_in(sample)

        # 3. 다운 블록
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

            # FiLM 레이어 적용
            # print(f"\nApplying FiLM layer in down block {down_block_idx}")
            sample = self.film_layers_down[down_block_idx](sample, encoder_hidden_states)

            down_block_res_samples += res_samples

        # 4. 중간 블록
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. 업 블록
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

            # FiLM 레이어 적용
            # print(f"\nApplying FiLM layer in up block {up_block_idx}")
            sample = self.film_layers_up[up_block_idx](sample, encoder_hidden_states)

        # 6. 후처리
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
        in_channels=2,  # x_t와 glyph 이미지가 concat 되었으므로 2
        out_channels=2,  # noise와 variance 예측,
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
        ## Time step embedding의 경우 UNet2DConditionModel에서 default로 정의되어 있음
        ## (time_proj): Timesteps()
        ## (time_embedding): TimestepEmbedding(
        ##   (linear_1): Linear(in_features=128, out_features=512, bias=True)
        ##   (act): SiLU()
        ##   (linear_2): Linear(in_features=512, out_features=512, bias=True)
        ##      )

        
        
        # print(f"UNet2DConditionModel: {self.unet}")
        # Writer Embedding
        self.writer_embedding = nn.Embedding(num_writers, writer_embed_dim)
        self.writer_norm = nn.LayerNorm(writer_embed_dim)
        
        self.writer_proj = nn.Linear(writer_embed_dim, writer_embed_dim * 4)
        
        
    
    def forward(self, x, glyph, timesteps, writer_ids=None, use_guidance=False, content_scale=3.0, style_scale=1.0):
        # print(f"Input x shape: {x.shape}")
        # print(f"Input glyph shape: {glyph.shape}")
        # Input normalization
        x = x * 2.0 - 1.0
        glyph = glyph * 2.0 - 1.0
        x_input = torch.cat([x, glyph], dim=1)
        # print(f"Concatenated input shape: {x_input.shape}")
        # Writer embedding with classifier-free guidance
        if use_guidance:
            # Unconditional branch
            uncond_ids = torch.zeros_like(writer_ids)
            uncond_emb = self.writer_embedding(uncond_ids)
            # print(f"Uncond embedding shape: {uncond_emb.shape}")
            uncond_emb = self.writer_norm(uncond_emb)
            uncond_emb = uncond_emb / torch.norm(uncond_emb, dim=-1, keepdim=True)
            uncond_emb = self.writer_proj(uncond_emb)
            uncond_hidden = uncond_emb.unsqueeze(1)
            # print(f"Uncond hidden shape: {uncond_hidden.shape}")
            
            # Conditional branch
            cond_emb = self.writer_embedding(writer_ids)
            # print(f"Cond embedding shape: {cond_emb.shape}")
            cond_emb = self.writer_norm(cond_emb)
            cond_emb = cond_emb / torch.norm(cond_emb, dim=-1, keepdim=True)
            cond_emb = self.writer_proj(cond_emb)
            cond_hidden = cond_emb.unsqueeze(1)
            # print(f"Cond hidden shape: {cond_hidden.shape}")
            
            # Style-free branch
            style_free_hidden = cond_hidden
            
            # Content-free branch
            # print("\nRunning content-free branch through UNet with FiLM layers...")
            content_free = self.unet(x_input, timesteps, encoder_hidden_states=uncond_hidden).sample
            content_free_pred, content_free_var = torch.split(content_free, 1, dim=1)
            # print(f"Content-free pred shape: {content_free_pred.shape}")
            # print(f"Content-free var shape: {content_free_var.shape}")
            
            # Style-free branch
            # print("\nRunning style-free branch through UNet with FiLM layers...")
            style_free = self.unet(x_input, timesteps, encoder_hidden_states=style_free_hidden).sample
            style_free_pred, style_free_var = torch.split(style_free, 1, dim=1)
            # print(f"Style-free pred shape: {style_free_pred.shape}")
            # print(f"Style-free var shape: {style_free_var.shape}")
            
            # Full conditional branch
            # print("\nRunning conditional branch through UNet with FiLM layers...")
            cond = self.unet(x_input, timesteps, encoder_hidden_states=cond_hidden).sample
            cond_pred, cond_var = torch.split(cond, 1, dim=1)
            # print(f"Conditional pred shape: {cond.shape}")
            # print(f"Conditional cond_pred shape: {cond_pred.shape}")
            # print(f"Conditional cond_var shape: {cond_var.shape}")
            
            # Combined guidance
            noise_pred = (cond_pred * content_scale) + (style_free_pred * style_scale) - ((content_scale + style_scale) * content_free_pred)
            var_pred = (cond_var * content_scale) + (style_free_var * style_scale) - ((content_scale + style_scale) * content_free_var)
        else:
            w_emb = self.writer_embedding(writer_ids)
            # print(f"Writer embedding shape: {w_emb.shape}")
            w_emb = self.writer_norm(w_emb)
            w_emb = w_emb / torch.norm(w_emb, dim=-1, keepdim=True)
            w_emb = self.writer_proj(w_emb)
            encoder_hidden_states = w_emb.unsqueeze(1)
            # print(f"Encoder hidden states shape: {encoder_hidden_states.shape}")
            
            # print("\nRunning through UNet with FiLM layers...")
            pred = self.unet(x_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred, var_pred = torch.split(pred, 1, dim=1)
            # print(f"Pred shape: {pred.shape}")
            # noise_pred, var_pred = self.unet(x_input, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            # print(f"Noise pred shape: {noise_pred.shape}")
            # print(f"Var pred shape: {var_pred.shape}")
    
        # # Split noise and variance predictions
        
        
        return noise_pred, var_pred
