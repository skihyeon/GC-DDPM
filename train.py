import torch
from hwdataset import IAMDataset
from model import GC_DDPM
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np
from config import IAMTrainingConfig
import torchvision.transforms as transforms

def generate_samples(config, epoch, model, batch_size=5):
    """
    수정된 GC_DDPM 모델을 사용하여 샘플을 생성합니다.
    논문 설정: DDIM 샘플러, 50 스텝, content_scale=3.0, style_scale=1.0
    """
    model.eval()
    
    # 데이터 준비를 한 번에 처리
    glyphs_dir = os.path.join(config.data_dir, 'glyphs')
    available_texts = [f.replace('.png', '') for f in os.listdir(glyphs_dir) 
                      if f.endswith('.png') and '_' not in f 
                      and len(f.replace('.png', '')) >= 3  # 3글자 이상
                      and not all(not c.isalnum() for c in f.replace('.png', ''))]  # 특수문자로만 이루어지지 않은 것
    selected_indices = np.random.choice(len(available_texts), batch_size, replace=False)
    
    # 글리프 이미지 한 번에 처리
    glyph_images = []
    transform = transforms.Compose([transforms.ToTensor()])
    
    for idx in selected_indices:
        text = available_texts[idx]
        img_path = os.path.join(glyphs_dir, f"{text}.png")
        img = Image.open(img_path).convert('L')
        img = img.resize((config.max_width, config.image_size), Image.LANCZOS)
        glyph_images.append(transform(img))
    
    glyph_images = torch.stack(glyph_images).to(config.device)
    
    # 각 writer ID에 대해 여러 번 생성
    all_samples = []
    for writer_id in range(1):  # 4명의 writer에 대해 생성
        writer_ids_tensor = torch.full((batch_size,), writer_id, device=config.device)
        
        # DDIM 샘플링으로 이미지 생성
        with torch.no_grad():
            samples = model.ddim_sample(
                glyph=glyph_images,
                writer_ids=writer_ids_tensor,
                use_guidance=True,
                content_scale=3.0,
                style_scale=1.0,
                num_inference_steps=100,
                eta=0.0
            )
            
            # 결과 이미지 처리
            samples_with_glyphs = torch.cat([glyph_images, samples], dim=3)  # 가로로 연결
            samples_with_glyphs = samples_with_glyphs.squeeze(1)  # [B, H, W]
            all_samples.append(samples_with_glyphs)
    
    # 모든 샘플을 하나의 이미지로 결합
    all_samples = torch.cat(all_samples, dim=0)  # [4*B, H, W]
    
    # numpy로 변환하고 값 범위 조정
    final_image = all_samples.cpu().numpy()  # [4*B, H, W]
    final_image = (final_image * 255).astype(np.uint8)
    
    # 이미지들을 세로로 쌓기 (5개의 이미지를 하나로 합치기)
    H, W = final_image.shape[1], final_image.shape[2]
    combined_image = np.zeros((H * final_image.shape[0], W), dtype=np.uint8)
    
    # 각 이미지를 세로로 쌓기
    for idx in range(final_image.shape[0]):
        combined_image[idx * H:(idx + 1) * H, :] = final_image[idx]
    
    # 저장
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    Image.fromarray(combined_image, mode='L').save(
        os.path.join(config.output_dir, 'samples', f'epoch_{epoch:04d}.png')
    )

def train(config):
    # 데이터셋 및 데이터로더 설정
    dataset = IAMDataset(config)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 모델 초기화 수정
    model = GC_DDPM(
        num_writers=config.num_writers,
        writer_embed_dim=256,
        image_size=config.image_size,
        max_width=config.max_width,
        in_channels=2,
        out_channels=2,
        betas=(config.beta_start, config.beta_end),
        n_timesteps=config.num_train_timesteps
    ).to(config.device)
    
    # 체크포인트 로드 로직 수정
    start_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
        if os.path.exists(checkpoint_path):
            print(f"load checkpoint: {checkpoint_path}")
            try:
                # 먼저 weights_only=True로 시도
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=lambda storage, loc: storage.cuda(config.gpu_num),
                    weights_only=True
                )
            except:
                # 실패하면 weights_only=False로 재시도
                print("Retrying checkpoint load with weights_only=False")
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=lambda storage, loc: storage.cuda(config.gpu_num),
                    weights_only=False
                )
                
            model.load_state_dict(checkpoint, strict=False)
            
            # checkpoint 파일명에서 epoch 파싱
            epoch = int(config.checkpoint_name.split('_')[-1].split('.')[0])
            start_epoch = epoch + 1
            
            print(f"resume from epoch {start_epoch}")
            generate_samples(
                config,
                epoch,
                model,
            )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    
    # Accelerator 설정
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 학습 루프
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            glyph_images = batch['glyph']
            writer_ids = batch['writer_id']
            
            # 10% 확률로 guidance 사용
            use_guidance = torch.rand(1).item() < 0.1
            
            if use_guidance:
                content_scale = 3.0
                style_scale = 1.0
            else:
                content_scale = 0.0
                style_scale = 0.0

            # 타임스텝 샘플링
            timesteps = torch.randint(
                0, 
                model.n_timesteps, 
                (clean_images.shape[0],), 
                device=clean_images.device,
                dtype=torch.long
            )
            
            # 노이즈 추가
            noise = torch.randn_like(clean_images)
            alpha_t = model.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            noisy_images = torch.sqrt(alpha_t) * clean_images + torch.sqrt(1 - alpha_t) * noise
            
            with accelerator.accumulate(model):
                # 모델 예측
                noise_pred, var_pred = model(
                    noisy_images, 
                    glyph_images, 
                    timesteps, 
                    writer_ids,
                    use_guidance=use_guidance,
                    content_scale=content_scale,
                    style_scale=style_scale
                )
                if torch.isnan(noise_pred).any() or torch.isnan(var_pred).any():
                    print("Model output contains NaN values")
                    continue
                # Noise prediction loss (MSE)
                noise_loss = F.mse_loss(noise_pred, noise)

                # Variance prediction loss (논문 수식 6)
                # Σθ(xt, g, w) = exp(νθ(xt, g, w) log βt + (1 - νθ(xt, g, w)) log β̃t)
                beta_t = model.betas[timesteps]
                beta_tilde_t = model.beta_tilde[timesteps]

                # numerical stability를 위한 clamp 추가
                log_beta_t = torch.log(torch.clamp(beta_t, min=1e-8))
                log_beta_tilde_t = torch.log(torch.clamp(beta_tilde_t, min=1e-8))

                # var_pred를 [0,1] 범위로 제한
                var_pred = var_pred.mean([2, 3]).squeeze(1)
                var_pred = torch.sigmoid(var_pred)  # νθ를 [0,1] 범위로 제한

                # target variance 계산
                target_var = torch.exp(var_pred * log_beta_t + (1 - var_pred) * log_beta_tilde_t)

                # variance loss (MSE)
                var_loss = F.mse_loss(var_pred, target_var)

                # total loss (논문에서는 구체적인 가중치를 명시하지 않음)
                loss = noise_loss + 0.01 * var_loss  # variance loss의 가중치를 0.1에서 0.01로 줄임
                
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=loss.item(),
                noise_loss=noise_loss.item(),
                var_loss=var_loss.item()
            )
        
        # 샘플 생성
        if accelerator.is_main_process:
            generate_samples(
                config, 
                epoch, 
                accelerator.unwrap_model(model)
            )
            
            # 모델 저장
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save(
                accelerator.unwrap_model(model).state_dict(),
                os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            )

if __name__ == "__main__":
    config = IAMTrainingConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    torch.cuda.set_device(config.gpu_num)
    train(config)