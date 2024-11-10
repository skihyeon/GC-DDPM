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
from PIL import ImageDraw, ImageFont
import time

def generate_samples(config, epoch, model, noise_scheduler, batch_size=5):
    """
    글리프 이미지와 writer ID를 기반으로 손글씨 이미지를 생성합니다.
    
    Args:
        config (IAMTrainingConfig): 설정 객체
        epoch (int): 현재 에폭
        model (GC_DDPM): 학습된 모델
        noise_scheduler: 노이즈 스케줄러
        batch_size (int): 한 번에 생성할 이미지 수
    """
    model.eval()
    
    # 글리프 이미지와 writer ID 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0)  # [0,1] -> [-1,1] 범위로 변환
    ])
    
    # writer IDs 로드
    with open(os.path.join(config.data_dir, 'writer_ids.txt'), 'r') as f:
        writer_ids = [line.strip() for line in f.readlines()]
    
    # 글리프 이미지 경로에서 사용 가능한 텍스트 목록 가져오기
    glyph_dir = os.path.join(config.data_dir, 'glyphs')
    available_texts = [f.replace('.png', '') for f in os.listdir(glyph_dir) if f.endswith('.png') and '_' not in f]
    
    # 랜덤 선택
    selected_texts = np.random.choice(available_texts, batch_size, replace=False)
    selected_writer_ids = np.random.choice(len(writer_ids), batch_size, replace=False)
    
    # 글리프 이미지 로드 및 전처리
    glyph_images = []
    for text in selected_texts:
        img_path = os.path.join(glyph_dir, f"{text}.png")
        img = Image.open(img_path).convert('L')
        img = img.resize((config.max_width, config.image_size), Image.LANCZOS)
        img_tensor = transform(img)
        glyph_images.append(img_tensor)
    
    glyph_images = torch.stack(glyph_images).to(config.device)
    writer_ids_tensor = torch.tensor(selected_writer_ids, dtype=torch.long).to(config.device)
    
    with torch.no_grad():
        # 노이즈 생성
        noise = torch.randn(
            batch_size,
            1,
            config.image_size,
            config.max_width
        ).to(config.device)
        
        # 샘플링
        sample = noise
        for t in tqdm(noise_scheduler.timesteps):
            noise_pred, _ = model(
                sample,
                glyph_images,
                t,
                writer_ids_tensor,
                use_guidance=True,
                content_scale=3.0,
                style_scale=1.0
            )
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
    
    # 결과 이미지 생성
    samples = []
    for i in range(batch_size):
        # 글리프 이미지와 생성된 이미지를 나란히 배치
        glyph = ((glyph_images[i].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        generated = ((sample[i].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        combined = np.concatenate([glyph[0], generated[0]], axis=1)
        samples.append(combined)
    
    # 모든 샘플을 세로로 쌓기
    final_image = np.concatenate(samples, axis=0)
    
    # 결과 저장
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    Image.fromarray(final_image).save(
        os.path.join(config.output_dir, 'samples', f'epoch_{epoch:04d}.png')
    )

def train():
    config = IAMTrainingConfig()

    # 데이터셋 및 데이터로더 설정
    dataset = IAMDataset(config)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # 모델 및 스케줄러 설정
    model = GC_DDPM(
        num_writers=config.num_writers,
        writer_embed_dim=256,
        image_size=config.image_size,
        max_width=config.max_width,
        in_channels=2,  # x_t와 glyph 이미지가 concat 되었으므로 2
        out_channels=2,  # noise와 variance 예측
    ).to(config.device)
    
    # 체크포인트 로드 로직 수정
    start_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
        if os.path.exists(checkpoint_path):
            print(f"체크포인트를 로드합니다: {checkpoint_path}")
            # map_location 파라미터 추가
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"에폭 {start_epoch}부터 학습을 재개합니다.")
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon"
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    
    # optimizer와 scheduler 상태 복원
    if config.resume_from_checkpoint and os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Accelerator 설정
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 학습 루프
    betas = noise_scheduler.betas.to(config.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
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
                # 논문에서 제안한 guidance scale 사용
                content_scale = 3.0  # content guidance scale (γ)
                style_scale = 1.0    # style guidance scale (η)
            else:
                content_scale = 0.0
                style_scale = 0.0

            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0, 
                noise_scheduler.config.num_train_timesteps, 
                (clean_images.shape[0],), 
                device=clean_images.device,
                dtype=torch.long
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                noise_pred, var_pred = model(
                    noisy_images, 
                    glyph_images, 
                    timesteps, 
                    writer_ids,
                    use_guidance=use_guidance,
                    content_scale=content_scale,
                    style_scale=style_scale
                )
                
                # Noise prediction loss
                noise_loss = F.mse_loss(noise_pred, noise)
                
                # Variance prediction loss
                betas_t = betas[timesteps]
                alphas_cumprod_t = alphas_cumprod[timesteps]
                alphas_cumprod_prev = alphas_cumprod[torch.clamp(timesteps - 1, min=0)]
                
                target_var = betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)
                var_pred = var_pred.mean([2, 3])
                var_loss = F.mse_loss(var_pred.squeeze(1), target_var)
                
                loss = noise_loss + 0.1 * var_loss
                
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
                accelerator.unwrap_model(model), 
                noise_scheduler,
            )
            
            # 모델 저장
            os.makedirs(config.checkpoint_dir, exist_ok=True) 
            torch.save({
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': loss.item(),
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))

if __name__ == "__main__":
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    train()