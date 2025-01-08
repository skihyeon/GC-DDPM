import torch
from hwdataset import IAMDataset
from model import GC_DDPM
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np
from config import TrainingConfig
import torchvision.transforms as transforms

def generate_samples(config, epoch, model, scheduler):
    """샘플 생성 함수"""
    model.eval()
    
    # 샘플 생성을 위한 설정
    writers_num, batch_size = 4, 5
    
    # 글리프 데이터 준비
    glyphs_dir = os.path.join(config.data_dir, 'glyphs')
    available_texts = [f.replace('.png', '') for f in os.listdir(glyphs_dir) 
                      if f.endswith('.png') and '_' not in f 
                      and len(f.replace('.png', '')) >= 3
                      and not all(not c.isalnum() for c in f.replace('.png', ''))]
    
    selected_indices = np.random.choice(len(available_texts), batch_size, replace=False)
    
    # 글리프 이미지 변환 및 로드
    transform = transforms.Compose([transforms.ToTensor()])
    glyph_images = []
    
    for idx in selected_indices:
        text = available_texts[idx]
        img = Image.open(os.path.join(glyphs_dir, f"{text}.png")).convert('L')
        img = img.resize((config.max_width, config.image_size), Image.LANCZOS)
        glyph_images.append(transform(img))
    
    # 배치 구성
    glyph_images = torch.stack(glyph_images).repeat(writers_num, 1, 1, 1).to(config.device)
    writer_ids = torch.arange(writers_num, device=config.device).repeat_interleave(batch_size)
    
    # 샘플 생성
    with torch.no_grad():
        samples = model.sample(
            glyph=glyph_images,
            writer_ids=writer_ids,
            scheduler=scheduler,
            use_guidance=True,
            content_scale=config.content_scale,
            style_scale=config.style_scale,
            num_inference_steps=config.num_inference_steps
        )
        
        samples_with_glyphs = torch.cat([glyph_images, samples], dim=3).squeeze(1)
    
    # 이미지 후처리 및 저장
    final_image = (samples_with_glyphs.cpu().numpy() * 255).astype(np.uint8)
    H, W = final_image.shape[1], final_image.shape[2]
    combined_image = np.zeros((H * writers_num, W * batch_size), dtype=np.uint8)
    
    # 글리프별로 writer's id 순으로 정렬
    for glyph_idx in range(batch_size):
        for writer_idx in range(writers_num):
            idx = writer_idx * batch_size + glyph_idx
            row = writer_idx
            col = glyph_idx
            combined_image[row * H:(row + 1) * H, col * W:(col + 1) * W] = final_image[idx]
    
    os.makedirs(os.path.join(config.output_dir, 'samples'), exist_ok=True)
    Image.fromarray(combined_image, mode='L').save(
        os.path.join(config.output_dir, 'samples', f'epoch_{epoch:04d}.png')
    )

def compute_loss(noise_pred, noise, var_pred, timesteps, noise_scheduler):
    """하이브리드 목적 함수 계산"""
    # 노이즈 예측 손실
    noise_loss = F.mse_loss(noise_pred, noise)
    
    # 분산 예측 손실
    beta_t = noise_scheduler.betas[timesteps]
    alpha_t = 1 - beta_t
    alpha_bar_t = noise_scheduler.alphas_cumprod[timesteps]
    beta_tilde_t = beta_t * (1 - alpha_bar_t) / (1 - alpha_t)

    log_beta_t = torch.log(torch.clamp(beta_t, min=1e-8))
    log_beta_tilde_t = torch.log(torch.clamp(beta_tilde_t, min=1e-8))

    var_pred = torch.sigmoid(var_pred.mean([2, 3]).squeeze(1))
    target_var = torch.exp(var_pred * log_beta_t + (1 - var_pred) * log_beta_tilde_t)
    var_loss = F.mse_loss(var_pred, target_var)

    return noise_loss + 0.01 * var_loss

def train_step(model, batch, noise_scheduler):
    """단일 학습 스텝 수행"""
    clean_images = batch['images']
    glyph_images = batch['glyph']
    writer_ids = batch['writer_id']
    
    # 각각 10% 확률로 null 조건 적용
    batch_size = clean_images.shape[0]
    use_null_glyph = torch.rand(batch_size, device=clean_images.device) < 0.1
    use_null_writer = torch.rand(batch_size, device=clean_images.device) < 0.1
    
    # Null 조건 적용
    current_glyph = torch.where(
        use_null_glyph.view(-1, 1, 1, 1),
        torch.zeros_like(glyph_images),
        glyph_images
    )
    current_writer_ids = torch.where(
        use_null_writer,
        torch.ones_like(writer_ids) * model.num_writers,
        writer_ids
    )
    
    # 노이즈 추가
    noise = torch.randn_like(clean_images)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                            (batch_size,), device=clean_images.device)
    noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
    
    # 모델 예측
    noise_pred, var_pred = model(
        noisy_images, 
        current_glyph, 
        timesteps, 
        current_writer_ids,
    )
    
    # Loss 계산
    loss = compute_loss(noise_pred, noise, var_pred, timesteps, noise_scheduler)
    
    return loss



def train(config):
    # 데이터 및 모델 초기화
    dataset = IAMDataset(config)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    # 학습용 DDPM 스케줄러
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon"
    )
    
    # 샘플링용 DDIM 스케줄러 - 동일한 베타 스케줄 사용
    sampling_scheduler = DDIMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        clip_sample=False,
        prediction_type="epsilon",
        steps_offset=1  # 중요: DDIM에서 타임스텝 오프셋 설정
    )
    
    # 스케줄러의 beta 값들을 GPU로 이동
    noise_scheduler.betas = noise_scheduler.betas.to(config.device)
    noise_scheduler.alphas = noise_scheduler.alphas.to(config.device)
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(config.device)
    
    sampling_scheduler.betas = sampling_scheduler.betas.to(config.device)
    sampling_scheduler.alphas = sampling_scheduler.alphas.to(config.device)
    sampling_scheduler.alphas_cumprod = sampling_scheduler.alphas_cumprod.to(config.device)
    
    # Accelerator 설정을 먼저 하고
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )

    # 모델 초기화
    model = GC_DDPM(
        num_writers=config.num_writers,
        writer_embed_dim=256,
        image_size=config.image_size,
        max_width=config.max_width,
        in_channels=2,
        out_channels=2,
        n_timesteps=config.num_train_timesteps
    )

    # 체크포인트 로드
    start_epoch = 0
    if config.resume_from_checkpoint:
        checkpoint_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
        if os.path.exists(checkpoint_path):
            print(f"체크포인트 로드: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']
            
            model_state_dict = model.state_dict()
            filtered_checkpoint = {k: v for k, v in checkpoint.items() 
                                if k in model_state_dict}
            model.load_state_dict(filtered_checkpoint, strict=False)
            
            start_epoch = int(config.checkpoint_name.split('_')[-1].split('.')[0]) + 1
            print(f"{start_epoch} 에폭부터 학습 재개")

    # 옵티마이저 및 스케줄러 설정
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay, 
        eps=config.eps
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    
    # Accelerator prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 체크포인트 로드 후 샘플 생성
    if config.resume_from_checkpoint and os.path.exists(checkpoint_path):
        generate_samples(config, start_epoch-1, accelerator.unwrap_model(model), sampling_scheduler)

    # 학습 루프
    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                # 학습 스텝 수행
                loss = train_step(model, batch, noise_scheduler)
                
                # Optimization
                accelerator.backward(loss)
                if accelerator.sync_gradients:  # 그래디언트 동기화가 필요한 경우에만
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
        
        # 샘플 생성 및 체크포인트 저장
        if accelerator.is_main_process:
            generate_samples(config, epoch, accelerator.unwrap_model(model), sampling_scheduler)
            

            if (epoch + 1) % config.save_per_epochs == 0:
                # 모델 저장
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                model_to_save = accelerator.unwrap_model(model).to('cpu')
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                )
                model_to_save.to(accelerator.device)

if __name__ == "__main__":
    from setproctitle import setproctitle
    setproctitle("Diffusion")
    
    config = TrainingConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_num)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    torch.cuda.set_device(config.gpu_num)
    
    os.makedirs(config.output_dir, exist_ok=True)
    config_path = os.path.join(config.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        print("="*50, file=f)
        print("Configuration:", file=f)
        for k, v in config.__dict__.items():
            print(f"{k}: {v}", file=f)
        print("="*50, file=f)
    
    train(config)