import torch
from tempData import MNISTHandwritingDataset
from model import GC_DDPM
from config import TrainingConfig
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
import os
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torchvision import datasets

def make_grid(images, rows=2, cols=5):
    w, h = images[0].size
    grid = Image.new('L', size=(cols*w, rows*h))
    for idx, image in enumerate(images):
        grid.paste(image, box=((idx%cols)*w, (idx//cols)*h))
    return grid

def generate_samples(config, epoch, model, noise_scheduler):
    model.eval()
    with torch.no_grad():
        # 에폭별로 다른 시드 설정
        torch.manual_seed(epoch)
        
        # MNIST에서 테스트용 글리프 이미지 가져오기
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.Resize(config.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        )
        
        # 각 writer_id(0-9)에 대해 하나씩 샘플 생성
        test_glyphs = []
        test_writer_ids = torch.arange(10).to(config.device)
        
        # 각 writer_id에 대한 글리프 이미지 준비 - 매 에폭마다 새로운 이미지 선택
        for _ in range(10):
            combined_glyph = torch.zeros(1, config.image_size, config.image_size * 10).to(config.device)
            for digit in range(10):
                indices = (test_dataset.targets == digit).nonzero().squeeze()
                torch.manual_seed(epoch * 10 + digit)
                random_idx = torch.randint(0, len(indices), (1,)).item()
                img, _ = test_dataset[indices[random_idx]]
                combined_glyph[:, :, digit*config.image_size:(digit+1)*config.image_size] = img
            test_glyphs.append(combined_glyph)
        
        test_glyphs = torch.stack(test_glyphs).to(config.device)
        
        # 노이즈 생성에도 에폭 기반 시드 사용
        torch.manual_seed(epoch * 100)
        sample = torch.randn(
            10,  # writer_id 당 하나씩
            1,
            config.image_size,
            config.image_size * 10  # 10개의 숫자가 이어진 이미지
        ).to(config.device)

        # Sampling loop
        for t in tqdm(noise_scheduler.timesteps):
            noise_pred, _ = model(
                sample, 
                test_glyphs, 
                t, 
                test_writer_ids,
                use_guidance=True,
                guidance_scale=3.0
            )
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample

        # 이미지 변환 및 저장
        images = []
        for i in range(10):
            image = sample[i].cpu().numpy()
            image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(image[0]))

        # 이미지 그리드 생성 및 저장 (2x5 그리드)
        image_grid = make_grid(images, rows=2, cols=5)
        os.makedirs(config.output_dir, exist_ok=True)
        image_grid.save(f"{config.output_dir}/epoch_{epoch:04d}.png")

def train():
    config = TrainingConfig()

    # 데이터셋 및 데이터로더 설정
    dataset = MNISTHandwritingDataset(config)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True
    )
    
    # 모델 및 스케줄러 설정
    model = GC_DDPM(
        num_writers=config.num_writers,
        writer_embed_dim=256,
        image_size=config.image_size,
        in_channels=1,
        out_channels=2,  # noise와 variance 예측
        freq_shift=0
    ).to(config.device)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        prediction_type="epsilon"  # variance 예측을 위해 변경
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
    betas = noise_scheduler.betas.to(config.device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch['images']
            glyph_images = batch['glyph']
            writer_ids = batch['writer_id']
            
            # Random classifier-free guidance training
            use_guidance = torch.rand(1).item() < 0.1
            
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
                    use_guidance=use_guidance
                )
                
                # Noise prediction loss
                noise_loss = F.mse_loss(noise_pred, noise)
                
                # Variance prediction loss - 수정
                betas_t = betas[timesteps]
                alphas_cumprod_t = alphas_cumprod[timesteps]
                alphas_cumprod_prev = alphas_cumprod[torch.clamp(timesteps - 1, min=0)]
                
                target_var = betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod_t)
                var_pred = var_pred.mean([2, 3])  # spatial dimensions만 평균
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
            
        # 샘플 생성 코드 수정
        if accelerator.is_main_process:
            generate_samples(
                config, 
                epoch, 
                accelerator.unwrap_model(model), 
                noise_scheduler
            )

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    torch.cuda.set_device(3)
    train()