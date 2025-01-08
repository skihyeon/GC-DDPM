import torch
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from torchvision import transforms
from model import GC_DDPM
from diffusers import DDIMScheduler
from config import InferenceConfig
import random
from typing import Optional, List, Union

class TextRenderer:
    def __init__(self, config: InferenceConfig):
        """텍스트를 글리프 이미지로 변환하는 클래스"""
        try:
            self.font = ImageFont.truetype(config.font_path, config.image_size)
        except OSError:
            raise RuntimeError(f"{config.font_path} 폰트를 찾을 수 없습니다.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.image_size = config.image_size
        self.max_width = config.max_width

    def render_text(self, text: str) -> torch.Tensor:
        """텍스트를 글리프 이미지로 변환

        Args:
            text: 변환할 텍스트

        Returns:
            torch.Tensor: 변환된 글리프 이미지 텐서
        """
        img = Image.new('L', (self.max_width, self.image_size), color=255)
        draw = ImageDraw.Draw(img)

        # 텍스트 너비 계산 및 중앙 정렬
        text_width = draw.textlength(text, font=self.font)
        x = (self.max_width - text_width) // 2
        draw.text((x, 0), text, font=self.font, fill=0)

        return self.transform(img)

class HandwritingGenerator:
    def __init__(
        self, 
        checkpoint_path: str,
        config: Optional[InferenceConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """손글씨 생성 모델 초기화
        
        Args:
            checkpoint_path: 모델 체크포인트 경로
            config: 설정 객체
            device: 실행 디바이스
        """
        if config is None:
            config = InferenceConfig()
            
        self.config = config
        self.device = device
        
        # 글리프 렌더러 초기화
        self.renderer = TextRenderer(config=config)
        
        # 모델 초기화
        self.model = GC_DDPM(
            num_writers=config.num_writers,
            writer_embed_dim=256,
            image_size=config.image_size,
            max_width=config.max_width,
            in_channels=2,
            out_channels=2,
            n_timesteps=config.num_train_timesteps
        ).to(device)
        
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        # DDIM 스케줄러 초기화
        self.scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=False,
            prediction_type="epsilon",
            steps_offset=1
        )
        
        # 스케줄러의 beta 값들을 device로 이동
        self.scheduler.betas = self.scheduler.betas.to(device)
        self.scheduler.alphas = self.scheduler.alphas.to(device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)

    def get_writer_ids(self) -> List[int]:
        """사용 가능한 writer ID 목록 반환"""
        return list(range(self.config.num_writers))

    def generate(
        self,
        text: Union[str, List[str]],
        writer_id: Optional[int] = None,
        num_samples: int = 1,
        content_scale: float = 3.0,
        style_scale: float = 1.0,
        num_inference_steps: int = 10,
    ) -> List[Image.Image]:
        """텍스트를 손글씨 이미지로 변환

        Args:
            text: 변환할 텍스트 또는 텍스트 리스트
            writer_id: 작성자 ID (None이면 랜덤 선택)
            num_samples: 생성할 샘플 수
            content_scale: 컨텐츠 가이던스 스케일
            style_scale: 스타일 가이던스 스케일
            num_inference_steps: 추론 스텝 수

        Returns:
            List[Image.Image]: 생성된 손글씨 이미지 리스트
        """
        if isinstance(text, str):
            text = [text] * num_samples
        elif len(text) != num_samples:
            raise ValueError("텍스트 리스트의 길이는 num_samples와 같아야 합니다.")
            
        # writer_id 설정
        if writer_id is None:
            writer_ids = torch.randint(0, self.config.num_writers, (num_samples,))
        else:
            if not (0 <= writer_id < self.config.num_writers):
                raise ValueError(f"writer_id는 0에서 {self.config.num_writers-1} 사이여야 합니다.")
            writer_ids = torch.full((num_samples,), writer_id)
        
        writer_ids = writer_ids.to(self.device)
        
        # 글리프 이미지 생성
        glyph_images = []
        for t in text:
            glyph = self.renderer.render_text(t)
            glyph_images.append(glyph)
        glyph_images = torch.stack(glyph_images).to(self.device)
        
        # 샘플 생성
        with torch.no_grad():
            samples = self.model.sample(
                glyph=glyph_images,
                writer_ids=writer_ids,
                scheduler=self.scheduler,
                use_guidance=True,
                content_scale=content_scale,
                style_scale=style_scale,
                num_inference_steps=num_inference_steps
            )
        
        # 텐서를 PIL 이미지로 변환
        samples = samples.cpu()
        images = []
        for sample in samples:
            # 검은 여백 제거
            img_array = (sample.squeeze(0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')
            
            # 여백 제거
            bbox = img.getbbox()
            if bbox:
                img = img.crop(bbox)
            
            images.append(img)
            
        return images

def main():
    # 설정 및 모델 초기화
    config = InferenceConfig()
    generator = HandwritingGenerator(
        checkpoint_path=os.path.join(config.checkpoint_dir, config.checkpoint_name),
        config=config
    )

    import argparse
    
    # 테스트 텍스트
    # texts = [
    #     "Hello World!",
    #     "This is a test.",
    #     "Handwriting Generation"
    # ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Hello World!")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    
    # 각 텍스트에 대해 2개의 샘플 생성
    images = generator.generate(args.text, num_samples=args.num_samples)
        
    # 결과 저장
    os.makedirs("generated_samples", exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"generated_samples/{args.text.replace(' ', '_')}_{i}.png")

if __name__ == "__main__":
    main() 