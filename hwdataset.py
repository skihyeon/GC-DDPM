from typing import Dict, Set, List, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from torchvision import transforms
from config import TrainingConfig

class IAMDataset(Dataset):
    def __init__(self, config: TrainingConfig):
        """
        IAM 손글씨 데이터셋 초기화
        
        Args:
            config: 학습 설정
        """
        self.config = config
        self.data_dir = config.data_dir
        self._setup_transforms()
        self._setup_font()
        self._load_dataset()
        self._setup_glyph_cache()
        self._save_writer_ids()

    def _setup_transforms(self) -> None:
        """이미지 변환 설정"""
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 논문에 따라 이미지 크기 조정
        self.image_size = self.config.image_size  # 64
        self.max_width = self.config.max_width    # 512

    def _setup_font(self) -> None:
        """폰트 설정"""
        try:
            self.font = ImageFont.truetype("NanumGothic.ttf", self.config.image_size)
        except OSError:
            raise RuntimeError("NanumGothic.ttf 폰트를 찾을 수 없습니다.")

    def _is_valid_text(self, text: str) -> bool:
        """텍스트 유효성 검사"""
        if len(text.strip()) < 2:
            return False
        if any(char in text for char in ['/', '.', '_']):
            return False
        return True

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """words.txt 파일의 한 줄을 파싱"""
        parts = line.strip().split()
        if len(parts) < 9:
            return None
            
        text = ' '.join(parts[8:])
        if not self._is_valid_text(text):
            return None
            
        image_name = parts[0]
        form_id = '-'.join(image_name.split('-')[:2])
        writer_id = self._extract_writer_id(form_id)
        
        if parts[1] != 'ok':
            return None
            
        image_path = self._get_image_path(form_id, image_name)
        if not os.path.exists(image_path):
            return None
            
        return {
            'image_name': image_name,
            'writer_id': writer_id,
            'text': text,
            'bbox': [int(x) for x in parts[3:7]]
        }

    def _load_dataset(self) -> None:
        """데이터셋 로드"""
        self.samples: List[Dict[str, Any]] = []
        self.writer_ids: Set[str] = set()
        words_file = os.path.join(self.data_dir, 'words.txt')
        
        if not os.path.exists(words_file):
            raise FileNotFoundError(f"words.txt 파일을 찾을 수 없습니다: {words_file}")
        
        with open(words_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                sample = self._parse_line(line)
                if sample:
                    self.writer_ids.add(sample['writer_id'])
                    self.samples.append(sample)
        
        self.writer_id_to_idx = {wid: idx for idx, wid in enumerate(sorted(self.writer_ids))}
        print(f"총 작성자 수: {len(self.writer_ids)}")
        print(f"총 샘플 수: {len(self.samples)}")

    def _setup_glyph_cache(self) -> None:
        """글리프 캐시 설정"""
        self.glyph_dir = os.path.join(self.data_dir, 'glyphs')
        os.makedirs(self.glyph_dir, exist_ok=True)
        self.glyph_cache: Dict[str, torch.Tensor] = {}

    def _save_writer_ids(self) -> None:
        """작성자 ID 목록 저장"""
        writer_ids_path = os.path.join(self.data_dir, 'writer_ids.txt')
        with open(writer_ids_path, 'w') as f:
            for wid in sorted(self.writer_ids):
                f.write(f"{wid}\n")

    @staticmethod
    def _extract_writer_id(form_id: str) -> str:
        """form ID에서 writer ID 추출"""
        return form_id.split('-')[1]

    def _get_image_path(self, form_id: str, image_name: str) -> str:
        """이미지 경로 생성"""
        return os.path.join(
            self.data_dir, 
            'words',
            form_id.split('-')[0],
            form_id,
            f"{image_name}.png"
        )

    def create_glyph_image(self, text: str) -> torch.Tensor:
        """텍스트를 글리프 이미지로 변환"""
        if text in self.glyph_cache:
            return self.glyph_cache[text]

        img = Image.new('L', (self.config.max_width, self.config.image_size), color=255)
        draw = ImageDraw.Draw(img)

        text_width = draw.textlength(text, font=self.font)
        x = (self.config.max_width - text_width) // 2
        draw.text((x, 0), text, font=self.font, fill=0)

        glyph_path = os.path.join(self.glyph_dir, f"{text}.png")
        img.save(glyph_path)

        glyph_tensor = self.transform(img)
        self.glyph_cache[text] = glyph_tensor
        return glyph_tensor

    def process_handwritten_image(self, image_path: str) -> torch.Tensor:
        """손글씨 이미지 전처리"""
        try:
            img = Image.open(image_path).convert('L')
        except Exception as e:
            raise RuntimeError(f"이미지 로드 실패: {image_path}, 에러: {str(e)}")

        aspect_ratio = min(img.size[0] / img.size[1], 8.0)
        new_width = int(self.config.image_size * aspect_ratio)
        img = img.resize((new_width, self.config.image_size), Image.LANCZOS)

        padded_img = Image.new('L', (self.config.max_width, self.config.image_size), color=255)
        x_offset = (self.config.max_width - new_width) // 2
        padded_img.paste(img, (x_offset, 0))

        return self.transform(padded_img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """데이터셋 아이템 반환"""
        sample = self.samples[idx]
        form_id = '-'.join(sample['image_name'].split('-')[:2])
        image_path = self._get_image_path(form_id, sample['image_name'])
        
        image = self.process_handwritten_image(image_path)
        glyph = self.create_glyph_image(sample['text'])
        
        if image.shape != glyph.shape:
            glyph = torch.nn.functional.interpolate(
                glyph.unsqueeze(0), 
                size=(self.config.image_size, self.config.max_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return {
            'images': image,
            'glyph': glyph,
            'writer_id': torch.tensor(self.writer_id_to_idx[sample['writer_id']], dtype=torch.long),
            'text': sample['text']
        }

    def __len__(self) -> int:
        """데이터셋 길이 반환"""
        return len(self.samples)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = TrainingConfig()
    dataset = IAMDataset(config)

    sample = dataset[0]
    images = sample['images'].squeeze(0).numpy()
    glyph = sample['glyph'].squeeze(0).numpy()
    combined = np.concatenate([images, glyph], axis=1)

    plt.imshow(combined, cmap='gray')
    plt.title(f"Writer ID: {sample['writer_id']}, Text: {sample['text']}")
    plt.savefig('sample.png')