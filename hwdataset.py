import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from torchvision import transforms
from config import IAMTrainingConfig

def extract_writer_id(form_id):
    """
    form ID에서 writer ID 추출
    예: 'a01-000u' -> '000u'
    """
    return form_id.split('-')[1]  # '000u'와 같은 전체 ID 반환

class IAMDataset:
    def __init__(self, config: IAMTrainingConfig):
        self.config = config
        self.data_dir = config.data_dir
        
        # transform 정의
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0)  # [0,1] -> [-1,1] 범위로 변환
        ])
        # font 설정 
        self.font = ImageFont.truetype("DejaVuSans.ttf", self.config.image_size)  # 폰트 크기를 image_size에 맞춤
        
        # config에서 필요한 값들 가져오기
        self.image_size = config.image_size
        self.max_width = config.max_width
        
        # words.txt 파일 읽기
        self.samples = []
        self.writer_ids = set()  # unique writer IDs
        words_file = os.path.join(self.data_dir, 'words.txt')
        
        with open(words_file, 'r') as f:
            for line in f:
                # 주석 라인 건너뛰기
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split()
                if len(parts) < 9:  # 최소 필요한 필드 수 확인
                    continue
                    
                image_name = parts[0]  # e.g., 'a01-000u-00-00'
                form_id = '-'.join(image_name.split('-')[:2])  # e.g., 'a01-000u'
                writer_id = extract_writer_id(form_id)  # e.g., '000u'
                
                # status가 'ok'인 샘플만 사용
                if parts[1] != 'ok':
                    continue
                
                self.writer_ids.add(writer_id)
                self.samples.append({
                    'image_name': image_name,
                    'writer_id': writer_id,
                    'text': ' '.join(parts[8:]),  # transcription (마지막 필드들을 모두 텍스트로)
                    'bbox': [int(x) for x in parts[3:7]]  # bounding box
                })
        
        # writer ID를 숫자 인덱스로 매핑 (정렬하여 일관된 인덱스 보장)
        self.writer_id_to_idx = {wid: idx for idx, wid in enumerate(sorted(self.writer_ids))}
        print(f"Total unique writers: {len(self.writer_ids)}")
        print(f"Total samples: {len(self.samples)}")

        # Glyph 이미지 사전 로드 (효율성을 위해 캐싱)
        self.glyph_dir = os.path.join(self.data_dir, 'glyphs')
        os.makedirs(self.glyph_dir, exist_ok=True)
        self.glyph_cache = {}
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 경로 구성
        # words/a01/a01-000u/a01-000u-00-00.png
        form_id = '-'.join(sample['image_name'].split('-')[:2])
        image_path = os.path.join(
            self.data_dir, 
            'words',
            form_id.split('-')[0],  # 'a01'
            form_id,                # 'a01-000u'
            f"{sample['image_name']}.png"
        )
        
        # 이미지 로드 및 전처리
        image = self.process_handwritten_image(image_path)
        glyph = self.create_glyph_image(sample['text'])
        
        # 이미지와 글리프의 크기를 동일하게 맞춤
        if image.shape != glyph.shape:
            glyph = torch.nn.functional.interpolate(
                glyph.unsqueeze(0), 
                size=(self.config.image_size, self.config.max_width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return {
            'images': image,        # (1, image_size, max_width)
            'glyph': glyph,         # (1, image_size, max_width)
            'writer_id': torch.tensor(self.writer_id_to_idx[sample['writer_id']], dtype=torch.long),
            'text': sample['text']
        }

    def __len__(self):
        return len(self.samples)
    
    def create_glyph_image(self, text):
        """텍스트를 글리프 이미지로 변환"""
        if text in self.glyph_cache:
            return self.glyph_cache[text]

        # 빈 이미지 생성
        img = Image.new('L', (self.config.max_width, self.config.image_size), color=0)
        draw = ImageDraw.Draw(img)

        # 텍스트 크기 측정
        text_width = draw.textlength(text, font=self.font)  # textbbox 대신 textsize 사용

        # 텍스트 중앙 정렬하여 그리기
        x = (self.config.max_width - text_width) // 2
        y = 0  # 상단 정렬
        draw.text((x, y), text, font=self.font, fill=255)

        # 변환 및 캐싱
        glyph_tensor = self.transform(img)
        self.glyph_cache[text] = glyph_tensor
        return glyph_tensor
    
    def process_handwritten_image(self, image_path):
        """손글씨 이미지 전처리"""
        img = Image.open(image_path).convert('L')

        # 높이를 image_size로 조정하면서 최대 aspect ratio 8 유지
        aspect_ratio = min(img.size[0] / img.size[1], 8.0)
        new_width = int(self.config.image_size * aspect_ratio)
        img = img.resize((new_width, self.config.image_size), Image.LANCZOS)

        # max_width에 맞춰 패딩 (양쪽 마진에 검은색 패딩)
        padded_img = Image.new('L', (self.config.max_width, self.config.image_size), color=0)
        x_offset = (self.config.max_width - new_width) // 2
        padded_img.paste(img, (x_offset, 0))

        # 변환
        return self.transform(padded_img)

    


if __name__ == "__main__":
    # 예시: 데이터셋 샘플 시각화
    import matplotlib.pyplot as plt
    from hwdataset import IAMDataset
    from config import IAMTrainingConfig

    config = IAMTrainingConfig()
    dataset = IAMDataset(config)

    sample = dataset[100]
    images = sample['images'].squeeze(0).numpy()  # (image_size, max_width)
    glyph = sample['glyph'].squeeze(0).numpy()    # (image_size, max_width)

    combined = np.concatenate([images, glyph], axis=1)  # 채널 차원에서 결합한 모습

    plt.imshow(combined, cmap='gray')
    plt.title(f"Writer ID: {sample['writer_id']}, Text: {sample['text']}")
    plt.savefig('sample.png')