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

class IAMDataset(Dataset):
    def __init__(self, config: IAMTrainingConfig):
        self.config = config
        self.data_dir = config.data_dir
        
        # transform 수정 - [-1,1] 변환 제거 (모델에서 처리)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # font 설정 
        self.font = ImageFont.truetype("NanumGothic.ttf", self.config.image_size)  # 한글 지원 폰트로 변경
        
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
                
                # 텍스트 추출 및 길이 체크
                text = ' '.join(parts[8:])  # transcription
                if len(text.strip()) < 2:  # 두 글자 미만 건너뛰기
                    continue
                    
                image_name = parts[0]  # e.g., 'a01-000u-00-00'
                form_id = '-'.join(image_name.split('-')[:2])  # e.g., 'a01-000u'
                writer_id = extract_writer_id(form_id)  # e.g., '000u'
                # print(writer_id)
                # status가 'ok'인 샘플만 사용
                if parts[1] != 'ok':
                    continue
                
                image_path = os.path.join(
                    self.data_dir, 
                    'words',
                    form_id.split('-')[0],
                    form_id,
                    f"{image_name}.png"
                )
                if not os.path.exists(image_path):
                    continue
                
                self.writer_ids.add(writer_id)
                self.samples.append({
                    'image_name': image_name,
                    'writer_id': writer_id,
                    'text': text,
                    'bbox': [int(x) for x in parts[3:7]]
                })
        
        # writer ID를 숫자 인덱스로 매핑 (정렬하여 일관된 인덱스 보장)
        self.writer_id_to_idx = {wid: idx for idx, wid in enumerate(sorted(self.writer_ids))}
        print(f"Total unique writers: {len(self.writer_ids)}")
        print(f"Total samples: {len(self.samples)}")

        # Glyph 이미지 사전 로드 (효율성을 위해 캐싱)
        self.glyph_dir = os.path.join(self.data_dir, 'glyphs')
        os.makedirs(self.glyph_dir, exist_ok=True)
        self.glyph_cache = {}
        
        # # writer ID 목록 파일 저장 (샘플 생성시 필요)
        # with open(os.path.join(self.data_dir, 'writer_ids.txt'), 'w') as f:
        #     for wid in sorted(self.writer_ids):
        #         f.write(f"{wid}\n")
                
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
        
        # 모든 이미지가 [0,1] 범위를 유지하도록 함
        return {
            'images': image,        # (1, image_size, max_width) in [0,1]
            'glyph': glyph,         # (1, image_size, max_width) in [0,1]
            'writer_id': torch.tensor(self.writer_id_to_idx[sample['writer_id']], dtype=torch.long),
            'text': sample['text']
        }

    def __len__(self):
        return len(self.samples)
    
    def create_glyph_image(self, text):
        """텍스트를 글리프 이미지로 변환"""
        # 캐시에서 확인
        if text in self.glyph_cache:
            return self.glyph_cache[text]

        # 새로운 글리프 이미지 생성
        img = Image.new('L', (self.config.max_width, self.config.image_size), color=255)
        draw = ImageDraw.Draw(img)

        # 텍스트 크기 측정
        text_width = draw.textlength(text, font=self.font)

        # 텍스트 중앙 정렬
        x = (self.config.max_width - text_width) // 2
        y = 0  # 상단 정렬
        draw.text((x, y), text, font=self.font, fill=0)

        # 글리프 이미지를 파일로 저장
        glyph_path = os.path.join(self.glyph_dir, f"{text}.png")
        img.save(glyph_path)

        # [0,1] 범위로 변환
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

        # max_width에 맞춰 패딩 (양쪽 마진에 흰색 패딩)
        padded_img = Image.new('L', (self.config.max_width, self.config.image_size), color=255)
        x_offset = (self.config.max_width - new_width) // 2
        padded_img.paste(img, (x_offset, 0))

        # [0,1] 범위로 변환
        return self.transform(padded_img)

    


if __name__ == "__main__":
    # 예시: 데이터셋 샘플 시각화
    import matplotlib.pyplot as plt
    from hwdataset import IAMDataset
    from config import IAMTrainingConfig

    config = IAMTrainingConfig()
    dataset = IAMDataset(config)

    sample = dataset[0]
    images = sample['images'].squeeze(0).numpy()  # (image_size, max_width)
    glyph = sample['glyph'].squeeze(0).numpy()    # (image_size, max_width)

    combined = np.concatenate([images, glyph], axis=1)  # 채널 차원에서 결합한 모습

    plt.imshow(combined, cmap='gray')
    plt.title(f"Writer ID: {sample['writer_id']}, Text: {sample['text']}")
    plt.savefig('sample.png')