import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from datasets import load_dataset

class MNISTHandwritingDataset(Dataset):
    def __init__(self, config):
        self.mnist = load_dataset('mnist', split="train")
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.combined_images = self.create_combined_images()
    
    def create_combined_images(self):
        # 각 숫자별로 이미지를 그룹화합니다
        digit_groups = [[] for _ in range(10)]
        for img, label in zip(self.mnist['image'], self.mnist['label']):
            digit_groups[label].append(img)
        
        # 가장 적은 이미지 수를 가진 그룹의 크기를 찾습니다
        # total 5422 images
        min_group_size = min(len(group) for group in digit_groups)
        
        combined_images = []
        for i in range(min_group_size):
            processed_images = []
            for digit in range(10):
                image = digit_groups[digit][i]
                processed_image = self.preprocess(image.convert("L"))
                processed_images.append(processed_image)
            
            # 전처리된 이미지들을 가로로 연결합니다
            combined_image = torch.cat(processed_images, dim=2)
            combined_images.append(combined_image)
        
        return combined_images

    def __len__(self):
        return len(self.combined_images)
    
    def __getitem__(self, idx):
        glyph = self.combined_images[idx].clone()
        return {"images": self.combined_images[idx],
                "glyph": glyph,
                "writer_id": torch.randint(0, 10, (1,), dtype=torch.long)[0]
                }
