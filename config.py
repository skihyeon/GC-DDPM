from dataclasses import dataclass
import torch
import os

@dataclass
class BaseConfig:
    """기본 설정"""
    # 모델 파라미터
    image_size: int = 32
    max_width: int = 256
    num_writers: int = 2218  # IAM+KOR 데이터셋의 작성자 수
    # num_writers: int = 252  # IAM 데이터셋의 작성자 수
    # num_writers: int = 1966  # kor 데이터셋의 작성자 수
    
    # 디퓨전 모델 파라미터
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    
    # 하드웨어 설정
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_num: int = 0

@dataclass
class TrainingConfig(BaseConfig):
    """학습 설정"""
    # 데이터 관련 설정
    data_dir: str = "./data/IAM+KOR"
    
    # 학습 하이퍼파라미터
    num_epochs: int = 1000
    train_batch_size: int = 16
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 100
    weight_decay: float = 0.01
    eps: float = 1e-8
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"
    content_scale: float = 3.0
    style_scale: float = 1.0
    num_inference_steps: int = 5
    
    # 저장 및 체크포인트 설정
    output_dir: str = "./saved_models/241129_iam_kor"
    save_per_epochs: int = 1
    checkpoint_dir: str = output_dir + "/checkpoints"
    resume_from_checkpoint: bool = True
    checkpoint_name: str = 'checkpoint_epoch_81.pt'

@dataclass
class InferenceConfig(BaseConfig):
    """추론 설정"""
    # 체크포인트 설정
    model_dir: str = "./saved_models/241129_iam_kor"
    checkpoint_dir: str = os.path.join(model_dir, "checkpoints")
    checkpoint_name: str = 'checkpoint_epoch_81.pt'
    
    # 샘플링 파라미터
    num_inference_steps: int = 5
    content_scale: float = 3.0
    style_scale: float = 1.0
    num_samples: int = 5  # 생성할 샘플 수
    
    # 폰트 설정
    font_path: str = "NanumGothic.ttf"
    
    # 서빙 설정
    server_name: str = "0.0.0.0"
    server_port: int = 7862
    share: bool = True  # gradio 임시 URL 생성 여부
    custom_path: str = "handwriting"  # 커스텀 URL 경로