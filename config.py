from dataclasses import dataclass
import torch

@dataclass
class TrainingConfig:
    image_size: int = 28  # MNIST 크기에 맞춤
    num_writers: int = 10  # MNIST의 클래스 수
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 1000
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    save_image_epochs: int = 5
    save_model_epochs: int = 10
    mixed_precision: str = 'fp16'
    output_dir: str = 'mnist_handwriting_output'
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    
    
@dataclass
class IAMTrainingConfig:
    image_size: int = 16
    max_width: int = 128
    train_batch_size: int = 20
    num_epochs: int = 100
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    mixed_precision: str = "fp16"
    gradient_accumulation_steps: int = 1
    device: str = "cuda"
    output_dir: str = "./saved_models/241115"
    num_writers: int = 252  # IAM 데이터셋의 작성자 수
    data_dir: str = "./data/IAM_words"
    checkpoint_dir: str = output_dir + "/checkpoints"
    resume_from_checkpoint = False  # 체크포인트에서 이어서 학습할지 여부
    checkpoint_name = 'checkpoint_epoch_34.pt'  # 로드할 체크포인트 파일명
