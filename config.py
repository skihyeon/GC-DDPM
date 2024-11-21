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
    # 데이터 관련 설정
    image_size: int = 32
    max_width: int = 256
    # num_writers: int = 252  # IAM 데이터셋의 작성자 수
    num_writers: int = 1966  # kor 데이터셋의 작성자 수
    # data_dir: str = "./data/IAM_words"
    data_dir: str = "./data/kor_hw"
    
    # 학습 하이퍼파라미터
    num_epochs: int = 1000
    train_batch_size: int = 16
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 100
    weight_decay: float = 0.01
    eps: float = 1e-8
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "fp16"
    
    # 디퓨전 모델 파라미터
    num_train_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    
    # 하드웨어 설정
    device: str = "cuda"
    gpu_num: int = 2
    
    # 저장 및 체크포인트 설정
    output_dir: str = "./saved_models/241120_new_kor"
    save_per_epochs: int = 1
    checkpoint_dir: str = output_dir + "/checkpoints"
    resume_from_checkpoint = True  # 체크포인트에서 이어서 학습할지 여부
    checkpoint_name = 'checkpoint_epoch_13.pt'  # 로드할 체크포인트 파일명

    # 샘플링 파라미터
    num_inference_steps: int = 10
    content_scale: float = 3.0
    style_scale: float = 1.0