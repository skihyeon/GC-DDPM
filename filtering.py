import torch
import torch.nn.functional as F
from typing import List, Dict, Any
import logging

class DataFilteringStrategy:
    def __init__(
        self, 
        ocr_model, 
        real_dataset, 
        synthetic_dataset, 
        threshold: float = 0.8,
        device: str = 'cuda'
    ):
        """
        Args:
            ocr_model: OCR 모델 (negative log posterior probability를 계산할 수 있어야 함)
            real_dataset: 실제 손글씨 데이터셋 R = {(xi, yi)}
            synthetic_dataset: GC-DDPM으로 생성된 합성 데이터셋 S = {(x̃j, ỹj)}
            threshold: 필터링 임계값 τ ∈ (0, 1]
            device: 계산에 사용할 디바이스
        """
        self.ocr_model = ocr_model
        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset
        self.threshold = threshold
        self.device = device
        self.selected_samples = []
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
    def calculate_negative_log_posterior(self, image: torch.Tensor, text: str) -> float:
        """
        L(x, y; M) = -log p(y|x; M) 계산
        
        Args:
            image: 입력 이미지
            text: 정답 텍스트
            
        Returns:
            float: negative log posterior probability
        """
        with torch.no_grad():
            logits = self.ocr_model(image.to(self.device))
            loss = self.ocr_model.calculate_loss(logits, text)  # OCR 모델은 이 메소드를 구현해야 함
            return loss.item()
            
    def calculate_confidence_score(
        self, 
        image: torch.Tensor, 
        conditional_text: str
    ) -> tuple[float, str]:
        """
        수식 (8) 구현: c(x̃j, ỹj; M) = L(x̃j, ŷj; M) / L(x̃j, ỹj; M)
        
        Args:
            image: 생성된 손글씨 이미지 x̃j
            conditional_text: 조건부 텍스트 ỹj
            
        Returns:
            tuple[float, str]: (confidence score, predicted text)
        """
        with torch.no_grad():
            # OCR 모델로 이미지 인식하여 ŷj 얻기
            logits = self.ocr_model(image.to(self.device))
            predicted_text = self.ocr_model.decode(logits)  # OCR 모델은 이 메소드를 구현해야 함
            
            # L(x̃j, ŷj; M) 계산
            loss_predicted = self.calculate_negative_log_posterior(image, predicted_text)
            
            # L(x̃j, ỹj; M) 계산
            loss_conditional = self.calculate_negative_log_posterior(image, conditional_text)
            
            # 신뢰도 점수 계산
            confidence_score = loss_predicted / loss_conditional
            
        return confidence_score, predicted_text
    
    def filter_samples(self) -> List[Dict[str, Any]]:
        """
        Algorithm 1의 3번째 줄 구현:
        S′ = {(x̃j, ỹj) ∈ S | c(x̃j, ỹj; M) ≥ τ}
        """
        filtered_samples = []
        total = len(self.synthetic_dataset)
        
        self.logger.info(f"Starting filtering process for {total} synthetic samples...")
        
        for idx, sample in enumerate(self.synthetic_dataset):
            image = sample['image']
            conditional_text = sample['text']
            
            confidence_score, predicted_text = self.calculate_confidence_score(
                image, conditional_text
            )
            
            if confidence_score >= self.threshold:
                filtered_samples.append(sample)
                
            if (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{total} samples...")
                
        self.logger.info(f"Filtering complete. Selected {len(filtered_samples)}/{total} samples")
        return filtered_samples
    
    def train_ocr_model(self, dataset):
        """
        Algorithm 1의 4번째 줄 구현:
        Train model M using R ∪ S′ starting from random weight initialization
        """
        self.ocr_model.reset_parameters()  # 랜덤 초기화
        self.ocr_model.train_model(dataset)  # OCR 모델은 이 메소드를 구현해야 함
    
    def progressive_filtering(self, num_rounds: int = 3) -> tuple[Any, List[Dict[str, Any]]]:
        """
        Algorithm 1 전체 구현
        
        Args:
            num_rounds: 필터링 라운드 수 N
            
        Returns:
            tuple[Any, List[Dict[str, Any]]]: (trained OCR model, selected synthetic samples)
        """
        self.logger.info(f"Starting progressive filtering with {num_rounds} rounds...")
        
        # 1. Train model M using R (Algorithm 1, line 1)
        if len(self.selected_samples) == 0:
            self.train_ocr_model(self.real_dataset)
        
        # Progressive filtering rounds
        for round_idx in range(num_rounds):
            self.logger.info(f"Starting round {round_idx + 1}/{num_rounds}")
            
            # 2. Filter synthetic samples (Algorithm 1, line 3)
            filtered_samples = self.filter_samples()
            
            # 3. Update selected samples
            self.selected_samples = filtered_samples
            
            # 4. Retrain OCR model with combined dataset (Algorithm 1, line 4)
            combined_dataset = self.real_dataset + self.selected_samples
            self.train_ocr_model(combined_dataset)
            
            self.logger.info(
                f"Round {round_idx + 1} complete: "
                f"Selected {len(filtered_samples)} synthetic samples"
            )
            
        return self.ocr_model, self.selected_samples