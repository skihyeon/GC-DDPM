import gradio as gr
import os
from generate import HandwritingGenerator
from config import InferenceConfig
from PIL import Image
import numpy as np
from typing import List

class HandwritingService:
    def __init__(self):
        """손글씨 생성 서비스 초기화"""
        self.config = InferenceConfig()
        self.generator = HandwritingGenerator(
            checkpoint_path=os.path.join(self.config.checkpoint_dir, self.config.checkpoint_name),
            config=self.config
        )
    
    def generate_handwriting(
        self, 
        text: str,
        writer_id: int,
        content_scale: float = 3.0,
        style_scale: float = 1.0,
        num_inference_steps: int = 5,
    ) -> tuple[np.ndarray]:
        """손글씨 이미지 생성"""
        if not text.strip():
            raise gr.Error("텍스트를 입력해주세요.")
            
        # writer_id가 -1이면 랜덤 선택
        writer_id = None if writer_id == -1 else writer_id
            
        # 이미지 생성
        images = self.generator.generate(
            text=text,
            writer_id=writer_id,
            num_samples=self.config.num_samples,
            content_scale=content_scale,
            style_scale=style_scale,
            num_inference_steps=num_inference_steps
        )
        
        # PIL 이미지를 리사이징하고 numpy 배열로 변환
        result = []
        for img in images:
            # 높이를 64로 고정하고 비율 유지하면서 리사이징
            w = int(img.size[0] * (64 / img.size[1]))
            resized_img = img.resize((w, 64), Image.Resampling.LANCZOS)
            
            # 512 크기의 검은색 배경 이미지 생성
            final_img = Image.new('L', (512, 64), 255)
            # 중앙에 리사이징된 이미지 붙이기
            x = (512 - w) // 2
            final_img.paste(resized_img, (x, 0))
            
            result.append(np.array(final_img))
            
        # 부족한 샘플 수만큼 None 추가
        result = result + [None] * (self.config.num_samples - len(result))
        return tuple(result)

def create_demo() -> gr.Interface:
    service = HandwritingService()
    
    def generate(
        text: str,
        writer_id: int,
        content_scale: float,
        style_scale: float,
        num_inference_steps: int,
    ) -> tuple[np.ndarray]:
        return service.generate_handwriting(
            text=text,
            writer_id=writer_id,
            content_scale=content_scale,
            style_scale=style_scale,
            num_inference_steps=num_inference_steps
        )
    
    # 예제 데이터 정의
    examples = [
        ["안녕하세요", -1, 3.0, 1.0, 20],
        ["테스트", -1, 3.0, 1.0, 20],
        ["손글씨", -1, 3.0, 1.0, 20],
        ["Hello!", -1, 3.0, 1.0, 20],
        ["Test", -1, 3.0, 1.0, 20],
        ["Handwriting", -1, 3.0, 1.0, 20]
    ]
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🖋️ AI 손글씨 생성기")
        gr.Markdown("텍스트를 입력하면 다양한 스타일의 손글씨 이미지로 변환합니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="텍스트",
                    placeholder="변환할 텍스트를 입력하세요...",
                    lines=2
                )
                writer_id = gr.Dropdown(
                    choices=[("랜덤", -1)] + [(f"작성자 {i}", i) for i in range(service.config.num_writers)],
                    value=-1,
                    label="작성자 선택"
                )
                content_scale = gr.Slider(
                    label="컨텐츠 가이던스 스케일",
                    minimum=0.0,
                    maximum=4.0,
                    step=0.1,
                    value=3.0,
                    info="높을수록 입력 텍스트와 더 일치하는 결과가 나오지만, 다양성이 줄어듭니다."
                )
                style_scale = gr.Slider(
                    label="스타일 가이던스 스케일",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    value=1.0,
                    info="생성된 이미지가 작성자의 스타일을 얼마나 따를지를 결정합니다."
                )
                num_inference_steps = gr.Slider(
                    label="추론 스텝 수",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=10
                )
                generate_btn = gr.Button("생성하기")
            
            with gr.Column(scale=2):
                output_images = [
                    gr.Image(label=f"샘플 {i+1}", show_label=True, type="numpy")
                    for i in range(service.config.num_samples)
                ]
        
        # Examples 컴포넌트 수정
        gr.Examples(
            examples=examples,
            inputs=[
                text_input,
                writer_id,
                content_scale,
                style_scale,
                num_inference_steps
            ],
            outputs=output_images,
            fn=generate,
            cache_examples=True,  # 캐시 비활성화
            label="예제",  # 예제 섹션 레이블 추가,
            run_on_click=True
        )
        
        # 이벤트 핸들러
        generate_btn.click(
            fn=generate,
            inputs=[text_input, writer_id, content_scale, style_scale, num_inference_steps],
            outputs=output_images,

        )
    
    return demo

def main():
    demo = create_demo()
    config = InferenceConfig()
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share,
    )

if __name__ == "__main__":
    main() 