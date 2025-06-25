---
base_model: google/gemma-7b-it
library_name: peft
---

# Gemma-7B-IT LoRA Fine-tuned Model (Rank 8)

이 모델은 Google의 Gemma-7B-IT(Instruction Tuned)를 기반으로 LoRA(Low-Rank Adaptation) 기법을 사용하여 fine-tuning된 대화형 언어 모델입니다.

## 모델 세부사항

### 모델 설명

이 모델은 Gemma-7B-IT를 기반으로 하여 LoRA 어댑터를 사용하여 fine-tuning된 대화형 언어 모델입니다. rank=8 설정으로 효율적인 파라미터 업데이트를 통해 지시 사항 수행과 대화 태스크에 특화되었습니다.

- **개발자:** Master Project
- **모델 타입:** Instruction-Following Language Model
- **언어:** 한국어/영어
- **라이센스:** Gemma License
- **베이스 모델:** google/gemma-7b-it

### 모델 구성

- **PEFT 타입:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 8
- **LoRA Alpha:** 16
- **LoRA Dropout:** 0.1
- **특화 영역:** 지시 사항 수행, 대화, Q&A

## 사용법

### 직접 사용

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 베이스 모델과 토크나이저 로드
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./model/gemma-7b-r8-it-master")

# 대화형 추론 실행
instruction = "다음 텍스트를 요약해주세요: [긴 텍스트]"
inputs = tokenizer(instruction, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 훈련 세부사항

### 훈련 설정

- **베이스 모델:** google/gemma-7b-it
- **LoRA 설정:** 
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
- **특화 영역:** 지시 사항 수행, 대화형 AI, 질문 답변

### 파일 구조

- `adapter_model.safetensors`: LoRA 어댑터 가중치
- `adapter_config.json`: LoRA 설정 파일
- `tokenizer.json`: 토크나이저 설정
- `trainer_state.json`: 훈련 상태 정보
- `training_args.bin`: 훈련 파라미터

## 성능 특징

- **지시 수행:** 다양한 지시 사항을 정확히 이해하고 수행
- **대화 능력:** 자연스러운 대화 진행
- **다중 태스크:** 요약, 번역, 창작, Q&A 등 다양한 태스크 지원
- **메모리 효율성:** LoRA를 통한 파라미터 효율적 fine-tuning
- **추론 속도:** 베이스 모델 대비 유사한 추론 속도

## 기술 사양

### 모델 아키텍처

- **베이스:** Transformer 기반 Instruction-Following Model
- **어댑터:** LoRA (Low-Rank Adaptation)
- **파라미터 수:** 베이스 모델 7B + LoRA 어댑터 (~18MB)
- **특화 영역:** 지시 사항 수행 및 대화형 AI

### 프레임워크

- **PEFT:** Parameter-Efficient Fine-Tuning
- **Transformers:** Hugging Face Transformers
- **PyTorch:** Deep Learning Framework