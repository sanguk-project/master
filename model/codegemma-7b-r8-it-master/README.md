---
base_model: google/codegemma-7b-it
library_name: peft
---

# CodeGemma-7B-IT LoRA Fine-tuned Model (Rank 8)

이 모델은 Google의 CodeGemma-7B-IT(Instruction Tuned)를 기반으로 LoRA(Low-Rank Adaptation) 기법을 사용하여 fine-tuning된 대화형 코드 생성 모델입니다.

## 모델 세부사항

### 모델 설명

이 모델은 CodeGemma-7B-IT를 기반으로 하여 LoRA 어댑터를 사용하여 fine-tuning된 코드 생성 언어 모델입니다. rank=8 설정으로 효율적인 파라미터 업데이트를 통해 지시 사항을 이해하고 코딩 태스크를 수행하는 대화형 코드 어시스턴트입니다.

- **개발자:** Master Project
- **모델 타입:** Instruction-Following Code Generation Model
- **언어:** 다양한 프로그래밍 언어 + 자연어 지시사항
- **라이센스:** Gemma License
- **베이스 모델:** google/codegemma-7b-it

### 모델 구성

- **PEFT 타입:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 8
- **LoRA Alpha:** 16
- **LoRA Dropout:** 0.1
- **특화 영역:** 지시 기반 코드 생성, 코드 설명, 디버깅

## 사용법

### 직접 사용

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 베이스 모델과 토크나이저 로드
base_model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b-it")
tokenizer = AutoTokenizer.from_pretrained("google/codegemma-7b-it")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./model/codegemma-7b-r8-it-master")

# 지시 기반 코드 생성
instruction = "파이썬으로 피보나치 수열을 생성하는 함수를 작성해주세요."
inputs = tokenizer(instruction, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 훈련 세부사항

### 훈련 설정

- **베이스 모델:** google/codegemma-7b-it
- **LoRA 설정:** 
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
- **특화 영역:** 지시 기반 코드 생성, 코드 분석, 대화형 프로그래밍

### 파일 구조

- `adapter_model.safetensors`: LoRA 어댑터 가중치
- `adapter_config.json`: LoRA 설정 파일
- `tokenizer.json`: 토크나이저 설정
- `trainer_state.json`: 훈련 상태 정보
- `training_args.bin`: 훈련 파라미터

## 성능 특징

- **지시 이해:** 자연어로 된 코딩 요청을 정확히 이해
- **코드 설명:** 복잡한 코드를 이해하기 쉽게 설명
- **디버깅 지원:** 버그 발견 및 수정 제안
- **코드 최적화:** 성능 및 가독성 개선 제안
- **대화형 프로그래밍:** 단계별 코드 개발 지원

## 기술 사양

### 모델 아키텍처

- **베이스:** Transformer 기반 Instruction-Following Code Model
- **어댑터:** LoRA (Low-Rank Adaptation)
- **파라미터 수:** 베이스 모델 7B + LoRA 어댑터 (~18MB)
- **특화 영역:** 대화형 코드 생성 및 프로그래밍 지원

### 프레임워크

- **PEFT:** Parameter-Efficient Fine-Tuning
- **Transformers:** Hugging Face Transformers
- **PyTorch:** Deep Learning Framework