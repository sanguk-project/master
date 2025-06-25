---
base_model: google/codegemma-7b
library_name: peft
---

# CodeGemma-7B LoRA Fine-tuned Model (Rank 4)

이 모델은 Google의 CodeGemma-7B를 기반으로 LoRA(Low-Rank Adaptation) 기법을 사용하여 fine-tuning된 효율적인 코드 생성 모델입니다.

## 모델 세부사항

### 모델 설명

이 모델은 CodeGemma-7B를 기반으로 하여 LoRA 어댑터를 사용하여 fine-tuning된 코드 생성 언어 모델입니다. rank=4 설정으로 최대한 효율적인 파라미터 업데이트를 통해 메모리 사용량을 최소화하면서 기본적인 코딩 태스크에 특화되었습니다.

- **개발자:** Master Project
- **모델 타입:** Lightweight Code Generation Language Model
- **언어:** 다양한 프로그래밍 언어 (Python, JavaScript, Java 등)
- **라이센스:** Gemma License
- **베이스 모델:** google/codegemma-7b

### 모델 구성

- **PEFT 타입:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 4
- **LoRA Alpha:** 16
- **LoRA Dropout:** 0.1
- **특화 영역:** 간단한 코드 생성, 기본 함수 작성

## 사용법

### 직접 사용

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 베이스 모델과 토크나이저 로드
base_model = AutoModelForCausalLM.from_pretrained("google/codegemma-7b")
tokenizer = AutoTokenizer.from_pretrained("google/codegemma-7b")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./model/codegemma-7b-r4-master")

# 간단한 코드 생성
inputs = tokenizer("def add_numbers(a, b):", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 훈련 세부사항

### 훈련 설정

- **베이스 모델:** google/codegemma-7b
- **LoRA 설정:** 
  - Rank: 4 (최소 rank로 메모리 효율성 극대화)
  - Alpha: 16
  - Dropout: 0.1
- **특화 영역:** 기본적인 코드 생성, 간단한 함수 작성

### 파일 구조

- `adapter_model.safetensors`: LoRA 어댑터 가중치
- `adapter_config.json`: LoRA 설정 파일
- `tokenizer.json`: 토크나이저 설정
- `trainer_state.json`: 훈련 상태 정보
- `training_args.bin`: 훈련 파라미터

## 성능 특징

- **메모리 효율성:** 가장 낮은 rank로 극도의 메모리 효율성
- **빠른 코드 생성:** 최소 파라미터로 빠른 코드 생성
- **기본 문법:** 프로그래밍 언어의 기본 문법에 특화
- **모델 크기:** 최소 어댑터 크기로 디스크 공간 절약

## 기술 사양

### 모델 아키텍처

- **베이스:** Transformer 기반 Code Generation Model
- **어댑터:** LoRA (Low-Rank Adaptation, Rank=4)
- **파라미터 수:** 베이스 모델 7B + LoRA 어댑터 (최소 크기)
- **특화 영역:** 효율적인 기본 코드 생성

### 프레임워크

- **PEFT:** Parameter-Efficient Fine-Tuning
- **Transformers:** Hugging Face Transformers
- **PyTorch:** Deep Learning Framework