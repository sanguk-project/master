---
base_model: google/gemma-7b
library_name: peft
---

# Gemma-7B LoRA Fine-tuned Model (Rank 8)

이 모델은 Google의 Gemma-7B를 기반으로 LoRA(Low-Rank Adaptation) 기법을 사용하여 fine-tuning된 모델입니다.

## 모델 세부사항

### 모델 설명

이 모델은 Gemma-7B를 기반으로 하여 LoRA 어댑터를 사용하여 fine-tuning된 언어 모델입니다. rank=8 설정으로 효율적인 파라미터 업데이트를 통해 특정 태스크에 최적화되었습니다.

- **개발자:** Master Project
- **모델 타입:** Causal Language Model
- **언어:** 한국어/영어
- **라이센스:** Gemma License
- **베이스 모델:** google/gemma-7b

### 모델 구성

- **PEFT 타입:** LoRA (Low-Rank Adaptation)
- **LoRA Rank (r):** 8
- **LoRA Alpha:** 16
- **LoRA Dropout:** 0.1
- **타겟 모듈:** fc_in, q_proj, fc_out, v_proj, out_proj, k_proj

## 사용법

### 직접 사용

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# 베이스 모델과 토크나이저 로드
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./model/gemma-7b-r8-master")

# 추론 실행
inputs = tokenizer("안녕하세요", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

## 훈련 세부사항

### 훈련 설정

- **베이스 모델:** google/gemma-7b
- **LoRA 설정:** 
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.1
- **타겟 모듈:** Attention 레이어의 핵심 projection 레이어들

### 파일 구조

- `adapter_model.safetensors`: LoRA 어댑터 가중치
- `adapter_config.json`: LoRA 설정 파일
- `tokenizer.json`: 토크나이저 설정
- `trainer_state.json`: 훈련 상태 정보
- `training_args.bin`: 훈련 파라미터

## 성능 특징

- **메모리 효율성:** LoRA를 통한 파라미터 효율적 fine-tuning
- **추론 속도:** 베이스 모델 대비 유사한 추론 속도
- **모델 크기:** 어댑터만 저장하여 디스크 공간 절약

## 제한사항

- 베이스 모델인 Gemma-7B의 제한사항을 상속
- 특정 도메인에 특화된 fine-tuning으로 일반적인 태스크에서는 성능 차이 가능
- LoRA rank가 낮아 복잡한 태스크에서는 제한적일 수 있음

## 기술 사양

### 모델 아키텍처

- **베이스:** Transformer 기반 Causal Language Model
- **어댑터:** LoRA (Low-Rank Adaptation)
- **파라미터 수:** 베이스 모델 7B + LoRA 어댑터 (~18MB)

### 프레임워크

- **PEFT:** Parameter-Efficient Fine-Tuning
- **Transformers:** Hugging Face Transformers
- **PyTorch:** Deep Learning Framework