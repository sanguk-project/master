# LLM 파인튜닝 프로젝트 (Large Language Model Fine-tuning)

본 프로젝트는 다양한 대규모 언어 모델(LLM)을 파인튜닝하여 특정 작업에 맞게 커스터마이징하는 실험적 연구 프로젝트입니다.

## 🎯 프로젝트 개요

이 리포지토리는 다음과 같은 LLM 모델들에 대한 파인튜닝 실험을 포함하고 있습니다:
- **Google Gemma** 시리즈 (2B, 7B)
- **Meta LLaMA** 시리즈 (3.2B Korean Bllossom)
- **CodeGemma** (코드 특화 모델)

## 📁 프로젝트 구조

```
├── dataset/                        # 훈련 데이터셋 저장소
│   ├── train_data.jsonl            # 훈련용 메인 데이터셋 (4.7MB)
│   ├── test_data.jsonl             # 테스트용 데이터셋 (1.2MB)
│   ├── solidity_vunerability_SWC-101_*.jsonl  # 정수 오버플로우 취약점
│   ├── solidity_vunerability_SWC-107_*.jsonl  # 재진입 공격 취약점
│   ├── solidity_vunerability_SWC-110_*.jsonl  # Assert 위반 취약점
│   ├── solidity_vunerability_SWC-113_*.jsonl  # DoS 공격 취약점
│   └── solidity_vunerability_SWC-114_*.jsonl  # 트랜잭션 순서 의존 취약점
├── model/                          # 학습된 모델 저장소
│   ├── gemma-7b-r{8,16}/           # Gemma 7B 파인튜닝 모델들
│   ├── codegemma-7b-r{8,16}/       # CodeGemma 7B 파인튜닝 모델들
│   └── runs/                       # 학습 로그 및 체크포인트
├── gemma_fine-tuning.ipynb         # Gemma 모델 파인튜닝 (메인)
├── gemma_2b_fine_tuning_l4.ipynb   # Gemma 2B L4 파인튜닝
├── gemma_7b_fine_tuning_fin.ipynb  # Gemma 7B 최종 파인튜닝
├── llama_fine_tuning.ipynb         # LLaMA 모델 파인튜닝
├── learning_llm_test.ipynb         # LLM 학습 테스트
├── llm_inference.ipynb             # 모델 추론 및 평가
├── huggigface_model.ipynb          # Hugging Face 모델 연동
└── README.md                       # 프로젝트 설명서
```

## 🚀 주요 기능

### 1. 다양한 모델 지원
- **Gemma 2B/7B**: Google의 경량화된 언어모델
- **CodeGemma 7B**: 코드 생성에 특화된 모델
- **LLaMA 3.2 Korean Bllossom**: 한국어 특화 모델

### 2. 최적화 기법
- **QLoRA (Quantized LoRA)**: 메모리 효율적인 파인튜닝
- **4비트 양자화**: BitsAndBytesConfig를 활용한 메모리 최적화
- **Flash Attention 2**: H100과 같은 최신 GPU에서 성능 최적화

### 3. 특화된 데이터셋
- **Solidity 취약점 분석**: 5가지 SWC 분류 기반 스마트 컨트랙트 보안 취약점 탐지
- **훈련/테스트 데이터**: 지도학습용 instruction-response 쌍
- **코드 보안 분석**: 취약점 식별 및 안전한 코드 제안

## 🛠️ 설치 및 설정

### 필수 요구사항
```bash
# GPU 환경 (CUDA 11.8 이상)
# Python 3.8 이상
# 최소 16GB GPU 메모리 권장 (H100 환경 최적화)
```

### 의존성 설치
```bash
# 기본 라이브러리
pip install torch torchvision torchaudio
pip install transformers datasets accelerate peft
pip install bitsandbytes

# Keras 및 JAX (Gemma 용)
pip install tensorflow keras keras-nlp jax[cuda]

# Flash Attention (선택사항 - H100 등 최신 GPU)
pip install flash-attn

# Hugging Face 및 Kaggle
pip install huggingface-hub kaggle
```

## 📚 사용법

### 1. Gemma 모델 파인튜닝
```python
# gemma_fine-tuning.ipynb 실행
# - dataset/train_data.jsonl 및 SWC 취약점 데이터셋 사용
# - JAX/Keras 프레임워크 기반
# - CodeGemma 7B 모델 활용
# - 스마트 컨트랙트 취약점 분석에 특화
```

### 2. LLaMA 모델 파인튜닝
```python
# llama_fine_tuning.ipynb 실행
# - dataset/train_data.jsonl 메인 훈련 데이터셋 사용
# - PyTorch/Transformers 기반
# - QLoRA 최적화 적용
# - 일반적인 instruction-following 태스크
```

### 3. 모델 추론 및 테스트
```python
# llm_inference.ipynb 실행
# - 학습된 모델 로드 및 추론
# - dataset/test_data.jsonl로 성능 평가
# - 실시간 취약점 분석 테스트
```

## 📊 데이터셋 정보

본 프로젝트에서 사용되는 주요 데이터셋은 `dataset/` 폴더에 저장되어 있습니다:

### 1. **훈련 및 테스트 데이터셋**
- **`train_data.jsonl`** (4.7MB): 모델 훈련용 메인 데이터셋
- **`test_data.jsonl`** (1.2MB): 모델 평가용 테스트 데이터셋

### 2. **Solidity 스마트 컨트랙트 취약점 데이터셋**
**SWC(Smart Contract Weakness Classification)** 기반의 특정 취약점 유형별 데이터셋:

- **`solidity_vunerability_SWC-101_Integer_Overflow_and_Underflow.jsonl`** (1.2MB)
  - 정수 오버플로우/언더플로우 취약점 예시
- **`solidity_vunerability_SWC-107_Reentrancy.jsonl`** (1.0MB) 
  - 재진입(Reentrancy) 공격 취약점 예시
- **`solidity_vunerability_SWC-110_Assert_Violation.jsonl`** (1.2MB)
  - Assert 위반 취약점 예시
- **`solidity_vunerability_SWC-113_DoS_with_Failed_Call.jsonl`** (1.2MB)
  - 실패한 호출로 인한 서비스 거부 공격 취약점 예시
- **`solidity_vunerability_SWC-114_Transaction_Order_Dependence.jsonl`** (1.3MB)
  - 트랜잭션 순서 의존성 취약점 예시

### 3. **데이터셋 구조**
각 데이터셋은 JSONL 형식으로 구성되며, 다음과 같은 필드를 포함:
```json
{
  "instruction": "취약한 코드에 대한 분석 요청 또는 문제 설명",
  "response": "보안 문제점 식별 및 수정된 코드 제공", 
  "category": "SWC-XXX (해당 취약점 분류)"
}
```

### 4. **데이터셋 특징**
- **총 데이터 규모**: 약 11.8MB (6개 파일)
- **형식**: Instruction-Response 쌍으로 구성된 지도학습용 데이터
- **언어**: Solidity 코드 및 영어 설명
- **취약점 유형**: 5가지 주요 스마트 컨트랙트 취약점 분류
- **활용 목적**: CodeGemma 및 Gemma 모델의 보안 코드 분석 능력 향상

### 5. **주요 취약점 유형 설명**
- **SWC-101**: 정수 연산 중 발생하는 오버플로우/언더플로우
- **SWC-107**: 재진입 공격으로 인한 자금 탈취 위험
- **SWC-110**: Assert 문 위반으로 인한 컨트랙트 실행 중단
- **SWC-113**: 외부 호출 실패로 인한 서비스 거부 공격
- **SWC-114**: 트랜잭션 순서에 의존하는 로직의 취약점

## ⚙️ 하이퍼파라미터 설정

### LoRA 설정
```python
# 일반적인 LoRA 설정
lora_config = LoraConfig(
    r=8,  # rank (r4, r8, r16 실험)
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)
```

### 4비트 양자화 설정
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False
)
```

## 📊 모델 성능

각 모델별로 다양한 rank 설정(r4, r8, r16)으로 실험을 진행하여 최적의 파라미터를 탐색합니다.

### 저장된 모델들
- `gemma-7b-r8-master/`: Gemma 7B (rank 8)
- `codegemma-7b-r16-master/`: CodeGemma 7B (rank 16)
- 각 모델에 대해 instruction-tuning 버전도 별도 저장

## 🖥️ 하드웨어 요구사항

### 권장 환경
- **GPU**: NVIDIA H100 (80GB VRAM)
- **메모리**: 32GB 이상 시스템 RAM
- **저장공간**: 100GB 이상 (모델 및 데이터셋 저장용)

### 최적화 특징
- Flash Attention 2 지원 (H100 환경)
- Mixed precision training (bfloat16)
- Gradient checkpointing
- Multi-GPU 지원 (device_map="auto")

## 🔬 실험 결과

각 노트북에서 수행한 실험들:
1. **모델 크기별 성능 비교** (2B vs 7B)
2. **LoRA rank 최적화** (r4, r8, r16)
3. **양자화 효과 분석** (4bit vs full precision)
4. **도메인별 특화** (코드 vs 일반 텍스트 vs 한국어)

## 📄 라이선스

본 프로젝트는 연구 목적으로 제작되었습니다. 사용하는 모델들의 라이선스를 준수해주세요:
- Gemma: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
- LLaMA: [LLaMA 2 License](https://github.com/facebookresearch/llama/blob/main/LICENSE)

## 📞 문의사항

- 프로젝트 관련 문의사항이 있으시면 Issue를 생성해주세요.
- E-mail: parksu9997@gmail.com
---