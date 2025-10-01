# Moon Rock Image Classification

PyTorch 기반 달 표면 암석 이미지 분류 프로젝트 (ResNet50 전이학습)

## 📌 프로젝트 개요

이 프로젝트는 사전학습된 ResNet50 모델을 활용한 전이학습(Transfer Learning)으로 달 표면 암석 이미지를 분류하는 딥러닝 모델입니다.

### 주요 기능
- ✅ ResNet50 사전학습 모델 활용
- ✅ 커스텀 FCL(Fully Connected Layer) 재정의
- ✅ 이미지 증강 및 전처리 파이프라인
- ✅ 학습/검증 데이터 자동 분할 (80:20)
- ✅ 모델 저장 및 로드 기능
- ✅ 실시간 예측 및 시각화

---

## 🚀 Python 3.12.1 마이그레이션 완료

### 마이그레이션 배경
- **기존 환경**: Python 3.8.10 + PyTorch 1.12.1
- **문제점**: GitHub Codespaces에서 가상환경 생성 오류 발생
- **해결책**: Python 3.12.1 + PyTorch 2.5.1로 업그레이드

---

## 📝 주요 변경 사항

### 1. **Import 경로 수정** (Cell 18, 20, 25)
```python
# ❌ 구버전 (Deprecated)
from torch.utils.data.sampler import SubsetRandomSampler

# ✅ 신버전 (Python 3.12 호환)
from torch.utils.data import SubsetRandomSampler
```

**영향받는 코드:**
- `load_split_train_test()` 함수
- `get_random_images()` 함수
- 데이터 샘플러 생성 코드

---

### 2. **Iterator 사용 패턴 개선** (Cell 25)
```python
# ❌ 구버전
dataiter = iter(loader)
images, labels = dataiter.next()

# ✅ 신버전 (Modern Python)
dataiter = iter(loader)
images, labels = next(dataiter)
```

**변경 이유:** Python 3.x에서 권장하는 built-in `next()` 함수 사용

---

### 3. **FCL(Fully Connected Layer) 재정의** (Cell 36)
```python
# ✅ 추가된 코드 (이전에 누락됨)
num_classes = len(trainloader.dataset.classes)
model.fc = nn.Sequential(
    nn.Linear(2048, 512),      # ResNet50 출력 → 512 뉴런
    nn.ReLU(),                 # 활성화 함수
    nn.Dropout(0.2),           # 과적합 방지 (20% 드롭아웃)
    nn.Linear(512, num_classes), # 최종 분류 레이어
    nn.LogSoftmax(dim=1)       # 로그 소프트맥스 출력
)
```

**아키텍처:**
- Input: 2048 (ResNet50 마지막 레이어)
- Hidden: 512 (ReLU + Dropout)
- Output: num_classes (데이터셋 클래스 수에 맞춤)

---

### 4. **NumPy GPU/CPU 호환성 개선** (Cell 54)
```python
# ❌ 구버전 (GPU 텐서에서 오류 발생 가능)
index = output.data.numpy().argmax()

# ✅ 신버전 (CPU 변환 후 NumPy 처리)
index = output.data.cpu().numpy().argmax()
```

**변경 이유:** GPU 텐서를 CPU로 명시적 변환 후 NumPy 연산 수행

---

## 📦 requirements.txt

```txt
# Core ML Libraries
torch==2.5.1                # PyTorch 최신 안정 버전
torchvision==0.20.1         # 컴퓨터 비전 도구 및 사전학습 모델

# Data Processing
numpy==2.1.3                # 수치 계산 라이브러리
pandas==2.2.3               # 데이터 분석

# Visualization
matplotlib==3.9.2           # 그래프 및 시각화

# Image Processing
Pillow==11.0.0              # 이미지 로딩 및 변환

# Jupyter Support
ipywidgets==8.1.5           # 노트북 위젯
ipykernel==6.29.5           # Jupyter 커널

# Utilities
tqdm==4.67.1                # 진행률 표시
```

### 버전 선택 기준
| 패키지 | 버전 | 이유 |
|--------|------|------|
| **torch** | 2.5.1 | Python 3.12 완전 지원, 성능 최적화 |
| **numpy** | 2.1.3 | PyTorch 2.x 호환성, Python 3.12 최적화 |
| **matplotlib** | 3.9.2 | NumPy 2.x 지원 |
| **Pillow** | 11.0.0 | 보안 패치, Python 3.12 지원 |

---

## 🛠️ 설치 및 실행 방법

### 1. 가상환경 생성 (Python 3.12.1)
```bash
# Linux/Mac/WSL
python3.12 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 2. 패키지 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 데이터 준비
```bash
# 프로젝트 루트에 data 디렉토리 생성
mkdir -p ./data

# 이미지 데이터를 클래스별 하위 디렉토리에 배치
# 예시 구조:
# ./data/
#   ├── class1/
#   │   ├── image1.jpg
#   │   └── image2.jpg
#   └── class2/
#       ├── image3.jpg
#       └── image4.jpg
```

### 4. Jupyter Notebook 실행
```bash
jupyter notebook ClassifySpaceRockCode.ipynb
```

### 5. 모델 학습 및 평가
- 노트북의 셀을 순서대로 실행
- 학습 완료 후 `moonrockmodel.pth` 파일 생성됨

---

## 📊 모델 아키텍처

### ResNet50 기반 전이학습
```
ResNet50 (Pretrained on ImageNet)
├── Conv Layers (Frozen) ❄️
├── Batch Normalization (Frozen) ❄️
├── ReLU Activations (Frozen) ❄️
└── Custom FCL (Trainable) 🔥
    ├── Linear(2048 → 512)
    ├── ReLU()
    ├── Dropout(0.2)
    ├── Linear(512 → num_classes)
    └── LogSoftmax(dim=1)
```

### 하이퍼파라미터
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| **Epochs** | 10 | 학습 반복 횟수 |
| **Batch Size** | 16 | 배치 크기 |
| **Learning Rate** | 0.003 | Adam optimizer 학습률 |
| **Validation Split** | 0.2 | 검증 데이터 비율 (20%) |
| **Optimizer** | Adam | 적응형 학습률 최적화 |
| **Loss Function** | NLLLoss | Negative Log Likelihood |

---

## 🎯 성능 모니터링

학습 과정에서 다음 메트릭이 출력됩니다:
- **Train Loss**: 학습 데이터 손실값
- **Test Loss**: 검증 데이터 손실값
- **Test Accuracy**: 검증 데이터 정확도

학습 완료 후 손실값 그래프가 자동 생성됩니다.

---

## 🔧 트러블슈팅

### 1. CUDA Out of Memory 오류
```python
# 배치 크기를 줄이기 (Cell 20, 함수 내부)
trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=8)  # 16 → 8
```

### 2. GPU 미사용 시 학습 속도 향상
```python
# Cell 30에서 디바이스 강제 지정
device = torch.device('cpu')  # GPU 없이 CPU만 사용
```

### 3. 이미지 데이터 경로 오류
```python
# Cell 4에서 데이터 경로 확인
data_dir = './data'  # 상대 경로
# 또는
data_dir = '/absolute/path/to/data'  # 절대 경로
```

### 4. 모델 로드 시 버전 불일치
```python
# Cell 50에서 weights_only 옵션 추가 (PyTorch 2.x)
model = torch.load('moonrockmodel.pth', weights_only=False)
```

### 5. 가상환경 활성화 확인
```bash
# 현재 Python 버전 확인
python --version  # Python 3.12.1 출력 확인

# pip 패키지 목록 확인
pip list | grep torch  # torch 2.5.1 확인
```

---

## 📚 기술 스택

### 프레임워크
- **PyTorch 2.5.1**: 딥러닝 프레임워크
- **torchvision 0.20.1**: 컴퓨터 비전 라이브러리

### 모델
- **ResNet50**: ImageNet 사전학습 모델
- **Transfer Learning**: 전이학습 기법

### 데이터 처리
- **RandomResizedCrop**: 랜덤 크롭 및 리사이즈 (224x224)
- **ToTensor**: PIL 이미지 → PyTorch 텐서 변환
- **SubsetRandomSampler**: 학습/검증 데이터 분할

### 시각화
- **Matplotlib**: 이미지 및 그래프 표시
- **PIL (Pillow)**: 이미지 로딩

---

## 📂 파일 구조

```
/mnt/d/dev/
├── ClassifySpaceRockCode.ipynb   # 메인 노트북 (수정됨 ✅)
├── requirements.txt               # 패키지 의존성 (Python 3.12 버전)
├── README.md                      # 이 파일
├── data/                          # 이미지 데이터 디렉토리
│   ├── class1/
│   └── class2/
└── moonrockmodel.pth              # 학습된 모델 (학습 후 생성)
```

---

## 🔄 이전 버전과의 차이점

| 항목 | Python 3.8.10 | Python 3.12.1 |
|------|---------------|---------------|
| PyTorch | 1.12.1 | 2.5.1 |
| NumPy | 1.24.2 | 2.1.3 |
| Import 경로 | `torch.utils.data.sampler` | `torch.utils.data` |
| Iterator | `dataiter.next()` | `next(dataiter)` |
| GPU 처리 | `.numpy()` | `.cpu().numpy()` |
| FCL 정의 | ❌ 누락 | ✅ 완전한 신경망 구조 |

---

## 🎓 학습 프로세스

### 1. 데이터 준비
- ImageFolder로 디렉토리 구조 기반 데이터 로드
- 80:20 비율로 학습/검증 데이터 분할
- 이미지 전처리 (크롭, 리사이즈, 텐서 변환)

### 2. 모델 구성
- ResNet50 사전학습 모델 로드
- 특징 추출 레이어 고정 (freeze)
- 커스텀 FCL 추가 및 학습 가능 상태로 설정

### 3. 학습 루프
- 순전파(Forward Pass): 입력 → 예측
- 손실 계산: NLLLoss 사용
- 역전파(Backward Pass): gradient 계산
- 파라미터 업데이트: Adam optimizer

### 4. 모델 평가
- 검증 데이터로 정확도 측정
- 손실값 그래프 시각화
- 모델 저장 (`.pth` 형식)

### 5. 예측
- 저장된 모델 로드
- 새로운 이미지에 대한 클래스 예측
- 결과 시각화

---

## 🆘 지원 및 문의

### 일반적인 문제
1. **가상환경 생성 실패**
   - Python 3.12.1이 설치되어 있는지 확인
   - `python3.12 --version` 명령으로 버전 확인

2. **패키지 설치 오류**
   - pip 최신 버전으로 업그레이드: `pip install --upgrade pip`
   - 개별 패키지 설치: `pip install torch==2.5.1`

3. **데이터 로드 오류**
   - `./data` 디렉토리 구조 확인
   - 이미지 파일 형식 확인 (JPG, PNG 등)

4. **학습 시간이 너무 오래 걸림**
   - GPU 사용 여부 확인: `torch.cuda.is_available()`
   - 배치 크기 조정 또는 에폭 수 감소

---

## 📄 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다.

---

## 📌 참고사항

- **최소 시스템 요구사항**:
  - RAM: 4GB 이상
  - 디스크 공간: 2GB 이상
  - Python: 3.12.1 이상

- **권장 사항**:
  - GPU (CUDA 지원): 학습 속도 향상
  - RAM: 8GB 이상
  - SSD: 데이터 로딩 속도 향상

- **개발 환경**:
  - WSL2 (Windows Subsystem for Linux)
  - GitHub Codespaces
  - Jupyter Notebook

---

**마지막 업데이트**: 2025-10-01
**Python 버전**: 3.12.1
**PyTorch 버전**: 2.5.1
