🧬 Embryo AI Analysis (Prototype)

배아 이미지 데이터를 기반으로 Stage / ICM / TE를 예측하고, 이식 가능 여부를 판단하는 딥러닝 기반 프로토타입 시스템입니다.

🚀 프로젝트 개요

본 프로젝트는 난임 시술 과정에서 생성되는 배아 이미지를 활용하여:

배아 발달 단계 (Stage)

내부세포괴 (ICM)

영양외배엽 (TE)

를 자동 분류하고,
이를 기반으로 이식 가능 여부를 보조적으로 판단하는 AI 모델을 구현합니다.

👉 본 모델은 의료적 판단을 대체하지 않으며, 판단을 보조하는 도구를 목표로 합니다.

🧠 주요 기능

📷 배아 이미지 입력

🧬 Stage (1~4 / 5~6) 분류

🧬 ICM (A/B/C) 분류

🧬 TE (A/B/C) 분류

✅ Gardner 기준 기반 이식 가능 여부 판단

📊 확률 기반 결과 출력

🏗️ 시스템 구조
Image Input
    ↓
[Stage Model] → Stage 예측
[ICM Model]   → ICM 예측
[TE Model]    → TE 예측
    ↓
Rule-based 판단
    ↓
이식 가능 여부 출력
🧪 모델 정보

Backbone: ResNet18 (Pretrained)

Framework: PyTorch

Input Size: 224x224

Augmentation:

RandomHorizontalFlip

RandomRotation

ColorJitter

📊 성능 (Validation 기준)
항목	정확도
Stage	~99%
ICM	~50%
TE	~59%

👉 ICM / TE 정확도가 낮은 이유:

데이터 수 부족

라벨링 주관성 (배아학자 간 편차)

형태 구분 난이도

⚠️ 주의사항

본 모델은 의료용 인증 모델이 아닙니다

실제 임상 판단에는 사용할 수 없습니다

연구 / 프로토타입 용도입니다

📁 프로젝트 구조
1.embryo_ai/
│
├── app_embryo_prototype.py   # Streamlit UI
├── embryo_dataset.py        # Dataset 클래스
├── train_*.py               # 학습 코드
├── evaluate_*.py            # 평가 코드
│
├── models/                  # (비어있음)
├── data/                    # (비어있음)
│
├── requirements.txt
├── README.md
└── .gitignore
📦 모델 파일 (중요)

⚠️ 모델 파일은 GitHub에 포함되어 있지 않습니다.

아래 파일을 models/ 폴더에 직접 넣어야 합니다:

models/
├── embryo_stage_resnet18.pth
├── embryo_icm_resnet18_best.pth
├── embryo_te_resnet18_best.pth
🖥️ 실행 방법
1. 환경 설치
pip install -r requirements.txt
2. 실행
streamlit run app_embryo_prototype.py
3. 접속

브라우저에서:

http://localhost:8501
🧬 이식 가능 판단 기준 (Prototype)
if stage == "5~6" and icm in ["A", "B"] and te in ["A", "B"]:
    return "이식 가능"
else:
    return "추가 평가 필요"

👉 실제 병원에서는 Gardner grading 기준을 사용합니다.

📊 데이터

AI Hub 배아 이미지 데이터 사용

Time-lapse + Microscope 이미지 포함

라벨:

Stage

ICM

TE

👨‍💻 개발 목적

의료 AI 모델 구조 이해

이미지 기반 classification pipeline 구현

실제 산업 데이터 기반 모델링 경험

📜 License

본 프로젝트는 연구 및 포트폴리오 용도로 제작되었습니다.

🙋‍♂️ Author

AI / Bioinformatics 학습 프로젝트

Embryo AI Prototype 개발