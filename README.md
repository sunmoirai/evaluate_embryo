# 🧬 Embryo AI Analysis (Prototype)

> **Deep Learning based Embryo Quality Assessment System**
> 배아 이미지를 분석하여 발달 단계(Stage)와 등급(ICM/TE)을 예측하고 이식 적합성을 보조 판단하는 딥러닝 프로토타입입니다.

---

## 🚀 프로젝트 개요
본 프로젝트는 난임 시술 과정에서 생성되는 배아 이미지를 활용하여 의료진의 의사결정을 보조하는 AI 파이프라인을 구축합니다.

* **핵심 기능**: 배아 이미지 자동 분류 및 Gardner 기준 기반 이식 가능 여부 판단 보조
* **분석 항목**: 
    * **Stage**: 발달 단계 (1~4 / 5~6)
    * **ICM (Inner Cell Mass)**: 내부세포괴 등급 (A/B/C)
    * **TE (Trophectoderm)**: 영양외배엽 등급 (A/B/C)

---

## 🧠 시스템 구조 (System Architecture)
본 시스템은 3개의 독립적인 **ResNet18** 모델이 병렬로 예측을 수행한 후, Rule-based 로직으로 최종 결과를 도출합니다.

1. **Image Input**: 배아 이미지 입력 (224x224)
2. **Multi-Model Inference**: Stage, ICM, TE 각각의 전용 모델이 예측 수행
3. **Rule-based Decision**: Gardner Grading 기준에 따른 이식 가능 여부 출력

---

## 🧪 모델 정보 (Model Details)
* **Backbone**: `ResNet18` (Pretrained)
* **Framework**: `PyTorch`
* **Input Size**: 224 x 224
* **Augmentation**: 
    * `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`
* **이식 가능 판단 기준**: 
    ```python
    if stage == "5~6" and icm in ["A", "B"] and te in ["A", "B"]:
        return "이식 가능 (Transferable)"
    ```

---

## 📊 성능 분석 (Validation Accuracy)
| 항목 | 정확도 (Accuracy) | 비고 |
| :--- | :--- | :--- |
| **Stage** | **~99%** | 형태적 차이가 명확하여 매우 높은 성능 기록 |
| **ICM** | **~50%** | 데이터 부족 및 라벨링 주관성으로 개선 필요 |
| **TE** | **~59%** | 형태 구분 난이도가 높음 |

> **⚠️ 성능 이슈 분석**: ICM/TE 정확도가 낮은 이유는 배아학자 간의 라벨링 주관성 편차와 미세한 형태적 차이 때문입니다. 향후 더 큰 데이터셋과 Attention 모델 도입이 필요합니다.

---

## 📁 프로젝트 구조
```text
embryo_ai/
├── app_embryo_prototype.py    # Streamlit UI 실행 파일
├── embryo_dataset.py          # Dataset 커스텀 클래스
├── train_.py                  # 모델 학습 스크립트
├── evaluate_.py               # 모델 평가 스크립트
├── models/                    # 학습된 모델(.pth) 저장 폴더
├── data/                      # 학습 데이터 폴더
└── requirements.txt           # 패키지 목록
```
---

## 🛠️ 실행 방법 (Usage)

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 모델 파일 배치
`models/` 폴더 내에 아래 가중치(`.pth`) 파일이 반드시 존재해야 합니다.

* `embryo_stage_resnet18.pth`
* `embryo_icm_resnet18_best.pth`
* `embryo_te_resnet18_best.pth`

### 3. 서비스 실행
아래 명령어를 터미널에 입력하여 대시보드를 실행합니다.

```bash
streamlit run app_embryo_prototype.py
```  

> **접속 주소**: [http://localhost:8501](http://localhost:8501)

---

## ⚠️ 주의사항 (Disclaimer)
본 프로젝트는 연구 및 포트폴리오 용도로 제작되었습니다.

의료기기 인증을 받지 않은 프로토타입이므로, 실제 임상적 판단이나 의료 목적으로는 사용할 수 없습니다.

---

## 👨‍💻 Author
**AI / Bioinformatics Researcher**

* **배아 이미지 기반 Classification 파이프라인 설계 및 구현**
* **실제 산업 데이터를 활용한 의료 AI 모델링 및 성능 최적화 경험**
* **데이터 전처리 및 하이퍼파라미터 튜닝을 통한 모델 성능 개선**

---