import io
import csv
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models
import streamlit as st


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(page_title="배아 형태학 보조 판독 프로토타입", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# -----------------------------
# 공통 스타일
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# 모델 로드 함수
# -----------------------------
@st.cache_resource
def load_stage_model(model_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_icm_model(model_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_te_model(model_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


# -----------------------------
# 예측 함수
# -----------------------------
def predict_one(model, image_pil: Image.Image):
    image = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].cpu().tolist()


# -----------------------------
# 보조 판정 로직
# -----------------------------
def judge_transferability(stage_label: str, icm_label: str, te_label: str) -> str:
    if icm_label == "C" and te_label == "C":
        return "이식 비권장"

    if icm_label in ["A", "B"] and te_label in ["A", "B"]:
        if stage_label == "5~6":
            return "이식 가능"
        return "추가 검토"

    return "추가 검토"


# -----------------------------
# UI 보조 함수
# -----------------------------
def get_decision_style(decision: str):
    if decision == "이식 가능":
        return "#d1fae5", "#065f46"
    elif decision == "추가 검토":
        return "#fef3c7", "#92400e"
    else:
        return "#fee2e2", "#991b1b"


def render_decision_badge(decision: str):
    bg, fg = get_decision_style(decision)
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:8px 14px;
            border-radius:999px;
            background:{bg};
            color:{fg};
            font-weight:700;
            font-size:16px;
            margin-top:4px;
            margin-bottom:8px;
        ">
            {decision}
        </div>
        """,
        unsafe_allow_html=True
    )


def render_confidence_warning(title: str, probs: list, threshold: float = 0.60):
    max_prob = max(probs)
    if max_prob < threshold:
        st.warning(f"{title} 예측 신뢰도가 낮습니다. 사람 판독과 함께 보시는 것이 좋습니다. (max prob: {max_prob:.2f})")


def render_prob_cards(title: str, labels: list, probs: list):
    st.markdown(f"#### {title}")

    max_idx = probs.index(max(probs))

    for i, (label, prob) in enumerate(zip(labels, probs)):
        is_best = i == max_idx

        bg = "#ecfdf5" if is_best else "#f8fafc"
        border = "#10b981" if is_best else "#e5e7eb"
        text = "#065f46" if is_best else "#111827"
        badge = "TOP" if is_best else ""

        st.markdown(
            f"""
            <div style="
                border:1px solid {border};
                background:{bg};
                border-radius:12px;
                padding:10px 14px;
                margin-bottom:8px;
                box-shadow:0 1px 2px rgba(0,0,0,0.04);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; color:{text};">{label}</div>
                    <div style="font-size:12px; font-weight:700; color:#10b981;">{badge}</div>
                </div>
                <div style="margin-top:6px; font-size:15px; color:{text};">{prob:.4f}</div>
                <div style="
                    margin-top:8px;
                    width:100%;
                    height:10px;
                    background:#e5e7eb;
                    border-radius:999px;
                    overflow:hidden;
                ">
                    <div style="
                        width:{prob * 100:.1f}%;
                        height:100%;
                        background:{'#10b981' if is_best else '#94a3b8'};
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_prediction_summary(stage_label, icm_label, te_label):
    st.markdown("#### 예측 요약")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stage", stage_label)
    with c2:
        st.metric("ICM", icm_label)
    with c3:
        st.metric("TE", te_label)


def rows_to_csv_bytes(rows):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "file_name",
        "stage_pred", "prob_stage_1_4", "prob_stage_5_6",
        "icm_pred", "prob_icm_A", "prob_icm_B", "prob_icm_C",
        "te_pred", "prob_te_A", "prob_te_B", "prob_te_C",
        "decision"
    ])
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


# -----------------------------
# 메인 UI
# -----------------------------
st.title("배아 형태학 보조 판독 프로토타입")
st.caption("Stage / ICM / TE 예측 기반 보조 도구")
st.info("이 프로그램은 연구용 보조 도구이며, 임상 최종 의사결정을 대체하지 않습니다.")

with st.sidebar:
    st.header("모델 파일 경로")
    stage_model_path = st.text_input("Stage 모델", "embryo_stage_resnet18.pth")
    icm_model_path = st.text_input("ICM 모델", "embryo_icm_resnet18_best_weighted_es.pth")
    te_model_path = st.text_input("TE 모델", "embryo_te_resnet18_best_weighted_es.pth")

    st.markdown("---")
    st.subheader("보조 판정 기준")
    st.markdown("""
    - **이식 가능**: ICM/TE가 양호하고 stage가 비교적 안정적
    - **추가 검토**: 경계 사례 또는 낮은 confidence
    - **이식 비권장**: ICM/TE가 모두 불량
    """)

    st.markdown("---")
    st.write(f"장치: `{DEVICE}`")

uploaded_files = st.file_uploader(
    "배아 이미지 파일들을 업로드하세요",
    type=["png", "jpg", "jpeg", "bmp"],
    accept_multiple_files=True
)

if uploaded_files:
    try:
        stage_model = load_stage_model(stage_model_path)
        icm_model = load_icm_model(icm_model_path)
        te_model = load_te_model(te_model_path)
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

    stage_map = {0: "1~4", 1: "5~6"}
    grade_map = {0: "A", 1: "B", 2: "C"}

    results = []

    st.subheader("판독 결과")

    for up in uploaded_files:
        try:
            image = Image.open(up).convert("RGB")

            stage_idx, stage_probs = predict_one(stage_model, image)
            icm_idx, icm_probs = predict_one(icm_model, image)
            te_idx, te_probs = predict_one(te_model, image)

            stage_label = stage_map[stage_idx]
            icm_label = grade_map[icm_idx]
            te_label = grade_map[te_idx]

            decision = judge_transferability(stage_label, icm_label, te_label)

            results.append([
                up.name,
                stage_label, round(stage_probs[0], 4), round(stage_probs[1], 4),
                icm_label, round(icm_probs[0], 4), round(icm_probs[1], 4), round(icm_probs[2], 4),
                te_label, round(te_probs[0], 4), round(te_probs[1], 4), round(te_probs[2], 4),
                decision
            ])

            with st.container():
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(image, caption=up.name, use_container_width=True)

                with col2:
                    render_prediction_summary(stage_label, icm_label, te_label)

                    st.markdown("#### 보조 판정")
                    render_decision_badge(decision)

                    render_confidence_warning("Stage", stage_probs, threshold=0.60)
                    render_confidence_warning("ICM", icm_probs, threshold=0.60)
                    render_confidence_warning("TE", te_probs, threshold=0.60)

                    tab1, tab2, tab3 = st.tabs(["Stage 확률", "ICM 확률", "TE 확률"])

                    with tab1:
                        render_prob_cards("Stage 확률", ["1~4", "5~6"], stage_probs)

                    with tab2:
                        render_prob_cards("ICM 확률", ["A", "B", "C"], icm_probs)

                    with tab3:
                        render_prob_cards("TE 확률", ["A", "B", "C"], te_probs)

                st.markdown("---")

        except Exception as e:
            st.error(f"{up.name} 처리 실패: {e}")

    if results:
        csv_bytes = rows_to_csv_bytes(results)
        st.download_button(
            label="CSV 다운로드",
            data=csv_bytes,
            file_name="embryo_morphology_prototype_results.csv",
            mime="text/csv"
        )