import os
import csv
import json
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


# -----------------------------
# 1. 모델 불러오기 함수
# -----------------------------
def load_stage_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("embryo_stage_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_icm_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("embryo_icm_resnet18_best_weighted_es.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_te_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("embryo_te_resnet18_best_weighted_es.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


# -----------------------------
# 2. 단일 이미지 예측 함수
# -----------------------------
def predict_one(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].cpu().tolist()


# -----------------------------
# 3. JSON에서 정답 읽기
# -----------------------------
def read_json_label(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stage = ""
    icm = ""
    te = ""

    categories = data.get("categories", {})
    properties = categories.get("properties", [])

    if isinstance(properties, list) and len(properties) > 0:
        prop = properties[0]
        stage = prop.get("stage", "")
        icm = prop.get("ICM", "")
        te = prop.get("TE", "")

    return stage, icm, te


# -----------------------------
# 4. 메인 실행
# -----------------------------
def main():
    # 테스트 이미지 폴더 / JSON 폴더
    image_folder = r"Validation\1.원천데이터\2.합성\image\microscope\d5"
    json_folder = r"Validation\2.라벨링데이터\2.합성\label\microscope\d5"

    output_csv = "evaluation_results.csv"
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 로드
    stage_model = load_stage_model(device)
    icm_model = load_icm_model(device)
    te_model = load_te_model(device)

    # 숫자 -> 라벨
    stage_map = {0: "1~4", 1: "5~6"}
    grade_map = {0: "A", 1: "B", 2: "C"}

    results = []

    # 정확도 계산용 카운터
    total_count = 0
    stage_correct = 0
    icm_correct = 0
    te_correct = 0

    for file_name in os.listdir(image_folder):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in valid_exts:
            continue

        image_path = os.path.join(image_folder, file_name)
        json_name = os.path.splitext(file_name)[0] + ".json"
        json_path = os.path.join(json_folder, json_name)

        if not os.path.exists(json_path):
            print(f"[JSON 없음] {file_name}")
            continue

        try:
            # 정답 읽기
            true_stage, true_icm, true_te = read_json_label(json_path)

            # 이미지 읽기
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # 예측
            pred_stage_idx, stage_probs = predict_one(stage_model, image)
            pred_icm_idx, icm_probs = predict_one(icm_model, image)
            pred_te_idx, te_probs = predict_one(te_model, image)

            pred_stage = stage_map[pred_stage_idx]
            pred_icm = grade_map[pred_icm_idx]
            pred_te = grade_map[pred_te_idx]

            # 정답 비교
            stage_match = (true_stage == pred_stage)
            icm_match = (true_icm == pred_icm)
            te_match = (true_te == pred_te)

            total_count += 1
            if stage_match:
                stage_correct += 1
            if icm_match:
                icm_correct += 1
            if te_match:
                te_correct += 1

            print(
                f"{file_name} | "
                f"Stage: {true_stage}->{pred_stage} | "
                f"ICM: {true_icm}->{pred_icm} | "
                f"TE: {true_te}->{pred_te}"
            )

            results.append([
                file_name,
                image_path,
                json_path,
                true_stage, pred_stage, stage_match,
                true_icm, pred_icm, icm_match,
                true_te, pred_te, te_match,
                round(stage_probs[0], 4), round(stage_probs[1], 4),
                round(icm_probs[0], 4), round(icm_probs[1], 4), round(icm_probs[2], 4),
                round(te_probs[0], 4), round(te_probs[1], 4), round(te_probs[2], 4)
            ])

        except Exception as e:
            print(f"[오류] {file_name} 처리 실패: {e}")

    # CSV 저장
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "image_path", "json_path",
            "true_stage", "pred_stage", "stage_match",
            "true_icm", "pred_icm", "icm_match",
            "true_te", "pred_te", "te_match",
            "prob_stage_1_4", "prob_stage_5_6",
            "prob_icm_A", "prob_icm_B", "prob_icm_C",
            "prob_te_A", "prob_te_B", "prob_te_C"
        ])
        writer.writerows(results)

    # 정확도 출력
    if total_count > 0:
        print("\n=== 최종 정확도 ===")
        print(f"총 이미지 수: {total_count}")
        print(f"Stage 정확도: {stage_correct / total_count * 100:.2f}%")
        print(f"ICM 정확도:   {icm_correct / total_count * 100:.2f}%")
        print(f"TE 정확도:    {te_correct / total_count * 100:.2f}%")
    else:
        print("평가된 이미지가 없습니다.")

    print(f"\n결과 저장 완료: {output_csv}")


if __name__ == "__main__":
    main()