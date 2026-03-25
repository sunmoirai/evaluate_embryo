import os
import csv
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


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
    model.load_state_dict(torch.load("embryo_icm_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def load_te_model(device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load("embryo_te_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_one(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].cpu().tolist()


def judge_transferability(stage_label, icm_label, te_label):
    # 임시 연구용 규칙
    if icm_label in ["A", "B"] and te_label in ["A", "B"]:
        return "이식 가능"

    # 둘 다 나쁨
    if icm_label == "C" and te_label == "C":
        return "이식 비권장"

    # 그 사이
    return "추가 검토"


def main():
    image_folder = "test_images"
    output_csv = "embryo_transferability_results.csv"
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 3개 불러오기
    stage_model = load_stage_model(device)
    icm_model = load_icm_model(device)
    te_model = load_te_model(device)

    # 숫자 → 실제 라벨
    stage_map = {0: "1~4", 1: "5~6"}
    grade_map = {0: "A", 1: "B", 2: "C"}

    results = []

    for file_name in os.listdir(image_folder):
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in valid_exts:
            continue

        image_path = os.path.join(image_folder, file_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # stage 예측
            stage_pred, stage_probs = predict_one(stage_model, image)
            stage_label = stage_map[stage_pred]

            # ICM 예측
            icm_pred, icm_probs = predict_one(icm_model, image)
            icm_label = grade_map[icm_pred]

            # TE 예측
            te_pred, te_probs = predict_one(te_model, image)
            te_label = grade_map[te_pred]

            # 최종 판단
            decision = judge_transferability(stage_label, icm_label, te_label)

            print(f"{file_name} -> stage:{stage_label}, ICM:{icm_label}, TE:{te_label}, 판단:{decision}")

            results.append([
                file_name,
                image_path,
                stage_label,
                round(stage_probs[0], 4),
                round(stage_probs[1], 4),
                icm_label,
                round(icm_probs[0], 4),
                round(icm_probs[1], 4),
                round(icm_probs[2], 4),
                te_label,
                round(te_probs[0], 4),
                round(te_probs[1], 4),
                round(te_probs[2], 4),
                decision
            ])

        except Exception as e:
            print(f"[오류] {file_name} 처리 실패: {e}")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name", "image_path",
            "stage_pred", "prob_stage_1_4", "prob_stage_5_6",
            "icm_pred", "prob_icm_A", "prob_icm_B", "prob_icm_C",
            "te_pred", "prob_te_A", "prob_te_B", "prob_te_C",
            "decision"
        ])
        writer.writerows(results)

    print(f"\n완료: {output_csv} 저장됨")


if __name__ == "__main__":
    main()