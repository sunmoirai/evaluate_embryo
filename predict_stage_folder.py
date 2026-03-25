import os
import csv
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


def main():
    # 1) 예측할 이미지 폴더
    image_folder = "test_images"   # ← 예측할 이미지들이 들어있는 폴더

    # 2) 결과 저장 CSV
    output_csv = "prediction_results.csv"

    # 3) 이미지 확장자
    valid_exts = [".png", ".jpg", ".jpeg", ".bmp"]

    # 4) 학습 때와 같은 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 5) 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6) 모델 구조 만들기
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 7) 학습된 모델 불러오기
    model.load_state_dict(torch.load("embryo_stage_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # 8) 숫자 라벨 → 실제 stage
    label_map = {
        0: "1~4",
        1: "5~6"
    }

    # 9) 결과 저장 리스트
    results = []

    # 10) 폴더 안 파일 하나씩 예측
    for file_name in os.listdir(image_folder):
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext not in valid_exts:
            continue

        image_path = os.path.join(image_folder, file_name)

        try:
            image = Image.open(image_path).convert("RGB")
            image = transform(image).unsqueeze(0)
            image = image.to(device)

            with torch.no_grad():
                outputs = model(image)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            pred_stage = label_map[pred]
            prob_1_4 = probs[0][0].item()
            prob_5_6 = probs[0][1].item()

            print(f"{file_name} -> 예측: {pred_stage}")

            results.append([
                file_name,
                image_path,
                pred_stage,
                round(prob_1_4, 4),
                round(prob_5_6, 4)
            ])

        except Exception as e:
            print(f"[오류] {file_name} 처리 실패: {e}")

    # 11) CSV 저장
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "image_path", "pred_stage", "prob_1_4", "prob_5_6"])
        writer.writerows(results)

    print(f"\n예측 완료. 결과 저장 파일: {output_csv}")


if __name__ == "__main__":
    main()