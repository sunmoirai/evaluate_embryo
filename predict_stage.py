from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models


def main():
    # 1) 예측할 이미지 경로
    image_path = "test_embryo-2.png"   # ← 여기만 바꾸면 됨

    # 2) 학습 때와 같은 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 3) 사용할 장치
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4) 모델 구조 다시 만들기
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    # 5) 학습된 가중치 불러오기
    model.load_state_dict(torch.load("embryo_stage_resnet18.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # 6) 이미지 열기
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)   # 배치 차원 추가
    image = image.to(device)

    # 7) 예측
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # 8) 숫자 라벨 → 실제 stage로 변환
    label_map = {
        0: "1~4",
        1: "5~6"
    }

    print("예측 결과:", label_map[pred])
    print("확률:")
    print(f"  stage 1~4 : {probs[0][0].item():.4f}")
    print(f"  stage 5~6 : {probs[0][1].item():.4f}")


if __name__ == "__main__":
    main()