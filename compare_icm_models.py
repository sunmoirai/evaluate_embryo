import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class EmbryoGradeDataset(Dataset):
    def __init__(self, csv_file, label_col="ICM", transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_col = label_col

        self.label_map = {
            "A": 0,
            "B": 1,
            "C": 2
        }

        self.idx_to_label = {v: k for k, v in self.label_map.items()}

        self.data = self.data[self.data[label_col].isin(self.label_map.keys())].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        label_str = row[self.label_col]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.label_map[label_str]

        return image, label


def load_model(model_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, dataloader, device):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def print_evaluation_result(model_name, y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n==============================")
    print(f"모델: {model_name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def main():
    # -----------------------------
    # 1) 비교할 모델 목록
    # -----------------------------
    model_paths = [
        "embryo_icm_resnet18_best.pth",
        "embryo_icm_resnet18_merged.pth"
    ]

    # -----------------------------
    # 2) 테스트용 CSV
    #    반드시 같은 테스트셋으로 비교해야 공정
    # -----------------------------
    test_csv = "test_labels_timelapse.csv"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = EmbryoGradeDataset(test_csv, label_col="ICM", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    class_names = ["A", "B", "C"]

    print(f"테스트 데이터 수: {len(dataset)}")

    for model_path in model_paths:
        model = load_model(model_path, num_classes=3, device=device)
        y_true, y_pred = evaluate_model(model, dataloader, device)
        print_evaluation_result(model_path, y_true, y_pred, class_names)


if __name__ == "__main__":
    main()