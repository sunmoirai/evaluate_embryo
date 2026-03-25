import copy
from collections import Counter

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


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


def train_one_model(
    label_col="ICM",
    save_path="best_model.pth",
    epochs=25,
    patience=5,
    csv_file="embryo_labels.csv"
):
    # train 전용 augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # validation 전용
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 전체 데이터
    full_data = pd.read_csv(csv_file)
    full_data = full_data[full_data[label_col].isin(["A", "B", "C"])].reset_index(drop=True)

    full_dataset_train = EmbryoGradeDataset(csv_file, label_col=label_col, transform=train_transform)
    full_dataset_val = EmbryoGradeDataset(csv_file, label_col=label_col, transform=val_transform)

    dataset_size = len(full_dataset_train)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    train_subset, val_subset = random_split(range(dataset_size), [train_size, val_size])

    train_dataset = torch.utils.data.Subset(full_dataset_train, train_subset.indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pretrained model
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    # class weight 자동 계산
    train_labels_str = full_data.iloc[train_subset.indices][label_col].tolist()
    counts = Counter(train_labels_str)

    num_A = counts["A"]
    num_B = counts["B"]
    num_C = counts["C"]
    total = num_A + num_B + num_C

    class_weights = torch.tensor([
        total / num_A,
        total / num_B,
        total / num_C
    ], dtype=torch.float32).to(device)

    print(f"\n[{label_col}] 클래스 분포 (train 기준)")
    print(f"A: {num_A}, B: {num_B}, C: {num_C}")
    print(f"[{label_col}] 클래스 가중치")
    print(
        f"A: {class_weights[0].item():.4f}, "
        f"B: {class_weights[1].item():.4f}, "
        f"C: {class_weights[2].item():.4f}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    print(f"\n=== {label_col} 모델 학습 시작 ===")
    print(f"CSV 파일: {csv_file}")
    print(f"데이터 수: {dataset_size}")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Epoch 수: {epochs}")
    print(f"Patience: {patience}")

    for epoch in range(epochs):
        # --------------------
        # Train
        # --------------------
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --------------------
        # Validation
        # --------------------
        model.eval()
        correct = 0
        total_count = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item()
                total_count += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total_count * 100

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # best model 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            early_stop_counter = 0
            print(f"  -> Best model 저장: {save_path} (Val Acc: {best_acc:.2f}%)")
        else:
            early_stop_counter += 1
            print(f"  -> 성능 향상 없음 (early stop counter: {early_stop_counter}/{patience})")

        # early stopping
        if early_stop_counter >= patience:
            print(f"\nEarly stopping 작동: {epoch+1} epoch에서 종료")
            break

    print(f"\n=== {label_col} 모델 학습 완료 ===")
    print(f"최고 Val Acc: {best_acc:.2f}%")
    print(f"저장 파일: {save_path}")


if __name__ == "__main__":
    # ICM 모델
    train_one_model(
        label_col="ICM",
        save_path="embryo_icm_resnet18_best_weighted_es.pth",
        epochs=25,
        patience=5,
        csv_file="embryo_labels.csv"
    )

    # TE 모델
    train_one_model(
        label_col="TE",
        save_path="embryo_te_resnet18_best_weighted_es.pth",
        epochs=25,
        patience=5,
        csv_file="embryo_labels.csv"
    )