import copy
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

        # A/B/C -> 0/1/2
        self.label_map = {
            "A": 0,
            "B": 1,
            "C": 2
        }

        # 해당 컬럼이 A/B/C인 데이터만 사용
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


def train_one_model(label_col="ICM", save_path="best_model.pth", epochs=15):
    # train 전용 augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # validation은 원본 느낌 유지
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 먼저 전체 CSV 읽고 train/val 분리용 base dataset 생성
    full_data = pd.read_csv("embryo_labels.csv")
    full_data = full_data[full_data[label_col].isin(["A", "B", "C"])].reset_index(drop=True)

    # 임시 CSV 대신 index 기반으로 나누기 위해 dataset 2개 생성
    full_dataset_train = EmbryoGradeDataset("embryo_labels.csv", label_col=label_col, transform=train_transform)
    full_dataset_val = EmbryoGradeDataset("embryo_labels.csv", label_col=label_col, transform=val_transform)

    dataset_size = len(full_dataset_train)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    train_subset, val_subset = random_split(range(dataset_size), [train_size, val_size])

    # 같은 index를 train/val dataset에 적용
    train_dataset = torch.utils.data.Subset(full_dataset_train, train_subset.indices)
    val_dataset = torch.utils.data.Subset(full_dataset_val, val_subset.indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # pretrained ResNet18
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n=== {label_col} 모델 학습 시작 ===")
    print(f"데이터 수: {dataset_size}")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Epoch 수: {epochs}")

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
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item()
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        val_acc = correct / total * 100

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
            print(f"  → Best model 저장됨: {save_path} (Val Acc: {best_acc:.2f}%)")

    print(f"\n=== {label_col} 모델 학습 완료 ===")
    print(f"최고 Val Acc: {best_acc:.2f}%")
    print(f"저장 파일: {save_path}")


if __name__ == "__main__":
    # ICM 모델 학습
    train_one_model(label_col="ICM", save_path="embryo_icm_resnet18_best.pth", epochs=15)

    # TE 모델 학습
    train_one_model(label_col="TE", save_path="embryo_te_resnet18_best.pth", epochs=15)