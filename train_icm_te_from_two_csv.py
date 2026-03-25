import copy
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights


class EmbryoGradeDataset(Dataset):
    def __init__(self, csv_files, label_col="ICM", transform=None):
        self.transform = transform
        self.label_col = label_col

        # 여러 CSV를 읽어서 하나로 합치기
        dataframes = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dataframes.append(df)

        self.data = pd.concat(dataframes, ignore_index=True)

        # 라벨 매핑
        self.label_map = {
            "A": 0,
            "B": 1,
            "C": 2
        }

        # A/B/C 값만 사용
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


def train_one_model(csv_files, label_col="ICM", save_path="best_model.pth", epochs=10):
    # train 전용 augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # validation 전용 transform
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 전체 데이터셋 2개 준비 (같은 데이터, transform만 다름)
    full_dataset_train = EmbryoGradeDataset(csv_files, label_col=label_col, transform=train_transform)
    full_dataset_val = EmbryoGradeDataset(csv_files, label_col=label_col, transform=val_transform)

    dataset_size = len(full_dataset_train)
    train_size = int(dataset_size * 0.8)
    val_size = dataset_size - train_size

    # 같은 인덱스로 train/val 나누기
    train_subset, val_subset = random_split(range(dataset_size), [train_size, val_size])

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
    print(f"사용 CSV: {csv_files}")
    print(f"전체 데이터 수: {dataset_size}")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Epoch 수: {epochs}")

    for epoch in range(epochs):
        # ------------------------
        # Train
        # ------------------------
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

        # ------------------------
        # Validation
        # ------------------------
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

        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"  -> Best model 저장: {save_path} (Val Acc: {best_acc:.2f}%)")

    print(f"\n=== {label_col} 모델 학습 완료 ===")
    print(f"최고 Val Acc: {best_acc:.2f}%")
    print(f"저장 파일: {save_path}")


if __name__ == "__main__":
    csv_files = ["embryo_labels.csv", "embryo_labels_SS.csv"]

    # ICM 학습
    train_one_model(
        csv_files=csv_files,
        label_col="ICM",
        save_path="embryo_icm_resnet18_merged.pth",
        epochs=10
    )

    # TE 학습
    train_one_model(
        csv_files=csv_files,
        label_col="TE",
        save_path="embryo_te_resnet18_merged.pth",
        epochs=10
    )