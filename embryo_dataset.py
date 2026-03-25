import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class EmbryoStageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSV 읽기
        self.data = pd.read_csv(csv_file)

        # transform 저장
        self.transform = transform

        # stage를 숫자로 바꾸는 기준
        self.label_map = {
            "1~4": 0,
            "5~6": 1
        }

        # stage 값이 우리가 원하는 것만 남기기
        self.data = self.data[self.data["stage"].isin(self.label_map.keys())].reset_index(drop=True)

    def __len__(self):
        # 데이터 개수 반환
        return len(self.data)

    def __getitem__(self, idx):
        # idx번째 행 가져오기
        row = self.data.iloc[idx]

        image_path = row["image_path"]
        stage = row["stage"]

        # 이미지 열기
        image = Image.open(image_path).convert("RGB")

        # transform 있으면 적용
        if self.transform:
            image = self.transform(image)

        # 라벨 숫자로 변환
        label = self.label_map[stage]

        return image, label
    
if __name__ == "__main__":
    dataset = EmbryoStageDataset("embryo_labels.csv")

    print("데이터 개수:", len(dataset))

    image, label = dataset[0]

    print("첫 번째 이미지 타입:", type(image))
    print("첫 번째 라벨:", label)