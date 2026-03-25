import os
import json
import csv
from pathlib import Path

# 1) 폴더 경로 설정
IMAGE_DIR = Path("./Training/1.원천데이터/image/microscope/d5")
LABEL_DIR = Path("./Training/2.라벨링데이터/label/microscope/d5")

# 2) 결과 저장 파일 이름
OUTPUT_CSV = "embryo_labels.csv"

# 3) 이미지 확장자 후보
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]

# 4) 이미지 파일들을 먼저 모아두기
#    key = 파일명(확장자 제외), value = 이미지 전체 경로
image_map = {}

for ext in IMAGE_EXTENSIONS:
    for img_path in IMAGE_DIR.rglob(f"*{ext}"):
        image_map[img_path.stem] = str(img_path)

# 5) CSV 파일 만들기
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as csvfile:
    writer = csv.writer(csvfile)

    # 첫 줄(컬럼명)
    writer.writerow(["image_id", "image_path", "json_path", "stage", "ICM", "TE"])

    # 6) 라벨 폴더 안의 모든 JSON 읽기
    for json_path in LABEL_DIR.rglob("*.json"):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[JSON 읽기 오류] {json_path} -> {e}")
            continue

        # 7) 파일명 기준으로 이미지 찾기
        image_id = json_path.stem
        image_path = image_map.get(image_id, "")

        # 8) stage / ICM / TE 기본값
        stage = ""
        icm = ""
        te = ""

        # 9) JSON 구조에서 값 추출
        #    지금까지 확인한 구조:
        #    {
        #      "objects": [],
        #      "categories": {
        #        "properties": [
        #          {
        #            "type": "radio",
        #            "stage": "5~6",
        #            "ICM": "B",
        #            "TE": "B"
        #          }
        #        ]
        #      }
        #    }

        categories = data.get("categories", {})
        properties = categories.get("properties", [])

        if isinstance(properties, list) and len(properties) > 0:
            prop = properties[0]
            stage = prop.get("stage", "")
            icm = prop.get("ICM", "")
            te = prop.get("TE", "")

        # 10) stage / ICM / TE가 있는 경우만 저장
        if stage or icm or te:
            writer.writerow([image_id, image_path, str(json_path), stage, icm, te])

print(f"완료: {OUTPUT_CSV} 파일이 생성되었습니다.")