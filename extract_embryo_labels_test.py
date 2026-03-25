import os
import json
import csv

# =============================
# 1. 경로 설정 (여기만 수정!)
# =============================
image_dir = r"./test_images/images"
json_dir = r"./test_images/labels"

output_csv = "test_labels_timelapse.csv"

# =============================
# 2. CSV 생성
# =============================
rows = []

for file_name in os.listdir(image_dir):
    if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_id = os.path.splitext(file_name)[0]

    image_path = os.path.join(image_dir, file_name)
    json_path = os.path.join(json_dir, image_id + ".json")

    if not os.path.exists(json_path):
        print(f"JSON 없음: {file_name}")
        continue

    # JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        prop = data["categories"]["properties"][0]

        stage = prop.get("stage", "")
        icm = prop.get("ICM", "")
        te = prop.get("TE", "")

        rows.append({
            "image_path": image_path,
            "stage": stage,
            "ICM": icm,
            "TE": te
        })

    except Exception as e:
        print(f"JSON 오류: {file_name}", e)

# =============================
# 3. CSV 저장
# =============================
with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "stage", "ICM", "TE"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV 생성 완료: {output_csv}")
print(f"총 데이터 수: {len(rows)}")