import csv
from collections import Counter

csv_file = "embryo_labels.csv"

stage_counter = Counter()
icm_counter = Counter()
te_counter = Counter()

with open(csv_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        stage_counter[row["stage"]] += 1
        icm_counter[row["ICM"]] += 1
        te_counter[row["TE"]] += 1

print("=== stage 분포 ===")
for k, v in stage_counter.items():
    print(k, v)

print("\n=== ICM 분포 ===")
for k, v in icm_counter.items():
    print(k, v)

print("\n=== TE 분포 ===")
for k, v in te_counter.items():
    print(k, v)