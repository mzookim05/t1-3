import json
from collections import defaultdict

from common import write_csv_atomic, write_jsonl_atomic
from settings import DATASET_MANIFEST_PATH, DEV_PATH, MERGED_SCORES_PATH, TEST_PATH, TRAIN_PATH


def assign_splits(rows):
    selected_rows = [row for row in rows if row["selected_for_sample"] == "예" and row["final_status"] == "pass"]
    families = []
    seen = set()
    for row in selected_rows:
        if row["family_id"] not in seen:
            seen.add(row["family_id"])
            families.append(row["family_id"])

    total = len(families)
    train_cutoff = max(1, round(total * 0.8)) if total else 0
    dev_cutoff = max(train_cutoff + 1, round(total * 0.9)) if total >= 3 else train_cutoff
    family_to_split = {}
    for index, family_id in enumerate(families):
        if index < train_cutoff:
            family_to_split[family_id] = "train"
        elif index < dev_cutoff:
            family_to_split[family_id] = "dev"
        else:
            family_to_split[family_id] = "test"

    manifest_rows = []
    train_rows, dev_rows, test_rows = [], [], []
    for row in selected_rows:
        split = family_to_split[row["family_id"]]
        payload = {
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "transformed_problem": row["transformed_problem"],
            "short_answer": row["short_answer"],
            "generated_explanation": row["generated_explanation"],
            "split": split,
        }
        if split == "train":
            train_rows.append(payload)
        elif split == "dev":
            dev_rows.append(payload)
        else:
            test_rows.append(payload)
        manifest_rows.append(
            {
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": split,
                "weighted_score": row["weighted_score"],
            }
        )

    return train_rows, dev_rows, test_rows, manifest_rows


def main(rows):
    train_rows, dev_rows, test_rows, manifest_rows = assign_splits(rows)
    write_jsonl_atomic(TRAIN_PATH, train_rows)
    write_jsonl_atomic(DEV_PATH, dev_rows)
    write_jsonl_atomic(TEST_PATH, test_rows)
    if manifest_rows:
        write_csv_atomic(DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()))
    return manifest_rows


if __name__ == "__main__":
    import csv

    with open(MERGED_SCORES_PATH, encoding="utf-8-sig", newline="") as input_file:
        main(list(csv.DictReader(input_file)))
