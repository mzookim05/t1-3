import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Pylance가 동적 sys.path import를 추론하지 못하므로 repo root 기반 절대 import로 고정한다.
PROJECT_ROOT = SCRIPT_DIR.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROBLEM_DIR = SCRIPT_DIR.parents[2] / "problem_generation" / "v2_objective"
if str(PROBLEM_DIR) not in sys.path:
    sys.path.insert(0, str(PROBLEM_DIR))

from scripts.aihub.problem_generation.v2_objective.common import write_csv_atomic, write_jsonl_atomic
from scripts.aihub.problem_generation.v2_objective.settings import (
    PROBLEM_AUDIT_QUEUE_PATH,
    PROBLEM_DATASET_MANIFEST_PATH,
    PROBLEM_DEV_PATH,
    PROBLEM_TEST_PATH,
    PROBLEM_TRAIN_PATH,
)


def assign_splits(rows):
    selected_rows = [row for row in rows if row["selected_for_seed"] == "예" and row["final_status"] == "pass"]
    trainable_rows = [row for row in selected_rows if row.get("train_eligible", "예") == "예"]
    audit_rows = [row for row in selected_rows if row.get("train_eligible", "예") != "예"]

    family_to_split = {}
    families_by_doc_type = {}
    seen = set()
    for row in trainable_rows:
        key = (row["doc_type_name"], row["family_id"])
        if key in seen:
            continue
        seen.add(key)
        families_by_doc_type.setdefault(row["doc_type_name"], []).append(row["family_id"])

    for doc_type_name, families in families_by_doc_type.items():
        total = len(families)
        if total >= 5:
            train_count = total - 2
            dev_count = 1
        elif total == 4:
            train_count = 2
            dev_count = 1
        elif total == 3:
            train_count = 1
            dev_count = 1
        elif total == 2:
            train_count = 1
            dev_count = 0
        elif total == 1:
            train_count = 1
            dev_count = 0
        else:
            train_count = 0
            dev_count = 0

        for index, family_id in enumerate(families):
            if index < train_count:
                family_to_split[family_id] = "train"
            elif index < train_count + dev_count:
                family_to_split[family_id] = "dev"
            else:
                family_to_split[family_id] = "test"

    manifest_rows = []
    train_rows, dev_rows, test_rows = [], [], []
    for row in trainable_rows:
        split = family_to_split[row["family_id"]]
        payload = {
            "problem_id": row["candidate_id"],
            "seed_sample_id": row["seed_sample_id"],
            "family_id": row["family_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row.get("sampling_lane", ""),
            "problem_task_type": row["problem_task_type"],
            "problem_generation_mode": row["problem_generation_mode"],
            "generated_stem": row["generated_stem"],
            "choice_a": row["choice_a"],
            "choice_b": row["choice_b"],
            "choice_c": row["choice_c"],
            "choice_d": row["choice_d"],
            "correct_choice": row["correct_choice"],
            "gold_short_answer": row["gold_short_answer"],
            "gold_reference_explanation": row["gold_reference_explanation"],
            "answer_mode": row.get("answer_mode", ""),
            "explanation_target": row.get("explanation_target", ""),
            "weighted_score": row["weighted_score"],
            "error_tags": row.get("error_tags", ""),
            "generator_model": row.get("generator_model", ""),
            "generation_mode": row.get("generation_mode", ""),
            "version_tag": row.get("version_tag", ""),
            "run_name": row.get("run_name", ""),
            "label_path": row.get("label_path", ""),
            "raw_path": row.get("raw_path", ""),
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
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": split,
                "dataset_disposition": split,
                "train_eligible": row.get("train_eligible", "예"),
                "audit_required": row.get("audit_required", "아니오"),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
            }
        )

    audit_payload_rows = []
    for row in audit_rows:
        audit_payload_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row.get("sampling_lane", ""),
                "generated_stem": row["generated_stem"],
                "choice_a": row["choice_a"],
                "choice_b": row["choice_b"],
                "choice_c": row["choice_c"],
                "choice_d": row["choice_d"],
                "correct_choice": row["correct_choice"],
                "gold_short_answer": row["gold_short_answer"],
                "error_tags": row.get("error_tags", ""),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
                "version_tag": row.get("version_tag", ""),
                "run_name": row.get("run_name", ""),
                "label_path": row.get("label_path", ""),
                "raw_path": row.get("raw_path", ""),
            }
        )
        manifest_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": "audit",
                "dataset_disposition": "audit",
                "train_eligible": row.get("train_eligible", "아니오"),
                "audit_required": row.get("audit_required", "예"),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
            }
        )

    return train_rows, dev_rows, test_rows, manifest_rows, audit_payload_rows


def main(rows):
    train_rows, dev_rows, test_rows, manifest_rows, audit_rows = assign_splits(rows)
    write_jsonl_atomic(PROBLEM_TRAIN_PATH, train_rows)
    write_jsonl_atomic(PROBLEM_DEV_PATH, dev_rows)
    write_jsonl_atomic(PROBLEM_TEST_PATH, test_rows)
    audit_fieldnames = [
        "problem_id",
        "seed_sample_id",
        "family_id",
        "doc_type_name",
        "source_subset",
        "sampling_lane",
        "generated_stem",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "correct_choice",
        "gold_short_answer",
        "error_tags",
        "audit_reason",
        "weighted_score",
        "version_tag",
        "run_name",
        "label_path",
        "raw_path",
    ]
    write_csv_atomic(
        PROBLEM_AUDIT_QUEUE_PATH,
        audit_rows,
        list(audit_rows[0].keys()) if audit_rows else audit_fieldnames,
    )
    if manifest_rows:
        write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()))
    return manifest_rows


if __name__ == "__main__":
    import csv
    from scripts.aihub.problem_generation.v2_objective.settings import MERGED_SCORES_PATH

    with open(MERGED_SCORES_PATH, encoding="utf-8-sig", newline="") as input_file:
        main(list(csv.DictReader(input_file)))
