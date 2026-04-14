import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPLANATION_DIR = SCRIPT_DIR.parent / "explanation_generation"
if str(EXPLANATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLANATION_DIR))

from common import write_csv_atomic, write_jsonl_atomic
from settings import AUDIT_QUEUE_PATH, DATASET_MANIFEST_PATH, DEV_PATH, MERGED_SCORES_PATH, TEST_PATH, TRAIN_PATH


def assign_splits(rows):
    selected_rows = [row for row in rows if row["selected_for_sample"] == "예" and row["final_status"] == "pass"]
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

    # `v6`처럼 sample 수가 작은 일반화 검증 런에서는 단순 8:1:1 컷이
    # 뒤쪽 문서유형만 dev/test로 몰아버릴 수 있어, 문서유형별로 최소 분산을 먼저 보장한다.
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
            "sample_id": row["sample_id"],
            "family_id": row["family_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row.get("sampling_lane", ""),
            "original_input": row.get("original_input", ""),
            "transformed_problem": row["transformed_problem"],
            "short_answer": row["short_answer"],
            "generated_explanation": row["generated_explanation"],
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
            "evidence_sentence_ids": row.get("evidence_sentence_ids", ""),
            "evidence_sentence_count": row.get("evidence_sentence_count", ""),
            "evidence_policy_name": row.get("evidence_policy_name", ""),
            "rule_basis": row.get("rule_basis", ""),
            "fact_basis": row.get("fact_basis", ""),
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
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row.get("sampling_lane", ""),
                "transformed_problem": row["transformed_problem"],
                "short_answer": row["short_answer"],
                "generated_explanation": row["generated_explanation"],
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
                "sample_id": row["sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": "",
                "dataset_disposition": "audit",
                "train_eligible": row.get("train_eligible", "아니오"),
                "audit_required": row.get("audit_required", "예"),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
            }
        )

    return train_rows, dev_rows, test_rows, manifest_rows, audit_payload_rows


def main(rows):
    # dataset_build 단계는 생성 결과를 그대로 저장하지 않고,
    # 최종 학습·평가셋 규칙에 맞는 split과 manifest만 확정한다.
    train_rows, dev_rows, test_rows, manifest_rows, audit_rows = assign_splits(rows)
    write_jsonl_atomic(TRAIN_PATH, train_rows)
    write_jsonl_atomic(DEV_PATH, dev_rows)
    write_jsonl_atomic(TEST_PATH, test_rows)
    audit_fieldnames = [
        "sample_id",
        "family_id",
        "doc_type_name",
        "source_subset",
        "sampling_lane",
        "transformed_problem",
        "short_answer",
        "generated_explanation",
        "error_tags",
        "audit_reason",
        "weighted_score",
        "version_tag",
        "run_name",
        "label_path",
        "raw_path",
    ]
    write_csv_atomic(
        AUDIT_QUEUE_PATH,
        audit_rows,
        list(audit_rows[0].keys()) if audit_rows else audit_fieldnames,
    )
    if manifest_rows:
        write_csv_atomic(DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()))
    return manifest_rows


if __name__ == "__main__":
    import csv

    with open(MERGED_SCORES_PATH, encoding="utf-8-sig", newline="") as input_file:
        main(list(csv.DictReader(input_file)))
