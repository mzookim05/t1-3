import csv
from collections import defaultdict

from common import (
    build_run_manifest,
    copy_file_to_run_inputs,
    ensure_run_dirs,
    load_jsonl,
    normalized_text,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from settings import (
    ANSWER_MODE_TO_PROBLEM_MODE,
    REFERENCE_TRAIN_PATH,
    RUN_MANIFEST_PATH,
    SEED_READY_PATH,
    SEED_REGISTRY_PATH,
)


def choose_generalization_rows(rows):
    rows_03 = [row for row in rows if row["source_subset"].startswith("03_")]
    rows_04 = [row for row in rows if row["source_subset"].startswith("04_")]
    selected = []
    selected.extend(sorted(rows_03, key=lambda row: row["sample_id"])[:2])
    if rows_04:
        selected.extend(sorted(rows_04, key=lambda row: row["sample_id"])[:1])
    else:
        selected.extend(sorted(rows_03, key=lambda row: row["sample_id"])[2:3])
    return selected


def choose_expansion_row(rows, doc_type_name):
    if doc_type_name == "법령_QA":
        preferred = [row for row in rows if row["source_subset"].startswith("01_")]
        if preferred:
            return sorted(preferred, key=lambda row: row["sample_id"])[0]
    preferred_02 = [row for row in rows if row["source_subset"].startswith("02_")]
    if preferred_02:
        return sorted(preferred_02, key=lambda row: row["sample_id"])[0]
    return sorted(rows, key=lambda row: row["sample_id"])[0]


def build_seed_row(row, selection_role, selection_note):
    answer_mode = row.get("answer_mode", "") or "criteria"
    return {
        "seed_sample_id": row["sample_id"],
        "reference_sample_id": row["sample_id"],
        "family_id": row["family_id"],
        "doc_type_name": row["doc_type_name"],
        "source_subset": row["source_subset"],
        "sampling_lane": row["sampling_lane"],
        "answer_mode": answer_mode,
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "standard_reframe"),
        "explanation_target": row.get("explanation_target", ""),
        "selection_role": selection_role,
        "selection_note": selection_note,
        "transformed_problem": row["transformed_problem"],
        "short_answer": row["short_answer"],
        "generated_explanation": row["generated_explanation"],
        "rule_basis": row.get("rule_basis", ""),
        "fact_basis": row.get("fact_basis", ""),
        "label_path": row.get("label_path", ""),
        "raw_path": row.get("raw_path", ""),
        "selected_at_utc": utc_now_iso(),
    }


def main():
    ensure_run_dirs()
    reference_rows = load_jsonl(REFERENCE_TRAIN_PATH)
    grouped = defaultdict(list)
    for row in reference_rows:
        grouped[row["doc_type_name"]].append(row)

    selected_seed_rows = []
    for doc_type_name in sorted(grouped):
        rows = grouped[doc_type_name]
        generalization_rows = [row for row in rows if row["sampling_lane"] == "generalization_03_04"]
        expansion_rows = [row for row in rows if row["sampling_lane"] == "expansion_01_02"]
        selected_seed_rows.extend(
            build_seed_row(
                row,
                selection_role="generalization_seed",
                selection_note="03 2개 + 04 1개 기준으로 generalization lane을 먼저 고정",
            )
            for row in choose_generalization_rows(generalization_rows)
        )
        selected_seed_rows.append(
            build_seed_row(
                choose_expansion_row(expansion_rows, doc_type_name),
                selection_role="expansion_seed",
                selection_note="문서유형별 expansion lane 1개를 유지해 01·02 연결성을 최소 검산",
            )
        )

    selected_seed_rows.sort(key=lambda row: (row["doc_type_name"], row["selection_role"], row["seed_sample_id"]))
    write_csv_atomic(SEED_REGISTRY_PATH, selected_seed_rows, list(selected_seed_rows[0].keys()))
    write_jsonl_atomic(SEED_READY_PATH, selected_seed_rows)
    copy_file_to_run_inputs(SEED_REGISTRY_PATH)
    copy_file_to_run_inputs(SEED_READY_PATH)

    manifest = build_run_manifest()
    manifest.update(
        {
            "seed_registry_count": len(selected_seed_rows),
            "seed_registry_doc_type_counts": {
                doc_type_name: sum(1 for row in selected_seed_rows if row["doc_type_name"] == doc_type_name)
                for doc_type_name in sorted({row["doc_type_name"] for row in selected_seed_rows})
            },
        }
    )
    write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return selected_seed_rows


if __name__ == "__main__":
    main()
