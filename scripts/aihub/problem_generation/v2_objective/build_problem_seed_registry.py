import csv
from collections import defaultdict

from common import (
    build_run_manifest,
    copy_file_to_run_inputs,
    ensure_run_dirs,
    load_csv_rows,
    load_jsonl,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from settings import (
    ANSWER_MODE_TO_PROBLEM_MODE,
    REFERENCE_PROBLEM_V1_MERGED_PATH,
    REFERENCE_TRAIN_PATH,
    RUN_MANIFEST_PATH,
    SEED_READY_PATH,
    SEED_REGISTRY_PATH,
)


BACKFILL_EXCLUDED_SAMPLE_IDS = {"v7_025"}


def build_seed_row(train_row, selection_role, selection_note, v1_row=None):
    answer_mode = train_row.get("answer_mode", "") or "criteria"
    return {
        "seed_sample_id": train_row["sample_id"],
        "reference_sample_id": train_row["sample_id"],
        "family_id": train_row["family_id"],
        "doc_type_name": train_row["doc_type_name"],
        "source_subset": train_row["source_subset"],
        "sampling_lane": train_row["sampling_lane"],
        "answer_mode": answer_mode,
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "single_best_rule"),
        "explanation_target": train_row.get("explanation_target", ""),
        "selection_role": selection_role,
        "selection_note": selection_note,
        "transformed_problem": train_row["transformed_problem"],
        "short_answer": train_row["short_answer"],
        "generated_explanation": train_row["generated_explanation"],
        "rule_basis": train_row.get("rule_basis", ""),
        "fact_basis": train_row.get("fact_basis", ""),
        "label_path": train_row.get("label_path", ""),
        "raw_path": train_row.get("raw_path", ""),
        "problem_v1_status": v1_row["final_status"] if v1_row else "",
        "problem_v1_generated_problem": v1_row["generated_problem"] if v1_row else "",
        "problem_v1_weighted_score": v1_row["weighted_score"] if v1_row else "",
        "selected_at_utc": utc_now_iso(),
    }


def choose_backfill_row(train_rows, used_sample_ids, used_family_ids):
    candidates = [
        row
        for row in train_rows
        if row["sample_id"] not in used_sample_ids
        and row["family_id"] not in used_family_ids
        and row["sample_id"] not in BACKFILL_EXCLUDED_SAMPLE_IDS
    ]

    def priority(row):
        return (
            row["doc_type_name"] != "결정례_QA",
            row["sampling_lane"] != "generalization_03_04",
            not row["source_subset"].startswith("03_"),
            row["sample_id"],
        )

    return sorted(candidates, key=priority)[0]


def main():
    ensure_run_dirs()

    train_rows = load_jsonl(REFERENCE_TRAIN_PATH)
    train_map = {row["sample_id"]: row for row in train_rows}
    v1_rows = load_csv_rows(REFERENCE_PROBLEM_V1_MERGED_PATH)

    selected_v1_rows = [
        row
        for row in v1_rows
        if row["selected_for_seed"] == "예" and row["final_status"] == "pass"
    ]
    selected_v1_rows.sort(key=lambda row: row["seed_sample_id"])

    seed_rows = []
    used_sample_ids = set()
    used_family_ids = set()

    for row in selected_v1_rows:
        train_row = train_map[row["seed_sample_id"]]
        seed_rows.append(
            build_seed_row(
                train_row,
                selection_role="v1_pass_seed",
                selection_note="problem_generation v1 strict-final pass family를 그대로 재사용",
                v1_row=row,
            )
        )
        used_sample_ids.add(train_row["sample_id"])
        used_family_ids.add(train_row["family_id"])

    backfill_row = choose_backfill_row(train_rows, used_sample_ids, used_family_ids)
    seed_rows.append(
        build_seed_row(
            backfill_row,
            selection_role="v2_backfill_seed",
            selection_note="v1 hard fail v7_025를 maintenance로 분리하고, 결정례_QA generalization family를 backfill",
        )
    )

    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["selection_role"], row["seed_sample_id"]))
    write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    copy_file_to_run_inputs(SEED_REGISTRY_PATH)
    copy_file_to_run_inputs(SEED_READY_PATH)

    lane_counts = defaultdict(int)
    doc_type_counts = defaultdict(int)
    role_counts = defaultdict(int)
    for row in seed_rows:
        lane_counts[row["sampling_lane"]] += 1
        doc_type_counts[row["doc_type_name"]] += 1
        role_counts[row["selection_role"]] += 1

    manifest = build_run_manifest()
    manifest.update(
        {
            "seed_registry_count": len(seed_rows),
            "seed_registry_lane_counts": dict(sorted(lane_counts.items())),
            "seed_registry_doc_type_counts": dict(sorted(doc_type_counts.items())),
            "seed_registry_role_counts": dict(sorted(role_counts.items())),
            "seed_backfill_sample_id": backfill_row["sample_id"],
            "seed_backfill_family_id": backfill_row["family_id"],
        }
    )
    write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return seed_rows


if __name__ == "__main__":
    main()
