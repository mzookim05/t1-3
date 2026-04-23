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
    TARGET_DOC_TYPE_COUNT,
    TARGET_EXPANSION_PER_DOC_TYPE,
    TARGET_GENERALIZATION_PER_DOC_TYPE,
    TARGET_SEED_COUNT,
)


# 첫 split-type pilot은 broad heuristic보다 고정된 seed set이 더 재현 가능하다.
# reviewer가 같은 16개 family를 다시 읽을 수 있게 sample id를 명시적으로 잠근다.
TARGET_SAMPLE_IDS_BY_DOC_TYPE = {
    "법령_QA": {
        "expansion_01_02": ["v7_001", "v7_002"],
        "generalization_03_04": ["v7_005", "v7_006"],
    },
    "해석례_QA": {
        "expansion_01_02": ["v7_011", "v7_012"],
        "generalization_03_04": ["v7_015", "v7_017"],
    },
    "결정례_QA": {
        "expansion_01_02": ["v7_021", "v7_023"],
        "generalization_03_04": ["v7_025", "v7_027"],
    },
    "판결문_QA": {
        "expansion_01_02": ["v7_031", "v7_033"],
        "generalization_03_04": ["v7_035", "v7_036"],
    },
}


def extract_multi_query_signal(question_text):
    normalized = normalized_text(question_text)
    signals = []
    if "와 관련된 핵심 판단 기준은 무엇인가요" in normalized:
        signals.append("reason_plus_standard")
    if "와 관련된 법원의 판단 기준은 무엇인가요" in normalized:
        signals.append("reason_plus_court_standard")
    if "의 판단 기준은 무엇인가요" in normalized:
        signals.append("embedded_question_plus_standard")
    if "아니면" in normalized:
        signals.append("explicit_alternative")
    if "여부" in normalized and "법적 입장" in normalized:
        signals.append("issue_plus_position")
    if not signals:
        signals.append("split_candidate_manual_lock")
    return "|".join(signals)


def build_split_focus_hint(question_text):
    normalized = normalized_text(question_text)
    replacements = [
        ("와 관련된 핵심 판단 기준은 무엇인가요?", ""),
        ("와 관련된 핵심 판단 기준은 무엇인가요", ""),
        ("와 관련된 법원의 판단 기준은 무엇인가요?", ""),
        ("와 관련된 법원의 판단 기준은 무엇인가요", ""),
        ("의 판단 기준은 무엇인가요?", ""),
        ("의 판단 기준은 무엇인가요", ""),
    ]
    simplified = normalized
    for old, new in replacements:
        simplified = simplified.replace(old, new)
    return normalized_text(simplified).rstrip(" ,")


def build_seed_row(train_row, selection_role, selection_note):
    answer_mode = train_row.get("answer_mode", "") or "criteria"
    return {
        "seed_sample_id": train_row["sample_id"],
        "reference_sample_id": train_row["sample_id"],
        "family_id": train_row["family_id"],
        "doc_type_name": train_row["doc_type_name"],
        "source_subset": train_row["source_subset"],
        "sampling_lane": train_row["sampling_lane"],
        "answer_mode": answer_mode,
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "split_single_issue_rule"),
        "explanation_target": train_row.get("explanation_target", ""),
        "selection_role": selection_role,
        "selection_note": selection_note,
        "multi_query_signal": extract_multi_query_signal(train_row["transformed_problem"]),
        "split_focus_hint": build_split_focus_hint(train_row["transformed_problem"]),
        "transformed_problem": train_row["transformed_problem"],
        "short_answer": train_row["short_answer"],
        "generated_explanation": train_row["generated_explanation"],
        "rule_basis": train_row.get("rule_basis", ""),
        "fact_basis": train_row.get("fact_basis", ""),
        "label_path": train_row.get("label_path", ""),
        "raw_path": train_row.get("raw_path", ""),
        "selected_at_utc": utc_now_iso(),
    }


def main():
    ensure_run_dirs()

    reference_rows = load_jsonl(REFERENCE_TRAIN_PATH)
    train_map = {row["sample_id"]: row for row in reference_rows}
    seed_rows = []

    for doc_type_name, lane_map in TARGET_SAMPLE_IDS_BY_DOC_TYPE.items():
        for sampling_lane, sample_ids in lane_map.items():
            selection_role = "expansion_split_seed" if sampling_lane == "expansion_01_02" else "generalization_split_seed"
            for sample_id in sample_ids:
                train_row = train_map.get(sample_id)
                if not train_row:
                    raise ValueError(f"v7_strict_final/train.jsonl에서 v3 target sample을 찾지 못했습니다: {sample_id}")
                if train_row["doc_type_name"] != doc_type_name:
                    raise ValueError(f"문서유형 불일치: expected={doc_type_name}, actual={train_row['doc_type_name']}")
                if train_row["sampling_lane"] != sampling_lane:
                    raise ValueError(f"lane 불일치: expected={sampling_lane}, actual={train_row['sampling_lane']}")
                seed_rows.append(
                    build_seed_row(
                        train_row,
                        selection_role=selection_role,
                        selection_note="복수 질의형 tail을 단일 쟁점 서술형으로 분리하는 v3 pilot 고정 seed",
                    )
                )

    if len(seed_rows) != TARGET_SEED_COUNT:
        raise ValueError(f"v3 seed registry count mismatch: expected={TARGET_SEED_COUNT}, actual={len(seed_rows)}")

    doc_type_counts = defaultdict(int)
    lane_counts = defaultdict(int)
    for row in seed_rows:
        doc_type_counts[row["doc_type_name"]] += 1
        lane_counts[row["sampling_lane"]] += 1

    for doc_type_name, count in doc_type_counts.items():
        if count != TARGET_DOC_TYPE_COUNT:
            raise ValueError(f"doc_type seed count mismatch: {doc_type_name} -> {count}")

    # `v3`는 문서유형별 2 expansion + 2 generalization을 고정해 split-type 개선을 비교한다.
    for doc_type_name in TARGET_SAMPLE_IDS_BY_DOC_TYPE:
        per_doc_rows = [row for row in seed_rows if row["doc_type_name"] == doc_type_name]
        expansion_count = sum(1 for row in per_doc_rows if row["sampling_lane"] == "expansion_01_02")
        generalization_count = sum(1 for row in per_doc_rows if row["sampling_lane"] == "generalization_03_04")
        if expansion_count != TARGET_EXPANSION_PER_DOC_TYPE or generalization_count != TARGET_GENERALIZATION_PER_DOC_TYPE:
            raise ValueError(
                f"lane count mismatch for {doc_type_name}: expansion={expansion_count}, generalization={generalization_count}"
            )

    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["sampling_lane"], row["seed_sample_id"]))
    write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    copy_file_to_run_inputs(SEED_REGISTRY_PATH)
    copy_file_to_run_inputs(SEED_READY_PATH)

    manifest = build_run_manifest()
    manifest.update(
        {
            "seed_registry_count": len(seed_rows),
            "seed_registry_doc_type_counts": dict(sorted(doc_type_counts.items())),
            "seed_registry_lane_counts": dict(sorted(lane_counts.items())),
            "seed_registry_strategy": "fixed_16_family_multiclause_split_pilot",
        }
    )
    write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return seed_rows


if __name__ == "__main__":
    main()
