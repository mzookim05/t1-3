from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `objective_interpretation_repair_pilot`은 D-slot hard/audit tail 2건만 남겼으므로,
# 같은 seed retry가 아니라 accepted 14개를 보존하고 fresh D-slot 2개로 final package를 재조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_repair_pilot as interpretation_pilot,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_repair_pilot_seed_preflight as preflight,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb6_non_law as pb6,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_accepted34_6slot_replacement as salvage,
)


VERSION_TAG = "objective_interpretation_repair_dslot_replacement_package"
# 새 llm_runs 폴더명은 실제 실행 시각을 따라야 하므로 날짜/시각을 하드코딩하지 않는다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_interpretation_repair_dslot_fresh_replacement"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PILOT_VERSION_TAG = "objective_interpretation_repair_pilot"
SOURCE_PILOT_RUN_PURPOSE = "objective_r2_interpretation_repair_api_pilot"
SOURCE_PILOT_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / SOURCE_PILOT_VERSION_TAG
    / "seed_registry.csv"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def resolve_source_pilot_run_dir() -> Path:
    # source API pilot도 실행 시각을 재계산하지 않고, 실제 tail seed가 들어 있는 artifact를 찾아 provenance를 고정한다.
    llm_runs_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    pattern = f"*_{SOURCE_PILOT_VERSION_TAG}_{SOURCE_PILOT_RUN_PURPOSE}"
    matched_dirs = []
    for candidate_dir in sorted(llm_runs_root.glob(pattern)):
        merged_path = candidate_dir / "merged" / f"merged_problem_scores_{SOURCE_PILOT_VERSION_TAG}.csv"
        if not merged_path.exists():
            continue
        rows = read_csv_rows(merged_path)
        seed_ids = {row.get("seed_sample_id", "") for row in rows}
        if {"interpretation_repair_preflight_004", "interpretation_repair_preflight_008"}.issubset(seed_ids):
            matched_dirs.append(candidate_dir)
    if not matched_dirs:
        raise FileNotFoundError("source interpretation repair pilot with 004/008 tails was not found")
    return matched_dirs[-1]


INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
REPLACEMENT_PLAN_CSV_PATH = RUN_EXPORTS_DIR / f"replacement_plan_{VERSION_TAG}.csv"
REPLACEMENT_PLAN_MD_PATH = RUN_EXPORTS_DIR / f"replacement_plan_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"replacement_merged_before_validator_{VERSION_TAG}.csv"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
TAIL_MEMO_MD_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.md"
VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

EXPECTED_TOTAL_SEED_COUNT = 2
EXPECTED_PRESERVED_COUNT = 14
EXPECTED_FINAL_PACKAGE_COUNT = 16
SUCCESS_USABLE_MIN = 16
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
REPLACEMENT_TARGET_LABEL_COUNTS = {"D": 2}
REPLACEMENT_SEED_ID_PREFIX = "interpretation_dslot_replacement"
EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 2}
EXPECTED_LANE_BY_DOC = {("해석례_QA", "expansion_01_02"): 2}
REPLACEMENT_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 1,
    "02_TL_유권해석_QA": 1,
}
BATCH_STATUS = "interpretation_repair_dslot_replacement_candidate_not_counted"
COUNT_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"
ORIGINAL_SALVAGE_NORMALIZE_PRESERVED_ROW = salvage.normalize_preserved_row

REPLACEMENT_TARGETS = [
    {
        "failed_seed_sample_id": "interpretation_repair_preflight_004",
        "source_subset": "01_TL_유권해석_QA",
        "sampling_lane": "expansion_01_02",
        "target_correct_choice": "D",
        "tail_class": "answer_uniqueness_same_direction_failure",
    },
    {
        "failed_seed_sample_id": "interpretation_repair_preflight_008",
        "source_subset": "02_TL_유권해석_QA",
        "sampling_lane": "expansion_01_02",
        "target_correct_choice": "D",
        "tail_class": "form_or_weak_audit_dslot_deficit",
    },
]

SOURCE_PILOT_RUN_DIR = resolve_source_pilot_run_dir()
SOURCE_PILOT_RUN_NAME = SOURCE_PILOT_RUN_DIR.name
SOURCE_PILOT_MERGED_PATH = SOURCE_PILOT_RUN_DIR / "merged" / f"merged_problem_scores_{SOURCE_PILOT_VERSION_TAG}.csv"


def classify_current_tail(row: dict[str, str]) -> str:
    # source tail class는 replacement 대상의 과거 상태이고, current tail class는 새 후보가 실제로 실패한 원인이다.
    if row.get("train_eligible") == "예" and row.get("audit_required") == "아니오" and row.get("final_status") == "pass":
        return "recovered_clean"
    reason_blob = "|".join(
        [
            row.get("validator_status", ""),
            row.get("validator_reasons", ""),
            row.get("error_tags", ""),
            row.get("answer_uniqueness", ""),
            row.get("distractor_direction", ""),
        ]
    )
    if (
        "answer_uniqueness" in reason_blob
        or "오답이 정답 가능" in reason_blob
        or row.get("answer_uniqueness") == "아니오"
        or row.get("distractor_direction") == "same_direction_blocked"
    ):
        return "answer_uniqueness_same_direction_failure"
    if row.get("validator_export_disposition") == "regenerate_required":
        return "weak_distractor_regenerate"
    if row.get("audit_required") == "예":
        return "form_or_weak_audit"
    return row.get("replacement_tail_class", "") or "unclassified_tail"


def infer_interpretation_axis(row: dict[str, str]) -> str:
    # 해석례 repair는 axis가 비어 있으면 다음 calibration 판단이 흐려지므로, 현재 실패 원인 기준으로 최소 축을 채운다.
    current_tail_class = row.get("current_tail_class", "") or classify_current_tail(row)
    if current_tail_class == "answer_uniqueness_same_direction_failure":
        return "answer_uniqueness"
    if "scope" in row.get("replacement_tail_class", ""):
        return "response_scope"
    if "form" in row.get("replacement_tail_class", ""):
        return "form_or_instruction"
    return row.get("interpretation_axis", "") or "interpretation_repair"


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def accepted_interpretation_rows() -> list[dict[str, str]]:
    rows = selected_rows(read_csv_rows(SOURCE_PILOT_MERGED_PATH))
    accepted = [
        dict(row)
        for row in rows
        if row.get("validator_export_disposition") == "export_ready"
        and row.get("train_eligible") == "예"
        and row.get("audit_required") == "아니오"
        and row.get("final_status") == "pass"
    ]
    if len(accepted) != EXPECTED_PRESERVED_COUNT:
        raise RuntimeError(f"interpretation accepted row count must be {EXPECTED_PRESERVED_COUNT}: {len(accepted)}")
    return accepted


def build_exclusion_sets(rows: list[dict[str, str]]) -> dict[str, set[str]]:
    return {
        "sample_ids": {
            row.get("sample_id", "") or row.get("seed_sample_id", "")
            for row in rows
            if row.get("sample_id", "") or row.get("seed_sample_id", "")
        },
        "reference_sample_ids": {row.get("reference_sample_id", "") for row in rows if row.get("reference_sample_id", "")},
        "family_ids": {row.get("family_id", "") for row in rows if row.get("family_id", "")},
        "label_paths": {row.get("label_path", "") for row in rows if row.get("label_path", "")},
        "raw_paths": {row.get("raw_path", "") for row in rows if row.get("raw_path", "")},
    }


def collect_excluded_rows() -> list[dict[str, str]]:
    # failed/audit source pilot seed도 seen seed로 묶어, 같은 seed retry가 아니라 fresh D-slot replacement로 남긴다.
    rows = preflight.collect_excluded_rows()
    rows.extend(salvage.load_csv_rows_if_exists(SOURCE_PILOT_SEED_REGISTRY_PATH))
    for source_row in selected_rows(read_csv_rows(SOURCE_PILOT_MERGED_PATH)):
        if source_row.get("seed_sample_id") in {target["failed_seed_sample_id"] for target in REPLACEMENT_TARGETS}:
            rows.append(
                {
                    "seed_sample_id": source_row.get("seed_sample_id", ""),
                    "reference_sample_id": source_row.get("reference_sample_id", ""),
                    "family_id": source_row.get("family_id", ""),
                    "label_path": source_row.get("label_path", ""),
                    "raw_path": source_row.get("raw_path", ""),
                }
            )
    return rows


def add_record_to_exclusion_sets(record: dict[str, str], exclusion_sets: dict[str, set[str]]) -> None:
    # package 내부에서 두 replacement가 같은 family/raw/label을 공유하면 fresh package 의미가 깨진다.
    exclusion_sets["sample_ids"].add(record["sample_id"])
    exclusion_sets["family_ids"].add(record["family_id"])
    exclusion_sets["label_paths"].add(record["label_path"])
    exclusion_sets["raw_paths"].add(record["raw_path"])


def matching_specs(source_subset: str, sampling_lane: str) -> list[dict[str, str]]:
    specs = []
    for spec in pb6.pb4.pb3.DATASET_SPECS:
        if (
            spec["doc_type_name"] == "해석례_QA"
            and spec["source_subset"] == source_subset
            and spec.get("sampling_lane") == sampling_lane
        ):
            copied = dict(spec)
            copied["sample_count"] = 1
            specs.append(copied)
    return specs


def select_fresh_record_for_target(
    target: dict[str, str],
    replacement_index: int,
    exclusion_sets: dict[str, set[str]],
) -> dict[str, str]:
    for spec in matching_specs(target["source_subset"], target["sampling_lane"]):
        label_paths = pb6.pb4.pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb6.pb4.pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb6.pb4.pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        candidate_indices = selected_indices + [index for index in range(len(label_paths)) if index not in set(selected_indices)]

        for candidate_index in candidate_indices:
            label_path = label_paths[candidate_index]
            payload = pb6.pb4.pb3.explanation_common.normalize_label_payload(
                label_path,
                pb6.pb4.pb3.explanation_common.load_json(label_path),
                spec["doc_type_name"],
            )
            passes_filter, _ = preflight.passes_interpretation_seed_filter(spec, payload)
            if not passes_filter:
                continue
            try:
                raw_path = pb6.pb4.pb3.explanation_common.locate_raw_path(
                    raw_paths,
                    spec["doc_type_name"],
                    payload["info"],
                )
            except FileNotFoundError:
                continue

            family_id = pb6.pb4.pb3.explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
            if family_id in exclusion_sets["family_ids"]:
                continue
            if str(label_path) in exclusion_sets["label_paths"]:
                continue
            if str(raw_path) in exclusion_sets["raw_paths"]:
                continue

            info = payload["info"]
            label = payload["label"]
            return {
                "sample_id": f"{REPLACEMENT_SEED_ID_PREFIX}_{replacement_index:03d}",
                "sample_order": replacement_index,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", ""),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb6.pb4.pb3.explanation_common.build_title(
                    {"info": info, "doc_type_name": spec["doc_type_name"]}
                ),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": replacement_index,
                "selected_index": candidate_index,
                "replacement_for_seed_sample_id": target["failed_seed_sample_id"],
                "replacement_tail_class": target["tail_class"],
                "target_correct_choice": target["target_correct_choice"],
        "selection_note": f"interpretation D-slot fresh replacement for {target['failed_seed_sample_id']}",
            }
    raise RuntimeError(f"fresh interpretation replacement seed not found for {target}")


def build_seed_row(record: dict[str, str]) -> dict[str, str]:
    row = preflight.build_seed_row(record)
    row["selection_role"] = "objective_interpretation_repair_dslot_replacement_seed"
    row["selection_note"] = record["selection_note"]
    row["replacement_for_seed_sample_id"] = record["replacement_for_seed_sample_id"]
    row["replacement_tail_class"] = record["replacement_tail_class"]
    row["pb9_replacement_for_seed_sample_id"] = record["replacement_for_seed_sample_id"]
    row["pb9_replacement_tail_class"] = record["replacement_tail_class"]
    preflight.augment_seed_row(row, 0)
    row["target_correct_choice"] = record["target_correct_choice"]
    row["source_tail_class"] = record["replacement_tail_class"]
    row["current_tail_class"] = ""
    row["interpretation_axis"] = "answer_uniqueness" if "answer_uniqueness" in record["replacement_tail_class"] else "form_or_instruction"
    row["interpretation_replacement_scope_note"] = (
        f"accepted{EXPECTED_PRESERVED_COUNT}_preserved_plus_{EXPECTED_TOTAL_SEED_COUNT}_fresh_d_slot_replacement_not_counted_until_signoff"
    )
    return row


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = interpretation_pilot.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## interpretation D-slot replacement 추가 지시
- 이번 seed는 `{seed.get('replacement_for_seed_sample_id') or seed.get('pb9_replacement_for_seed_sample_id', '')}`의 D-slot hard/audit tail을 fresh seed로 대체하기 위한 replacement다.
- target label은 `{seed.get('target_correct_choice', '')}`이며, final package의 `A/B/C/D = 4/4/4/4` 균형을 회복해야 한다.
- preserved source row는 `{EXPECTED_PRESERVED_COUNT}`개이고, 이번 fresh replacement slot은 `{EXPECTED_TOTAL_SEED_COUNT}`개다.
- `유권해석_QA` expansion에서는 오답이 같은 회답 방향을 유지한 채 조건만 넓히거나 좁히면 안 된다.
- 오답은 반드시 원문 전제조건, 예외, 적용범위, 회답 방향 중 하나를 명확히 깨뜨려야 한다.
- 다른 선택지가 원문 사실관계에서 같은 결론으로도 정답 가능하면 answer uniqueness failure로 본다.
- 정답은 하나의 회답 결론에만 닫고, 회답 결론과 회답 이유를 한 stem에서 동시에 묻지 않는다.
"""
    return messages


def apply_validator_to_replacements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    target_by_seed = salvage.target_schedule_by_seed()
    metadata_by_seed = salvage.replacement_seed_metadata_by_seed()
    validated_rows = []
    report_rows = []
    for row in selected_rows(rows):
        seed_metadata = metadata_by_seed.get(row["seed_sample_id"], {})
        row["replacement_for_seed_sample_id"] = seed_metadata.get("replacement_for_seed_sample_id", "")
        row["replacement_tail_class"] = seed_metadata.get("replacement_tail_class", "")
        row["source_tail_class"] = row["replacement_tail_class"]
        row["pb9_replacement_for_seed_sample_id"] = seed_metadata.get("pb9_replacement_for_seed_sample_id", "")
        row["pb9_replacement_tail_class"] = seed_metadata.get("pb9_replacement_tail_class", "")
        target_label = target_by_seed[row["seed_sample_id"]]
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices,
            row.get("correct_choice", ""),
            target_label,
        )
        action, status, reasons = interpretation_pilot.choose_interpretation_validator_action(row, match_count)
        shuffle_ok = recalculated_label == target_label
        metadata_ok = True
        metadata_reasons: list[str] = []
        if not shuffle_ok:
            action = "hard_block"
            status = "correct_choice_recalc_block"
            reasons = [*reasons, "correct_choice_recalc_mismatch"]

        row["upstream_final_status"] = row.get("final_status", "")
        row["upstream_audit_required"] = row.get("audit_required", "")
        row["target_correct_choice"] = target_label
        row["validator_recalculated_correct_choice"] = recalculated_label or ""
        row["validator_correct_choice_match_count"] = str(match_count)
        row["validator_shuffle_recalc_ok"] = "예" if shuffle_ok else "아니오"

        if action in {"accept", "audit"} and shuffle_ok:
            original_correct_choice = row.get("correct_choice", "")
            row["choice_a"] = shuffled_choices["A"]
            row["choice_b"] = shuffled_choices["B"]
            row["choice_c"] = shuffled_choices["C"]
            row["choice_d"] = shuffled_choices["D"]
            validator_replay.remap_label_keyed_metadata(row, original_correct_choice, target_label)
            row["correct_choice"] = recalculated_label or row.get("correct_choice", "")
            metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(row)
        else:
            metadata_ok = None
            metadata_reasons = []

        disposition, split_allowed, _ = salvage.replacement_export_disposition(action, metadata_ok)
        gate_fields = interpretation_pilot.row_gate_fields(row, match_count)
        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["validator_reason_short"] = reasons[0] if reasons else "validator_clean"
        row["metadata_remap_ok"] = "예" if metadata_ok is True else "아니오" if metadata_ok is False else "대상아님"
        row["metadata_remap_reasons"] = "|".join(metadata_reasons)
        row["validator_export_disposition"] = disposition
        row["split_allowed"] = split_allowed
        row["count_allowed"] = "아니오"
        row["count_disposition"] = COUNT_REFLECTION_STATUS
        row["export_correct_choice"] = row.get("correct_choice", "")
        row["batch_status"] = BATCH_STATUS
        row["count_reflection_status"] = COUNT_REFLECTION_STATUS
        row["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
        row["replacement_package_role"] = "replacement_candidate"
        row.update(gate_fields)

        if disposition == "export_ready":
            row["final_status"] = "pass"
            row["audit_required"] = "아니오"
            row["audit_reason"] = ""
            row["train_eligible"] = "예"
        elif disposition == "audit_queue":
            row["final_status"] = "pass"
            row["audit_required"] = "예"
            row["audit_reason"] = "validator_audit"
            row["train_eligible"] = "아니오"
        elif disposition == "regenerate_required":
            row["final_status"] = "soft_fail"
            row["audit_required"] = "아니오"
            row["train_eligible"] = "아니오"
        else:
            row["final_status"] = "hard_fail"
            row["audit_required"] = "아니오"
            row["train_eligible"] = "아니오"

        row["current_tail_class"] = classify_current_tail(row)
        row["interpretation_axis"] = infer_interpretation_axis(row)

        report_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "replacement_for_seed_sample_id": row.get("replacement_for_seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "source_tail_class": row.get("source_tail_class", ""),
                "current_tail_class": row.get("current_tail_class", ""),
                "tail_class": row.get("current_tail_class", ""),
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "validator_action": action,
                "validator_status": status,
                "validator_reasons": row["validator_reasons"],
                "validator_export_disposition": disposition,
                "metadata_remap_ok": row["metadata_remap_ok"],
                "answer_uniqueness": row.get("answer_uniqueness", ""),
                "distractor_direction": row.get("distractor_direction", ""),
                "same_direction_distractor": row.get("same_direction_distractor", ""),
                "same_direction_guard_ok": row.get("same_direction_guard_ok", ""),
                "interpretation_axis": row.get("interpretation_axis", ""),
                "split_allowed": row.get("split_allowed", ""),
                "count_allowed": row.get("count_allowed", ""),
                "final_status": row["final_status"],
                "train_eligible": row["train_eligible"],
                "audit_required": row["audit_required"],
                "export_correct_choice": row["export_correct_choice"],
            }
        )
        validated_rows.append(row)

    if rows:
        pb6.pb4.pb3.base.write_csv_atomic(REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH, rows, list(rows[0].keys()))
    pb6.pb4.pb3.base.write_csv_atomic(VALIDATOR_REPORT_CSV_PATH, report_rows, list(report_rows[0].keys()))
    salvage.write_validator_report_md(report_rows)
    return validated_rows


def write_validator_report_md(report_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# validator report `{VERSION_TAG}`",
        "",
        "## replacement actions",
        "| seed | failed_seed | source | lane | target | source_tail | current_tail | action | disposition | answer_uniqueness | direction | split/count | final |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['replacement_for_seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['target_correct_choice']}` | `{row['source_tail_class']}` | `{row['current_tail_class']}` | `{row['validator_action']}` | `{row['validator_export_disposition']}` | `{row['answer_uniqueness']}` | `{row['distractor_direction']}` | `{row['split_allowed']}/{row['count_allowed']}` | `{row['final_status']}` |"
        )
    pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_REPORT_MD_PATH, "\n".join(lines) + "\n")


def write_tail_memo(package: list[dict[str, str]]) -> list[dict[str, str]]:
    tail_rows = []
    for row in selected_rows(package):
        is_tail = row.get("train_eligible") != "예" or row.get("audit_required") == "예" or row.get("final_status") != "pass"
        if not is_tail:
            continue
        tail_rows.append(
            {
                "seed_sample_id": row.get("seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "validator_action": row.get("validator_action", ""),
                "validator_status": row.get("validator_status", ""),
                "validator_reasons": row.get("validator_reasons", ""),
                "validator_export_disposition": row.get("validator_export_disposition", ""),
                "target_correct_choice": row.get("target_correct_choice", ""),
                "export_correct_choice": row.get("export_correct_choice", ""),
                "source_tail_class": row.get("source_tail_class", row.get("replacement_tail_class", "")),
                "current_tail_class": row.get("current_tail_class", classify_current_tail(row)),
                "answer_uniqueness": row.get("answer_uniqueness", ""),
                "distractor_direction": row.get("distractor_direction", ""),
                "same_direction_distractor": row.get("same_direction_distractor", ""),
                "same_direction_guard_ok": row.get("same_direction_guard_ok", ""),
                "interpretation_axis": row.get("interpretation_axis", ""),
                "split_allowed": row.get("split_allowed", ""),
                "count_allowed": row.get("count_allowed", ""),
            }
        )
    fieldnames = [
        "seed_sample_id",
        "source_subset",
        "sampling_lane",
        "final_status",
        "audit_required",
        "error_tags",
        "validator_action",
        "validator_status",
        "validator_reasons",
        "validator_export_disposition",
        "target_correct_choice",
        "export_correct_choice",
        "source_tail_class",
        "current_tail_class",
        "answer_uniqueness",
        "distractor_direction",
        "same_direction_distractor",
        "same_direction_guard_ok",
        "interpretation_axis",
        "split_allowed",
        "count_allowed",
    ]
    pb6.pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, fieldnames)
    lines = [f"# tail memo `{VERSION_TAG}`", ""]
    if tail_rows:
        lines.extend(
            [
                "| seed | source | lane | status | action | source_tail | current_tail | answer_uniqueness | direction | split/count |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in tail_rows:
            lines.append(
                f"| `{row['seed_sample_id']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['validator_action']}` | `{row['source_tail_class']}` | `{row['current_tail_class']}` | `{row['answer_uniqueness']}` | `{row['distractor_direction']}` | `{row['split_allowed']}/{row['count_allowed']}` |"
            )
    else:
        lines.append("- tail_count: `0`")
        lines.append("- csv_policy: `header_only_no_placeholder_row`")
    pb6.pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def normalize_preserved_interpretation_row(row: dict[str, str]) -> dict[str, str]:
    normalized = ORIGINAL_SALVAGE_NORMALIZE_PRESERVED_ROW(row)
    normalized["replacement_package_role"] = "preserved_interpretation_repair_pilot_accept"
    normalized["batch_status"] = BATCH_STATUS
    normalized["count_reflection_status"] = COUNT_REFLECTION_STATUS
    normalized["count_disposition"] = COUNT_REFLECTION_STATUS
    normalized["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
    normalized["count_allowed"] = "아니오"
    return normalized


def sync_processed_outputs_with_replacement_fields() -> None:
    # salvage helper가 만든 split/manifest에 line-specific provenance field를 추가해 다음 count review parity를 맞춘다.
    if not MERGED_SCORES_PATH.exists():
        return
    package_rows = selected_rows(read_csv_rows(MERGED_SCORES_PATH))
    selected_by_problem_id = {row["candidate_id"]: row for row in package_rows}
    extra_fields = [
        "target_correct_choice",
        "export_correct_choice",
        "validator_action",
        "validator_export_disposition",
        "validator_reason_short",
        "validator_recalculated_correct_choice",
        "metadata_remap_ok",
        "split_allowed",
        "count_allowed",
        "count_disposition",
        "interpretation_axis",
        "interpretation_seed_action",
        "interpretation_failure_class",
        "replacement_for_seed_sample_id",
        "replacement_tail_class",
        "source_tail_class",
        "current_tail_class",
        "same_direction_guard_ok",
    ]
    for path in (PROBLEM_TRAIN_PATH, PROBLEM_DEV_PATH, PROBLEM_TEST_PATH):
        if not path.exists():
            continue
        payload_rows = pb6.pb4.pb3.base.load_jsonl(path)
        enriched_rows = []
        for payload in payload_rows:
            source = selected_by_problem_id.get(payload.get("problem_id", ""), {})
            enriched = dict(payload)
            enriched["count_disposition"] = COUNT_REFLECTION_STATUS
            for field in extra_fields:
                if field == "count_disposition":
                    enriched[field] = COUNT_REFLECTION_STATUS
                else:
                    enriched[field] = source.get(field, enriched.get(field, ""))
            enriched_rows.append(enriched)
        pb6.pb4.pb3.base.write_jsonl_atomic(path, enriched_rows)

    if PROBLEM_DATASET_MANIFEST_PATH.exists():
        manifest_rows = read_csv_rows(PROBLEM_DATASET_MANIFEST_PATH)
        enriched_rows = []
        for row in manifest_rows:
            source = selected_by_problem_id.get(row.get("problem_id", ""), {})
            enriched = dict(row)
            enriched["count_disposition"] = COUNT_REFLECTION_STATUS
            for field in extra_fields:
                if field == "count_disposition":
                    enriched[field] = COUNT_REFLECTION_STATUS
                else:
                    enriched[field] = source.get(field, enriched.get(field, ""))
            enriched_rows.append(enriched)
        fieldnames = []
        for row in enriched_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        pb6.pb4.pb3.base.write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, enriched_rows, fieldnames)


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict[str, object]:
    summary = salvage.compute_package_summary(merged_rows)
    tail_rows = read_csv_rows(TAIL_MEMO_CSV_PATH) if TAIL_MEMO_CSV_PATH.exists() else []
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "run_id": RUN_NAME,
        "run_dir": str(RUN_DIR),
        "created_at_utc": pb6.pb4.pb3.base.utc_now_iso(),
        "source_interpretation_pilot_run_name": SOURCE_PILOT_RUN_NAME,
        "source_pilot_run_name": SOURCE_PILOT_RUN_NAME,
        "source_interpretation_pilot_merged_path": str(SOURCE_PILOT_MERGED_PATH),
        "source_package_run_name": SOURCE_PILOT_RUN_NAME,
        "source_package_merged_path": str(SOURCE_PILOT_MERGED_PATH),
        "preserved_row_count": EXPECTED_PRESERVED_COUNT,
        "replacement_seed_count": len(seed_rows),
        "replacement_row_count": len(seed_rows),
        "seed_registry_csv_path": str(SEED_REGISTRY_PATH),
        "generation_count": pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": pb6.pb4.pb3.base.load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": pb6.pb4.pb3.base.load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": pb6.pb4.pb3.base.load_jsonl_count(DISTRACTORFIT_LOG_PATH),
        "judge_nearmiss_count": pb6.pb4.pb3.base.load_jsonl_count(NEARMISS_LOG_PATH),
        "final_package_count": summary["selected_count"],
        "selected_pass_count": summary["selected_pass_count"],
        "selected_hard_fail_count": summary["selected_hard_fail_count"],
        "selected_soft_fail_count": summary["selected_soft_fail_count"],
        "selected_train_eligible_count": summary["selected_train_eligible_count"],
        "selected_audit_required_count": summary["selected_audit_required_count"],
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": pb6.pb4.pb3.base.load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": pb6.pb4.pb3.base.load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "validator_summary": summary,
        "success_criteria": {
            "final_package_count": EXPECTED_FINAL_PACKAGE_COUNT,
            "usable_min": SUCCESS_USABLE_MIN,
            "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
            "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
            "audit_max": SUCCESS_AUDIT_MAX,
            "final_exact_export_label_balance": TARGET_LABEL_COUNTS,
            "shuffle_recalc_mismatch": 0,
            "metadata_remap_mismatch": 0,
            "count_reflection": "reviewer_signoff_required",
        },
        "success_result": {
            "usable": summary["selected_train_eligible_count"],
            "hard_fail": summary["selected_hard_fail_count"],
            "soft_fail": summary["selected_soft_fail_count"],
            "audit": summary["selected_audit_required_count"],
            "export_label_counts": summary["export_label_counts"],
            "selected_label_counts": summary["selected_label_counts"],
            "shuffle_recalc_mismatch_count": summary["shuffle_recalc_mismatch_count"],
            "metadata_remap_mismatch_count": summary["metadata_remap_mismatch_count"],
            "passed": summary["final_package_success_passed"],
        },
        "current_count_decision": "not_counted_until_reviewer_signoff_after_dslot_replacement_review",
        "count_reflection_status": COUNT_REFLECTION_STATUS,
        "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
        "tail_memo_count": len(tail_rows),
        "manifest_artifact_sync_status": "synced",
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight_csv": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
            "replacement_plan_csv": str(REPLACEMENT_PLAN_CSV_PATH),
            "replacement_plan_md": str(REPLACEMENT_PLAN_MD_PATH),
            "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
            "judge_nearmiss_log": str(NEARMISS_LOG_PATH),
            "replacement_merged_before_validator": str(REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "validator_report_csv": str(VALIDATOR_REPORT_CSV_PATH),
            "validator_report_md": str(VALIDATOR_REPORT_MD_PATH),
            "final_package_csv": str(FINAL_PACKAGE_CSV_PATH),
            "final_package_md": str(FINAL_PACKAGE_MD_PATH),
            "tail_memo_csv": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md": str(TAIL_MEMO_MD_PATH),
            "problem_train": str(PROBLEM_TRAIN_PATH),
            "problem_dev": str(PROBLEM_DEV_PATH),
            "problem_test": str(PROBLEM_TEST_PATH),
            "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
        },
    }
    pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def split_dataset_with_interpretation_replacement_package(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    manifest_rows = salvage.split_dataset_with_salvage_package(rows)
    sync_processed_outputs_with_replacement_fields()
    return manifest_rows


def configure_replacement_globals() -> None:
    salvage.VERSION_TAG = VERSION_TAG
    salvage.RUN_DATE = RUN_DATE
    salvage.RUN_PURPOSE = RUN_PURPOSE
    salvage.RUN_NAME = RUN_NAME
    salvage.SOURCE_PB9_RUN_NAME = SOURCE_PILOT_RUN_NAME
    salvage.SOURCE_PB9_RUN_DIR = SOURCE_PILOT_RUN_DIR
    salvage.SOURCE_PB9_MERGED_PATH = SOURCE_PILOT_MERGED_PATH
    salvage.SOURCE_PB9_SEED_REGISTRY_PATH = SOURCE_PILOT_SEED_REGISTRY_PATH
    salvage.INTERIM_DIR = INTERIM_DIR
    salvage.PROCESSED_DIR = PROCESSED_DIR
    salvage.RUN_DIR = RUN_DIR
    salvage.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    salvage.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    salvage.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    salvage.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    salvage.RUN_MERGED_DIR = RUN_MERGED_DIR
    salvage.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    salvage.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    salvage.SEED_READY_PATH = SEED_READY_PATH
    salvage.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    salvage.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    salvage.REPLACEMENT_PLAN_CSV_PATH = REPLACEMENT_PLAN_CSV_PATH
    salvage.REPLACEMENT_PLAN_MD_PATH = REPLACEMENT_PLAN_MD_PATH
    salvage.TARGET_LABEL_SCHEDULE_CSV_PATH = TARGET_LABEL_SCHEDULE_CSV_PATH
    salvage.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    salvage.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    salvage.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    salvage.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    salvage.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    salvage.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    salvage.REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH = REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH
    salvage.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    salvage.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    salvage.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    salvage.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    salvage.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    salvage.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    salvage.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    salvage.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    salvage.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    salvage.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    salvage.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    salvage.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    salvage.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    salvage.FINAL_PACKAGE_CSV_PATH = FINAL_PACKAGE_CSV_PATH
    salvage.FINAL_PACKAGE_MD_PATH = FINAL_PACKAGE_MD_PATH
    salvage.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    salvage.MANIFEST_HEADER_GATE_MD_PATH = MANIFEST_HEADER_GATE_MD_PATH
    salvage.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    salvage.EXPECTED_PRESERVED_COUNT = EXPECTED_PRESERVED_COUNT
    salvage.EXPECTED_FINAL_PACKAGE_COUNT = EXPECTED_FINAL_PACKAGE_COUNT
    salvage.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    salvage.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    salvage.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    salvage.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    salvage.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    salvage.REPLACEMENT_TARGET_LABEL_COUNTS = REPLACEMENT_TARGET_LABEL_COUNTS
    salvage.REPLACEMENT_SEED_ID_PREFIX = REPLACEMENT_SEED_ID_PREFIX
    salvage.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    salvage.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    salvage.PB9_REPLACEMENT_SOURCE_COUNTS = REPLACEMENT_SOURCE_COUNTS
    salvage.BATCH_STATUS = BATCH_STATUS
    salvage.COUNT_REFLECTION_STATUS = COUNT_REFLECTION_STATUS
    salvage.DOWNSTREAM_CONSUMPTION_ALLOWED = DOWNSTREAM_CONSUMPTION_ALLOWED
    salvage.REPLACEMENT_TARGETS = REPLACEMENT_TARGETS
    salvage.REPLACEMENT_PLAN_ROWS = []
    salvage.accepted_pb9_rows = accepted_interpretation_rows
    salvage.collect_excluded_rows = collect_excluded_rows
    salvage.select_fresh_record_for_target = select_fresh_record_for_target
    salvage.build_seed_row = build_seed_row
    salvage.build_generation_messages = build_generation_messages
    salvage.apply_validator_to_replacements = apply_validator_to_replacements
    salvage.write_validator_report_md = write_validator_report_md
    salvage.write_tail_memo = write_tail_memo
    salvage.build_run_manifest = build_run_manifest
    salvage.normalize_preserved_row = normalize_preserved_interpretation_row

    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = "interpretation repair accepted14 + D-slot replacement package"
    pb6.SEED_ID_PREFIX = REPLACEMENT_SEED_ID_PREFIX
    pb6.SEED_SELECTION_ROLE = "objective_interpretation_repair_dslot_replacement_seed"
    pb6.SEED_SELECTION_NOTE = "interpretation repair pilot accepted14 보존 후 D-slot hard/audit tail fresh replacement seed"
    pb6.SEED_FILTER_NOTE = "interpretation_repair_dslot_seen_seed_pool_excluded"
    pb6.SCOPE_NOTE = "해석례_QA only; accepted 14 preserved plus 2 fresh D-slot replacements; current count 미합산"
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.PB6_SOURCE_COUNTS = REPLACEMENT_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pb6.SUCCESS_LAW_ROW_COUNT = 0
    pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_interpretation_repair_dslot_replacement_package"
    pb6.SEED_REGISTRY_STRATEGY = "interpretation_repair_accepted14_plus_fresh_dslot_replacement"
    pb6.LAW_STATUS_NOTE = "interpretation_repair_replacement_candidate_not_counted"
    pb6.OVERLAP_CHECK_LABEL = "no current/failed/repaired/interpretation-pilot seed overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        "`objective_interpretation_repair_pilot` accepted `14개`는 보존하고 `004/008`만 fresh D-slot replacement로 대체한다.",
        "source pilot seed registry와 기존 reviewer-managed seed registry를 모두 seen seed로 제외한다.",
        "final `16-slot` package가 통과해도 reviewer sign-off 전 current count에는 합산하지 않는다.",
    ]
    pb6.INTERIM_DIR = INTERIM_DIR
    pb6.PROCESSED_DIR = PROCESSED_DIR
    pb6.RUN_DIR = RUN_DIR
    pb6.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb6.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb6.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb6.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb6.SEED_READY_PATH = SEED_READY_PATH
    pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb6.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb6.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb6.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb6.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb6.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb6.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb6.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb6.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb6.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb6.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb6.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb6.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb6.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb6.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb6.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pb6.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    pb6.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    pb6.build_seed_registry = salvage.build_seed_registry
    pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages
    pb6.build_batch_summary = salvage.build_batch_summary
    pb6.build_run_manifest = build_run_manifest
    pb6.pb4.pb3.base.split_dataset = split_dataset_with_interpretation_replacement_package


def main() -> dict[str, object]:
    configure_replacement_globals()
    return pb6.main()


if __name__ == "__main__":
    main()
