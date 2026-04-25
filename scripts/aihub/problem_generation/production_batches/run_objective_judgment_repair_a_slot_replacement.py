from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# reviewer가 요구한 다음 stop line은 전체 `판결문_QA` 재실행이 아니라,
# accepted 15개를 보존하고 `judgment_repair_preflight_013` A-slot만 fresh seed로 대체하는 것이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_repair_pilot as judgment_pilot,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_repair_pilot_seed_preflight as preflight,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb6_non_law as pb6,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_accepted34_6slot_replacement as salvage,
)
from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402


VERSION_TAG = "objective_judgment_repair_a_slot_replacement_package"
RUN_PURPOSE = "objective_r2_judgment_repair_a_slot_fresh_replacement"


# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 공용 helper로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PILOT_RUN_NAME = judgment_pilot.RUN_NAME
SOURCE_PILOT_RUN_DIR = judgment_pilot.RUN_DIR
SOURCE_PILOT_MERGED_PATH = judgment_pilot.MERGED_SCORES_PATH
SOURCE_PILOT_SEED_REGISTRY_PATH = judgment_pilot.SEED_REGISTRY_PATH

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
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

EXPECTED_TOTAL_SEED_COUNT = 1
EXPECTED_PRESERVED_COUNT = 15
EXPECTED_FINAL_PACKAGE_COUNT = 16
SUCCESS_USABLE_MIN = 16
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
REPLACEMENT_TARGET_LABEL_COUNTS = {"A": 1}
REPLACEMENT_SEED_ID_PREFIX = "judgment_repair_a_slot"
EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 1}
EXPECTED_LANE_BY_DOC = {("판결문_QA", "generalization_03_04"): 1}
REPLACEMENT_SOURCE_COUNTS = {"04_TL_판결문_QA": 1}
BATCH_STATUS = "judgment_repair_a_slot_replacement_candidate_not_counted"
COUNT_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"
ORIGINAL_NORMALIZE_PRESERVED_ROW = salvage.normalize_preserved_row

REPLACEMENT_TARGETS = [
    {
        "failed_seed_sample_id": "judgment_repair_preflight_013",
        "source_subset": "04_TL_판결문_QA",
        "sampling_lane": "generalization_03_04",
        "target_correct_choice": "A",
        "tail_class": "judgment_weak_distractor_nearmiss_audit",
    }
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def selected_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def accepted_judgment_rows() -> list[dict[str, str]]:
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
        raise RuntimeError(f"judgment accepted row count must be {EXPECTED_PRESERVED_COUNT}: {len(accepted)}")
    return accepted


def collect_excluded_rows() -> list[dict[str, str]]:
    # 같은 `013` seed retry가 아니라 fresh A-slot replacement로 남도록,
    # 기존 exclusion pool과 source pilot seed registry를 모두 seen seed로 묶는다.
    rows = preflight.collect_excluded_rows()
    rows.extend(salvage.load_csv_rows_if_exists(SOURCE_PILOT_SEED_REGISTRY_PATH))
    return rows


def matching_specs(source_subset: str, sampling_lane: str) -> list[dict[str, str]]:
    specs = []
    for spec in pb6.pb4.pb3.DATASET_SPECS:
        if (
            spec["doc_type_name"] == "판결문_QA"
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
            passes_filter, _ = preflight.passes_judgment_seed_filter(spec, payload)
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
                "selection_note": f"judgment A-slot fresh replacement for {target['failed_seed_sample_id']}",
            }
    raise RuntimeError(f"fresh judgment replacement seed not found for {target}")


def build_seed_row(record: dict[str, str]) -> dict[str, str]:
    row = pb6.pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_judgment_repair_a_slot_replacement_seed"
    row["selection_note"] = record["selection_note"]
    # 기존 salvage helper와의 호환용 `pb9_*` field는 유지하되, reviewer-facing artifact에는
    # judgment line 전용 alias도 함께 남겨 `pb9` package처럼 오해되지 않게 한다.
    row["replacement_for_seed_sample_id"] = record["replacement_for_seed_sample_id"]
    row["replacement_tail_class"] = record["replacement_tail_class"]
    row["pb9_replacement_for_seed_sample_id"] = record["replacement_for_seed_sample_id"]
    row["pb9_replacement_tail_class"] = record["replacement_tail_class"]
    row["target_correct_choice"] = record["target_correct_choice"]
    row["judgment_replacement_scope_note"] = "accepted15_preserved_plus_1_fresh_a_slot_replacement_not_counted_until_signoff"
    preflight.augment_seed_row(row, 0)
    row["target_correct_choice"] = record["target_correct_choice"]
    return row


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = judgment_pilot.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## judgment A-slot replacement 추가 지시
- 이번 seed는 `{seed.get('replacement_for_seed_sample_id') or seed.get('pb9_replacement_for_seed_sample_id', '')}`의 A-slot weak distractor audit tail을 fresh seed로 대체하기 위한 단일 replacement다.
- target label은 `{seed.get('target_correct_choice', '')}`이며, final package의 `A/B/C/D = 4/4/4/4` 균형을 회복해야 한다.
- 정답은 판결문상 하나의 판단 기준 또는 적용 사실에만 닫혀야 한다.
- 오답 3개는 같은 판결문 anchor를 공유하되, 각각 청구·쟁점·판단기준·적용사실 중 정확히 한 축만 어긋나야 한다.
- 오답이 너무 포괄적이거나 정답과 멀면 `weak distractor`로, 별도 쟁점으로도 정답 가능하면 `answer uniqueness failure`로 본다.
"""
    return messages


def choose_replacement_validator_action(row: dict[str, str], answer_match_count: int) -> tuple[str, str, list[str]]:
    return judgment_pilot.choose_judgment_validator_action(row, answer_match_count)


def apply_validator_to_replacements(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    target_by_seed = salvage.target_schedule_by_seed()
    metadata_by_seed = salvage.replacement_seed_metadata_by_seed()
    validated_rows = []
    report_rows = []
    for row in selected_rows(rows):
        seed_metadata = metadata_by_seed.get(row["seed_sample_id"], {})
        # row-level provenance와 reviewer-facing alias를 먼저 채워 이후 report/CSV가 line-neutral하게 읽히게 한다.
        row["replacement_for_seed_sample_id"] = seed_metadata.get("replacement_for_seed_sample_id") or seed_metadata.get("pb9_replacement_for_seed_sample_id", "")
        row["replacement_tail_class"] = seed_metadata.get("replacement_tail_class") or seed_metadata.get("pb9_replacement_tail_class", "")
        row["pb9_replacement_for_seed_sample_id"] = seed_metadata.get("pb9_replacement_for_seed_sample_id", "")
        row["pb9_replacement_tail_class"] = seed_metadata.get("pb9_replacement_tail_class", "")
        target_label = target_by_seed[row["seed_sample_id"]]
        choices = validator_replay.choice_map(row)
        shuffled_choices, recalculated_label, match_count = validator_replay.shuffled_choices_for_target(
            choices,
            row.get("correct_choice", ""),
            target_label,
        )
        action, status, reasons = choose_replacement_validator_action(row, match_count)
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
        row["validator_action"] = action
        row["validator_status"] = status
        row["validator_reasons"] = "|".join(reasons)
        row["metadata_remap_ok"] = "예" if metadata_ok is True else "아니오" if metadata_ok is False else "대상아님"
        row["metadata_remap_reasons"] = "|".join(metadata_reasons)
        row["validator_export_disposition"] = disposition
        row["split_allowed"] = split_allowed
        row["count_allowed"] = "아니오"
        row["export_correct_choice"] = row.get("correct_choice", "")
        row["batch_status"] = BATCH_STATUS
        row["count_reflection_status"] = COUNT_REFLECTION_STATUS
        row["downstream_consumption_allowed"] = DOWNSTREAM_CONSUMPTION_ALLOWED
        row["replacement_package_role"] = "replacement_candidate"

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

        report_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "replacement_for_seed_sample_id": row.get("replacement_for_seed_sample_id") or row.get("pb9_replacement_for_seed_sample_id", ""),
                "source_subset": row.get("source_subset", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "tail_class": row.get("replacement_tail_class") or row.get("pb9_replacement_tail_class", ""),
                "target_correct_choice": target_label,
                "recalculated_correct_choice": row["validator_recalculated_correct_choice"],
                "validator_action": action,
                "validator_status": status,
                "validator_reasons": row["validator_reasons"],
                "validator_export_disposition": disposition,
                "metadata_remap_ok": row["metadata_remap_ok"],
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


def normalize_preserved_judgment_row(row: dict[str, str]) -> dict[str, str]:
    # `pb9` salvage helper를 재사용하더라도 보존 row의 역할명은 judgment repair source를 가리켜야 한다.
    normalized = ORIGINAL_NORMALIZE_PRESERVED_ROW(row)
    normalized["replacement_package_role"] = "preserved_judgment_repair_pilot_accept"
    return normalized


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict[str, object]:
    summary = salvage.compute_package_summary(merged_rows)
    tail_rows = read_csv_rows(TAIL_MEMO_CSV_PATH)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "run_id": RUN_NAME,
        "run_dir": str(RUN_DIR),
        "created_at_utc": pb6.pb4.pb3.base.utc_now_iso(),
        "source_judgment_pilot_run_name": SOURCE_PILOT_RUN_NAME,
        "source_judgment_pilot_merged_path": str(SOURCE_PILOT_MERGED_PATH),
        "preserved_row_count": EXPECTED_PRESERVED_COUNT,
        "replacement_row_count": len(seed_rows),
        "replacement_seed_count": len(seed_rows),
        "seed_registry_csv_path": str(SEED_REGISTRY_PATH),
        "seed_preflight_csv_path": str(SEED_PREFLIGHT_CSV_PATH),
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
        "current_count_decision": "not_counted_until_reviewer_signoff_after_a_slot_replacement_review",
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
    salvage.accepted_pb9_rows = accepted_judgment_rows
    salvage.collect_excluded_rows = collect_excluded_rows
    salvage.select_fresh_record_for_target = select_fresh_record_for_target
    salvage.build_seed_row = build_seed_row
    salvage.build_generation_messages = build_generation_messages
    salvage.apply_validator_to_replacements = apply_validator_to_replacements
    salvage.build_run_manifest = build_run_manifest
    salvage.normalize_preserved_row = normalize_preserved_judgment_row

    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = "judgment repair accepted15 + A-slot replacement package"
    pb6.SEED_ID_PREFIX = REPLACEMENT_SEED_ID_PREFIX
    pb6.SEED_SELECTION_ROLE = "objective_judgment_repair_a_slot_replacement_seed"
    pb6.SEED_SELECTION_NOTE = "judgment repair pilot accepted15 보존 후 A-slot audit tail fresh replacement seed"
    pb6.SEED_FILTER_NOTE = "judgment_repair_a_slot_seen_seed_pool_excluded"
    pb6.SCOPE_NOTE = "판결문_QA only; accepted 15 preserved plus 1 fresh A-slot replacement; current count 미합산"
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
    pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_judgment_repair_a_slot_replacement_package"
    pb6.SEED_REGISTRY_STRATEGY = "judgment_repair_accepted15_plus_fresh_a_slot_replacement"
    pb6.LAW_STATUS_NOTE = "judgment_repair_replacement_candidate_not_counted"
    pb6.OVERLAP_CHECK_LABEL = "no current/failed/repaired/judgment-pilot seed overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        "`objective_judgment_repair_pilot` accepted `15개`는 보존하고 `judgment_repair_preflight_013`만 fresh A-slot replacement로 대체한다.",
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
    pb6.pb4.pb3.base.split_dataset = salvage.split_dataset_with_salvage_package


def main() -> dict[str, object]:
    configure_replacement_globals()
    return pb6.main()


if __name__ == "__main__":
    main()
