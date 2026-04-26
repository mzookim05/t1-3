from __future__ import annotations

import sys
from pathlib import Path

# `205615` package는 D-slot 1개만 남겼으므로, 같은 seed retry가 아니라
# export-ready 15개를 보존하고 fresh D-slot 1개로 final 16-slot package를 다시 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_repair_dslot_replacement as dslot_base,
)


VERSION_TAG = "objective_interpretation_repair_dslot_final_replacement_package"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_interpretation_repair_remaining_dslot_fresh_replacement"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = dslot_base.PROJECT_ROOT
SOURCE_PACKAGE_VERSION_TAG = "objective_interpretation_repair_dslot_replacement_package"
SOURCE_PACKAGE_RUN_PURPOSE = "objective_r2_interpretation_repair_dslot_fresh_replacement"
SOURCE_PACKAGE_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / SOURCE_PACKAGE_VERSION_TAG
    / "seed_registry.csv"
)


def read_seed_registry_keys(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    # source package run stamp는 실행 시점마다 달라질 수 있으므로,
    # 하드코딩된 날짜 대신 locked seed registry와 동일한 run artifact를 찾아 provenance를 고정한다.
    rows = dslot_base.read_csv_rows(path)
    return [
        (
            row.get("seed_sample_id", ""),
            row.get("reference_sample_id", ""),
            row.get("family_id", ""),
            row.get("label_path", ""),
            row.get("raw_path", ""),
            row.get("target_correct_choice", ""),
        )
        for row in rows
    ]


def resolve_source_package_run_dir() -> Path:
    # 같은 목적의 run이 여러 개 있어도 실제 source package seed 구성이 같은 최신 artifact만 채택한다.
    llm_runs_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    pattern = f"*_{SOURCE_PACKAGE_VERSION_TAG}_{SOURCE_PACKAGE_RUN_PURPOSE}"
    source_registry_keys = read_seed_registry_keys(SOURCE_PACKAGE_SEED_REGISTRY_PATH)
    matched_dirs: list[Path] = []
    for candidate_dir in sorted(llm_runs_root.glob(pattern)):
        candidate_registry = candidate_dir / "inputs" / "seed_registry.csv"
        if not candidate_registry.exists():
            continue
        if read_seed_registry_keys(candidate_registry) == source_registry_keys:
            matched_dirs.append(candidate_dir)
    if not matched_dirs:
        raise FileNotFoundError(f"source package run not found for {SOURCE_PACKAGE_SEED_REGISTRY_PATH}")
    return matched_dirs[-1]


SOURCE_PACKAGE_RUN_DIR = resolve_source_package_run_dir()
SOURCE_PACKAGE_RUN_NAME = SOURCE_PACKAGE_RUN_DIR.name
SOURCE_PACKAGE_MERGED_PATH = (
    SOURCE_PACKAGE_RUN_DIR
    / "merged"
    / f"merged_problem_scores_{SOURCE_PACKAGE_VERSION_TAG}.csv"
)

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

EXPECTED_TOTAL_SEED_COUNT = 1
EXPECTED_PRESERVED_COUNT = 15
EXPECTED_FINAL_PACKAGE_COUNT = 16
SUCCESS_USABLE_MIN = 16
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
REPLACEMENT_TARGET_LABEL_COUNTS = {"D": 1}
REPLACEMENT_SEED_ID_PREFIX = "interpretation_dslot_final_replacement"
EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 1}
EXPECTED_LANE_BY_DOC = {("해석례_QA", "expansion_01_02"): 1}
REPLACEMENT_SOURCE_COUNTS = {"02_TL_유권해석_QA": 1}
BATCH_STATUS = "interpretation_repair_dslot_final_replacement_candidate_not_counted"
COUNT_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

REPLACEMENT_TARGETS = [
    {
        "failed_seed_sample_id": "interpretation_dslot_replacement_002",
        "source_subset": "02_TL_유권해석_QA",
        "sampling_lane": "expansion_01_02",
        "target_correct_choice": "D",
        "tail_class": "answer_uniqueness_same_direction_failure",
    },
]


def configure_final_replacement_globals() -> None:
    # 기존 D-slot replacement runner의 검증 로직은 그대로 쓰되, source package와 남은 slot 범위만 1개로 좁힌다.
    dslot_base.VERSION_TAG = VERSION_TAG
    dslot_base.RUN_DATE = RUN_DATE
    dslot_base.RUN_PURPOSE = RUN_PURPOSE
    dslot_base.RUN_NAME = RUN_NAME
    dslot_base.SOURCE_PILOT_RUN_DIR = SOURCE_PACKAGE_RUN_DIR
    dslot_base.SOURCE_PILOT_RUN_NAME = SOURCE_PACKAGE_RUN_NAME
    dslot_base.SOURCE_PILOT_MERGED_PATH = SOURCE_PACKAGE_MERGED_PATH
    dslot_base.SOURCE_PILOT_SEED_REGISTRY_PATH = SOURCE_PACKAGE_SEED_REGISTRY_PATH
    dslot_base.INTERIM_DIR = INTERIM_DIR
    dslot_base.PROCESSED_DIR = PROCESSED_DIR
    dslot_base.RUN_DIR = RUN_DIR
    dslot_base.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    dslot_base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    dslot_base.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    dslot_base.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    dslot_base.RUN_MERGED_DIR = RUN_MERGED_DIR
    dslot_base.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    dslot_base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    dslot_base.SEED_READY_PATH = SEED_READY_PATH
    dslot_base.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    dslot_base.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    dslot_base.REPLACEMENT_PLAN_CSV_PATH = REPLACEMENT_PLAN_CSV_PATH
    dslot_base.REPLACEMENT_PLAN_MD_PATH = REPLACEMENT_PLAN_MD_PATH
    dslot_base.TARGET_LABEL_SCHEDULE_CSV_PATH = TARGET_LABEL_SCHEDULE_CSV_PATH
    dslot_base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    dslot_base.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    dslot_base.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    dslot_base.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    dslot_base.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    dslot_base.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    dslot_base.REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH = REPLACEMENT_MERGED_BEFORE_VALIDATOR_PATH
    dslot_base.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    dslot_base.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    dslot_base.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    dslot_base.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    dslot_base.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    dslot_base.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    dslot_base.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    dslot_base.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    dslot_base.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    dslot_base.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    dslot_base.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    dslot_base.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    dslot_base.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    dslot_base.FINAL_PACKAGE_CSV_PATH = FINAL_PACKAGE_CSV_PATH
    dslot_base.FINAL_PACKAGE_MD_PATH = FINAL_PACKAGE_MD_PATH
    dslot_base.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    dslot_base.MANIFEST_HEADER_GATE_MD_PATH = MANIFEST_HEADER_GATE_MD_PATH
    dslot_base.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    dslot_base.EXPECTED_PRESERVED_COUNT = EXPECTED_PRESERVED_COUNT
    dslot_base.EXPECTED_FINAL_PACKAGE_COUNT = EXPECTED_FINAL_PACKAGE_COUNT
    dslot_base.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    dslot_base.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    dslot_base.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    dslot_base.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    dslot_base.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    dslot_base.REPLACEMENT_TARGET_LABEL_COUNTS = REPLACEMENT_TARGET_LABEL_COUNTS
    dslot_base.REPLACEMENT_SEED_ID_PREFIX = REPLACEMENT_SEED_ID_PREFIX
    dslot_base.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    dslot_base.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    dslot_base.REPLACEMENT_SOURCE_COUNTS = REPLACEMENT_SOURCE_COUNTS
    dslot_base.BATCH_STATUS = BATCH_STATUS
    dslot_base.COUNT_REFLECTION_STATUS = COUNT_REFLECTION_STATUS
    dslot_base.DOWNSTREAM_CONSUMPTION_ALLOWED = DOWNSTREAM_CONSUMPTION_ALLOWED
    dslot_base.REPLACEMENT_TARGETS = REPLACEMENT_TARGETS


def main() -> dict[str, object]:
    configure_final_replacement_globals()
    return dslot_base.main()


if __name__ == "__main__":
    main()
