from __future__ import annotations

import csv
import sys
from pathlib import Path

# `pb9` salvage가 `39/40`까지 좁혀졌으므로, 전체 batch를 다시 열지 않고
# 남은 C-slot weak distractor tail 1개만 fresh seed로 대체해 final package를 재조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_accepted34_6slot_replacement as salvage,
)


VERSION_TAG = "pb9_cslot_final_replacement_package"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_pb9_remaining_cslot_salvage_package"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = salvage.PROJECT_ROOT
SOURCE_SALVAGE_RUN_NAME = (
    "2026-04-26_053025_pb9_accepted34_6slot_replacement_package_objective_r2_pb9_tail_6slot_salvage_package"
)
SOURCE_SALVAGE_RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / SOURCE_SALVAGE_RUN_NAME
SOURCE_SALVAGE_MERGED_PATH = (
    SOURCE_SALVAGE_RUN_DIR / "merged" / "merged_problem_scores_pb9_accepted34_6slot_replacement_package.csv"
)
SOURCE_SALVAGE_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb9_accepted34_6slot_replacement_package"
    / "seed_registry.csv"
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
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"

REPLACEMENT_TARGET = {
    "failed_seed_sample_id": "pb9_replacement_006",
    "source_subset": "04_TL_결정례_QA",
    "sampling_lane": "generalization_03_04",
    "target_correct_choice": "C",
    "tail_class": "04tl_weak_distractor_final_cslot",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def accepted_salvage_rows() -> list[dict[str, str]]:
    rows = salvage.selected_rows(read_csv_rows(SOURCE_SALVAGE_MERGED_PATH))
    accepted = [
        dict(row)
        for row in rows
        if row.get("validator_export_disposition") == "export_ready"
        and row.get("train_eligible") == "예"
        and row.get("audit_required") == "아니오"
        and row.get("final_status") == "pass"
    ]
    if len(accepted) != 39:
        raise RuntimeError(f"pb9 salvage accepted row count must be 39: {len(accepted)}")
    return accepted


def collect_excluded_rows_for_cslot() -> list[dict[str, str]]:
    # 기존 current/failed/repair seed와 이번 failed replacement seed를 모두 제외해야
    # 마지막 C-slot도 같은 family 재시도가 아니라 fresh replacement로 해석된다.
    rows = salvage.pb9_smoke.collect_excluded_rows()
    rows.extend(salvage.load_csv_rows_if_exists(salvage.SOURCE_PB9_SEED_REGISTRY_PATH))
    rows.extend(salvage.load_csv_rows_if_exists(salvage.REFERENCE_04TL_CALIBRATION_SEED_REGISTRY_PATH))
    rows.extend(salvage.load_csv_rows_if_exists(salvage.REFERENCE_04TL_CSLOT_REPLACEMENT_SEED_REGISTRY_PATH))
    rows.extend(salvage.load_csv_rows_if_exists(SOURCE_SALVAGE_SEED_REGISTRY_PATH))
    for source_row in salvage.selected_rows(read_csv_rows(SOURCE_SALVAGE_MERGED_PATH)):
        if source_row.get("seed_sample_id") == "pb9_replacement_006":
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


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = salvage.pb9_api.build_generation_messages(seed, reference_v2)
    messages[1]["content"] += f"""

## pb9 remaining C-slot final replacement 추가 지시
- 이번 seed는 `pb9` salvage package에서 남은 `{seed.get('pb9_replacement_for_seed_sample_id', '')}` C-slot weak distractor tail을 fresh seed로 대체하기 위한 마지막 replacement다.
- target label은 반드시 `{seed.get('target_correct_choice', '')}`이며, final package의 `A/B/C/D = 10/10/10/10` 균형을 회복해야 한다.
- 정답은 반드시 `gold_short_answer`의 판단 기준과 닫혀야 한다.
- 오답 3개는 정답과 같은 결정 이유, 판단 기준, 적용 사실 중 하나를 공유하되 각각 정확히 한 축만 다르게 비튼다.
- 오답 중 하나라도 일반 정의, 기관 역할, 상식적 반대말, 너무 포괄적인 문장, 법적 anchor가 먼 문장으로 빠지면 weak distractor로 본다.
- 마지막 C-slot replacement이므로 `오답약함`, `near_miss_부족`, `all_three_near_miss = 아니오`가 나오면 package 전체가 count sign-off로 갈 수 없다.
"""
    return messages


def configure_cslot_globals() -> None:
    salvage.VERSION_TAG = VERSION_TAG
    salvage.RUN_DATE = RUN_DATE
    salvage.RUN_PURPOSE = RUN_PURPOSE
    salvage.RUN_NAME = RUN_NAME
    salvage.SOURCE_PB9_RUN_NAME = SOURCE_SALVAGE_RUN_NAME
    salvage.SOURCE_PB9_RUN_DIR = SOURCE_SALVAGE_RUN_DIR
    salvage.SOURCE_PB9_MERGED_PATH = SOURCE_SALVAGE_MERGED_PATH
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
    salvage.EXPECTED_TOTAL_SEED_COUNT = 1
    salvage.EXPECTED_PRESERVED_COUNT = 39
    salvage.EXPECTED_FINAL_PACKAGE_COUNT = 40
    salvage.SUCCESS_USABLE_MIN = 40
    salvage.SUCCESS_HARD_FAIL_MAX = 0
    salvage.SUCCESS_SOFT_FAIL_MAX = 0
    salvage.SUCCESS_AUDIT_MAX = 0
    salvage.REPLACEMENT_TARGET_LABEL_COUNTS = {"C": 1}
    salvage.REPLACEMENT_SEED_ID_PREFIX = "pb9_cslot_replacement"
    salvage.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 1}
    salvage.EXPECTED_LANE_BY_DOC = {("결정례_QA", "generalization_03_04"): 1}
    salvage.PB9_REPLACEMENT_SOURCE_COUNTS = {"04_TL_결정례_QA": 1}
    salvage.BATCH_STATUS = "pb9_final_cslot_replacement_candidate_not_counted"
    salvage.COUNT_REFLECTION_STATUS = "not_counted"
    salvage.DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"
    salvage.REPLACEMENT_TARGETS = [REPLACEMENT_TARGET]
    salvage.REPLACEMENT_PLAN_ROWS = []
    salvage.accepted_pb9_rows = accepted_salvage_rows
    salvage.collect_excluded_rows = collect_excluded_rows_for_cslot
    salvage.build_generation_messages = build_generation_messages


def main() -> dict[str, object]:
    configure_cslot_globals()
    return salvage.main()


if __name__ == "__main__":
    main()
