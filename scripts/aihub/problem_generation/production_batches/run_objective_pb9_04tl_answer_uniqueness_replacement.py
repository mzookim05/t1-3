from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# `pb9_04tl_calibration_003`은 weak distractor tail이 아니라 answer uniqueness hard fail이므로,
# 같은 seed retry 대신 failed seed를 제외한 fresh C-slot replacement 1개만 확인한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb9_04tl_calibration_pilot as pilot,
)


VERSION_TAG = "pb9_04tl_decision_answer_uniqueness_1slot_replacement"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_04tl_cslot_answer_uniqueness_replacement"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pilot.PROJECT_ROOT
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
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"

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
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
JUDGE_STRUCTURED_CONTRACT_MD_PATH = RUN_EXPORTS_DIR / f"judge_structured_contract_{VERSION_TAG}.md"
STRUCTURED_FIELD_GATE_MD_PATH = RUN_EXPORTS_DIR / f"structured_field_gate_{VERSION_TAG}.md"

SOURCE_COUNTS = {"04_TL_결정례_QA": 1}
EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 1}
EXPECTED_LANE_BY_DOC = {("결정례_QA", "generalization_03_04"): 1}
TARGET_LABEL_COUNTS = {"C": 1}
SUCCESS_USABLE_MIN = 1
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 0
SUCCESS_LAW_ROW_COUNT = 0

BATCH_STATUS = "replacement_not_counted"
COUNT_REFLECTION_STATUS = "not_counted"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"

ORIGINAL_COLLECT_EXCLUDED_ROWS = pilot.collect_excluded_rows_for_calibration
ORIGINAL_BUILD_GENERATION_MESSAGES = pilot.build_generation_messages


def collect_excluded_rows_for_replacement() -> list[dict[str, str]]:
    # 기존 current/failed/pb9 seen seed와 직전 calibration 8개를 모두 제외해야
    # `003`만 갈아끼운 fresh replacement라는 해석이 유지된다.
    rows = ORIGINAL_COLLECT_EXCLUDED_ROWS()
    rows.extend(pilot.load_csv_rows_if_exists(pilot.SEED_REGISTRY_PATH))
    return rows


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## 04TL answer-uniqueness C-slot replacement 추가 지시
- 이번 run은 `pb9_04tl_calibration_003`의 answer uniqueness hard fail을 fresh seed 1개로 대체하는 C-slot replacement다.
- 정답 선택지는 반드시 `gold_short_answer`의 법적 판단 기준과 같은 결론이어야 한다.
- 오답은 `gold_reference_explanation`에 등장하는 다른 독립 각하/기각/배척 사유를 그대로 충족하면 안 된다.
- 복수 각하 사유, 기간 경과, 원처분주의, 청구기간, 당사자적격처럼 여러 기준이 함께 있으면 stem이 묻는 단일 기준 하나에만 정답을 고정한다.
- 다른 choice가 별도 독립 사유로도 정답 가능하게 읽히면 answer uniqueness failure로 본다.
"""
    return messages


def write_target_label_schedule(selected: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for row in selected:
        rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "candidate_id": row["candidate_id"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "target_correct_choice": "C",
            }
        )
    counts = Counter(row["target_correct_choice"] for row in rows)
    if dict(counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"C-slot replacement target label mismatch: {dict(counts)}")
    pilot.micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pilot.micro.pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    return rows


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = pilot.micro.pb8.pb6.pb4.pb3.summarize_rows(rows)
    selected = pilot.selected_rows(rows)
    summary_rows = [
        {"metric": "seed_count", "value": str(len(selected))},
        {"metric": "selected_pass", "value": str(summary["selected_pass_count"])},
        {"metric": "selected_hard_fail", "value": str(summary["selected_hard_fail_count"])},
        {"metric": "selected_soft_fail", "value": str(summary["selected_soft_fail_count"])},
        {"metric": "train_eligible", "value": str(summary["selected_train_eligible_count"])},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "pilot_success_passed", "value": str(pilot.VALIDATOR_SUMMARY.get("pilot_success_passed", False))},
    ]
    pilot.micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    pilot.micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(
        BATCH_LANE_SUMMARY_CSV_PATH,
        [{"sampling_lane": "generalization_03_04", "count": str(len(selected))}],
        ["sampling_lane", "count"],
    )
    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected)}`",
        "- doc_type_counts: `{'결정례_QA': 1}`",
        "- lane_counts: `{'generalization_03_04': 1}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        f"- validator_action_counts: `{pilot.VALIDATOR_SUMMARY.get('validator_action_counts', {})}`",
        f"- replacement_success_passed: `{pilot.VALIDATOR_SUMMARY.get('pilot_success_passed', False)}`",
        "",
        "## success criteria",
        "| criterion | target | result |",
        "| --- | --- | --- |",
        f"| usable | `>= {SUCCESS_USABLE_MIN} / 1` | `{summary['selected_train_eligible_count']}` |",
        f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
        f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
        f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
        f"| structured field missing/parse | `0` | `{pilot.VALIDATOR_SUMMARY.get('structured_missing_count', 0)}` |",
        f"| weak label consistency failure | `0` | `{pilot.VALIDATOR_SUMMARY.get('weak_label_consistency_failure_count', 0)}` |",
        f"| export-ready weak distractor | `0` | `{pilot.VALIDATOR_SUMMARY.get('export_ready_weak_distractor_count', 0)}` |",
        f"| export-ready all_three_near_miss = 아니오 | `0` | `{pilot.VALIDATOR_SUMMARY.get('export_ready_all_three_near_miss_no_count', 0)}` |",
        f"| shuffle/metadata mismatch | `0` | `shuffle {pilot.VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)} / metadata {pilot.VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |",
        "- count_reflection: `not_counted`",
    ]
    pilot.micro.pb8.pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def configure_replacement_globals() -> None:
    # 검증된 `pb9_04tl` calibration runner를 재사용하되, seed 수와 target label만 C-slot 1개로 좁힌다.
    pilot.VERSION_TAG = VERSION_TAG
    pilot.RUN_DATE = RUN_DATE
    pilot.RUN_PURPOSE = RUN_PURPOSE
    pilot.RUN_NAME = RUN_NAME
    pilot.INTERIM_DIR = INTERIM_DIR
    pilot.PROCESSED_DIR = PROCESSED_DIR
    pilot.RUN_DIR = RUN_DIR
    pilot.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pilot.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pilot.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pilot.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pilot.RUN_MERGED_DIR = RUN_MERGED_DIR
    pilot.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pilot.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pilot.SEED_READY_PATH = SEED_READY_PATH
    pilot.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pilot.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pilot.TARGET_LABEL_SCHEDULE_CSV_PATH = TARGET_LABEL_SCHEDULE_CSV_PATH
    pilot.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pilot.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pilot.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pilot.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pilot.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pilot.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pilot.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pilot.RAW_MERGED_BEFORE_VALIDATOR_PATH = RAW_MERGED_BEFORE_VALIDATOR_PATH
    pilot.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pilot.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pilot.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pilot.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pilot.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pilot.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pilot.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pilot.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pilot.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    pilot.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    pilot.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    pilot.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    pilot.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    pilot.JUDGE_STRUCTURED_CONTRACT_MD_PATH = JUDGE_STRUCTURED_CONTRACT_MD_PATH
    pilot.STRUCTURED_FIELD_GATE_MD_PATH = STRUCTURED_FIELD_GATE_MD_PATH
    pilot.SOURCE_COUNTS = SOURCE_COUNTS
    pilot.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pilot.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pilot.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    pilot.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    pilot.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    pilot.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    pilot.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    pilot.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    pilot.BATCH_STATUS = BATCH_STATUS
    pilot.COUNT_REFLECTION_STATUS = COUNT_REFLECTION_STATUS
    pilot.DOWNSTREAM_CONSUMPTION_ALLOWED = DOWNSTREAM_CONSUMPTION_ALLOWED
    pilot.VALIDATOR_SUMMARY = {}
    pilot.collect_excluded_rows_for_calibration = collect_excluded_rows_for_replacement
    pilot.build_generation_messages = build_generation_messages
    pilot.write_target_label_schedule = write_target_label_schedule
    pilot.build_batch_summary = build_batch_summary


def main() -> dict:
    configure_replacement_globals()
    pilot.configure_calibration_pilot_globals()
    pilot.micro.configure_micro_globals()
    pilot.micro.pb8.pb6.RUN_LABEL = "pb9 04TL answer uniqueness C-slot replacement"
    pilot.micro.pb8.pb6.SEED_ID_PREFIX = "pb9_04tl_cslot_replacement"
    pilot.micro.pb8.pb6.SEED_SELECTION_ROLE = "objective_pb9_04tl_answer_uniqueness_replacement_seed"
    pilot.micro.pb8.pb6.SEED_SELECTION_NOTE = "pb9_04tl_calibration_003 answer uniqueness hard fail을 fresh C-slot seed로 대체하는 seed"
    pilot.micro.pb8.pb6.SEED_FILTER_NOTE = "pb9_04tl_calibration_seen_seed_pool_excluded"
    pilot.micro.pb8.pb6.SCOPE_NOTE = "04_TL_결정례_QA generalization_03_04 only; 1-slot C replacement; current count 미합산"
    pilot.micro.pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = 1
    pilot.micro.pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_pb9_04tl_answer_uniqueness_replacement"
    pilot.micro.pb8.pb6.SEED_REGISTRY_STRATEGY = "fresh_04tl_decision_generalization_pool_excluding_current_failed_repair_pb9_and_calibration_seen_seed"
    pilot.micro.pb8.pb6.LAW_STATUS_NOTE = "04tl_decision_cslot_replacement_not_counted"
    pilot.micro.pb8.pb6.OVERLAP_CHECK_LABEL = "no current/failed/repaired/pb9/calibration-seen seed overlap"
    pilot.micro.pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`pb9_04tl_calibration_003`은 same-seed retry가 아니라 seed-specific exclusion으로 처리한다.",
        "`pb9_04tl_decision_weak_distractor_calibration_pilot`의 8개 seed를 모두 seen seed로 제외한다.",
        "이번 run은 `04_TL_결정례_QA generalization_03_04` `C-slot 1개` replacement signal 전용이며 current count에는 합산하지 않는다.",
    ]
    pilot.micro.pb8.pb6.pb4.pb3.base.build_judge_prompt = pilot.build_structured_judge_prompt
    pilot.micro.pb8.pb6.pb4.pb3.base.build_judge_row = pilot.build_structured_judge_row
    pilot.write_judge_structured_contract_md()
    return pilot.micro.pb8.pb6.main()


if __name__ == "__main__":
    main()
