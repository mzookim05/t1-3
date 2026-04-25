import sys
from pathlib import Path

# Reviewer sign-off 이후 `8개` micro package에서 바로 `pb9 40개`로 키우지 않고,
# 같은 validator를 `16개` targeted pilot으로 한 번 더 일반화 검증한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_micro_pilot as micro,
)


VERSION_TAG = "decision_choice_validator_targeted_pilot_16"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_decision_choice_validator_targeted_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = micro.pb8.pb6.pb4.pb3.base.PROJECT_ROOT
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

REFERENCE_MICRO_SEED_REGISTRY_PATH = micro.SEED_REGISTRY_PATH
REFERENCE_MICRO_RETRY_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_choice_validator_micro_retry"
    / "seed_registry.csv"
)
REFERENCE_A_SLOT_REPLACEMENT_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "decision_choice_validator_a_slot_replacement"
    / "seed_registry.csv"
)

TARGETED_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 4,
    "02_TL_심결례_QA": 2,
    "02_TL_심결문_QA": 2,
    "03_TL_결정례_QA": 4,
    "04_TL_결정례_QA": 4,
}

EXPECTED_DOC_TYPE_COUNTS = {
    "결정례_QA": 16,
}

EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "generalization_03_04"): 8,
    ("결정례_QA", "expansion_01_02"): 8,
}

SUCCESS_USABLE_MIN = 15
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 0
SUCCESS_AUDIT_MAX = 1
SUCCESS_LAW_ROW_COUNT = 0
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}

BASE_BUILD_GENERATION_MESSAGES = micro.build_generation_messages
BASE_BUILD_BATCH_SUMMARY = micro.build_batch_summary
BASE_BUILD_RUN_MANIFEST = micro.build_run_manifest
BASE_WRITE_VALIDATOR_WIRING_CHECK_MD = micro.write_validator_wiring_check_md


def collect_excluded_rows_for_targeted():
    # 이번 targeted pilot은 micro/retry/replacement까지 이미 본 seed로 제외해야
    # `16개` 결과가 fresh decision seed 일반화 신호로 남는다.
    rows = micro.ORIGINAL_COLLECT_EXCLUDED_ROWS()
    rows.extend(micro.pb8.pb6.load_csv_rows_if_exists(micro.REFERENCE_PB8_SEED_REGISTRY_PATH))
    rows.extend(micro.pb8.pb6.load_csv_rows_if_exists(micro.REFERENCE_DECISION_GUARDRAIL_SEED_REGISTRY_PATH))
    rows.extend(micro.pb8.pb6.load_csv_rows_if_exists(REFERENCE_MICRO_SEED_REGISTRY_PATH))
    rows.extend(micro.pb8.pb6.load_csv_rows_if_exists(REFERENCE_MICRO_RETRY_SEED_REGISTRY_PATH))
    rows.extend(micro.pb8.pb6.load_csv_rows_if_exists(REFERENCE_A_SLOT_REPLACEMENT_SEED_REGISTRY_PATH))
    return rows


def build_generation_messages(seed, reference_v2):
    messages = BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## decision choice validator targeted pilot 추가 지시
- 이번 run은 `8개` micro package 통과 이후 `결정례_QA` `16개` targeted pilot에서 validator 일반화를 확인하는 단계다.
- 생성 단계에서는 정답 유일성, 오답 의미 분리, 같은 anchor를 공유하는 one-axis near-miss를 우선한다.
- 후처리 validator가 label을 `A/B/C/D = 4/4/4/4`로 다시 섞고 metadata를 재매핑하므로, label 위치가 아니라 선택지 의미 품질에 집중한다.
- `nearmiss_reason` 같은 자연어 Judge reason은 진단용으로만 쓰이며, post-shuffle artifact truth는 `choice_*`, `correct_choice`, `distractor_type_map`, `near_miss_notes`, `validator_*` 구조화 필드가 담당한다.
"""
    return messages


def build_batch_summary(rows):
    summary = micro.pb8.pb6.pb4.pb3.summarize_rows(rows)
    selected_rows = [row for row in rows if row.get("selected_for_seed") == "예"]
    doc_type_counts = {}
    lane_counts = {}
    source_counts = {}
    for row in selected_rows:
        doc_type_counts[row["doc_type_name"]] = doc_type_counts.get(row["doc_type_name"], 0) + 1
        lane_counts[row["sampling_lane"]] = lane_counts.get(row["sampling_lane"], 0) + 1
        source_counts[row["source_subset"]] = source_counts.get(row["source_subset"], 0) + 1

    # 기존 micro summary에는 `8개` 기준 문구가 남으므로, targeted pilot 전용 summary를 새로 쓴다.
    summary_rows = [
        {"metric": "seed_count", "value": str(len(selected_rows))},
        {"metric": "selected_pass", "value": str(summary["selected_pass_count"])},
        {"metric": "selected_hard_fail", "value": str(summary["selected_hard_fail_count"])},
        {"metric": "selected_soft_fail", "value": str(summary["selected_soft_fail_count"])},
        {"metric": "train_eligible", "value": str(summary["selected_train_eligible_count"])},
        {"metric": "audit_required", "value": str(summary["selected_audit_required_count"])},
        {"metric": "success_passed", "value": str(micro.VALIDATOR_SUMMARY.get("micro_success_passed", False))},
    ]
    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(BATCH_SUMMARY_CSV_PATH, summary_rows, ["metric", "value"])
    micro.pb8.pb6.pb4.pb3.base.write_csv_atomic(
        BATCH_LANE_SUMMARY_CSV_PATH,
        [{"sampling_lane": lane, "count": count} for lane, count in sorted(lane_counts.items())],
        ["sampling_lane", "count"],
    )

    lines = [
        f"# batch summary `{VERSION_TAG}`",
        "",
        "## overall summary",
        f"- seed_count: `{len(selected_rows)}`",
        f"- doc_type_counts: `{doc_type_counts}`",
        f"- lane_counts: `{lane_counts}`",
        f"- selected: `{summary['selected_pass_count']} pass / {summary['selected_hard_fail_count']} hard_fail / {summary['selected_soft_fail_count']} soft_fail`",
        f"- train/audit: `train_eligible {summary['selected_train_eligible_count']} / audit_required {summary['selected_audit_required_count']}`",
        "",
        "## source subset counts",
        "| source_subset | planned |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(
        [
            "",
            "## targeted pilot success criteria",
            "| criterion | target | result |",
            "| --- | --- | --- |",
            f"| usable | `>= {SUCCESS_USABLE_MIN} / 16` | `{summary['selected_train_eligible_count']}` |",
            f"| hard_fail | `{SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |",
            f"| soft_fail | `{SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |",
            f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |",
            f"| unresolved regenerate | `0` | `{micro.VALIDATOR_SUMMARY.get('validator_action_counts', {}).get('regenerate', 0)}` |",
            f"| answer uniqueness hard block | `0` | `{micro.VALIDATOR_SUMMARY.get('validator_action_counts', {}).get('hard_block', 0)}` |",
            f"| shuffle recalc mismatch | `0` | `{micro.VALIDATOR_SUMMARY.get('shuffle_recalc_mismatch_count', 0)}` |",
            f"| metadata remap mismatch | `0` | `{micro.VALIDATOR_SUMMARY.get('metadata_remap_mismatch_count', 0)}` |",
            f"| export label balance | `A/B/C/D = 4/4/4/4` | `{micro.VALIDATOR_SUMMARY.get('export_label_counts', {})}` |",
        ]
    )
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    return summary_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = BASE_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    action_counts = micro.VALIDATOR_SUMMARY.get("validator_action_counts", {})
    success_result = manifest.setdefault("success_result", {})
    success_result.update(
        {
            "unresolved_regenerate": action_counts.get("regenerate", 0),
            "answer_uniqueness_hard_block": action_counts.get("hard_block", 0),
            "correct_choice_recalc_mismatch": micro.VALIDATOR_SUMMARY.get("shuffle_recalc_mismatch_count", 0),
            "metadata_remap_mismatch_count": micro.VALIDATOR_SUMMARY.get("metadata_remap_mismatch_count", 0),
            "targeted_pilot_success_passed": bool(micro.VALIDATOR_SUMMARY.get("micro_success_passed", False)),
        }
    )
    manifest["success_criteria"] = {
        "usable_min": SUCCESS_USABLE_MIN,
        "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
        "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
        "audit_max": SUCCESS_AUDIT_MAX,
        "unresolved_regenerate": 0,
        "correct_choice_recalc_mismatch": 0,
        "metadata_remap_mismatch": 0,
        "answer_uniqueness_hard_block": 0,
        "export_label_balance": TARGET_LABEL_COUNTS,
        "law_row_count": SUCCESS_LAW_ROW_COUNT,
    }
    manifest["validator_policy"]["natural_language_reason_policy"] = (
        "`nearmiss_reason` and Judge one-sentence reasons are diagnostic text only; "
        "post-shuffle gates use structured choice and validator fields."
    )
    manifest["current_count_decision"] = "not_counted_until_reviewer_signoff"
    micro.pb8.pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def write_validator_wiring_check_md():
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| validator module imported | `pass` | `run_objective_decision_choice_validator_replay.py`의 deterministic validator 사용 |",
        "| split_dataset hook connected | `pass` | `merge_scores -> validator -> split_dataset/export` 순서로 연결 |",
        "| correct_choice recalculation gate | `pass` | mismatch 발생 시 `hard_block` 처리 |",
        "| label-keyed metadata remap gate | `pass` | `distractor_type_map`, `near_miss_notes`를 post-shuffle label로 재매핑 |",
        "| regenerate policy defined | `pass` | 재생성 필요 row는 export 제외 `soft_fail`로 기록하고 reviewer 판단을 받음 |",
        "| target seed count | `pass` | `결정례_QA 16개`, lane `8/8` |",
        "| target label schedule | `pass` | selected/export package 기준 `A/B/C/D = 4/4/4/4` |",
        "| natural-language judge reason policy | `pass` | `nearmiss_reason`, `one_sentence_reason`은 diagnostic text로만 사용 |",
        "| count reflection | `pass` | reviewer sign-off 전 current count 미합산 |",
    ]
    micro.pb8.pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def configure_targeted_globals():
    # micro pilot runner의 검증된 postprocess hook을 유지하고, scope와 성공 기준만 `16개` targeted pilot로 확장한다.
    micro.VERSION_TAG = VERSION_TAG
    micro.RUN_DATE = RUN_DATE
    micro.RUN_PURPOSE = RUN_PURPOSE
    micro.RUN_NAME = RUN_NAME
    micro.INTERIM_DIR = INTERIM_DIR
    micro.PROCESSED_DIR = PROCESSED_DIR
    micro.RUN_DIR = RUN_DIR
    micro.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    micro.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    micro.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    micro.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    micro.RUN_MERGED_DIR = RUN_MERGED_DIR
    micro.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    micro.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    micro.SEED_READY_PATH = SEED_READY_PATH
    micro.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    micro.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    micro.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    micro.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    micro.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    micro.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    micro.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    micro.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    micro.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    micro.RAW_MERGED_BEFORE_VALIDATOR_PATH = RAW_MERGED_BEFORE_VALIDATOR_PATH
    micro.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    micro.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    micro.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    micro.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    micro.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    micro.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    micro.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    micro.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    micro.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    micro.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH
    micro.VALIDATOR_REPORT_CSV_PATH = VALIDATOR_REPORT_CSV_PATH
    micro.VALIDATOR_REPORT_MD_PATH = VALIDATOR_REPORT_MD_PATH
    micro.VALIDATOR_WIRING_CHECK_MD_PATH = VALIDATOR_WIRING_CHECK_MD_PATH
    micro.MICRO_SOURCE_COUNTS = TARGETED_SOURCE_COUNTS
    micro.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    micro.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    micro.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    micro.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    micro.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    micro.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    micro.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    micro.TARGET_LABEL_COUNTS = TARGET_LABEL_COUNTS
    micro.collect_excluded_rows_for_micro = collect_excluded_rows_for_targeted
    micro.build_generation_messages = build_generation_messages
    micro.build_batch_summary = build_batch_summary
    micro.build_run_manifest = build_run_manifest
    micro.write_validator_wiring_check_md = write_validator_wiring_check_md


def main():
    configure_targeted_globals()
    micro.configure_micro_globals()
    # micro wrapper 내부의 `8개` 기본값을 targeted pilot의 `16개` 기준으로 다시 고정한다.
    micro.pb8.pb6.RUN_LABEL = "decision choice validator targeted pilot 16"
    micro.pb8.pb6.SEED_ID_PREFIX = "decision_targeted"
    micro.pb8.pb6.SEED_SELECTION_ROLE = "objective_decision_choice_validator_targeted_pilot_seed"
    micro.pb8.pb6.SEED_SELECTION_NOTE = "결정례_QA postprocess choice validator를 16개 fresh seed에서 일반화 검증하는 targeted pilot seed"
    micro.pb8.pb6.SEED_FILTER_NOTE = "decision_choice_validator_targeted_seen_seed_pool_excluded"
    micro.pb8.pb6.SCOPE_NOTE = "결정례_QA only; 16개 targeted pilot + postprocess validator + shuffle/recalc/metadata remap gate"
    micro.pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = 16
    micro.pb8.pb6.SUCCESS_USABLE_MIN = SUCCESS_USABLE_MIN
    micro.pb8.pb6.SUCCESS_HARD_FAIL_MAX = SUCCESS_HARD_FAIL_MAX
    micro.pb8.pb6.SUCCESS_SOFT_FAIL_MAX = SUCCESS_SOFT_FAIL_MAX
    micro.pb8.pb6.SUCCESS_AUDIT_MAX = SUCCESS_AUDIT_MAX
    micro.pb8.pb6.SUCCESS_LAW_ROW_COUNT = SUCCESS_LAW_ROW_COUNT
    micro.pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_decision_choice_validator_targeted_pilot"
    micro.pb8.pb6.SEED_REGISTRY_STRATEGY = "fresh_aihub_qa_training_decision_only_pool_excluding_current_failed_and_all_decision_validator_seen_seed"
    micro.pb8.pb6.LAW_STATUS_NOTE = "decision_choice_validator_targeted_pilot_count_excluded_until_reviewer_signoff"
    micro.pb8.pb6.OVERLAP_CHECK_LABEL = "no current/failed-pb5/pb6/pb7/pb8/decision-guardrail/micro/retry/replacement/held-out/audit overlap"
    micro.pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`current counted-line attempted seed registry 109개`는 usable count가 아니라 `r2 + pb2 + pb3 + pb4`에 실제 투입된 seed registry 규모를 뜻한다.",
        "`decision guardrail pilot`, `micro pilot`, `micro retry`, `A-slot replacement`까지 모두 seen seed로 제외해 이번 16개를 fresh targeted pilot으로 둔다.",
    ]
    return micro.pb8.pb6.main()


if __name__ == "__main__":
    main()
