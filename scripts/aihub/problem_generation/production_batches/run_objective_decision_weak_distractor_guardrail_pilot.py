import sys
from pathlib import Path

# `pb8`에서 hard/soft fail은 없었지만 weak distractor audit이 남았으므로,
# 바로 `pb9` 40개로 키우기 전에 결정례 오답 변별력 guardrail만 16개로 검증한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as pb8


VERSION_TAG = "decision_weak_distractor_guardrail_pilot"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-25_203320"
RUN_PURPOSE = "objective_r2_decision_weak_distractor_guardrail_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb8.pb6.pb4.pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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

REFERENCE_PB8_SEED_REGISTRY_PATH = (
    pb8.pb6.pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb8_decision_only_objective_current_r2"
    / "seed_registry.csv"
)

PILOT_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 3,
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

ORIGINAL_COLLECT_EXCLUDED_ROWS = pb8.collect_excluded_rows
ORIGINAL_BUILD_GENERATION_MESSAGES = pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES


def collect_excluded_rows():
    # `pb8`은 미합산 batch지만 이미 generation/Judge를 본 seed이므로,
    # targeted pilot에서도 fresh 검증 착시를 막기 위해 seen pool로 보호한다.
    rows = ORIGINAL_COLLECT_EXCLUDED_ROWS()
    rows.extend(pb8.pb6.load_csv_rows_if_exists(REFERENCE_PB8_SEED_REGISTRY_PATH))
    return rows


def build_generation_messages(seed, reference_v2):
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += """

## decision weak distractor guardrail 추가 지시
- 이번 run은 `결정례_QA` weak distractor / near-miss 부족 audit tail을 줄이기 위한 targeted pilot이다.
- 오답 중 최소 `2개`는 정답과 같은 결정 이유, 판단 기준, 적용 사실 중 하나를 공유하되, 판단 요소 `1개`만 다르게 비틀 것.
- 너무 넓은 일반론, 배경 설명, 사건과 무관한 상식형 오답을 피하고, 같은 사건 맥락 안의 near-miss 오답으로 만들 것.
- 오답끼리 같은 의미를 반복하지 말고, 정답보다 모호하거나 포괄적인 표현으로 도망가지 말 것.
- stem은 하나의 판단 기준 또는 하나의 적용 사실만 묻고, 정답 선택지를 직접 노출하지 말 것.
"""
    return messages


def configure_pilot_globals():
    # `pb8`까지 검증된 wrapper 구조를 유지하고, seed scope와 success criteria만
    # 16개 decision weak distractor targeted pilot에 맞게 좁힌다.
    pb8.pb6.VERSION_TAG = VERSION_TAG
    pb8.pb6.RUN_DATE = RUN_DATE
    pb8.pb6.RUN_PURPOSE = RUN_PURPOSE
    pb8.pb6.RUN_NAME = RUN_NAME
    pb8.pb6.RUN_LABEL = "decision weak distractor pilot"
    pb8.pb6.SEED_ID_PREFIX = "decision_guardrail"
    pb8.pb6.SEED_SELECTION_ROLE = "objective_decision_weak_distractor_guardrail_pilot_seed"
    pb8.pb6.SEED_SELECTION_NOTE = "pb8 결정례 weak distractor audit tail을 줄이기 위한 targeted pilot seed"
    pb8.pb6.SEED_FILTER_NOTE = "decision_only_guardrail_seen_seed_pool_excluded"
    pb8.pb6.SCOPE_NOTE = "결정례_QA only; weak distractor / near-miss 부족 guardrail targeted pilot"
    pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = 16
    pb8.pb6.SUCCESS_USABLE_MIN = 15
    pb8.pb6.SUCCESS_HARD_FAIL_MAX = 0
    pb8.pb6.SUCCESS_SOFT_FAIL_MAX = 0
    pb8.pb6.SUCCESS_AUDIT_MAX = 1
    pb8.pb6.SUCCESS_LAW_ROW_COUNT = 0
    pb8.pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_decision_weak_distractor_guardrail_candidate"
    pb8.pb6.SEED_REGISTRY_STRATEGY = "fresh_aihub_qa_training_decision_only_pool_excluding_current_law_targeted_failed_pb5_failed_pb6_failed_pb7_failed_pb8_heldout_audit_rows"
    pb8.pb6.LAW_STATUS_NOTE = "decision_guardrail_targeted_pilot_count_excluded_until_reviewer_signoff"
    pb8.pb6.OVERLAP_CHECK_LABEL = "no current/law-targeted/failed-pb5/pb6/pb7/pb8/held-out/audit overlap"
    pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`current counted-line attempted seed registry 109개`는 usable count가 아니라 `r2 + pb2 + pb3 + pb4`에 실제 투입된 seed registry 규모를 뜻한다.",
        "`law targeted pilot 16개`, failed `pb5 40개`, failed `pb6 45개`, failed `pb7 40개`, failed `pb8 40개`까지 더해, 이번 결정례 targeted pilot에서는 seen objective seed `290개`를 제외 대상으로 본다.",
    ]

    pb8.pb6.INTERIM_DIR = INTERIM_DIR
    pb8.pb6.PROCESSED_DIR = PROCESSED_DIR
    pb8.pb6.RUN_DIR = RUN_DIR
    pb8.pb6.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb8.pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb8.pb6.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb8.pb6.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb8.pb6.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb8.pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb8.pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb8.pb6.SEED_READY_PATH = SEED_READY_PATH
    pb8.pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb8.pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb8.pb6.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb8.pb6.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb8.pb6.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb8.pb6.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb8.pb6.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb8.pb6.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb8.pb6.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb8.pb6.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb8.pb6.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb8.pb6.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb8.pb6.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb8.pb6.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb8.pb6.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb8.pb6.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb8.pb6.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pb8.pb6.TAIL_MEMO_CSV_PATH = TAIL_MEMO_CSV_PATH
    pb8.pb6.TAIL_MEMO_MD_PATH = TAIL_MEMO_MD_PATH

    pb8.pb6.PB6_SOURCE_COUNTS = PILOT_SOURCE_COUNTS
    pb8.pb6.PB6_DATASET_SPECS = pb8.pb6.build_pb6_dataset_specs()
    pb8.pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb8.pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb8.pb6.collect_excluded_rows = collect_excluded_rows
    pb8.pb6.passes_pb6_seed_filter = pb8.passes_pb8_seed_filter
    pb8.pb6.classify_tail = pb8.classify_pb8_tail
    pb8.pb6.ORIGINAL_BUILD_GENERATION_MESSAGES = build_generation_messages


def main():
    configure_pilot_globals()
    return pb8.pb6.main()


if __name__ == "__main__":
    main()
