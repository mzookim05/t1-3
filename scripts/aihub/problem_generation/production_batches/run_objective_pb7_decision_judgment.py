import sys
from pathlib import Path

# `pb7`은 `pb6` 이후 새로 열린 count 확보 트랙이다. `법령_QA`와
# `해석례_QA`는 repair 검토 대상으로 분리하고, hard/soft fail 없이 버틴
# `결정례_QA`와 `판결문_QA`만 current recipe로 다시 검증한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6


VERSION_TAG = "pb7_decision_judgment_objective_current_r2"
RUN_DATE = "2026-04-25"
RUN_PURPOSE = "objective_r2_decision_judgment_controlled_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb6.pb4.pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb6.pb4.pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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

REFERENCE_PB6_SEED_REGISTRY_PATH = (
    pb6.pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb6_non_law_objective_current_r2"
    / "seed_registry.csv"
)

PB7_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 3,
    "02_TL_심결문_QA": 2,
    "03_TL_결정례_QA": 6,
    "04_TL_결정례_QA": 6,
    "01_TL_판결문_QA": 4,
    "02_TL_판결문_QA": 4,
    "03_TL_판결문_QA": 6,
    "04_TL_판결문_QA": 6,
}

EXPECTED_DOC_TYPE_COUNTS = {
    "결정례_QA": 20,
    "판결문_QA": 20,
}

EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "generalization_03_04"): 12,
    ("결정례_QA", "expansion_01_02"): 8,
    ("판결문_QA", "generalization_03_04"): 12,
    ("판결문_QA", "expansion_01_02"): 8,
}

ORIGINAL_COLLECT_EXCLUDED_ROWS = pb6.collect_excluded_rows


def collect_excluded_rows():
    # `pb6`도 failed controlled batch지만 이미 생성과 Judge를 본 seed라서
    # `pb7` fresh 검증에서는 재사용하지 않는다.
    rows = ORIGINAL_COLLECT_EXCLUDED_ROWS()
    rows.extend(pb6.load_csv_rows_if_exists(REFERENCE_PB6_SEED_REGISTRY_PATH))
    return rows


def passes_pb7_seed_filter(spec, payload):
    if spec["doc_type_name"] in {"법령_QA", "해석례_QA"}:
        return False, "law_or_interpretation_doc_type_restricted"
    return pb6.pb4.passes_seed_quality_filter(
        spec["doc_type_name"],
        payload["label"]["input"],
        payload["label"]["output"],
    )


def configure_pb7_globals():
    # `pb6`의 검증된 runner 본체를 쓰되, batch identity와 seed scope만
    # `pb7` 결정례/판결문 전용 stop line으로 재배선한다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = "pb7 decision/judgment"
    pb6.SEED_ID_PREFIX = "pb7_dj"
    pb6.SEED_SELECTION_ROLE = "objective_pb7_decision_judgment_current_r2_seed"
    pb6.SEED_SELECTION_NOTE = "법령_QA와 해석례_QA repair 검토 중 결정례_QA/판결문_QA만 이어가는 seed"
    pb6.SEED_FILTER_NOTE = "law_and_interpretation_doc_types_excluded_and_seen_seed_pool_excluded"
    pb6.SCOPE_NOTE = "결정례_QA/판결문_QA only; 법령_QA와 해석례_QA는 repair track에서 별도 처리"
    pb6.EXPECTED_TOTAL_SEED_COUNT = 40
    pb6.SUCCESS_USABLE_MIN = 36
    pb6.SUCCESS_HARD_FAIL_MAX = 0
    pb6.SUCCESS_SOFT_FAIL_MAX = 1
    pb6.SUCCESS_AUDIT_MAX = 4
    pb6.SUCCESS_LAW_ROW_COUNT = 0
    pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_decision_judgment_current"
    pb6.SEED_REGISTRY_STRATEGY = "fresh_aihub_qa_training_decision_judgment_pool_excluding_current_law_targeted_failed_pb5_failed_pb6_heldout_audit_rows"
    pb6.LAW_STATUS_NOTE = "law_and_interpretation_repair_tracks_excluded_from_pb7"
    pb6.EXCLUSION_WORDING_LINES = [
        "`current counted-line attempted seed registry 109개`는 usable count가 아니라 `r2 + pb2 + pb3 + pb4`에 실제 투입된 seed registry 규모를 뜻한다.",
        "`law targeted pilot 16개`, failed `pb5 40개`, failed `pb6 45개`까지 더해, 이번 결정례/판결문 batch에서는 seen objective seed `210개`를 제외 대상으로 본다.",
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

    pb6.PB6_SOURCE_COUNTS = PB7_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.collect_excluded_rows = collect_excluded_rows
    pb6.passes_pb6_seed_filter = passes_pb7_seed_filter


def main():
    configure_pb7_globals()
    return pb6.main()


if __name__ == "__main__":
    main()
