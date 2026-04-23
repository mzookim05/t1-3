from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# `problem_generation v2`는 설명형 서술형 `v1`을 유지한 채,
# 첫 객관식 prototype을 별도 mainline으로 여는 실행이다.
VERSION_TAG = "v2"
RUN_DATE = "2026-04-14"
RUN_PURPOSE = "qa_objective_single_best"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

# objective line의 기준 산출물을 전용 폴더에 모아 다른 subtype/batch와 섞이지 않게 한다.
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "v2_objective"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "v2_objective"
REFERENCE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final"
REFERENCE_PROBLEM_V1_PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "v1_descriptive"
)
PROMPT_DIR = SCRIPT_DIR / "prompts"

REFERENCE_TRAIN_PATH = REFERENCE_PROCESSED_DIR / "train.jsonl"
REFERENCE_MANIFEST_PATH = REFERENCE_PROCESSED_DIR / "dataset_manifest.csv"
REFERENCE_PROBLEM_V1_MANIFEST_PATH = REFERENCE_PROBLEM_V1_PROCESSED_DIR / "dataset_manifest.csv"
REFERENCE_PROBLEM_V1_MERGED_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-14_v1_qa_descriptive_prototype"
    / "merged"
    / "merged_problem_scores_v1.csv"
)

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"

RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
PROBLEM_EXAMPLES_MD_PATH = RUN_EXPORTS_DIR / f"problem_examples_{VERSION_TAG}.md"
PROBLEM_EXAMPLES_CSV_PATH = RUN_EXPORTS_DIR / f"problem_examples_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

GENERATOR_MODEL_CANDIDATES = ("gpt-5.4",)
JUDGE_MODEL_CANDIDATES = ("gemini-2.5-pro",)

GENERATOR_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 900
GENERATOR_API_TIMEOUT_SECONDS = 45
JUDGE_API_TIMEOUT_SECONDS = 60

GENERATOR_MAIN_CHECKPOINT_EVERY = 4
GENERATOR_STRICT_CHECKPOINT_EVERY = 2
JUDGE_MAIN_MAX_WORKERS = 4
JUDGE_MAIN_MAX_ATTEMPTS = 4
JUDGE_MAIN_RETRY_BASE_SECONDS = 3
JUDGE_MAIN_SUCCESS_SLEEP_SECONDS = 0.1
JUDGE_MAIN_CHECKPOINT_EVERY = 12
JUDGE_STRICT_MAX_WORKERS = 1
JUDGE_STRICT_MAX_ATTEMPTS = 0
JUDGE_STRICT_RETRY_BASE_SECONDS = 5
JUDGE_STRICT_SUCCESS_SLEEP_SECONDS = 1.0
JUDGE_STRICT_CHECKPOINT_EVERY = 6

PROBLEM_TASK_TYPE = "objective_single_best"
TARGET_SEED_COUNT = 16

ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "single_best_rule",
    "application": "single_best_application",
    "requirement": "single_best_rule",
    "scope": "single_best_scope",
}

DOC_TYPE_PROMPT_HINTS = {
    "법령_QA": "요건, 절차, 효과를 구분해 단일정답 선택지가 되게 만든다.",
    "해석례_QA": "회답 이유와 판단 기준을 중심으로 하나의 정답을 고르게 만든다.",
    "결정례_QA": "복수 쟁점을 섞지 말고 핵심 판단 이유 하나를 단일정답으로 고르게 만든다.",
    "판결문_QA": "판시 기준과 사건 적용 중 핵심 하나를 정답으로 만들고 나머지는 오답으로 분리한다.",
}

ALLOWED_ERROR_TAGS = [
    "정답 누설",
    "원문 외 사실 추가",
    "정답 비유일",
    "복수 쟁점 혼합",
    "형식 부적합",
    "오답이 정답 가능",
    "선택지 중복",
]

HARD_FAIL_TAGS = {
    "정답 누설",
    "원문 외 사실 추가",
    "정답 비유일",
    "복수 쟁점 혼합",
    "오답이 정답 가능",
    "선택지 중복",
}

SCORE_WEIGHTS = {
    "Grounding": 0.35,
    "Keyedness": 0.40,
    "DistractorFit": 0.25,
}
