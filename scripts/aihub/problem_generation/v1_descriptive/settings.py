from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]

# `problem_generation v1`은 `v7 strict final`을 seed pool로 삼아,
# 설명형 서술형 문제 생성 축을 첫 prototype 수준으로 여는 실행이다.
VERSION_TAG = "v1"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-14_183615"
RUN_PURPOSE = "qa_descriptive_prototype"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

# subtype별 산출물이 섞이지 않도록 v1 전용 interim/processed 폴더를 고정한다.
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "v1_descriptive"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "v1_descriptive"
REFERENCE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final"
PROMPT_DIR = SCRIPT_DIR / "prompts"

REFERENCE_TRAIN_PATH = REFERENCE_PROCESSED_DIR / "train.jsonl"
REFERENCE_MANIFEST_PATH = REFERENCE_PROCESSED_DIR / "dataset_manifest.csv"
REFERENCE_AUDIT_PATH = REFERENCE_PROCESSED_DIR / "audit_queue.csv"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"

RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWERABILITY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answerability_{VERSION_TAG}.jsonl"
TASKFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_taskfit_{VERSION_TAG}.jsonl"
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
GENERATOR_MAX_TOKENS = 500
GENERATOR_API_TIMEOUT_SECONDS = 45
JUDGE_API_TIMEOUT_SECONDS = 60

# `v1`은 seed 수가 작지만 `Judge` 역할이 3개라 호출 수가 생각보다 quickly 커진다.
# 메인 런은 worker를 넉넉히 두고, strict finalize만 보수적으로 잠가 wall-clock을 줄인다.
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

PROBLEM_TASK_TYPE = "descriptive_qa"
TARGET_SEED_COUNT = 16

ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "standard_reframe",
    "application": "issue_application",
    "requirement": "requirement_check",
    "scope": "scope_boundary",
}

DOC_TYPE_PROMPT_HINTS = {
    "법령_QA": "조문 기준, 요건, 절차를 분명히 묻는 설명형 문제로 정리한다.",
    "해석례_QA": "회답과 이유를 바탕으로 판단 기준이나 법적 입장을 설명하게 만든다.",
    "결정례_QA": "핵심 법리와 사안 적용을 함께 설명하게 만든다.",
    "판결문_QA": "판시 기준과 사건 적용 관계를 설명하게 만든다.",
}

ALLOWED_ERROR_TAGS = [
    "정답 누설",
    "원문 외 사실 추가",
    "정답 비유일",
    "복수 쟁점 혼합",
    "형식 부적합",
    "근거 누락",
]

HARD_FAIL_TAGS = {
    "정답 누설",
    "원문 외 사실 추가",
    "정답 비유일",
    "복수 쟁점 혼합",
}

SCORE_WEIGHTS = {
    "Grounding": 0.40,
    "Answerability": 0.35,
    "TaskFit": 0.25,
}
