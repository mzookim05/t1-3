from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# `problem_generation v3`는 `train_v7` 안의 복수 질의형 seed를 다시 읽어,
# 단일 쟁점 서술형으로 분리하는 첫 split-descriptive mainline이다.
VERSION_TAG = "v3"
RUN_DATE = "2026-04-22"
RUN_PURPOSE = "qa_split_descriptive_multiclause"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "dataset_build"
PROMPT_DIR = SCRIPT_DIR / "prompts"

REFERENCE_TRAIN_PATH = PROCESSED_DIR / "train_v7.jsonl"
REFERENCE_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest_v7.csv"
REFERENCE_PROBLEM_V2_MANIFEST_PATH = PROCESSED_DIR / "problem_dataset_manifest_v2.csv"

SEED_REGISTRY_PATH = INTERIM_DIR / f"problem_seed_registry_{VERSION_TAG}.csv"
SEED_READY_PATH = INTERIM_DIR / f"problem_seed_ready_{VERSION_TAG}.jsonl"

RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWERABILITY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answerability_{VERSION_TAG}.jsonl"
TASKFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_taskfit_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"
PROBLEM_EXAMPLES_MD_PATH = RUN_EXPORTS_DIR / f"problem_examples_{VERSION_TAG}.md"
PROBLEM_EXAMPLES_CSV_PATH = RUN_EXPORTS_DIR / f"problem_examples_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / f"problem_train_{VERSION_TAG}.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / f"problem_dev_{VERSION_TAG}.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / f"problem_test_{VERSION_TAG}.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / f"problem_dataset_manifest_{VERSION_TAG}.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / f"problem_audit_queue_{VERSION_TAG}.csv"

GENERATOR_MODEL_CANDIDATES = ("gpt-5.4",)
JUDGE_MODEL_CANDIDATES = ("gemini-2.5-pro",)

GENERATOR_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 600
GENERATOR_API_TIMEOUT_SECONDS = 45
JUDGE_API_TIMEOUT_SECONDS = 60

# `v3`도 `v1/v2`와 같은 운영 철학을 유지해 메인은 빠르게, strict finalize는 보수적으로 잠근다.
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

PROBLEM_TASK_TYPE = "descriptive_qa_split"
TARGET_SEED_COUNT = 16
TARGET_DOC_TYPE_COUNT = 4
TARGET_EXPANSION_PER_DOC_TYPE = 2
TARGET_GENERALIZATION_PER_DOC_TYPE = 2

ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "split_single_issue_rule",
    "application": "split_single_issue_application",
    "requirement": "split_single_issue_requirement",
    "scope": "split_single_issue_scope",
}

DOC_TYPE_PROMPT_HINTS = {
    "법령_QA": "절차·요건·효과가 겹치면 한 축만 남기고 단일 쟁점 설명형으로 바꾼다.",
    "해석례_QA": "회답 이유와 법적 입장이 함께 섞여 있으면 하나의 판단 포인트만 남긴다.",
    "결정례_QA": "이유와 핵심 판단 기준이 같이 들어오면 실질 질문 하나만 남긴다.",
    "판결문_QA": "판단 이유와 법원의 기준이 섞여 있으면 단일 판단 요소만 설명하게 만든다.",
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
