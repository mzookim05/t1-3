from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# `v7`은 `v6 strict final`의 표본을 그대로 재사용해,
# 남은 `hard fail 4 + audit 1`을 줄이는 최소 안정화 런이다.
VERSION_TAG = "v7"
RUN_DATE = "2026-04-14"
RUN_PURPOSE = "tail_stabilization_full_01_04"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "explanation_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "explanation_generation"
# 구조 개편 이후 explanation 최종셋은 dataset_build 루트가 아니라
# `data/processed/aihub/explanation_generation/{version}` 아래에 버전별로 잠근다.
PROCESSED_VERSION_DIRNAME = "v7_strict_final"
PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aihub"
    / "explanation_generation"
    / PROCESSED_VERSION_DIRNAME
)
PROMPT_DIR = SCRIPT_DIR / "prompts"

SAMPLE_REGISTRY_PATH = INTERIM_DIR / f"sample_registry_{VERSION_TAG}.csv"
REFERENCE_SAMPLE_REGISTRY_PATH = INTERIM_DIR / "sample_registry_v6.csv"
EVIDENCE_CARDS_PATH = INTERIM_DIR / f"evidence_cards_{VERSION_TAG}.jsonl"
TRANSFORMED_SAMPLES_PATH = INTERIM_DIR / f"transformed_samples_{VERSION_TAG}.jsonl"
JUDGE_READY_SAMPLES_PATH = INTERIM_DIR / f"judge_ready_samples_{VERSION_TAG}.jsonl"

RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
RUN_SELECTED_SAMPLES_PATH = RUN_DIR / "selected_samples.csv"
GENERATIONS_PATH = RUN_GENERATIONS_DIR / f"generated_explanations_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWER_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answer_{VERSION_TAG}.jsonl"
PEDAGOGY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_pedagogy_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_judge_scores_{VERSION_TAG}.csv"
MEETING_EXAMPLES_MD_PATH = RUN_EXPORTS_DIR / f"meeting_examples_{VERSION_TAG}.md"
MEETING_EXAMPLES_CSV_PATH = RUN_EXPORTS_DIR / f"meeting_examples_{VERSION_TAG}.csv"
ABLATION_SUMMARY_PATH = RUN_EXPORTS_DIR / f"ablation_summary_{VERSION_TAG}.csv"
MEETING_EXAMPLES_TITLE = f"meeting_examples_{VERSION_TAG}"

# 버전 정보는 폴더명에 이미 드러나므로, 내부 파일명은 역할명만 유지해
# rerun 시에도 current folderized 구조와 문서 설명이 어긋나지 않게 맞춘다.
TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
DEV_PATH = PROCESSED_DIR / "dev.jsonl"
TEST_PATH = PROCESSED_DIR / "test.jsonl"
DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

# 비밀값이 아닌 실행 상수는 `.env`가 아니라 여기에 두어,
# 어떤 모델과 규칙으로 실행했는지 문서와 코드가 같이 움직이게 한다.
# `v7`도 생성·판정 모델 조합은 유지하고,
# evidence/guardrail/Judge 운영 방식만 바꿔 tail risk를 줄인다.
# 자동으로 다른 상용 모델로 하향하지 않고, 실패 시에는 로컬 fallback이 드러나게 남긴다.
GENERATOR_MODEL_CANDIDATES = ("gpt-5.4",)
JUDGE_MODEL_CANDIDATES = ("gemini-2.5-pro",)
ACTIVE_GENERATION_VARIANT = "without_long_answer"
# `v7`도 release variant는 그대로 고정해
# evidence/guardrail 수정 효과를 해석하기 쉽게 둔다.
GENERATION_INPUT_VARIANTS = (
    {
        "name": "without_long_answer",
        "label": "long_answer 제외",
        "include_long_answer": False,
    },
)

GENERATOR_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 500
GENERATOR_API_TIMEOUT_SECONDS = 45
# Judge 호출 자체 타임아웃은 유지하되,
# `v7`에서는 main 런과 strict finalize 런의 worker/retry/sleep 정책을 분리한다.
JUDGE_API_TIMEOUT_SECONDS = 60
JUDGE_MAX_WORKERS = 4
JUDGE_MAIN_MAX_WORKERS = 4
JUDGE_MAIN_MAX_ATTEMPTS = 4
JUDGE_MAIN_RETRY_BASE_SECONDS = 3
JUDGE_MAIN_SUCCESS_SLEEP_SECONDS = 0.2
JUDGE_MAIN_CHECKPOINT_EVERY = 18
JUDGE_STRICT_MAX_WORKERS = 1
JUDGE_STRICT_MAX_ATTEMPTS = 0
JUDGE_STRICT_RETRY_BASE_SECONDS = 5
JUDGE_STRICT_SUCCESS_SLEEP_SECONDS = 1.0
JUDGE_STRICT_CHECKPOINT_EVERY = 6

HARD_FAIL_TAGS = {"결론 불일치", "근거 왜곡", "원문 외 사실 추가"}
TRAIN_BLOCK_TAGS_BY_DOC_TYPE = {
    "법령_QA": {"법리 누락", "근거 누락"},
    "해석례_QA": {"법리 누락", "근거 누락"},
    "결정례_QA": {"법리 누락", "근거 누락", "적용 약함"},
    "판결문_QA": {"법리 누락", "근거 누락", "적용 약함"},
}
ALLOWED_ERROR_TAGS = [
    "근거 누락",
    "법리 누락",
    "적용 약함",
    "문체 장황",
    "결론 불일치",
    "근거 왜곡",
    "원문 외 사실 추가",
]

SCORE_WEIGHTS = {
    "Grounding": 0.45,
    "Answer": 0.35,
    "Pedagogy": 0.20,
}

DOC_TYPE_RULES = {
    "법령_QA": {
        "transform_type": "law_reframe",
        "template_name": "정의·요건·범위 -> 조문 근거 -> 결론",
        "target_sentences": 3,
        "target_word_range": "25∼60",
        "styles": ("single",),
        "preferred_sections": ("조문", "항", "호"),
    },
    "해석례_QA": {
        "transform_type": "condition_reframe",
        "template_name": "쟁점 -> 이유 -> 결론",
        "target_sentences": 3,
        "target_word_range": "25∼60",
        "styles": ("single",),
        "preferred_sections": ("회답", "이유"),
    },
    "결정례_QA": {
        "transform_type": "issue_reframe",
        "template_name": "쟁점 -> 핵심 법리 -> 사실 적용 -> 결론",
        "target_sentences": 4,
        "target_word_range": "40∼100",
        "styles": ("legal_priority", "fact_priority"),
        "preferred_sections": ("전문",),
    },
    "판결문_QA": {
        "transform_type": "holding_reframe",
        "template_name": "쟁점 -> 판시 기준 -> 사건 적용 -> 결론",
        "target_sentences": 4,
        "target_word_range": "40∼100",
        "styles": ("legal_priority", "fact_priority"),
        "preferred_sections": ("판시사항", "판결요지", "판례내용", "참조조문"),
    },
}

DATASET_SPECS = [
    {
        "source_subset": "01_TL_법령_QA",
        "domain": "01",
        "doc_type_name": "법령_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/01.*/*/1.데이터/Training/02.라벨링데이터/TL_01. 민사법_002. 법령_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/01.*/*/1.데이터/Training/01.원천데이터/TS_01. 민사법_002. 법령/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "02_TL_법령_QA",
        "domain": "02",
        "doc_type_name": "법령_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_002. 법령_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_002. 법령/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "03_TL_법령_QA",
        "domain": "03",
        "doc_type_name": "법령_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "04_TL_법령_QA",
        "domain": "04",
        "doc_type_name": "법령_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "01_TL_유권해석_QA",
        "domain": "01",
        "doc_type_name": "해석례_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/01.*/*/1.데이터/Training/02.라벨링데이터/TL_01. 민사법_004. 유권해석_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/01.*/*/1.데이터/Training/01.원천데이터/TS_01. 민사법_004. 유권해석/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "02_TL_유권해석_QA",
        "domain": "02",
        "doc_type_name": "해석례_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_005. 유권해석_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_005. 유권해석/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "03_TL_해석례_QA",
        "domain": "03",
        "doc_type_name": "해석례_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_해석례_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_해석례/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "04_TL_해석례_QA",
        "domain": "04",
        "doc_type_name": "해석례_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_해석례_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_해석례/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "01_TL_심결례_QA",
        "domain": "01",
        "doc_type_name": "결정례_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/01.*/*/1.데이터/Training/02.라벨링데이터/TL_01. 민사법_003. 심결례_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/01.*/*/1.데이터/Training/01.원천데이터/TS_01. 민사법_003. 심결례/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "02_TL_심결례_QA",
        "domain": "02",
        "doc_type_name": "결정례_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_003. 심결례_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_003. 심결례/*.json",
        "sample_count": 1,
    },
    {
        "source_subset": "02_TL_심결문_QA",
        "domain": "02",
        "doc_type_name": "결정례_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_004. 심결문_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_004. 심결문/*.json",
        "sample_count": 1,
    },
    {
        "source_subset": "03_TL_결정례_QA",
        "domain": "03",
        "doc_type_name": "결정례_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_결정례_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_결정례/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "04_TL_결정례_QA",
        "domain": "04",
        "doc_type_name": "결정례_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_결정례_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_결정례/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "01_TL_판결문_QA",
        "domain": "01",
        "doc_type_name": "판결문_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/01.*/*/1.데이터/Training/02.라벨링데이터/TL_01. 민사법_001. 판결문_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/01.*/*/1.데이터/Training/01.원천데이터/TS_01. 민사법_001. 판결문/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "02_TL_판결문_QA",
        "domain": "02",
        "doc_type_name": "판결문_QA",
        "sampling_lane": "expansion_01_02",
        "label_glob": "data/raw/aihub/02.*/*/1.데이터/Training/02.라벨링데이터/TL_02. 지식재산권법_001. 판결문_0001. 질의응답/*.json",
        "raw_glob": "data/raw/aihub/02.*/*/1.데이터/Training/01.원천데이터/TS_02. 지식재산권법_001. 판결문/*.json",
        "sample_count": 2,
    },
    {
        "source_subset": "03_TL_판결문_QA",
        "domain": "03",
        "doc_type_name": "판결문_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_판결문_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_판결문/*.csv",
        "sample_count": 3,
    },
    {
        "source_subset": "04_TL_판결문_QA",
        "domain": "04",
        "doc_type_name": "판결문_QA",
        "sampling_lane": "generalization_03_04",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_판결문_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_판결문/*.csv",
        "sample_count": 3,
    },
]

MEETING_EXAMPLE_PRIORITY = ("법령_QA", "결정례_QA", "판결문_QA")
