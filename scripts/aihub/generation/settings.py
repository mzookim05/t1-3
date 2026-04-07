from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

VERSION_TAG = "v5"
RUN_NAME = "2026-04-07_v5_law_refine"
RUN_DIR = PROJECT_ROOT / "analysis" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "generation"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "generation"
PROMPT_DIR = SCRIPT_DIR / "prompts"

SAMPLE_REGISTRY_PATH = INTERIM_DIR / f"sample_registry_{VERSION_TAG}.csv"
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

TRAIN_PATH = PROCESSED_DIR / f"train_{VERSION_TAG}.jsonl"
DEV_PATH = PROCESSED_DIR / f"dev_{VERSION_TAG}.jsonl"
TEST_PATH = PROCESSED_DIR / f"test_{VERSION_TAG}.jsonl"
DATASET_MANIFEST_PATH = PROCESSED_DIR / f"dataset_manifest_{VERSION_TAG}.csv"

# 비밀값이 아닌 실행 상수는 `.env`가 아니라 여기에 두어,
# 어떤 모델과 규칙으로 실행했는지 문서와 코드가 같이 움직이게 한다.
# 생성은 최신 mini 계열을 우선 사용하되, 호환성 문제에 대비해 안정적인
# `gpt-4.1-mini`를 2순위로 둔다.
GENERATOR_MODEL_CANDIDATES = ("gpt-5.4-mini", "gpt-4.1-mini")
# `v4`부터는 동일 런 안에서 Judge 백본을 하나로 통일해 비교 가능성을 높였고,
# `v5`에서도 같은 조건을 유지해 법령 보정 효과만 보게 한다.
JUDGE_MODEL_CANDIDATES = ("gemini-2.5-flash",)
ACTIVE_GENERATION_VARIANT = "without_long_answer"
# `v5`에서도 `long_answer` 제거가 기본 경로이고, 포함 여부는 ablation으로만 비교한다.
GENERATION_INPUT_VARIANTS = (
    {
        "name": "without_long_answer",
        "label": "long_answer 제외",
        "include_long_answer": False,
    },
    {
        "name": "with_long_answer",
        "label": "long_answer 포함",
        "include_long_answer": True,
    },
)

GENERATOR_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 500

HARD_FAIL_TAGS = {"결론 불일치", "근거 왜곡", "원문 외 사실 추가"}
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
        "source_subset": "03_TL_법령_QA",
        "domain": "03",
        "doc_type_name": "법령_QA",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 2,
    },
    {
        "source_subset": "04_TL_법령_QA",
        "domain": "04",
        "doc_type_name": "법령_QA",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_법령_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_법령/*.csv",
        "sample_count": 1,
    },
    {
        "source_subset": "03_TL_해석례_QA",
        "domain": "03",
        "doc_type_name": "해석례_QA",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_해석례_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_해석례/*.csv",
        "sample_count": 2,
    },
    {
        "source_subset": "03_TL_결정례_QA",
        "domain": "03",
        "doc_type_name": "결정례_QA",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_결정례_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_결정례/*.csv",
        "sample_count": 2,
    },
    {
        "source_subset": "04_TL_결정례_QA",
        "domain": "04",
        "doc_type_name": "결정례_QA",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_결정례_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_결정례/*.csv",
        "sample_count": 1,
    },
    {
        "source_subset": "03_TL_판결문_QA",
        "domain": "03",
        "doc_type_name": "판결문_QA",
        "label_glob": "data/raw/aihub/03.*/*/1.데이터/Training/02.라벨링데이터/TL_판결문_QA/*.json",
        "raw_glob": "data/raw/aihub/03.*/*/1.데이터/Training/01.원천데이터/TS_판결문/*.csv",
        "sample_count": 1,
    },
    {
        "source_subset": "04_TL_판결문_QA",
        "domain": "04",
        "doc_type_name": "판결문_QA",
        "label_glob": "data/raw/aihub/04.*/*/1.데이터/Training/02.라벨링데이터/TL_판결문_QA/*.json",
        "raw_glob": "data/raw/aihub/04.*/*/1.데이터/Training/01.원천데이터/TS_판결문/*.csv",
        "sample_count": 1,
    },
]

MEETING_EXAMPLE_PRIORITY = ("법령_QA", "결정례_QA", "판결문_QA")
