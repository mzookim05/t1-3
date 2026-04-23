import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

# production batch runner가 line 폴더로 분리되어도 공용 helper를 절대 import로 찾게 한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[4]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.shared.production_batch_common import (
    PROJECT_ROOT,
    call_gemini_json,
    call_openai_json,
    copy_file_to_run_inputs,
    ensure_dirs,
    load_csv_count,
    load_jsonl,
    load_jsonl_count,
    load_prompt,
    load_selected_seed_ids,
    normalized_text,
    render_prompt,
    snapshot_prompts,
    split_sentences,
    tokenize,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


# 기존 `v3` strict-final을 건드리지 않고, 같은 split-descriptive recipe로 남은 seed만 별도 생산 배치로 돌린다.
VERSION_TAG = "pb1_descriptive"
RUN_DATE = "2026-04-22"
RUN_PURPOSE = "descriptive_v3_default_production_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
REFERENCE_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "explanation_generation" / "v7_strict_final"
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"

REFERENCE_TRAIN_PATH = REFERENCE_PROCESSED_DIR / "train.jsonl"
REFERENCE_MERGED_PATH = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-22_v3_qa_split_descriptive_multiclause"
    / "merged"
    / "merged_problem_scores_v3.csv"
)

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
ANSWERABILITY_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_answerability_{VERSION_TAG}.jsonl"
TASKFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_taskfit_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

GENERATOR_MODEL_CANDIDATES = ("gpt-5.4",)
JUDGE_MODEL_CANDIDATES = ("gemini-2.5-pro",)
GENERATOR_TEMPERATURE = 0.2
JUDGE_TEMPERATURE = 0.1
GENERATOR_MAX_TOKENS = 600
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

PROBLEM_TASK_TYPE = "descriptive_qa_split"
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
ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "split_single_issue_rule",
    "application": "split_single_issue_application",
    "requirement": "split_single_issue_requirement",
    "scope": "split_single_issue_scope",
}
ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding_descriptive.md",
    "Answerability": "judge_answerability_descriptive.md",
    "TaskFit": "judge_taskfit_descriptive.md",
}
ROLE_TO_LOG_PATH = {
    "Grounding": GROUNDING_LOG_PATH,
    "Answerability": ANSWERABILITY_LOG_PATH,
    "TaskFit": TASKFIT_LOG_PATH,
}
PROBLEMATIC_TAIL_PATTERNS = (
    "와 관련된 핵심 판단 기준은 무엇인가요",
    "와 관련된 법원의 판단 기준은 무엇인가요",
    "의 판단 기준은 무엇인가요",
)


def ensure_run_dirs():
    # 새 배치는 기존 `v3` 결과와 완전히 분리된 경로로만 기록해 baseline 비교면을 보존한다.
    ensure_dirs(
        INTERIM_DIR,
        PROCESSED_DIR,
        RUN_DIR,
        RUN_PROMPTS_DIR,
        RUN_INPUTS_DIR,
        RUN_GENERATIONS_DIR,
        RUN_JUDGE_LOGS_DIR,
        RUN_MERGED_DIR,
    )


def extract_multi_query_signal(question_text):
    normalized = normalized_text(question_text)
    signals = []
    if "와 관련된 핵심 판단 기준은 무엇인가요" in normalized:
        signals.append("reason_plus_standard")
    if "와 관련된 법원의 판단 기준은 무엇인가요" in normalized:
        signals.append("reason_plus_court_standard")
    if "의 판단 기준은 무엇인가요" in normalized:
        signals.append("embedded_question_plus_standard")
    if "아니면" in normalized:
        signals.append("explicit_alternative")
    if "여부" in normalized and "법적 입장" in normalized:
        signals.append("issue_plus_position")
    if not signals:
        signals.append("split_candidate_production_batch")
    return "|".join(signals)


def build_split_focus_hint(question_text):
    normalized = normalized_text(question_text)
    replacements = [
        ("와 관련된 핵심 판단 기준은 무엇인가요?", ""),
        ("와 관련된 핵심 판단 기준은 무엇인가요", ""),
        ("와 관련된 법원의 판단 기준은 무엇인가요?", ""),
        ("와 관련된 법원의 판단 기준은 무엇인가요", ""),
        ("의 판단 기준은 무엇인가요?", ""),
        ("의 판단 기준은 무엇인가요", ""),
    ]
    simplified = normalized
    for old, new in replacements:
        simplified = simplified.replace(old, new)
    return normalized_text(simplified).rstrip(" ,")


def build_seed_row(train_row):
    answer_mode = train_row.get("answer_mode", "") or "criteria"
    return {
        "seed_sample_id": train_row["sample_id"],
        "reference_sample_id": train_row["sample_id"],
        "family_id": train_row["family_id"],
        "doc_type_name": train_row["doc_type_name"],
        "source_subset": train_row["source_subset"],
        "sampling_lane": train_row["sampling_lane"],
        "answer_mode": answer_mode,
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "split_single_issue_rule"),
        "explanation_target": train_row.get("explanation_target", ""),
        "selection_role": "descriptive_default_batch_seed",
        "selection_note": "v3 strict-final selected row에 아직 들어가지 않은 train_v7 seed를 descriptive 기본 생산선으로 추가 생성",
        "multi_query_signal": extract_multi_query_signal(train_row["transformed_problem"]),
        "split_focus_hint": build_split_focus_hint(train_row["transformed_problem"]),
        "transformed_problem": train_row["transformed_problem"],
        "short_answer": train_row["short_answer"],
        "generated_explanation": train_row["generated_explanation"],
        "rule_basis": train_row.get("rule_basis", ""),
        "fact_basis": train_row.get("fact_basis", ""),
        "label_path": train_row.get("label_path", ""),
        "raw_path": train_row.get("raw_path", ""),
        "selected_at_utc": utc_now_iso(),
    }


def build_seed_registry():
    ensure_run_dirs()
    selected_seed_ids = load_selected_seed_ids(REFERENCE_MERGED_PATH)
    train_rows = load_jsonl(REFERENCE_TRAIN_PATH)
    seed_rows = [build_seed_row(row) for row in train_rows if row["sample_id"] not in selected_seed_ids]
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["sampling_lane"], row["seed_sample_id"]))
    write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def overlap_ratio(base_text, compare_text):
    base_tokens = set(tokenize(base_text))
    compare_tokens = set(tokenize(compare_text))
    if not base_tokens:
        return 0.0
    return len(base_tokens & compare_tokens) / len(base_tokens)


def strip_meta_tail(text):
    cleaned = normalized_text(text)
    replacements = [
        (r"와 관련된 핵심 판단 기준은 무엇인가요\??$", ""),
        (r"와 관련된 법원의 판단 기준은 무엇인가요\??$", ""),
        (r"의 판단 기준은 무엇인가요\??$", ""),
    ]
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned)
    return normalized_text(cleaned).rstrip(" ,")


def convert_question_to_descriptive(text):
    cleaned = strip_meta_tail(text)
    replacements = [
        (r"어떻게 진행되나요\??$", "절차를 설명하시오."),
        (r"어떻게 되나요\??$", "설명하시오."),
        (r"무엇인가요\??$", "설명하시오."),
        (r"무엇입니까\??$", "설명하시오."),
        (r"무엇입니까$", "설명하시오."),
        (r"무엇인가$", "설명하시오."),
        (r"왜 .+ 판단하였습니까\??$", "그 판단 이유를 설명하시오."),
        (r"왜 .+ 판단되었나요\??$", "그 판단 이유를 설명하시오."),
        (r"이유는 무엇인가요\??$", "이유를 설명하시오."),
        (r"어떤 요소를 고려해야 하나요\??$", "고려해야 하는 요소를 설명하시오."),
        (r"해당하나요\??$", "해당 여부를 설명하시오."),
    ]
    for pattern, replacement in replacements:
        if re.search(pattern, cleaned):
            return re.sub(pattern, replacement, cleaned)

    if "아니면" in cleaned:
        return cleaned.rstrip("?") + "를 어떻게 판단해야 하는지 설명하시오."
    if cleaned.endswith("?"):
        return cleaned[:-1] + "를 설명하시오."
    if cleaned.endswith(("설명하시오.", "서술하시오.", "밝히시오.")):
        return cleaned
    return cleaned.rstrip(".") + "를 설명하시오."


def build_local_fallback_problem(seed):
    # production batch도 `v3`와 같은 split guard를 그대로 써서 meta-tail과 복수 질의를 강하게 잘라낸다.
    focus_text = seed.get("split_focus_hint") or seed["transformed_problem"]
    simplified = convert_question_to_descriptive(focus_text)
    prefix_map = {
        "법령_QA": "다음 법령 상황에 관하여, ",
        "해석례_QA": "다음 해석례 상황에 관하여, ",
        "결정례_QA": "다음 결정례 상황에 관하여, ",
        "판결문_QA": "다음 판결문 상황에 관하여, ",
    }
    prefix = prefix_map.get(seed["doc_type_name"], "다음 상황에 관하여, ")
    return normalized_text(prefix + simplified)


def contains_split_failure(text):
    normalized = normalized_text(text)
    if normalized.count("?") >= 2 or "아니면" in normalized:
        return True
    return any(pattern in normalized for pattern in PROBLEMATIC_TAIL_PATTERNS)


def postprocess_problem(seed, generated_problem):
    cleaned = normalized_text(generated_problem)
    if not cleaned.endswith(("설명하시오.", "서술하시오.", "밝히시오.")):
        cleaned = convert_question_to_descriptive(cleaned)
    if contains_split_failure(cleaned):
        cleaned = build_local_fallback_problem(seed)
    if overlap_ratio(seed["short_answer"], cleaned) >= 0.72:
        cleaned = build_local_fallback_problem(seed)
    return cleaned


def build_generation_messages(seed):
    system_prompt = load_prompt("generator_system_descriptive.txt")
    user_template = load_prompt("generator_user_template_descriptive.md")
    user_prompt = render_prompt(
        user_template,
        {
            "doc_type_name": seed["doc_type_name"],
            "source_subset": seed["source_subset"],
            "problem_generation_mode": seed["problem_generation_mode"],
            "doc_type_prompt_hint": DOC_TYPE_PROMPT_HINTS[seed["doc_type_name"]],
            "transformed_problem": seed["transformed_problem"],
            "split_focus_hint": seed.get("split_focus_hint", ""),
            "multi_query_signal": seed.get("multi_query_signal", ""),
            "short_answer": seed["short_answer"],
            "generated_explanation": seed["generated_explanation"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
        },
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_existing_generation_rows():
    if not GENERATED_PROBLEMS_PATH.exists():
        return []
    return [row for row in load_jsonl(GENERATED_PROBLEMS_PATH) if row.get("generation_mode") == "openai_api"]


def checkpoint_generation_rows(rows, strict_mode):
    checkpoint_every = GENERATOR_STRICT_CHECKPOINT_EVERY if strict_mode else GENERATOR_MAIN_CHECKPOINT_EVERY
    if rows and len(rows) % checkpoint_every == 0:
        write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)


def generate_one(seed, strict_mode):
    candidate_id = f"{seed['seed_sample_id']}::split_descriptive_{VERSION_TAG}"
    while True:
        try:
            response = call_openai_json(
                build_generation_messages(seed),
                response_label=candidate_id,
                model_candidates=GENERATOR_MODEL_CANDIDATES,
                temperature=GENERATOR_TEMPERATURE,
                max_tokens=GENERATOR_MAX_TOKENS,
                timeout_seconds=GENERATOR_API_TIMEOUT_SECONDS,
            )
            generated_problem = response["json"]["generated_problem"].strip()
            split_strategy = normalized_text(response["json"].get("split_strategy", "model_single_issue_reframe"))
            focus_issue = normalized_text(response["json"].get("focus_issue", seed.get("split_focus_hint", "")))
            generator_model = response["model"]
            generation_mode = "openai_api"
        except RuntimeError as exc:
            if strict_mode:
                continue
            generated_problem = build_local_fallback_problem(seed)
            split_strategy = "local_single_issue_fallback"
            focus_issue = seed.get("split_focus_hint", "")
            generator_model = "local_template_fallback"
            generation_mode = f"fallback:{str(exc)[:160]}"

        generated_problem = postprocess_problem(seed, generated_problem)
        return {
            "seed_sample_id": seed["seed_sample_id"],
            "reference_sample_id": seed["reference_sample_id"],
            "candidate_id": candidate_id,
            "problem_task_type": PROBLEM_TASK_TYPE,
            "problem_generation_mode": seed["problem_generation_mode"],
            "doc_type_name": seed["doc_type_name"],
            "source_subset": seed["source_subset"],
            "sampling_lane": seed["sampling_lane"],
            "family_id": seed["family_id"],
            "generated_problem": generated_problem,
            "split_strategy": split_strategy or "model_single_issue_reframe",
            "focus_issue": focus_issue or seed.get("split_focus_hint", ""),
            "multi_query_signal": seed.get("multi_query_signal", ""),
            "split_focus_hint": seed.get("split_focus_hint", ""),
            "gold_short_answer": seed["short_answer"],
            "gold_reference_explanation": seed["generated_explanation"],
            "answer_mode": seed.get("answer_mode", ""),
            "explanation_target": seed.get("explanation_target", ""),
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
            "label_path": seed.get("label_path", ""),
            "raw_path": seed.get("raw_path", ""),
            "generation_model": generator_model,
            "generation_mode": generation_mode,
            "generated_at_utc": utc_now_iso(),
        }


def run_generation(mode="main"):
    snapshot_prompts(["generator_system_descriptive.txt", "generator_user_template_descriptive.md"], RUN_PROMPTS_DIR)
    strict_mode = mode == "strict_finalize"
    seeds = load_jsonl(SEED_READY_PATH)
    rows = load_existing_generation_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for seed in seeds:
        candidate_id = f"{seed['seed_sample_id']}::split_descriptive_{VERSION_TAG}"
        if candidate_id in completed_candidate_ids:
            continue
        rows.append(generate_one(seed, strict_mode=strict_mode))
        completed_candidate_ids.add(candidate_id)
        checkpoint_generation_rows(rows, strict_mode)

    write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)
    return rows


def build_judge_prompt(seed, generation, role_name):
    template = load_prompt(ROLE_TO_PROMPT[role_name])
    return render_prompt(
        template,
        {
            "doc_type_name": seed["doc_type_name"],
            "problem_generation_mode": generation["problem_generation_mode"],
            "generated_problem": generation["generated_problem"],
            "gold_short_answer": generation["gold_short_answer"],
            "gold_reference_explanation": generation["gold_reference_explanation"],
            "source_problem": seed["transformed_problem"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
        },
    )


def build_local_judge_response(seed, generation, role_name):
    generated_problem = generation["generated_problem"]
    source_text = normalized_text(
        " ".join(
            [
                seed["transformed_problem"],
                seed.get("rule_basis", ""),
                seed.get("fact_basis", ""),
                generation["gold_short_answer"],
            ]
        )
    )
    source_overlap = overlap_ratio(source_text, generated_problem)
    answer_overlap = overlap_ratio(generation["gold_short_answer"], generated_problem)
    error_tags = []

    if role_name == "Grounding":
        if source_overlap >= 0.30:
            score = 5
            reason = "문제 문장이 source와 evidence 범위를 크게 벗어나지 않습니다."
        elif source_overlap >= 0.18:
            score = 4
            reason = "문제 문장이 source에 대체로 닫히지만 일부 표현이 느슨합니다."
        else:
            score = 2
            error_tags.append("원문 외 사실 추가")
            reason = "문제 문장이 source와 evidence에 충분히 닫히지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    elif role_name == "Answerability":
        if answer_overlap >= 0.70:
            score = 2
            error_tags.append("정답 누설")
            reason = "문제 문장에 정답 표현이 과하게 드러납니다."
        elif contains_split_failure(generated_problem):
            score = 2
            error_tags.append("복수 쟁점 혼합")
            reason = "복수 질의 흔적이 남아 단일 정답형으로 보기 어렵습니다."
        elif len(tokenize(generation["gold_short_answer"])) <= 2:
            score = 4
            reason = "짧은 정답으로 답변 가능한 구조입니다."
        else:
            score = 5
            reason = "정답과 문제의 대응이 유지됩니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    else:
        sentence_count = len(split_sentences(generated_problem))
        if sentence_count > 3 or generated_problem.count("?") >= 1:
            score = 3
            error_tags.append("형식 부적합")
            reason = "설명형 문제이지만 문장 형식이 아직 질문형에 가깝습니다."
        elif not generated_problem.endswith(("설명하시오.", "서술하시오.", "밝히시오.")):
            score = 3
            error_tags.append("형식 부적합")
            reason = "설명형 서술형 문제 종결 표현이 약합니다."
        else:
            score = 5
            reason = "설명형 서술형 문제 형식에 잘 맞습니다."
        pass_or_fail = "pass" if score >= 3 else "fail"

    return {
        "score": score,
        "pass_or_fail": pass_or_fail,
        "error_tags": [tag for tag in error_tags if tag in ALLOWED_ERROR_TAGS],
        "one_sentence_reason": reason,
    }


def build_judge_row(seed, generation, role_name, response):
    parsed = response["json"]
    return {
        "seed_sample_id": seed["seed_sample_id"],
        "candidate_id": generation["candidate_id"],
        "role_name": role_name,
        "doc_type_name": generation["doc_type_name"],
        "score": int(parsed["score"]),
        "pass_or_fail": parsed["pass_or_fail"],
        "error_tags": parsed.get("error_tags", []),
        "one_sentence_reason": parsed["one_sentence_reason"],
        "judge_model": response["model"],
        "judge_mode": response.get("judge_mode", "gemini_api"),
        "judge_error": response.get("judge_error", ""),
        "judge_attempt_count": response.get("judge_attempt_count", 1),
        "judge_elapsed_seconds": response.get("judge_elapsed_seconds", 0.0),
        "judged_at_utc": utc_now_iso(),
    }


def get_mode_config(mode_name):
    if mode_name == "strict_finalize":
        return {
            "mode_name": mode_name,
            "max_workers": JUDGE_STRICT_MAX_WORKERS,
            "max_attempts": JUDGE_STRICT_MAX_ATTEMPTS,
            "retry_base_seconds": JUDGE_STRICT_RETRY_BASE_SECONDS,
            "success_sleep_seconds": JUDGE_STRICT_SUCCESS_SLEEP_SECONDS,
            "checkpoint_every": JUDGE_STRICT_CHECKPOINT_EVERY,
            "allow_local_fallback": False,
        }

    return {
        "mode_name": mode_name,
        "max_workers": JUDGE_MAIN_MAX_WORKERS,
        "max_attempts": JUDGE_MAIN_MAX_ATTEMPTS,
        "retry_base_seconds": JUDGE_MAIN_RETRY_BASE_SECONDS,
        "success_sleep_seconds": JUDGE_MAIN_SUCCESS_SLEEP_SECONDS,
        "checkpoint_every": JUDGE_MAIN_CHECKPOINT_EVERY,
        "allow_local_fallback": True,
    }


def evaluate_one(seed, generation, role_name, mode_config):
    attempt_count = 0
    while True:
        attempt_count += 1
        started_at = time.monotonic()
        try:
            response = call_gemini_json(
                build_judge_prompt(seed, generation, role_name),
                response_label=f"{generation['candidate_id']}::{role_name}",
                model_candidates=JUDGE_MODEL_CANDIDATES,
                temperature=JUDGE_TEMPERATURE,
                timeout_seconds=JUDGE_API_TIMEOUT_SECONDS,
                allowed_error_tags=ALLOWED_ERROR_TAGS,
            )
            response["judge_mode"] = "gemini_api"
            response["judge_error"] = ""
            response["judge_attempt_count"] = attempt_count
            response["judge_elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            row = build_judge_row(seed, generation, role_name, response)
            time.sleep(mode_config["success_sleep_seconds"])
            return role_name, row
        except Exception as exc:  # noqa: BLE001
            if mode_config["allow_local_fallback"] and mode_config["max_attempts"] and attempt_count >= mode_config["max_attempts"]:
                response = {
                    "json": build_local_judge_response(seed, generation, role_name),
                    "model": "local_rule_fallback",
                    "judge_mode": "local_rule_fallback",
                    "judge_error": str(exc)[:300],
                    "judge_attempt_count": attempt_count,
                    "judge_elapsed_seconds": round(time.monotonic() - started_at, 3),
                }
                row = build_judge_row(seed, generation, role_name, response)
                return role_name, row

            wait_seconds = min(60, mode_config["retry_base_seconds"] * attempt_count)
            print(
                "[problem judge retry]",
                f"mode={mode_config['mode_name']}",
                f"seed_sample_id={seed['seed_sample_id']}",
                f"candidate_id={generation['candidate_id']}",
                f"role={role_name}",
                f"attempt={attempt_count}",
                f"wait={wait_seconds}s",
                f"error={str(exc)[:300]}",
                flush=True,
            )
            time.sleep(wait_seconds)


def checkpoint_judge_outputs(outputs):
    write_jsonl_atomic(GROUNDING_LOG_PATH, outputs["Grounding"])
    write_jsonl_atomic(ANSWERABILITY_LOG_PATH, outputs["Answerability"])
    write_jsonl_atomic(TASKFIT_LOG_PATH, outputs["TaskFit"])


def load_existing_judge_outputs():
    outputs = {role_name: [] for role_name in ROLE_TO_PROMPT}
    for role_name, log_path in ROLE_TO_LOG_PATH.items():
        if not Path(log_path).exists():
            continue
        for row in load_jsonl(log_path):
            if row.get("judge_mode") == "gemini_api":
                outputs[role_name].append(row)
    return outputs


def run_judges(mode="main"):
    mode_config = get_mode_config(mode)
    snapshot_prompts(["judge_grounding_descriptive.md", "judge_answerability_descriptive.md", "judge_taskfit_descriptive.md"], RUN_PROMPTS_DIR)
    seed_map = {seed["seed_sample_id"]: seed for seed in load_jsonl(SEED_READY_PATH)}
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    outputs = load_existing_judge_outputs()
    completed_keys = {
        (row["candidate_id"], role_name)
        for role_name, rows in outputs.items()
        for row in rows
    }

    pending_jobs = []
    for generation in generations:
        seed = seed_map[generation["seed_sample_id"]]
        for role_name in ROLE_TO_PROMPT:
            key = (generation["candidate_id"], role_name)
            if key in completed_keys:
                continue
            pending_jobs.append((seed, generation, role_name))

    if not pending_jobs:
        checkpoint_judge_outputs(outputs)
        return outputs

    with ThreadPoolExecutor(max_workers=mode_config["max_workers"]) as executor:
        future_map = {
            executor.submit(evaluate_one, seed, generation, role_name, mode_config): (seed, generation, role_name)
            for seed, generation, role_name in pending_jobs
        }
        completed_since_checkpoint = 0
        for future in as_completed(future_map):
            role_name, row = future.result()
            outputs[role_name].append(row)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= mode_config["checkpoint_every"]:
                checkpoint_judge_outputs(outputs)
                completed_since_checkpoint = 0

    checkpoint_judge_outputs(outputs)
    return outputs


def index_rows(rows):
    return {row["candidate_id"]: row for row in rows}


def merge_tags(*tag_lists):
    merged = []
    for tag_list in tag_lists:
        for tag in tag_list:
            if tag not in merged:
                merged.append(tag)
    return merged


def finalize_status(grounding_score, answerability_score, taskfit_score, error_tags, weighted_score):
    if grounding_score < 4 or answerability_score < 4:
        return "hard_fail"
    if any(tag in HARD_FAIL_TAGS for tag in error_tags):
        return "hard_fail"
    if taskfit_score < 3 or weighted_score < 3.8:
        return "soft_fail"
    return "pass"


def build_selected_flags(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["seed_sample_id"], []).append(row)
    selected_candidate_ids = set()
    for seed_sample_id, seed_rows in grouped.items():
        best_row = sorted(
            seed_rows,
            key=lambda row: (row["final_status"] != "pass", -row["weighted_score"], row["candidate_id"]),
        )[0]
        selected_candidate_ids.add(best_row["candidate_id"])
    return selected_candidate_ids


def merge_scores():
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    grounding_map = index_rows(load_jsonl(GROUNDING_LOG_PATH))
    answerability_map = index_rows(load_jsonl(ANSWERABILITY_LOG_PATH))
    taskfit_map = index_rows(load_jsonl(TASKFIT_LOG_PATH))
    rows = []

    for generation in generations:
        grounding = grounding_map[generation["candidate_id"]]
        answerability = answerability_map[generation["candidate_id"]]
        taskfit = taskfit_map[generation["candidate_id"]]
        error_tags = merge_tags(
            grounding.get("error_tags", []),
            answerability.get("error_tags", []),
            taskfit.get("error_tags", []),
        )
        weighted_score = round(
            grounding["score"] * SCORE_WEIGHTS["Grounding"]
            + answerability["score"] * SCORE_WEIGHTS["Answerability"]
            + taskfit["score"] * SCORE_WEIGHTS["TaskFit"],
            4,
        )
        final_status = finalize_status(
            grounding_score=grounding["score"],
            answerability_score=answerability["score"],
            taskfit_score=taskfit["score"],
            error_tags=error_tags,
            weighted_score=weighted_score,
        )
        audit_required = "예" if final_status == "pass" and error_tags else "아니오"
        train_eligible = "예" if final_status == "pass" and audit_required == "아니오" else "아니오"
        rows.append(
            {
                "seed_sample_id": generation["seed_sample_id"],
                "candidate_id": generation["candidate_id"],
                "problem_task_type": generation["problem_task_type"],
                "problem_generation_mode": generation["problem_generation_mode"],
                "doc_type_name": generation["doc_type_name"],
                "source_subset": generation["source_subset"],
                "sampling_lane": generation["sampling_lane"],
                "family_id": generation["family_id"],
                "generated_problem": generation["generated_problem"],
                "split_strategy": generation.get("split_strategy", ""),
                "focus_issue": generation.get("focus_issue", ""),
                "multi_query_signal": generation.get("multi_query_signal", ""),
                "split_focus_hint": generation.get("split_focus_hint", ""),
                "gold_short_answer": generation["gold_short_answer"],
                "gold_reference_explanation": generation["gold_reference_explanation"],
                "answer_mode": generation.get("answer_mode", ""),
                "explanation_target": generation.get("explanation_target", ""),
                "rule_basis": generation.get("rule_basis", ""),
                "fact_basis": generation.get("fact_basis", ""),
                "grounding_score": grounding["score"],
                "answerability_score": answerability["score"],
                "taskfit_score": taskfit["score"],
                "weighted_score": weighted_score,
                "error_tags": "|".join(error_tags),
                "final_status": final_status,
                "audit_required": audit_required,
                "audit_reason": "|".join(error_tags) if audit_required == "예" else "",
                "train_eligible": train_eligible,
                "generator_model": generation["generation_model"],
                "generation_mode": generation["generation_mode"],
                "grounding_judge_model": grounding["judge_model"],
                "answerability_judge_model": answerability["judge_model"],
                "taskfit_judge_model": taskfit["judge_model"],
                "version_tag": VERSION_TAG,
                "run_name": RUN_NAME,
                "label_path": generation.get("label_path", ""),
                "raw_path": generation.get("raw_path", ""),
            }
        )

    selected_candidate_ids = build_selected_flags(rows)
    for row in rows:
        row["selected_for_seed"] = "예" if row["candidate_id"] in selected_candidate_ids else "아니오"
    rows.sort(key=lambda row: (row["seed_sample_id"], row["candidate_id"]))
    write_csv_atomic(MERGED_SCORES_PATH, rows, list(rows[0].keys()))
    return rows


def assign_splits(rows):
    selected_rows = [row for row in rows if row["selected_for_seed"] == "예" and row["final_status"] == "pass"]
    trainable_rows = [row for row in selected_rows if row.get("train_eligible", "예") == "예"]
    audit_rows = [row for row in selected_rows if row.get("train_eligible", "예") != "예"]

    family_to_split = {}
    families_by_doc_type = {}
    seen = set()
    for row in trainable_rows:
        key = (row["doc_type_name"], row["family_id"])
        if key in seen:
            continue
        seen.add(key)
        families_by_doc_type.setdefault(row["doc_type_name"], []).append(row["family_id"])

    for doc_type_name, families in families_by_doc_type.items():
        total = len(families)
        if total >= 5:
            train_count = total - 2
            dev_count = 1
        elif total == 4:
            train_count = 2
            dev_count = 1
        elif total == 3:
            train_count = 1
            dev_count = 1
        elif total == 2:
            train_count = 1
            dev_count = 0
        elif total == 1:
            train_count = 1
            dev_count = 0
        else:
            train_count = 0
            dev_count = 0

        for index, family_id in enumerate(families):
            if index < train_count:
                family_to_split[family_id] = "train"
            elif index < train_count + dev_count:
                family_to_split[family_id] = "dev"
            else:
                family_to_split[family_id] = "test"

    manifest_rows = []
    train_rows, dev_rows, test_rows = [], [], []
    for row in trainable_rows:
        split = family_to_split[row["family_id"]]
        payload = {
            "problem_id": row["candidate_id"],
            "seed_sample_id": row["seed_sample_id"],
            "family_id": row["family_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row.get("sampling_lane", ""),
            "problem_task_type": row["problem_task_type"],
            "problem_generation_mode": row["problem_generation_mode"],
            "generated_problem": row["generated_problem"],
            "split_strategy": row.get("split_strategy", ""),
            "focus_issue": row.get("focus_issue", ""),
            "multi_query_signal": row.get("multi_query_signal", ""),
            "split_focus_hint": row.get("split_focus_hint", ""),
            "gold_short_answer": row["gold_short_answer"],
            "gold_reference_explanation": row["gold_reference_explanation"],
            "answer_mode": row.get("answer_mode", ""),
            "explanation_target": row.get("explanation_target", ""),
            "weighted_score": row["weighted_score"],
            "error_tags": row.get("error_tags", ""),
            "generator_model": row.get("generator_model", ""),
            "generation_mode": row.get("generation_mode", ""),
            "version_tag": row.get("version_tag", ""),
            "run_name": row.get("run_name", ""),
            "label_path": row.get("label_path", ""),
            "raw_path": row.get("raw_path", ""),
            "split": split,
        }
        if split == "train":
            train_rows.append(payload)
        elif split == "dev":
            dev_rows.append(payload)
        else:
            test_rows.append(payload)
        manifest_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": split,
                "dataset_disposition": split,
                "train_eligible": row.get("train_eligible", "예"),
                "audit_required": row.get("audit_required", "아니오"),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
            }
        )

    audit_rows_payload = []
    for row in audit_rows:
        audit_rows_payload.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row.get("sampling_lane", ""),
                "generated_problem": row["generated_problem"],
                "split_strategy": row.get("split_strategy", ""),
                "focus_issue": row.get("focus_issue", ""),
                "gold_short_answer": row["gold_short_answer"],
                "error_tags": row.get("error_tags", ""),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
                "version_tag": row.get("version_tag", ""),
                "run_name": row.get("run_name", ""),
                "label_path": row.get("label_path", ""),
                "raw_path": row.get("raw_path", ""),
            }
        )
        manifest_rows.append(
            {
                "problem_id": row["candidate_id"],
                "seed_sample_id": row["seed_sample_id"],
                "family_id": row["family_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "split": "",
                "dataset_disposition": "audit",
                "train_eligible": row.get("train_eligible", "아니오"),
                "audit_required": row.get("audit_required", "예"),
                "audit_reason": row.get("audit_reason", ""),
                "weighted_score": row["weighted_score"],
            }
        )

    return train_rows, dev_rows, test_rows, manifest_rows, audit_rows_payload


def split_dataset(rows):
    train_rows, dev_rows, test_rows, manifest_rows, audit_rows = assign_splits(rows)
    write_jsonl_atomic(PROBLEM_TRAIN_PATH, train_rows)
    write_jsonl_atomic(PROBLEM_DEV_PATH, dev_rows)
    write_jsonl_atomic(PROBLEM_TEST_PATH, test_rows)
    audit_fieldnames = [
        "problem_id",
        "seed_sample_id",
        "family_id",
        "doc_type_name",
        "source_subset",
        "sampling_lane",
        "generated_problem",
        "split_strategy",
        "focus_issue",
        "gold_short_answer",
        "error_tags",
        "audit_reason",
        "weighted_score",
        "version_tag",
        "run_name",
        "label_path",
        "raw_path",
    ]
    write_csv_atomic(
        PROBLEM_AUDIT_QUEUE_PATH,
        audit_rows,
        list(audit_rows[0].keys()) if audit_rows else audit_fieldnames,
    )
    if manifest_rows:
        write_csv_atomic(PROBLEM_DATASET_MANIFEST_PATH, manifest_rows, list(manifest_rows[0].keys()))
    return manifest_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows):
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "created_at_utc": utc_now_iso(),
        "reference_train_path": str(REFERENCE_TRAIN_PATH),
        "reference_merged_path": str(REFERENCE_MERGED_PATH),
        "seed_registry_strategy": "exclude_v3_selected_seed_ids",
        "seed_registry_count": len(seed_rows),
        "generation_count": load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_answerability_count": load_jsonl_count(ANSWERABILITY_LOG_PATH),
        "judge_taskfit_count": load_jsonl_count(TASKFIT_LOG_PATH),
        "merged_count": load_csv_count(MERGED_SCORES_PATH),
        "selected_pass_count": sum(1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "pass"),
        "selected_hard_fail_count": sum(1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "hard_fail"),
        "selected_soft_fail_count": sum(1 for row in merged_rows if row["selected_for_seed"] == "예" and row["final_status"] == "soft_fail"),
        "dataset_manifest_count": len(manifest_rows),
        "problem_train_count": load_jsonl_count(PROBLEM_TRAIN_PATH),
        "problem_dev_count": load_jsonl_count(PROBLEM_DEV_PATH),
        "problem_test_count": load_jsonl_count(PROBLEM_TEST_PATH),
        "problem_audit_count": load_csv_count(PROBLEM_AUDIT_QUEUE_PATH),
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "generated_problems": str(GENERATED_PROBLEMS_PATH),
            "judge_grounding_log": str(GROUNDING_LOG_PATH),
            "judge_answerability_log": str(ANSWERABILITY_LOG_PATH),
            "judge_taskfit_log": str(TASKFIT_LOG_PATH),
            "merged_scores": str(MERGED_SCORES_PATH),
            "problem_train": str(PROBLEM_TRAIN_PATH),
            "problem_dev": str(PROBLEM_DEV_PATH),
            "problem_test": str(PROBLEM_TEST_PATH),
            "problem_dataset_manifest": str(PROBLEM_DATASET_MANIFEST_PATH),
            "problem_audit_queue": str(PROBLEM_AUDIT_QUEUE_PATH),
        },
    }
    write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def main():
    seed_rows = build_seed_registry()
    run_generation(mode="main")
    run_generation(mode="strict_finalize")
    run_judges(mode="main")
    run_judges(mode="strict_finalize")
    merged_rows = merge_scores()
    manifest_rows = split_dataset(merged_rows)
    return build_run_manifest(seed_rows, merged_rows, manifest_rows)


if __name__ == "__main__":
    main()
