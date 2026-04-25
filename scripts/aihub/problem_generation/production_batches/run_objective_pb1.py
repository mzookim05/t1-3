import hashlib
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
    json_dumps_stable,
    load_csv_count,
    load_jsonl,
    load_jsonl_count,
    load_prompt,
    load_selected_seed_ids,
    normalized_text,
    render_prompt,
    snapshot_prompts,
    tokenize,
    utc_now_iso,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)


# 기존 `v2` strict-final을 덮어쓰지 않고, 같은 objective recipe로 남은 seed만 별도 생산 배치로 돌린다.
VERSION_TAG = "pb1_objective"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-22_164512"
RUN_PURPOSE = "objective_v2_default_production_batch"
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
    / "2026-04-14_203007_v2_qa_objective_single_best"
    / "merged"
    / "merged_problem_scores_v2.csv"
)

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
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
ANSWER_MODE_TO_PROBLEM_MODE = {
    "criteria": "single_best_rule",
    "application": "single_best_application",
    "requirement": "single_best_rule",
    "scope": "single_best_scope",
}
MODE_TO_STEM_ENDING = {
    "single_best_rule": "옳은 설명을 고르시오.",
    "single_best_application": "가장 적절한 설명을 고르시오.",
    "single_best_scope": "올바른 적용 범위를 고르시오.",
}
CHOICE_LABELS = ("A", "B", "C", "D")
ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding_objective.md",
    "Keyedness": "judge_keyedness_objective.md",
    "DistractorFit": "judge_distractorfit_objective.md",
}
ROLE_TO_LOG_PATH = {
    "Grounding": GROUNDING_LOG_PATH,
    "Keyedness": KEYEDNESS_LOG_PATH,
    "DistractorFit": DISTRACTORFIT_LOG_PATH,
}


def ensure_run_dirs():
    # 새 배치는 기존 `v2` 산출물과 경로가 절대 겹치지 않도록 별도 폴더를 강제한다.
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
        "problem_generation_mode": ANSWER_MODE_TO_PROBLEM_MODE.get(answer_mode, "single_best_rule"),
        "explanation_target": train_row.get("explanation_target", ""),
        "selection_role": "objective_default_batch_seed",
        "selection_note": "v2 strict-final selected row에 아직 들어가지 않은 v7_strict_final/train.jsonl seed를 objective 기본 생산선으로 추가 생성",
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


def compress_question_stem(text):
    stem = normalized_text(text).rstrip("?")
    replacements = [
        (r"와 관련된 핵심 판단 기준은 무엇인가요$", ""),
        (r"의 핵심 판단 기준은 무엇인가요$", ""),
        (r"의 이유는 무엇인가요$", ""),
        (r"무엇인가요$", ""),
        (r"무엇입니까$", ""),
    ]
    for pattern, replacement in replacements:
        stem = re.sub(pattern, replacement, stem)
    return stem.rstrip(" ,")


def short_answer_to_choice_text(short_answer):
    first_sentence = normalized_text(short_answer).split(". ")[0].strip()
    return first_sentence.rstrip(".") + "."


def build_distractor_variants(seed):
    doc_type = seed["doc_type_name"]
    if doc_type == "법령_QA":
        return [
            ("요건 일부 누락", "일부 요건만 충족하면 언제나 허용된다고 본다."),
            ("효과/절차 혼동", "절차 규정이므로 법적 효과와는 직접 관련이 없다고 본다."),
            ("범위 과대/과소", "적용 범위를 모든 경우로 넓혀 해석한다."),
        ]
    if doc_type == "해석례_QA":
        return [
            ("요건 일부 누락", "회답의 전제 요건을 고려하지 않고 결론만 일반화한다."),
            ("효과/절차 혼동", "회답 이유 대신 절차적 표현만으로 결론을 바꾼다."),
            ("범위 과대/과소", "해석 범위를 지나치게 넓히거나 좁힌다."),
        ]
    if doc_type == "결정례_QA":
        return [
            ("요건 일부 누락", "핵심 판단 이유 한 부분만 떼어 내 단정한다."),
            ("효과/절차 혼동", "판단 기준과 결론의 이유를 뒤섞어 설명한다."),
            ("범위 과대/과소", "사안의 적용 범위를 과도하게 확장한다."),
        ]
    return [
        ("요건 일부 누락", "판시 기준의 일부만 반영해 결론을 단정한다."),
        ("효과/절차 혼동", "판단 기준과 적용 결과를 서로 바꿔 설명한다."),
        ("범위 과대/과소", "판시 범위를 일반론으로 과도하게 확장한다."),
    ]


def stable_correct_choice(seed_sample_id):
    digest = hashlib.sha256(seed_sample_id.encode("utf-8")).hexdigest()
    return CHOICE_LABELS[int(digest[:2], 16) % len(CHOICE_LABELS)]


def build_local_fallback_problem(seed):
    stem_core = compress_question_stem(seed["transformed_problem"])
    stem = normalized_text(f"{stem_core} {MODE_TO_STEM_ENDING[seed['problem_generation_mode']]}")
    correct_choice = stable_correct_choice(seed["seed_sample_id"])
    correct_text = short_answer_to_choice_text(seed["short_answer"])
    choices = {}
    distractor_type_map = {}
    distractors = build_distractor_variants(seed)
    distractor_labels = [label for label in CHOICE_LABELS if label != correct_choice]
    if len(distractor_labels) != len(distractors):
        raise RuntimeError("객관식 production fallback 선택지 수와 오답 후보 수가 일치하지 않습니다.")
    # Pylance가 `zip(strict=True)` overload를 좁히지 못해, 길이는 위에서 직접 검증하고 일반 zip을 사용한다.
    for label, (dtype, text) in zip(distractor_labels, distractors):
        choices[label] = text
        distractor_type_map[label] = dtype
    choices[correct_choice] = correct_text
    distractor_type_map[correct_choice] = "정답"
    return {
        "generated_stem": stem,
        "choice_a": choices["A"],
        "choice_b": choices["B"],
        "choice_c": choices["C"],
        "choice_d": choices["D"],
        "correct_choice": correct_choice,
        "distractor_type_map": distractor_type_map,
    }


def build_generation_messages(seed):
    system_prompt = load_prompt("generator_system_objective.txt")
    user_template = load_prompt("generator_user_template_objective.md")
    user_prompt = render_prompt(
        user_template,
        {
            "doc_type_name": seed["doc_type_name"],
            "source_subset": seed["source_subset"],
            "problem_generation_mode": seed["problem_generation_mode"],
            "doc_type_prompt_hint": DOC_TYPE_PROMPT_HINTS[seed["doc_type_name"]],
            "transformed_problem": seed["transformed_problem"],
            "short_answer": seed["short_answer"],
            "generated_explanation": seed["generated_explanation"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
            "problem_v1_generated_problem": "",
        },
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def validate_generated_payload(payload):
    correct_choice = normalized_text(payload["correct_choice"]).upper()
    if correct_choice not in CHOICE_LABELS:
        raise RuntimeError("correct_choice가 A/B/C/D가 아닙니다.")

    choices = [normalized_text(payload[f"choice_{label.lower()}"]) for label in CHOICE_LABELS]
    if any(not choice for choice in choices):
        raise RuntimeError("빈 선택지가 있습니다.")
    if len(set(choices)) != 4:
        raise RuntimeError("선택지 중복이 있습니다.")

    stem = normalized_text(payload["generated_stem"])
    if not stem.endswith(("고르시오.", "고르세요.", "고르십시오.")):
        payload["generated_stem"] = stem.rstrip(".?") + " 옳은 설명을 고르시오."

    return payload


def postprocess_problem(seed, payload):
    payload = validate_generated_payload(payload)
    stem = normalized_text(payload["generated_stem"])
    if overlap_ratio(seed["short_answer"], stem) >= 0.72:
        return build_local_fallback_problem(seed)
    return payload


def load_existing_generation_rows():
    if not GENERATED_PROBLEMS_PATH.exists():
        return []
    return [row for row in load_jsonl(GENERATED_PROBLEMS_PATH) if row.get("generation_mode") == "openai_api"]


def checkpoint_generation_rows(rows, strict_mode):
    checkpoint_every = GENERATOR_STRICT_CHECKPOINT_EVERY if strict_mode else GENERATOR_MAIN_CHECKPOINT_EVERY
    if rows and len(rows) % checkpoint_every == 0:
        write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)


def generate_one(seed, strict_mode):
    candidate_id = f"{seed['seed_sample_id']}::objective_{VERSION_TAG}"
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
            payload = response["json"]
            generator_model = response["model"]
            generation_mode = "openai_api"
        except RuntimeError as exc:
            if strict_mode:
                continue
            payload = build_local_fallback_problem(seed)
            generator_model = "local_template_fallback"
            generation_mode = f"fallback:{str(exc)[:160]}"

        try:
            payload = postprocess_problem(seed, payload)
        except RuntimeError:
            if strict_mode:
                continue
            payload = build_local_fallback_problem(seed)
            generator_model = "local_template_fallback"
            generation_mode = "fallback:postprocess_guard"

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
            "generated_stem": payload["generated_stem"],
            "choice_a": payload["choice_a"],
            "choice_b": payload["choice_b"],
            "choice_c": payload["choice_c"],
            "choice_d": payload["choice_d"],
            "correct_choice": payload["correct_choice"],
            "distractor_type_map": payload.get("distractor_type_map", {}),
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
    snapshot_prompts(["generator_system_objective.txt", "generator_user_template_objective.md"], RUN_PROMPTS_DIR)
    strict_mode = mode == "strict_finalize"
    seeds = load_jsonl(SEED_READY_PATH)
    rows = load_existing_generation_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for seed in seeds:
        candidate_id = f"{seed['seed_sample_id']}::objective_{VERSION_TAG}"
        if candidate_id in completed_candidate_ids:
            continue
        rows.append(generate_one(seed, strict_mode=strict_mode))
        completed_candidate_ids.add(candidate_id)
        checkpoint_generation_rows(rows, strict_mode)

    write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)
    return rows


def choices_text(generation):
    return " ".join([generation["choice_a"], generation["choice_b"], generation["choice_c"], generation["choice_d"]])


def build_judge_prompt(seed, generation, role_name):
    template = load_prompt(ROLE_TO_PROMPT[role_name])
    return render_prompt(
        template,
        {
            "doc_type_name": seed["doc_type_name"],
            "problem_generation_mode": generation["problem_generation_mode"],
            "generated_stem": generation["generated_stem"],
            "choice_a": generation["choice_a"],
            "choice_b": generation["choice_b"],
            "choice_c": generation["choice_c"],
            "choice_d": generation["choice_d"],
            "correct_choice": generation["correct_choice"],
            "gold_short_answer": generation["gold_short_answer"],
            "gold_reference_explanation": generation["gold_reference_explanation"],
            "source_problem": seed["transformed_problem"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
        },
    )


def build_local_judge_response(seed, generation, role_name):
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
    option_map = {
        "A": generation["choice_a"],
        "B": generation["choice_b"],
        "C": generation["choice_c"],
        "D": generation["choice_d"],
    }
    correct_choice = generation["correct_choice"]
    correct_text = option_map.get(correct_choice, "")
    all_choices = list(option_map.values())
    error_tags = []

    if role_name == "Grounding":
        source_overlap = overlap_ratio(source_text, normalized_text(generation["generated_stem"] + " " + choices_text(generation)))
        if source_overlap >= 0.30:
            score = 5
            reason = "문제와 선택지가 source와 evidence 범위를 크게 벗어나지 않습니다."
        elif source_overlap >= 0.18:
            score = 4
            reason = "문제와 선택지가 source에 대체로 닫히지만 일부 표현이 느슨합니다."
        else:
            score = 2
            error_tags.append("원문 외 사실 추가")
            reason = "선택지나 문제 문장이 source 범위를 충분히 따르지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    elif role_name == "Keyedness":
        duplicate_choices = len(set(normalized_text(choice) for choice in all_choices)) != 4
        if duplicate_choices:
            score = 1
            error_tags.append("선택지 중복")
            pass_or_fail = "fail"
            reason = "선택지 중복이 있어 단일정답형으로 볼 수 없습니다."
        else:
            correct_overlap = overlap_ratio(generation["gold_short_answer"], correct_text)
            competing_overlaps = [
                overlap_ratio(generation["gold_short_answer"], choice)
                for label, choice in option_map.items()
                if label != correct_choice
            ]
            if correct_overlap < 0.18:
                score = 2
                error_tags.append("정답 비유일")
                pass_or_fail = "fail"
                reason = "정답 선택지와 기준 정답의 대응이 약합니다."
            elif any(overlap >= correct_overlap - 0.05 for overlap in competing_overlaps):
                score = 2
                error_tags.append("오답이 정답 가능")
                pass_or_fail = "fail"
                reason = "오답 선택지 중 기준 정답과 지나치게 가까운 선택지가 있습니다."
            else:
                score = 5
                pass_or_fail = "pass"
                reason = "단일정답 구조가 유지됩니다."
    else:
        stem_overlap = overlap_ratio(generation["gold_short_answer"], generation["generated_stem"])
        if stem_overlap >= 0.70:
            score = 2
            error_tags.append("정답 누설")
            reason = "문제 본문에 정답이 과하게 노출됩니다."
        elif "?" in generation["generated_stem"] and generation["generated_stem"].count("?") >= 2:
            score = 2
            error_tags.append("복수 쟁점 혼합")
            reason = "문제 본문에 복수 질의가 섞여 있습니다."
        else:
            distractor_lengths = [len(tokenize(choice)) for label, choice in option_map.items() if label != correct_choice]
            if distractor_lengths and min(distractor_lengths) < 3:
                score = 3
                reason = "오답 선택지 길이가 짧아 품질이 다소 약합니다."
            else:
                score = 5
                reason = "오답 선택지가 그럴듯하지만 정답과 구분됩니다."
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
    write_jsonl_atomic(KEYEDNESS_LOG_PATH, outputs["Keyedness"])
    write_jsonl_atomic(DISTRACTORFIT_LOG_PATH, outputs["DistractorFit"])


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
    snapshot_prompts(["judge_grounding_objective.md", "judge_keyedness_objective.md", "judge_distractorfit_objective.md"], RUN_PROMPTS_DIR)
    seed_map = {seed["seed_sample_id"]: seed for seed in load_jsonl(SEED_READY_PATH)}
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    outputs = load_existing_judge_outputs()
    completed_keys = {
        (row["candidate_id"], role_name)
        for role_name, rows in outputs.items()
        for row in rows
    }

    futures = []
    with ThreadPoolExecutor(max_workers=mode_config["max_workers"]) as executor:
        for generation in generations:
            seed = seed_map[generation["seed_sample_id"]]
            for role_name in outputs:
                key = (generation["candidate_id"], role_name)
                if key in completed_keys:
                    continue
                futures.append(executor.submit(evaluate_one, seed, generation, role_name, mode_config))

        for completed_index, future in enumerate(as_completed(futures), start=1):
            resolved_role_name, row = future.result()
            outputs[resolved_role_name].append(row)
            if completed_index % max(1, mode_config["checkpoint_every"]) == 0:
                checkpoint_judge_outputs(outputs)

    for role_name in outputs:
        outputs[role_name].sort(key=lambda row: (row["seed_sample_id"], row["candidate_id"]))
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


def finalize_status(grounding_score, keyedness_score, distractorfit_score, error_tags, weighted_score):
    if grounding_score < 4 or keyedness_score < 4:
        return "hard_fail"
    if any(tag in HARD_FAIL_TAGS for tag in error_tags):
        return "hard_fail"
    if distractorfit_score < 3 or weighted_score < 3.8:
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
    keyedness_map = index_rows(load_jsonl(KEYEDNESS_LOG_PATH))
    distractorfit_map = index_rows(load_jsonl(DISTRACTORFIT_LOG_PATH))
    rows = []

    for generation in generations:
        grounding = grounding_map[generation["candidate_id"]]
        keyedness = keyedness_map[generation["candidate_id"]]
        distractorfit = distractorfit_map[generation["candidate_id"]]
        error_tags = merge_tags(
            grounding.get("error_tags", []),
            keyedness.get("error_tags", []),
            distractorfit.get("error_tags", []),
        )
        weighted_score = round(
            grounding["score"] * SCORE_WEIGHTS["Grounding"]
            + keyedness["score"] * SCORE_WEIGHTS["Keyedness"]
            + distractorfit["score"] * SCORE_WEIGHTS["DistractorFit"],
            4,
        )
        final_status = finalize_status(
            grounding_score=grounding["score"],
            keyedness_score=keyedness["score"],
            distractorfit_score=distractorfit["score"],
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
                "generated_stem": generation["generated_stem"],
                "choice_a": generation["choice_a"],
                "choice_b": generation["choice_b"],
                "choice_c": generation["choice_c"],
                "choice_d": generation["choice_d"],
                "correct_choice": generation["correct_choice"],
                "distractor_type_map": json_dumps_stable(generation.get("distractor_type_map", {})),
                "gold_short_answer": generation["gold_short_answer"],
                "gold_reference_explanation": generation["gold_reference_explanation"],
                "answer_mode": generation.get("answer_mode", ""),
                "explanation_target": generation.get("explanation_target", ""),
                "rule_basis": generation.get("rule_basis", ""),
                "fact_basis": generation.get("fact_basis", ""),
                "grounding_score": grounding["score"],
                "keyedness_score": keyedness["score"],
                "distractorfit_score": distractorfit["score"],
                "weighted_score": weighted_score,
                "error_tags": "|".join(error_tags),
                "final_status": final_status,
                "audit_required": audit_required,
                "audit_reason": "|".join(error_tags) if audit_required == "예" else "",
                "train_eligible": train_eligible,
                "generator_model": generation["generation_model"],
                "generation_mode": generation["generation_mode"],
                "grounding_judge_model": grounding["judge_model"],
                "keyedness_judge_model": keyedness["judge_model"],
                "distractorfit_judge_model": distractorfit["judge_model"],
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
            "generated_stem": row["generated_stem"],
            "choice_a": row["choice_a"],
            "choice_b": row["choice_b"],
            "choice_c": row["choice_c"],
            "choice_d": row["choice_d"],
            "correct_choice": row["correct_choice"],
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
                "generated_stem": row["generated_stem"],
                "choice_a": row["choice_a"],
                "choice_b": row["choice_b"],
                "choice_c": row["choice_c"],
                "choice_d": row["choice_d"],
                "correct_choice": row["correct_choice"],
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
                "split": "audit",
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
        "generated_stem",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "correct_choice",
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
        "seed_registry_strategy": "exclude_v2_selected_seed_ids",
        "seed_registry_count": len(seed_rows),
        "generation_count": load_jsonl_count(GENERATED_PROBLEMS_PATH),
        "judge_grounding_count": load_jsonl_count(GROUNDING_LOG_PATH),
        "judge_keyedness_count": load_jsonl_count(KEYEDNESS_LOG_PATH),
        "judge_distractorfit_count": load_jsonl_count(DISTRACTORFIT_LOG_PATH),
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
            "judge_keyedness_log": str(KEYEDNESS_LOG_PATH),
            "judge_distractorfit_log": str(DISTRACTORFIT_LOG_PATH),
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
