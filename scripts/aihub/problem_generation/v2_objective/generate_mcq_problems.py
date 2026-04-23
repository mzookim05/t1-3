import argparse
import hashlib
import re

from common import (
    call_openai_json,
    load_jsonl,
    load_prompt,
    normalized_text,
    render_prompt,
    snapshot_prompts,
    tokenize,
    utc_now_iso,
    write_jsonl_atomic,
)
from settings import (
    DOC_TYPE_PROMPT_HINTS,
    GENERATED_PROBLEMS_PATH,
    GENERATOR_MAIN_CHECKPOINT_EVERY,
    GENERATOR_STRICT_CHECKPOINT_EVERY,
    PROBLEM_TASK_TYPE,
    SEED_READY_PATH,
)


CHOICE_LABELS = ("A", "B", "C", "D")


MODE_TO_STEM_ENDING = {
    "single_best_rule": "옳은 설명을 고르시오.",
    "single_best_application": "가장 적절한 설명을 고르시오.",
    "single_best_scope": "올바른 적용 범위를 고르시오.",
}


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
        raise RuntimeError("객관식 fallback 선택지 수와 오답 후보 수가 일치하지 않습니다.")
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


def build_messages(seed):
    system_prompt = load_prompt("generator_system.txt")
    user_template = load_prompt("generator_user_template.md")
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
            "problem_v1_generated_problem": seed.get("problem_v1_generated_problem", ""),
        },
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_existing_rows():
    if not GENERATED_PROBLEMS_PATH.exists():
        return []
    return [
        row
        for row in load_jsonl(GENERATED_PROBLEMS_PATH)
        if row.get("generation_mode") == "openai_api"
    ]


def checkpoint_outputs(rows, strict_mode):
    checkpoint_every = GENERATOR_STRICT_CHECKPOINT_EVERY if strict_mode else GENERATOR_MAIN_CHECKPOINT_EVERY
    if rows and len(rows) % checkpoint_every == 0:
        write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)


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


def generate_one(seed, strict_mode):
    candidate_id = f"{seed['seed_sample_id']}::objective_v2"
    while True:
        try:
            response = call_openai_json(build_messages(seed), response_label=candidate_id)
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
            "problem_v1_status": seed.get("problem_v1_status", ""),
            "problem_v1_generated_problem": seed.get("problem_v1_generated_problem", ""),
            "label_path": seed.get("label_path", ""),
            "raw_path": seed.get("raw_path", ""),
            "generation_model": generator_model,
            "generation_mode": generation_mode,
            "generated_at_utc": utc_now_iso(),
        }


def main(mode="main"):
    snapshot_prompts(["generator_system.txt", "generator_user_template.md"])
    strict_mode = mode == "strict_finalize"
    seeds = load_jsonl(SEED_READY_PATH)
    rows = load_existing_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for seed in seeds:
        candidate_id = f"{seed['seed_sample_id']}::objective_v2"
        if candidate_id in completed_candidate_ids:
            continue
        rows.append(generate_one(seed, strict_mode=strict_mode))
        completed_candidate_ids.add(candidate_id)
        checkpoint_outputs(rows, strict_mode)

    write_jsonl_atomic(GENERATED_PROBLEMS_PATH, rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "strict_finalize"), default="main")
    args = parser.parse_args()
    main(mode=args.mode)
