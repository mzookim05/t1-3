import argparse
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
    SEED_READY_PATH,
)


QUESTION_ENDINGS = {
    "standard_reframe": "관련 판단 기준을 설명하시오.",
    "issue_application": "해당 사안에서의 법적 판단과 이유를 설명하시오.",
    "requirement_check": "성립 요건 또는 충족 요건을 서술하시오.",
    "scope_boundary": "적용 범위 또는 해당 여부를 설명하시오.",
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
        (r"의 판단 기준은 무엇인가요$", ""),
        (r"의 조건은 무엇인가요$", ""),
        (r"의 요건은 무엇인가요$", ""),
        (r"의 범위는 무엇인가요$", ""),
        (r"어떻게 되나요$", ""),
        (r"무엇인가요$", ""),
        (r"무엇입니까$", ""),
    ]
    for pattern, replacement in replacements:
        stem = re.sub(pattern, replacement, stem)
    return stem.rstrip(" ,")


def build_local_fallback_problem(seed):
    stem = compress_question_stem(seed["transformed_problem"])
    ending = QUESTION_ENDINGS.get(seed["problem_generation_mode"], "관련 쟁점을 설명하시오.")
    if seed["doc_type_name"] == "법령_QA":
        prefix = "다음 법령 상황에 관하여, "
    elif seed["doc_type_name"] == "해석례_QA":
        prefix = "다음 해석례 상황에 관하여, "
    elif seed["doc_type_name"] == "결정례_QA":
        prefix = "다음 결정례 상황에 관하여, "
    else:
        prefix = "다음 판결문 상황에 관하여, "
    return normalized_text(f"{prefix}{stem} {ending}")


def postprocess_problem(seed, generated_problem):
    cleaned = normalized_text(generated_problem)
    if not cleaned.endswith(("설명하시오.", "서술하시오.", "밝히시오.")):
        cleaned = cleaned.rstrip(".?") + " 설명하시오."

    # 문제 문장에 정답이 과하게 노출되면 fallback으로 되돌려 정답 누설을 줄인다.
    if overlap_ratio(seed["short_answer"], cleaned) >= 0.72:
        cleaned = build_local_fallback_problem(seed)

    return cleaned


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


def generate_one(seed, strict_mode):
    candidate_id = f"{seed['seed_sample_id']}::descriptive_v1"
    while True:
        try:
            response = call_openai_json(build_messages(seed), response_label=candidate_id)
            generated_problem = response["json"]["generated_problem"].strip()
            generator_model = response["model"]
            generation_mode = "openai_api"
        except RuntimeError as exc:
            if strict_mode:
                continue
            generated_problem = build_local_fallback_problem(seed)
            generator_model = "local_template_fallback"
            generation_mode = f"fallback:{str(exc)[:160]}"

        generated_problem = postprocess_problem(seed, generated_problem)
        return {
            "seed_sample_id": seed["seed_sample_id"],
            "reference_sample_id": seed["reference_sample_id"],
            "candidate_id": candidate_id,
            "problem_task_type": "descriptive_qa",
            "problem_generation_mode": seed["problem_generation_mode"],
            "doc_type_name": seed["doc_type_name"],
            "source_subset": seed["source_subset"],
            "sampling_lane": seed["sampling_lane"],
            "family_id": seed["family_id"],
            "generated_problem": generated_problem,
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


def main(mode="main"):
    snapshot_prompts(["generator_system.txt", "generator_user_template.md"])
    strict_mode = mode == "strict_finalize"
    seeds = load_jsonl(SEED_READY_PATH)
    rows = load_existing_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for seed in seeds:
        candidate_id = f"{seed['seed_sample_id']}::descriptive_v1"
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
