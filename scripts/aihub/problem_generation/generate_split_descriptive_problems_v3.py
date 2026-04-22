import argparse
import re

from common_v3 import (
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
from settings_v3 import (
    DOC_TYPE_PROMPT_HINTS,
    GENERATED_PROBLEMS_PATH,
    GENERATOR_MAIN_CHECKPOINT_EVERY,
    GENERATOR_STRICT_CHECKPOINT_EVERY,
    SEED_READY_PATH,
)


QUESTION_ENDINGS = {
    "split_single_issue_rule": "핵심 기준을 설명하시오.",
    "split_single_issue_application": "법적 판단 이유를 설명하시오.",
    "split_single_issue_requirement": "성립 요건을 설명하시오.",
    "split_single_issue_scope": "적용 범위를 설명하시오.",
}


PROBLEMATIC_TAIL_PATTERNS = (
    "와 관련된 핵심 판단 기준은 무엇인가요",
    "와 관련된 법원의 판단 기준은 무엇인가요",
    "의 판단 기준은 무엇인가요",
)


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
    # `v3` fallback은 meta-tail을 제거한 뒤 단일 쟁점 설명형으로 강제 전환한다.
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

    # split-type `v3`는 meta-tail이 남거나 대안 질문이 남는 순간 바로 fallback으로 되돌린다.
    if contains_split_failure(cleaned):
        cleaned = build_local_fallback_problem(seed)

    # 문제 문장에 정답이 과하게 노출되면 fallback으로 되돌려 정답 누설을 줄인다.
    if overlap_ratio(seed["short_answer"], cleaned) >= 0.72:
        cleaned = build_local_fallback_problem(seed)

    return cleaned


def build_messages(seed):
    system_prompt = load_prompt("generator_system_v3.txt")
    user_template = load_prompt("generator_user_template_v3.md")
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
    candidate_id = f"{seed['seed_sample_id']}::split_descriptive_v3"
    while True:
        try:
            response = call_openai_json(build_messages(seed), response_label=candidate_id)
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
            "problem_task_type": "descriptive_qa_split",
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


def main(mode="main"):
    snapshot_prompts(["generator_system_v3.txt", "generator_user_template_v3.md"])
    strict_mode = mode == "strict_finalize"
    seeds = load_jsonl(SEED_READY_PATH)
    rows = load_existing_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for seed in seeds:
        candidate_id = f"{seed['seed_sample_id']}::split_descriptive_v3"
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
