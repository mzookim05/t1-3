import argparse
import re

from common import (
    call_openai_json,
    load_jsonl,
    load_prompt,
    normalized_text,
    render_prompt,
    snapshot_prompts,
    split_sentences,
    tokenize,
    utc_now_iso,
    write_jsonl_atomic,
)
from settings import DOC_TYPE_RULES, GENERATIONS_PATH, GENERATION_INPUT_VARIANTS, JUDGE_READY_SAMPLES_PATH


def style_label(style_name):
    if style_name == "legal_priority":
        return "법리 우선형"
    if style_name == "fact_priority":
        return "사실 우선형"
    return "단일"


def build_evidence_text(evidence_sentences):
    lines = []
    for sentence in evidence_sentences:
        lines.append(
            f"- [{sentence['sentence_id']}] {sentence['section']}: {sentence['text']}"
        )
    return "\n".join(lines)


def build_long_answer_block(sample, generation_variant):
    if generation_variant["include_long_answer"]:
        return f"- `long_answer`: `{sample['long_answer']}`"
    return "- `long_answer`: 이번 ablation에서는 입력에서 제외"


def build_law_required_terms_text(sample):
    required_terms = sample.get("law_required_terms", [])
    return ", ".join(required_terms) if required_terms else "없음"


def overlap_ratio(base_text, compare_text):
    base_tokens = set(tokenize(base_text))
    compare_tokens = set(tokenize(compare_text))
    if not base_tokens:
        return 0.0
    return len(base_tokens & compare_tokens) / len(base_tokens)


def enforce_law_conclusion(sample, generated_explanation):
    cleaned_explanation = normalized_text(generated_explanation)
    if sample["doc_type_name"] != "법령_QA":
        return cleaned_explanation

    sentences = [normalized_text(sentence) for sentence in split_sentences(cleaned_explanation) if normalized_text(sentence)]
    closing_target = normalized_text(sample.get("law_closing_target") or sample["short_answer"])
    if not sentences:
        return closing_target

    last_sentence = sentences[-1]
    required_terms = sample.get("law_required_terms", [])
    missing_terms = [term for term in required_terms if term and term not in last_sentence]
    closing_overlap = overlap_ratio(closing_target, last_sentence)
    law_question_mode = sample.get("law_question_mode", "")

    should_replace = False
    if law_question_mode == "definition":
        should_replace = bool(missing_terms) or closing_overlap < 0.75
    elif law_question_mode in ("requirement", "scope"):
        should_replace = closing_overlap < 0.65
    else:
        should_replace = closing_overlap < 0.60

    # `v5`에서는 법령형 설명의 마지막 결론문을 short_answer 기준으로 고정해
    # 정의형 핵심 요소가 마지막 문장에서 다시 축약되는 문제를 줄인다.
    if should_replace:
        sentences[-1] = closing_target

    return " ".join(sentence for sentence in sentences if sentence)


def relax_unsupported_final_citation(sample, generated_explanation):
    cleaned_explanation = normalized_text(generated_explanation)
    if sample["doc_type_name"] not in ("해석례_QA", "결정례_QA", "판결문_QA"):
        return cleaned_explanation

    sentences = [normalized_text(sentence) for sentence in split_sentences(cleaned_explanation) if normalized_text(sentence)]
    if not sentences:
        return cleaned_explanation

    evidence_text = normalized_text(
        " ".join(sentence["text"] for sentence in sample["evidence_card"]["evidence_sentences"])
    )
    last_sentence = sentences[-1]
    citation_match = re.match(
        r"^((?:구\s*)?[「」A-Za-z가-힣·ㆍ\s]+제\d+조(?:제\d+항)?(?:제\d+호)?)(?:에 따르면|에 따라),?\s*",
        last_sentence,
    )
    if citation_match and normalized_text(citation_match.group(1)) not in evidence_text:
        sentences[-1] = last_sentence[citation_match.end() :].strip()

    return " ".join(sentence for sentence in sentences if sentence)


def postprocess_generated_explanation(sample, generated_explanation):
    processed = enforce_law_conclusion(sample, generated_explanation)
    processed = relax_unsupported_final_citation(sample, processed)
    return processed


def build_messages(sample, style_name, generation_variant):
    system_prompt = load_prompt("generator_system.txt")
    user_template = load_prompt("generator_user_template.md")
    rule = DOC_TYPE_RULES[sample["doc_type_name"]]
    user_prompt = render_prompt(
        user_template,
        {
            "doc_type_name": sample["doc_type_name"],
            "style_name": style_label(style_name),
            "transformed_problem": sample["transformed_problem"],
            "short_answer": sample["short_answer"],
            "long_answer_block": build_long_answer_block(sample, generation_variant),
            "answer_mode": sample.get("answer_mode", ""),
            "explanation_target": sample.get("explanation_target", ""),
            "law_question_mode": sample.get("law_question_mode", ""),
            "law_generation_hint": sample.get("law_generation_hint", ""),
            "law_closing_target": sample.get("law_closing_target", ""),
            "law_required_terms": build_law_required_terms_text(sample),
            "issue": sample["evidence_card"]["issue"],
            "conclusion": sample["evidence_card"]["conclusion"],
            "rule_basis": sample["evidence_card"]["rule_basis"],
            "fact_basis": sample["evidence_card"]["fact_basis"],
            "evidence_bullets": build_evidence_text(sample["evidence_card"]["evidence_sentences"]),
            "template_name": sample["generator_template_name"],
            "target_sentences": rule["target_sentences"],
            "target_word_range": rule["target_word_range"],
        },
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_local_fallback_explanation(sample, style_name, generation_variant):
    evidence = sample["evidence_card"]
    issue_sentence = f"이 사안의 쟁점은 {evidence['issue']}에 관한 판단 기준을 정리하는 데 있다."
    fallback_support = sample["long_answer"] if generation_variant["include_long_answer"] else sample["short_answer"]
    rule_sentence = evidence["rule_basis"] or fallback_support
    fact_sentence = evidence["fact_basis"] or fallback_support
    conclusion_sentence = sample["short_answer"]

    if sample["doc_type_name"] in ("법령_QA", "해석례_QA"):
        sentences = [issue_sentence, rule_sentence, conclusion_sentence]
    elif style_name == "fact_priority":
        sentences = [issue_sentence, fact_sentence, rule_sentence, conclusion_sentence]
    else:
        sentences = [issue_sentence, rule_sentence, fact_sentence, conclusion_sentence]

    cleaned = [sentence.strip() for sentence in sentences if sentence.strip()]
    return " ".join(cleaned)


def checkpoint_outputs(rows):
    write_jsonl_atomic(GENERATIONS_PATH, rows)


def load_existing_rows():
    if not GENERATIONS_PATH.exists():
        return []
    return [
        row
        for row in load_jsonl(GENERATIONS_PATH)
        if row.get("generation_mode") == "openai_api"
    ]


def generate_one(sample, style_name, generation_variant, strict_mode):
    candidate_id = f"{sample['sample_id']}::{style_name}::{generation_variant['name']}"
    while True:
        try:
            response = call_openai_json(
                build_messages(sample, style_name, generation_variant),
                response_label=candidate_id,
            )
            generated_explanation = response["json"]["generated_explanation"].strip()
            generator_model = response["model"]
            generation_mode = "openai_api"
        except RuntimeError as exc:
            if strict_mode:
                continue
            generated_explanation = build_local_fallback_explanation(sample, style_name, generation_variant)
            generator_model = "local_template_fallback"
            generation_mode = f"fallback:{str(exc)[:160]}"
        generated_explanation = postprocess_generated_explanation(sample, generated_explanation)
        return {
            "sample_id": sample["sample_id"],
            "candidate_id": candidate_id,
            "style_name": style_name,
            "ablation_variant": generation_variant["name"],
            "ablation_label": generation_variant["label"],
            "include_long_answer": generation_variant["include_long_answer"],
            "doc_type_name": sample["doc_type_name"],
            "family_id": sample["family_id"],
            "source_subset": sample["source_subset"],
            "sampling_lane": sample.get("sampling_lane", ""),
            "transformed_problem": sample["transformed_problem"],
            "original_input": sample["original_input"],
            "short_answer": sample["short_answer"],
            "long_answer": sample["long_answer"],
            "answer_mode": sample.get("answer_mode", ""),
            "explanation_target": sample.get("explanation_target", ""),
            "generator_template_name": sample.get("generator_template_name", ""),
            "generated_explanation": generated_explanation,
            "generator_model": generator_model,
            "generation_mode": generation_mode,
            "generated_at_utc": utc_now_iso(),
        }


def main(mode="main"):
    snapshot_prompts(["generator_system.txt", "generator_user_template.md"])
    samples = load_jsonl(JUDGE_READY_SAMPLES_PATH)
    strict_mode = mode == "strict_finalize"
    rows = load_existing_rows()
    completed_candidate_ids = {row["candidate_id"] for row in rows}

    for sample in samples:
        for generation_variant in GENERATION_INPUT_VARIANTS:
            for style_name in sample["candidate_styles"]:
                candidate_id = f"{sample['sample_id']}::{style_name}::{generation_variant['name']}"
                if candidate_id in completed_candidate_ids:
                    continue
                rows.append(generate_one(sample, style_name, generation_variant, strict_mode))
                completed_candidate_ids.add(candidate_id)
                if len(rows) % 4 == 0:
                    checkpoint_outputs(rows)

    checkpoint_outputs(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("main", "strict_finalize"),
        default="main",
        help="생성 실행 모드",
    )
    args = parser.parse_args()
    main(mode=args.mode)
