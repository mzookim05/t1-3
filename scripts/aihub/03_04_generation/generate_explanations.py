from common import (
    call_openai_json,
    load_jsonl,
    load_prompt,
    render_prompt,
    snapshot_prompts,
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
    return "- `long_answer`: 이번 `v3` ablation에서는 입력에서 제외"


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


def main():
    snapshot_prompts(["generator_system.txt", "generator_user_template.md"])
    samples = load_jsonl(JUDGE_READY_SAMPLES_PATH)
    rows = []

    for sample in samples:
        for generation_variant in GENERATION_INPUT_VARIANTS:
            for style_name in sample["candidate_styles"]:
                candidate_id = f"{sample['sample_id']}::{style_name}::{generation_variant['name']}"
                try:
                    response = call_openai_json(
                        build_messages(sample, style_name, generation_variant),
                        response_label=candidate_id,
                    )
                    generated_explanation = response["json"]["generated_explanation"].strip()
                    generator_model = response["model"]
                    generation_mode = "openai_api"
                except RuntimeError as exc:
                    generated_explanation = build_local_fallback_explanation(sample, style_name, generation_variant)
                    generator_model = "local_template_fallback"
                    generation_mode = f"fallback:{str(exc)[:160]}"
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "candidate_id": candidate_id,
                        "style_name": style_name,
                        "ablation_variant": generation_variant["name"],
                        "ablation_label": generation_variant["label"],
                        "include_long_answer": generation_variant["include_long_answer"],
                        "doc_type_name": sample["doc_type_name"],
                        "family_id": sample["family_id"],
                        "source_subset": sample["source_subset"],
                        "transformed_problem": sample["transformed_problem"],
                        "short_answer": sample["short_answer"],
                        "long_answer": sample["long_answer"],
                        "generated_explanation": generated_explanation,
                        "generator_model": generator_model,
                        "generation_mode": generation_mode,
                        "generated_at_utc": utc_now_iso(),
                    }
                )

    write_jsonl_atomic(GENERATIONS_PATH, rows)
    return rows


if __name__ == "__main__":
    main()
