from common import count_words, load_jsonl, pick_long_answer, pick_short_answer, normalized_text, write_jsonl_atomic
from settings import DOC_TYPE_RULES, EVIDENCE_CARDS_PATH, JUDGE_READY_SAMPLES_PATH, TRANSFORMED_SAMPLES_PATH


LAW_MODE_RULES = {
    "definition": {
        "transform_type": "definition_reframe",
        "template_name": "의미 -> 핵심 구성요소 -> 결론",
    },
    "requirement": {
        "transform_type": "requirement_reframe",
        "template_name": "요건 -> 조문 근거 -> 결론",
    },
    "criteria": {
        "transform_type": "apply_rule",
        "template_name": "판단 기준 -> 조문 적용 -> 결론",
    },
}


def classify_law_question(original_input):
    normalized = normalized_text(original_input)
    if any(keyword in normalized for keyword in ("무엇", "의미", "뜻", "정의", "란 무엇")):
        return "definition"
    if any(keyword in normalized for keyword in ("요건", "성립", "조건", "갖추어야", "요소")):
        return "requirement"
    return "criteria"


def build_transformed_problem(card):
    title = card["title"]
    original_input = card["original_input"].rstrip(" ?")
    doc_type_name = card["doc_type_name"]
    law_question_mode = card.get("law_question_mode", "criteria")

    if doc_type_name == "법령_QA":
        # `v2`의 일괄적인 "판단 기준" 문구가 정의형 질문까지 설명형으로 밀어
        # Answer fail을 만들었기 때문에, `v3`에서는 정의형/요건형을 보수적으로 유지한다.
        if law_question_mode in ("definition", "requirement"):
            return f"{title} 기준으로, {original_input}"
        return f"{title} 기준으로 보면, {original_input}에 대한 판단 기준은 무엇인가요?"
    if doc_type_name == "해석례_QA":
        return f"{title} 해석례 기준으로, {original_input}의 판단 기준은 무엇인가요?"
    if doc_type_name == "결정례_QA":
        return f"{title} 결정례에서 {original_input}와 관련된 핵심 판단 기준은 무엇인가요?"
    return f"{title} 판결에서 {original_input}와 관련된 법원의 판단 기준은 무엇인가요?"


def build_transformed_sample(card):
    short_answer = pick_short_answer(card["label_output"])
    long_answer = pick_long_answer(card["label_output"])
    rule = DOC_TYPE_RULES[card["doc_type_name"]]
    law_question_mode = classify_law_question(card["original_input"]) if card["doc_type_name"] == "법령_QA" else ""
    law_rule = LAW_MODE_RULES.get(law_question_mode, {})
    transform_type = law_rule.get("transform_type", rule["transform_type"])
    template_name = law_rule.get("template_name", rule["template_name"])

    enriched_card = {
        **card,
        "law_question_mode": law_question_mode,
    }

    return {
        **enriched_card,
        "transform_type": transform_type,
        "transformed_problem": build_transformed_problem(enriched_card),
        "short_answer": short_answer,
        "long_answer": long_answer,
        "long_answer_word_count": count_words(long_answer),
        "candidate_styles": list(rule["styles"]),
        "generator_template_name": template_name,
    }


def main():
    cards = load_jsonl(EVIDENCE_CARDS_PATH)
    transformed = [build_transformed_sample(card) for card in cards]
    write_jsonl_atomic(TRANSFORMED_SAMPLES_PATH, transformed)
    write_jsonl_atomic(JUDGE_READY_SAMPLES_PATH, transformed)
    return transformed


if __name__ == "__main__":
    main()
