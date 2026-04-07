import re

from common import (
    classify_law_question,
    count_words,
    extract_law_subject,
    load_jsonl,
    normalized_text,
    pick_long_answer,
    pick_short_answer,
    split_sentences,
    strip_statute_lead,
    write_jsonl_atomic,
)
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
    "scope": {
        "transform_type": "scope_reframe",
        "template_name": "범위 -> 조문 근거 -> 결론",
    },
    "criteria": {
        "transform_type": "apply_rule",
        "template_name": "판단 기준 -> 조문 적용 -> 결론",
    },
}


def ensure_question(text):
    cleaned = normalized_text(text).rstrip("?")
    return f"{cleaned}?"


def clean_law_item(text):
    cleaned = normalized_text(text)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    return cleaned


def extract_law_item_texts(card):
    item_texts = []
    for sentence in card["evidence_card"]["evidence_sentences"]:
        text = clean_law_item(sentence["text"])
        if not text:
            continue
        if sentence["section"] in ("호", "목") or re.match(r"^\d+\.\s*", sentence["text"].strip()):
            item_texts.append(text)
    return item_texts


def to_formal_clause(text, final=False):
    clause = normalized_text(text).rstrip(".")
    replacements = [
        ("초과하지 아니할 것", "초과하지 않아야 합니다" if final else "초과하지 않아야 하고"),
        ("하지 아니할 것", "하지 않아야 합니다" if final else "하지 않아야 하고"),
        ("아니할 것", "않아야 합니다" if final else "않아야 하고"),
        ("보유할 예정일 것", "보유할 예정이어야 합니다" if final else "보유할 예정이어야 하고"),
        ("일 것", "이어야 합니다" if final else "이어야 하고"),
        ("할 것", "해야 합니다" if final else "해야 하고"),
    ]
    for source, target in replacements:
        if clause.endswith(source):
            return clause[: -len(source)] + target
    if final:
        return clause if clause.endswith("합니다") else f"{clause}입니다"
    return clause


def extract_definition_body(text):
    body = strip_statute_lead(text)
    body = re.sub(r"\s*<개정[^>]+>", "", body)
    if "란 " in body:
        body = body.split("란 ", 1)[1]
    return normalized_text(body)


def build_law_short_answer(card, law_question_mode):
    evidence_sentences = card["evidence_card"]["evidence_sentences"]
    if not evidence_sentences:
        return pick_short_answer(card["label_output"])

    subject = extract_law_subject(card["original_input"])
    main_text = evidence_sentences[0]["text"]
    item_texts = extract_law_item_texts(card)

    if law_question_mode == "definition":
        body = extract_definition_body(main_text)
        split_token = "로서 계약체결 당시 다음 각 호의 요건을 충족하는 거래를 말한다."
        if split_token in body and item_texts:
            base = body.split(split_token, 1)[0]
            normalized_items = [
                to_formal_clause(item, final=(index == len(item_texts) - 1))
                for index, item in enumerate(item_texts)
            ]
            condition_text = normalized_items[0] if len(normalized_items) == 1 else " ".join(normalized_items)
            return f"{subject}란 {base}로서, 계약체결 당시 {condition_text}."
        if body:
            body = re.sub(r"말한다\.$", "말합니다.", body)
            body = re.sub(r"말한다$", "말합니다.", body)
            if subject and not body.startswith(subject):
                return f"{subject}란 {body}"
            return body

    if law_question_mode == "requirement":
        body = extract_definition_body(main_text)
        split_token = "로서 계약체결 당시 다음 각 호의 요건을 충족하는 거래를 말한다."
        if split_token in body:
            base = body.split(split_token, 1)[0]
            if item_texts:
                normalized_items = [
                    to_formal_clause(item, final=(index == len(item_texts) - 1))
                    for index, item in enumerate(item_texts)
                ]
                if len(normalized_items) == 1:
                    condition_text = normalized_items[0]
                else:
                    condition_text = " ".join(normalized_items)
                return f"{subject}란 {base}로서, 계약체결 당시 {condition_text}."

    if law_question_mode == "scope" and item_texts:
        joined_items = " 또는 ".join(item_texts)
        return f"{subject}의 범위는 {joined_items}입니다."

    fallback = pick_short_answer(card["label_output"])
    return re.sub(r"\.(\d+\.)", r". \1", fallback)


def build_law_generation_hint(law_question_mode):
    if law_question_mode == "definition":
        return "정의형 질문이므로 첫 문장에서 정의 대상을 바로 설명하고, 마지막 문장은 short_answer의 핵심 요소를 축약하지 말라. law_required_terms가 있으면 모두 남겨라."
    if law_question_mode == "requirement":
        return "요건형 질문이므로 열거된 각 요건을 빠뜨리지 말고, 마지막 문장은 short_answer와 같은 요건 구성을 유지하라."
    if law_question_mode == "scope":
        return "범위형 질문이므로 해당 대상의 범위를 닫힌 형태로 열거하고, 마지막 문장은 short_answer와 같은 범위로 닫아라."
    return "법령형 질문이므로 조문 근거와 결론을 짧고 닫힌 형태로 정리하라."


def build_law_required_terms(card, short_answer, law_question_mode):
    item_texts = extract_law_item_texts(card)
    required_terms = []

    if law_question_mode == "definition":
        for phrase in (
            "등록ㆍ신고",
            "가상자산",
            "업(業)으로 하는 행위",
            "업으로 하는 행위",
            "다음 각 호의 어느 하나에 해당하는 행위",
        ):
            if phrase in short_answer and phrase not in required_terms:
                required_terms.append(phrase)
    elif law_question_mode in ("requirement", "scope"):
        required_terms.extend(item_texts[:2])

    return required_terms


def build_transformed_problem(card):
    title = card["title"]
    original_input = normalized_text(card["original_input"]).rstrip("?")
    doc_type_name = card["doc_type_name"]
    law_question_mode = card.get("law_question_mode", "criteria")

    if doc_type_name == "법령_QA":
        subject = extract_law_subject(card["original_input"])
        if law_question_mode == "definition":
            return ensure_question(f"{title} 기준으로, {subject}의 의미는 무엇인가요")
        if law_question_mode == "requirement":
            return ensure_question(f"{title} 기준으로, {subject}의 요건은 무엇인가요")
        if law_question_mode == "scope":
            return ensure_question(f"{title} 기준으로, {subject}의 범위는 무엇인가요")
        return ensure_question(f"{title} 기준으로, {subject}의 판단 기준은 무엇인가요")
    if doc_type_name == "해석례_QA":
        return ensure_question(f"{title} 해석례 기준으로, {original_input}의 판단 기준은 무엇인가요")
    if doc_type_name == "결정례_QA":
        return ensure_question(f"{title} 결정례에서 {original_input}와 관련된 핵심 판단 기준은 무엇인가요")
    return ensure_question(f"{title} 판결에서 {original_input}와 관련된 법원의 판단 기준은 무엇인가요")


def build_transformed_sample(card):
    rule = DOC_TYPE_RULES[card["doc_type_name"]]
    law_question_mode = card.get("law_question_mode") or (classify_law_question(card["original_input"]) if card["doc_type_name"] == "법령_QA" else "")
    law_rule = LAW_MODE_RULES.get(law_question_mode, {})
    transform_type = law_rule.get("transform_type", rule["transform_type"])
    template_name = law_rule.get("template_name", rule["template_name"])

    short_answer = (
        build_law_short_answer(card, law_question_mode)
        if card["doc_type_name"] == "법령_QA"
        else pick_short_answer(card["label_output"])
    )
    long_answer = pick_long_answer(card["label_output"])

    enriched_card = {
        **card,
        "law_question_mode": law_question_mode,
        "law_generation_hint": build_law_generation_hint(law_question_mode) if card["doc_type_name"] == "법령_QA" else "",
        "law_required_terms": build_law_required_terms(card, short_answer, law_question_mode) if card["doc_type_name"] == "법령_QA" else [],
        "law_closing_target": short_answer if card["doc_type_name"] == "법령_QA" else "",
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
