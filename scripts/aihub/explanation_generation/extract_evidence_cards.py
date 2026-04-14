import json
import re

from common import (
    classify_law_question,
    is_statute_heading_only,
    lexical_overlap_score,
    load_csv_rows,
    load_raw_rows,
    normalized_text,
    pick_short_answer,
    tokenize,
    write_jsonl_atomic,
)
from settings import EVIDENCE_CARDS_PATH, SAMPLE_REGISTRY_PATH

MAX_EVIDENCE_SENTENCE_COUNTS = {
    "법령_QA": 3,
    "해석례_QA": 3,
    "결정례_QA": 4,
    "판결문_QA": 4,
}


def score_law_row(row, sm_class, query_tokens, answer_tokens):
    text = normalized_text(row["내용"])
    score = lexical_overlap_score(text, query_tokens) + (lexical_overlap_score(text, answer_tokens) * 2)

    if sm_class and sm_class in text:
        score += 3
    if row["구분"] in ("조문", "항"):
        score += 1
    if any(token in text for token in ("보상", "손실", "허가", "대상", "범위", "요건")):
        score += 2
    if is_statute_heading_only(text):
        score -= 4

    return score


def find_law_anchor_index(rows, sm_class, label_input, label_output):
    short_answer = pick_short_answer(label_output)
    query_tokens = tokenize(f"{label_input} {short_answer} {label_output}")
    answer_tokens = tokenize(short_answer)
    scored = [
        (score_law_row(row, sm_class, query_tokens, answer_tokens), index)
        for index, row in enumerate(rows)
    ]
    _, best_index = max(scored, default=(0, 0))
    return best_index


def find_previous_law_heading_index(rows, start_index):
    for index in range(start_index, -1, -1):
        if rows[index]["구분"] == "조문":
            return index
    return None


def find_next_substantive_law_index(rows, start_index):
    for index in range(start_index + 1, min(start_index + 6, len(rows))):
        text = rows[index]["내용"].strip()
        if text:
            return index
    return None


def choose_law_rows(rows, sm_class, law_question_mode, label_input, label_output):
    start_index = find_law_anchor_index(rows, sm_class, label_input, label_output)
    selected_rows = []
    heading_index = find_previous_law_heading_index(rows, start_index)
    if heading_index is not None and heading_index != start_index:
        selected_rows.append(rows[heading_index])
    selected_rows.append(rows[start_index])
    anchor_index = start_index
    main_row = rows[start_index]

    # `01·02`는 `smClass`가 없어 본문 문장이 anchor가 될 수 있다.
    # 이 경우 바로 앞 조문 제목을 함께 담아 evidence가 문맥 없이 잘리지 않게 한다.
    if is_statute_heading_only(main_row["내용"]):
        next_index = find_next_substantive_law_index(rows, start_index)
        if next_index is not None:
            if rows[next_index] not in selected_rows:
                selected_rows.append(rows[next_index])
            anchor_index = next_index

    anchor_text = rows[anchor_index]["내용"]
    include_items = (
        law_question_mode in ("requirement", "scope")
        or "다음 각 호" in anchor_text
        or "다음 각 목" in anchor_text
    )
    if not include_items and law_question_mode == "definition":
        return selected_rows

    max_item_count = 2 if include_items else 1
    short_answer = pick_short_answer(label_output)
    answer_tokens = tokenize(short_answer)
    nearby_rows = []
    for row in rows[anchor_index + 1 : anchor_index + 10]:
        text = row["내용"].strip()
        if not text:
            continue
        if row["구분"] == "조문":
            break
        nearby_rows.append(
            (
                lexical_overlap_score(text, answer_tokens) + (1 if row["구분"] in ("호", "목", "항") else 0),
                row,
            )
        )

    for _, row in sorted(nearby_rows, key=lambda item: item[0], reverse=True):
        if row not in selected_rows:
            selected_rows.append(row)
        if len(selected_rows) >= max_item_count + 2:
            break
    return selected_rows


def choose_interpretation_rows(rows, label_input, label_output):
    query_tokens = tokenize(f"{label_input} {label_output}")
    answer_tokens = tokenize(pick_short_answer(label_output))

    def score_row(row):
        text = normalized_text(row["내용"])
        score = lexical_overlap_score(text, query_tokens) + (lexical_overlap_score(text, answer_tokens) * 2)
        if row["구분"] == "회답":
            score += 4
        elif row["구분"] == "이유":
            score += 2
        if any(marker in text for marker in ("따라서", "타당", "해당", "적용", "가능", "없습니다")):
            score += 1
        return score

    answer_rows = sorted(
        [row for row in rows if row["구분"] == "회답"],
        key=score_row,
        reverse=True,
    )
    reason_rows = sorted(
        [row for row in rows if row["구분"] == "이유"],
        key=score_row,
        reverse=True,
    )

    selected_rows = []
    if answer_rows:
        selected_rows.append(answer_rows[0])
    for row in reason_rows:
        if row not in selected_rows:
            selected_rows.append(row)
        if len(selected_rows) >= 3:
            break
    if selected_rows:
        return sorted(selected_rows, key=lambda row: int(row["문장번호"]))

    scored_rows = sorted(
        rows,
        key=lambda row: (score_row(row), -int(row["문장번호"])),
        reverse=True,
    )
    return sorted(scored_rows[:3], key=lambda row: int(row["문장번호"]))


def choose_decision_rows(rows, label_input, label_output):
    query_tokens = tokenize(f"{label_input} {label_output}")
    scored_rows = sorted(
        rows,
        key=lambda row: (lexical_overlap_score(row["내용"], query_tokens), -int(row["문장번호"])),
        reverse=True,
    )
    top_rows = sorted(scored_rows[:4], key=lambda row: int(row["문장번호"]))
    return top_rows


def choose_precedent_rows(rows, label_input, label_output):
    query_tokens = tokenize(f"{label_input} {label_output}")
    answer_tokens = tokenize(pick_short_answer(label_output))
    scored_rows = []

    def is_rule_candidate(row):
        text = normalized_text(row["내용"])
        section = row["구분"]
        return section in ("판시사항", "판결요지", "참조조문") or "해당법조" in text or bool(re.search(r"제\d+조", text))

    def is_application_candidate(row):
        text = normalized_text(row["내용"])
        section = row["구분"]
        return section in ("이유", "판단", "기초사실", "범죄사실", "전문") or any(
            marker in text for marker in ("이 사건", "피고", "원고", "따라서", "인정", "침해", "지급할 의무")
        )

    def is_short_heading(row):
        text = normalized_text(row["내용"])
        return len(text) <= 12 and bool(re.match(r"^(?:\d+\)|\d+\.|○)", text))

    for index, row in enumerate(rows):
        text = row["내용"].strip()
        section = row["구분"]
        score = lexical_overlap_score(text, query_tokens)
        score += lexical_overlap_score(text, answer_tokens) * 2
        if is_rule_candidate(row):
            score += 2
        if is_application_candidate(row):
            score += 2
        if any(marker in text for marker in ("상표권을 침해", "지연손해금", "지급할 의무", "혼동", "유사")):
            score += 2
        if is_short_heading(row):
            score -= 3
        scored_rows.append((score, index, row))

    scored_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    selected_indices = []

    for predicate in (is_rule_candidate, is_application_candidate):
        candidates = [item for item in scored_rows if predicate(item[2]) and item[0] > 0]
        if candidates:
            candidate_index = candidates[0][1]
            if candidate_index not in selected_indices:
                selected_indices.append(candidate_index)

    for score, index, row in scored_rows:
        if score <= 0:
            continue
        if index not in selected_indices:
            selected_indices.append(index)
        if len(selected_indices) >= 4:
            break

    if len(selected_indices) < 4:
        for _, index, _ in scored_rows:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= 4:
                break

    selected_rows = [rows[index] for index in sorted(selected_indices)]
    return selected_rows


def build_card(record):
    info = json.loads(record["info_json"])
    label_input = record["label_input"]
    label_output = record["label_output"]
    rows = load_raw_rows(record["raw_path"], record["doc_type_name"])
    law_question_mode = classify_law_question(label_input) if record["doc_type_name"] == "법령_QA" else ""

    if record["doc_type_name"] == "법령_QA":
        selected = choose_law_rows(
            rows,
            info.get("smClass", "").strip(),
            law_question_mode,
            label_input,
            label_output,
        )
    elif record["doc_type_name"] == "해석례_QA":
        selected = choose_interpretation_rows(rows, label_input, label_output)
    elif record["doc_type_name"] == "결정례_QA":
        selected = choose_decision_rows(rows, label_input, label_output)
    else:
        selected = choose_precedent_rows(rows, label_input, label_output)

    selected = selected[: MAX_EVIDENCE_SENTENCE_COUNTS[record["doc_type_name"]]]

    evidence_sentences = [
        {
            "sentence_id": int(row["문장번호"]),
            "section": row["구분"],
            "text": row["내용"].strip(),
        }
        for row in selected
    ]
    conclusion = pick_short_answer(label_output)
    issue = label_input.rstrip(" ?")
    substantive_sentences = [
        sentence for sentence in evidence_sentences if sentence["text"] and not is_statute_heading_only(sentence["text"])
    ]
    rule_basis = substantive_sentences[0]["text"] if substantive_sentences else (evidence_sentences[0]["text"] if evidence_sentences else "")
    fact_basis = " ".join(sentence["text"] for sentence in substantive_sentences[1:]).strip()

    return {
        "sample_id": record["sample_id"],
        "sample_order": int(record["sample_order"]),
        "source_subset": record["source_subset"],
        "domain": record["domain"],
        "doc_type_name": record["doc_type_name"],
        "sampling_lane": record.get("sampling_lane", "generalization_03_04"),
        "family_id": record["family_id"],
        "title": record["title"],
        "label_path": record["label_path"],
        "raw_path": record["raw_path"],
        "info": info,
        "original_input": label_input,
        "label_output": label_output,
        "law_question_mode": law_question_mode,
        "evidence_card": {
            "issue": issue,
            "conclusion": conclusion,
            "rule_basis": rule_basis,
            "fact_basis": fact_basis,
            "evidence_sentence_ids": [sentence["sentence_id"] for sentence in evidence_sentences],
            "evidence_sentences": evidence_sentences,
            "evidence_sentence_count": len(evidence_sentences),
            "evidence_policy_name": "v6_doc_type_specific_limits",
        },
    }


def main():
    registry_rows = load_csv_rows(SAMPLE_REGISTRY_PATH)
    cards = [build_card(row) for row in registry_rows]
    write_jsonl_atomic(EVIDENCE_CARDS_PATH, cards)
    return cards


if __name__ == "__main__":
    main()
