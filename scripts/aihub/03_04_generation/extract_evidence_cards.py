import json
import re

from common import (
    classify_law_question,
    is_statute_heading_only,
    lexical_overlap_score,
    load_csv_rows,
    load_json,
    pick_short_answer,
    tokenize,
    write_jsonl_atomic,
)
from settings import EVIDENCE_CARDS_PATH, SAMPLE_REGISTRY_PATH


def find_law_start_index(rows, sm_class):
    if sm_class:
        for index, row in enumerate(rows):
            text = row["내용"].strip()
            if text.startswith(sm_class):
                return index
        for index, row in enumerate(rows):
            text = row["내용"].strip()
            if row["구분"] == "조문" and sm_class in text:
                return index
    return 0


def find_next_substantive_law_index(rows, start_index):
    for index in range(start_index + 1, min(start_index + 6, len(rows))):
        text = rows[index]["내용"].strip()
        if text:
            return index
    return None


def choose_law_rows(rows, sm_class, law_question_mode):
    start_index = find_law_start_index(rows, sm_class)
    selected_rows = [rows[start_index]]
    anchor_index = start_index
    main_row = rows[start_index]

    # 조문 제목만 잡히는 경우가 있어, `v5`부터는 바로 뒤의 실질 조문/항을 함께 담는다.
    # `v4_001`처럼 제목만 evidence에 남으면 Grounding이 무너져도 원인을 코드에서 못 잡는다.
    if is_statute_heading_only(main_row["내용"]):
        next_index = find_next_substantive_law_index(rows, start_index)
        if next_index is not None:
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

    max_item_count = 2 if include_items else 0
    item_count = 0
    for row in rows[anchor_index + 1 : anchor_index + 8]:
        text = row["내용"].strip()
        if not text:
            continue
        if row["구분"] in ("호", "목") and re.match(r"^(?:\d+\.|[가-하]\.)\s*", text):
            selected_rows.append(row)
            item_count += 1
            if item_count >= max_item_count:
                break
        elif row["구분"] == "조문" and item_count > 0:
            break
    return selected_rows


def choose_interpretation_rows(rows):
    answer_rows = [row for row in rows if row["구분"] == "회답"][:1]
    reason_rows = [row for row in rows if row["구분"] == "이유"][:3]
    return answer_rows + reason_rows


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

    for index, row in enumerate(rows):
        text = row["내용"].strip()
        section = row["구분"]
        score = lexical_overlap_score(text, query_tokens)
        score += lexical_overlap_score(text, answer_tokens) * 2

        # `판결문_QA`는 실제 raw에서 `판시사항/판결요지`가 비어 있고 `판례내용`만 남는 경우가 많다.
        # 그래서 section heuristic만으로는 Grounding이 약해져, 질문/정답과 가까운 이유 문장을 같이 찾는다.
        if section in ("판시사항", "판결요지"):
            score += 2
        if any(marker in text for marker in ("【이 유】", "【판단】", "판단", "이유", "판시")):
            score += 1
        if any(marker in text for marker in ("【주문】", "결론", "인용", "기각")):
            score += 1

        scored_rows.append((score, index, row))

    scored_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    anchor_indices = [index for score, index, _ in scored_rows if score > 0][:2]
    if not anchor_indices:
        anchor_indices = [index for _, index, _ in scored_rows[:2]]

    selected_indices = []
    for anchor_index in anchor_indices:
        for candidate_index in (anchor_index, anchor_index + 1, anchor_index - 1):
            if 0 <= candidate_index < len(rows) and candidate_index not in selected_indices:
                selected_indices.append(candidate_index)
            if len(selected_indices) >= 4:
                break
        if len(selected_indices) >= 4:
            break

    if len(selected_indices) < 4:
        for _, index, _ in scored_rows:
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= 4:
                break

    selected_rows = [rows[index] for index in selected_indices]
    return sorted(selected_rows, key=lambda row: int(row["문장번호"]))


def build_card(record):
    info = json.loads(record["info_json"])
    label_input = record["label_input"]
    label_output = record["label_output"]
    rows = load_csv_rows(record["raw_path"])
    law_question_mode = classify_law_question(label_input) if record["doc_type_name"] == "법령_QA" else ""

    if record["doc_type_name"] == "법령_QA":
        selected = choose_law_rows(rows, info.get("smClass", "").strip(), law_question_mode)
    elif record["doc_type_name"] == "해석례_QA":
        selected = choose_interpretation_rows(rows)
    elif record["doc_type_name"] == "결정례_QA":
        selected = choose_decision_rows(rows, label_input, label_output)
    else:
        selected = choose_precedent_rows(rows, label_input, label_output)

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
        },
    }


def main():
    registry_rows = load_csv_rows(SAMPLE_REGISTRY_PATH)
    cards = [build_card(row) for row in registry_rows]
    write_jsonl_atomic(EVIDENCE_CARDS_PATH, cards)
    return cards


if __name__ == "__main__":
    main()
