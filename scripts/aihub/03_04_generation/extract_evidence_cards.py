import json

from common import (
    lexical_overlap_score,
    load_csv_rows,
    load_json,
    pick_short_answer,
    tokenize,
    write_jsonl_atomic,
)
from settings import EVIDENCE_CARDS_PATH, SAMPLE_REGISTRY_PATH


def choose_law_rows(rows, sm_class):
    matching_indices = [index for index, row in enumerate(rows) if sm_class and sm_class in row["내용"]]
    if matching_indices:
        start_index = matching_indices[0]
        return rows[start_index : start_index + 3]
    return rows[:3]


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

    if record["doc_type_name"] == "법령_QA":
        selected = choose_law_rows(rows, info.get("smClass", "").strip())
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
    rule_basis = evidence_sentences[0]["text"] if evidence_sentences else ""
    fact_basis = " ".join(sentence["text"] for sentence in evidence_sentences[1:]).strip()

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
