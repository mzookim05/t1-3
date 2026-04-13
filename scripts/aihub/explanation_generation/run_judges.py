import time

from common import (
    call_gemini_json,
    load_jsonl,
    load_prompt,
    render_prompt,
    split_sentences,
    snapshot_prompts,
    tokenize,
    utc_now_iso,
    write_jsonl_atomic,
)
from settings import ANSWER_LOG_PATH, DOC_TYPE_RULES, GROUNDING_LOG_PATH, JUDGE_READY_SAMPLES_PATH, PEDAGOGY_LOG_PATH, GENERATIONS_PATH


ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding.md",
    "Answer": "judge_answer.md",
    "Pedagogy": "judge_pedagogy.md",
}


def build_evidence_text(evidence_sentences):
    return "\n".join(
        f"- [{sentence['sentence_id']}] {sentence['section']}: {sentence['text']}"
        for sentence in evidence_sentences
    )


def build_prompt(sample, generation, role_name):
    template = load_prompt(ROLE_TO_PROMPT[role_name])
    return render_prompt(
        template,
        {
            "doc_type_name": sample["doc_type_name"],
            "transformed_problem": sample["transformed_problem"],
            "short_answer": sample["short_answer"],
            "long_answer": sample["long_answer"],
            "generated_explanation": generation["generated_explanation"],
            "evidence_bullets": build_evidence_text(sample["evidence_card"]["evidence_sentences"]),
        },
    )


def overlap_ratio(base_text, compare_text):
    base_tokens = set(tokenize(base_text))
    compare_tokens = set(tokenize(compare_text))
    if not base_tokens:
        return 0.0
    return len(base_tokens & compare_tokens) / len(base_tokens)


def build_local_judge_response(sample, generation, role_name):
    explanation = generation["generated_explanation"]
    explanation_sentences = split_sentences(explanation)
    evidence_text = " ".join(
        sentence["text"] for sentence in sample["evidence_card"]["evidence_sentences"]
    )
    evidence_overlap = overlap_ratio(evidence_text, explanation)
    answer_overlap = overlap_ratio(sample["short_answer"], explanation)
    last_sentence = explanation_sentences[-1] if explanation_sentences else ""
    last_sentence_overlap = overlap_ratio(sample["short_answer"], last_sentence)
    error_tags = []

    if role_name == "Grounding":
        if evidence_overlap >= 0.55:
            score = 5
            reason = "핵심 설명이 evidence 문장과 전반적으로 정렬됩니다."
        elif evidence_overlap >= 0.30:
            score = 4
            reason = "대부분의 설명이 evidence 문장과 연결되지만 일부는 느슨합니다."
        else:
            score = 2
            error_tags.append("근거 누락")
            reason = "해설이 evidence 문장과 충분히 겹치지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    elif role_name == "Answer":
        if last_sentence_overlap >= 0.75 or answer_overlap >= 0.65:
            score = 5
            reason = "해설의 결론이 정답과 잘 맞습니다."
        elif last_sentence_overlap >= 0.45 or answer_overlap >= 0.40:
            score = 4
            reason = "결론 방향은 같지만 표현이 완전히 맞지는 않습니다."
        else:
            score = 2
            error_tags.append("결론 불일치")
            reason = "해설의 결론이 정답과 충분히 맞지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    else:
        target_sentence_count = DOC_TYPE_RULES[sample["doc_type_name"]]["target_sentences"]
        sentence_gap = abs(len(explanation_sentences) - target_sentence_count)
        if sentence_gap == 0 and len(explanation) <= 700:
            score = 5
            reason = "문장 수와 길이가 학습용 설명 형식에 잘 맞습니다."
        elif sentence_gap <= 1:
            score = 4
            reason = "설명 형식은 대체로 맞지만 문장 수나 길이가 약간 흔들립니다."
        elif sentence_gap <= 2:
            score = 3
            error_tags.append("문체 장황")
            reason = "설명 구조는 유지하지만 길이 조정이 더 필요합니다."
        else:
            score = 2
            error_tags.append("문체 장황")
            reason = "설명 구조가 목표 형식에서 많이 벗어납니다."
        pass_or_fail = "pass" if score >= 3 else "fail"

    return {
        "score": score,
        "pass_or_fail": pass_or_fail,
        "error_tags": error_tags,
        "one_sentence_reason": reason,
    }


def build_row(sample, generation, role_name, response):
    parsed = response["json"]
    return {
        "sample_id": sample["sample_id"],
        "candidate_id": generation["candidate_id"],
        "ablation_variant": generation.get("ablation_variant", ""),
        "role_name": role_name,
        "doc_type_name": sample["doc_type_name"],
        "score": int(parsed["score"]),
        "pass_or_fail": parsed["pass_or_fail"],
        "error_tags": parsed.get("error_tags", []),
        "one_sentence_reason": parsed["one_sentence_reason"],
        "judge_model": response["model"],
        "judge_mode": response.get("judge_mode", "gemini_api"),
        "judged_at_utc": utc_now_iso(),
    }


def evaluate_one(sample, generation, role_name):
    try:
        response = call_gemini_json(
            build_prompt(sample, generation, role_name),
            response_label=f"{generation['candidate_id']}::{role_name}",
        )
        response["judge_mode"] = "gemini_api"
        # `v4`는 Judge 백본을 하나로 통일했고, 추가적인 429 대응은 공통 호출부에서
        # 재시도로 처리하므로 요청 간 간격은 과도하게 길지 않게 둔다.
        time.sleep(1.2)
    except RuntimeError:
        response = {
            "model": "local_rule_fallback",
            "response_label": f"{generation['candidate_id']}::{role_name}",
            "json": build_local_judge_response(sample, generation, role_name),
            "judge_mode": "local_rule_fallback",
        }
    return role_name, build_row(sample, generation, role_name, response)


def main():
    snapshot_prompts(["judge_grounding.md", "judge_answer.md", "judge_pedagogy.md"])
    sample_map = {sample["sample_id"]: sample for sample in load_jsonl(JUDGE_READY_SAMPLES_PATH)}
    generations = load_jsonl(GENERATIONS_PATH)

    outputs = {
        "Grounding": [],
        "Answer": [],
        "Pedagogy": [],
    }

    for generation in generations:
        sample = sample_map[generation["sample_id"]]
        for role_name in outputs:
            resolved_role_name, row = evaluate_one(sample, generation, role_name)
            outputs[resolved_role_name].append(row)

    for role_name in outputs:
        outputs[role_name].sort(key=lambda row: (row["sample_id"], row["candidate_id"]))

    write_jsonl_atomic(GROUNDING_LOG_PATH, outputs["Grounding"])
    write_jsonl_atomic(ANSWER_LOG_PATH, outputs["Answer"])
    write_jsonl_atomic(PEDAGOGY_LOG_PATH, outputs["Pedagogy"])
    return outputs


if __name__ == "__main__":
    main()
