import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

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
from settings import (
    ANSWER_LOG_PATH,
    DOC_TYPE_RULES,
    GENERATIONS_PATH,
    GROUNDING_LOG_PATH,
    JUDGE_MAIN_CHECKPOINT_EVERY,
    JUDGE_MAIN_MAX_ATTEMPTS,
    JUDGE_MAIN_MAX_WORKERS,
    JUDGE_MAIN_RETRY_BASE_SECONDS,
    JUDGE_MAIN_SUCCESS_SLEEP_SECONDS,
    JUDGE_READY_SAMPLES_PATH,
    PEDAGOGY_LOG_PATH,
    JUDGE_STRICT_CHECKPOINT_EVERY,
    JUDGE_STRICT_MAX_ATTEMPTS,
    JUDGE_STRICT_MAX_WORKERS,
    JUDGE_STRICT_RETRY_BASE_SECONDS,
    JUDGE_STRICT_SUCCESS_SLEEP_SECONDS,
)


ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding.md",
    "Answer": "judge_answer.md",
    "Pedagogy": "judge_pedagogy.md",
}

ROLE_TO_LOG_PATH = {
    "Grounding": GROUNDING_LOG_PATH,
    "Answer": ANSWER_LOG_PATH,
    "Pedagogy": PEDAGOGY_LOG_PATH,
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
            "answer_mode": sample.get("answer_mode", ""),
            "explanation_target": sample.get("explanation_target", ""),
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
        fact_basis = sample["evidence_card"].get("fact_basis", "")
        rule_basis = sample["evidence_card"].get("rule_basis", "")
        fact_overlap = overlap_ratio(fact_basis, explanation) if fact_basis else 0.0
        rule_overlap = overlap_ratio(rule_basis, explanation) if rule_basis else 0.0

        if sample["doc_type_name"] in ("결정례_QA", "판결문_QA") and fact_basis and fact_overlap < 0.15:
            score = 3
            error_tags.append("적용 약함")
            reason = "핵심 사실관계나 적용 판단이 충분히 드러나지 않습니다."
        elif sample["doc_type_name"] in ("법령_QA", "해석례_QA") and rule_basis and rule_overlap < 0.20:
            score = 3
            error_tags.append("법리 누락")
            reason = "결론은 있으나 기준이 되는 법리 설명이 충분하지 않습니다."
        elif sentence_gap == 0 and len(explanation) <= 700:
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
        "judge_error": response.get("judge_error", ""),
        "judge_attempt_count": response.get("judge_attempt_count", 1),
        "judge_elapsed_seconds": response.get("judge_elapsed_seconds", 0.0),
        "judged_at_utc": utc_now_iso(),
    }


def get_mode_config(mode_name):
    if mode_name == "strict_finalize":
        return {
            "mode_name": mode_name,
            "max_workers": JUDGE_STRICT_MAX_WORKERS,
            "max_attempts": JUDGE_STRICT_MAX_ATTEMPTS,
            "retry_base_seconds": JUDGE_STRICT_RETRY_BASE_SECONDS,
            "success_sleep_seconds": JUDGE_STRICT_SUCCESS_SLEEP_SECONDS,
            "checkpoint_every": JUDGE_STRICT_CHECKPOINT_EVERY,
            "allow_local_fallback": False,
        }

    return {
        "mode_name": mode_name,
        "max_workers": JUDGE_MAIN_MAX_WORKERS,
        "max_attempts": JUDGE_MAIN_MAX_ATTEMPTS,
        "retry_base_seconds": JUDGE_MAIN_RETRY_BASE_SECONDS,
        "success_sleep_seconds": JUDGE_MAIN_SUCCESS_SLEEP_SECONDS,
        "checkpoint_every": JUDGE_MAIN_CHECKPOINT_EVERY,
        "allow_local_fallback": True,
    }


def evaluate_one(sample, generation, role_name, mode_config):
    attempt_count = 0
    while True:
        attempt_count += 1
        started_at = time.monotonic()
        try:
            response = call_gemini_json(
                build_prompt(sample, generation, role_name),
                response_label=f"{generation['candidate_id']}::{role_name}",
            )
            response["judge_mode"] = "gemini_api"
            response["judge_error"] = ""
            response["judge_attempt_count"] = attempt_count
            response["judge_elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            row = build_row(sample, generation, role_name, response)
            # 메인 런은 속도를, strict finalize는 안정성을 우선한다.
            time.sleep(mode_config["success_sleep_seconds"])
            return role_name, row
        except Exception as exc:  # noqa: BLE001
            if mode_config["allow_local_fallback"] and mode_config["max_attempts"] and attempt_count >= mode_config["max_attempts"]:
                response = {
                    "json": build_local_judge_response(sample, generation, role_name),
                    "model": "local_rule_fallback",
                    "judge_mode": "local_rule_fallback",
                    "judge_error": str(exc)[:300],
                    "judge_attempt_count": attempt_count,
                    "judge_elapsed_seconds": round(time.monotonic() - started_at, 3),
                }
                row = build_row(sample, generation, role_name, response)
                return role_name, row

            wait_seconds = min(60, mode_config["retry_base_seconds"] * attempt_count)
            print(
                "[judge retry]",
                f"mode={mode_config['mode_name']}",
                f"sample_id={sample['sample_id']}",
                f"candidate_id={generation['candidate_id']}",
                f"role={role_name}",
                f"attempt={attempt_count}",
                f"wait={wait_seconds}s",
                f"error={str(exc)[:300]}",
                flush=True,
            )
            time.sleep(wait_seconds)


def checkpoint_outputs(outputs):
    write_jsonl_atomic(GROUNDING_LOG_PATH, outputs["Grounding"])
    write_jsonl_atomic(ANSWER_LOG_PATH, outputs["Answer"])
    write_jsonl_atomic(PEDAGOGY_LOG_PATH, outputs["Pedagogy"])


def load_existing_outputs():
    outputs = {role_name: [] for role_name in ROLE_TO_PROMPT}
    for role_name, log_path in ROLE_TO_LOG_PATH.items():
        path = Path(log_path)
        if not path.exists():
            continue
        for row in load_jsonl(path):
            if row.get("judge_mode") == "gemini_api":
                outputs[role_name].append(row)
    return outputs


def main(mode="main"):
    mode_config = get_mode_config(mode)
    snapshot_prompts(["judge_grounding.md", "judge_answer.md", "judge_pedagogy.md"])
    sample_map = {sample["sample_id"]: sample for sample in load_jsonl(JUDGE_READY_SAMPLES_PATH)}
    generations = load_jsonl(GENERATIONS_PATH)
    outputs = load_existing_outputs()

    completed_keys = {
        (row["candidate_id"], role_name)
        for role_name, rows in outputs.items()
        for row in rows
    }

    futures = []
    with ThreadPoolExecutor(max_workers=mode_config["max_workers"]) as executor:
        for generation in generations:
            sample = sample_map[generation["sample_id"]]
            for role_name in outputs:
                key = (generation["candidate_id"], role_name)
                if key in completed_keys:
                    continue
                futures.append(executor.submit(evaluate_one, sample, generation, role_name, mode_config))

        for completed_index, future in enumerate(as_completed(futures), start=1):
            resolved_role_name, row = future.result()
            outputs[resolved_role_name].append(row)
            if completed_index % max(1, mode_config["checkpoint_every"]) == 0:
                checkpoint_outputs(outputs)

    for role_name in outputs:
        outputs[role_name].sort(key=lambda row: (row["sample_id"], row["candidate_id"]))

    checkpoint_outputs(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("main", "strict_finalize"),
        default="main",
        help="Judge 실행 모드",
    )
    args = parser.parse_args()
    main(mode=args.mode)
