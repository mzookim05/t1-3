import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from common_v3 import (
    call_gemini_json,
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
from settings_v3 import (
    ALLOWED_ERROR_TAGS,
    ANSWERABILITY_LOG_PATH,
    GENERATED_PROBLEMS_PATH,
    GROUNDING_LOG_PATH,
    JUDGE_MAIN_CHECKPOINT_EVERY,
    JUDGE_MAIN_MAX_ATTEMPTS,
    JUDGE_MAIN_MAX_WORKERS,
    JUDGE_MAIN_RETRY_BASE_SECONDS,
    JUDGE_MAIN_SUCCESS_SLEEP_SECONDS,
    JUDGE_STRICT_CHECKPOINT_EVERY,
    JUDGE_STRICT_MAX_ATTEMPTS,
    JUDGE_STRICT_MAX_WORKERS,
    JUDGE_STRICT_RETRY_BASE_SECONDS,
    JUDGE_STRICT_SUCCESS_SLEEP_SECONDS,
    SEED_READY_PATH,
    TASKFIT_LOG_PATH,
)


ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding.md",
    "Answerability": "judge_answerability.md",
    "TaskFit": "judge_taskfit.md",
}

ROLE_TO_LOG_PATH = {
    "Grounding": GROUNDING_LOG_PATH,
    "Answerability": ANSWERABILITY_LOG_PATH,
    "TaskFit": TASKFIT_LOG_PATH,
}


def overlap_ratio(base_text, compare_text):
    base_tokens = set(tokenize(base_text))
    compare_tokens = set(tokenize(compare_text))
    if not base_tokens:
        return 0.0
    return len(base_tokens & compare_tokens) / len(base_tokens)


def contains_split_failure(text):
    normalized = normalized_text(text)
    return (
        normalized.count("?") >= 2
        or "아니면" in normalized
        or "와 관련된 핵심 판단 기준은 무엇인가요" in normalized
        or "와 관련된 법원의 판단 기준은 무엇인가요" in normalized
        or "의 판단 기준은 무엇인가요" in normalized
    )


def build_prompt(seed, generation, role_name):
    template = load_prompt(ROLE_TO_PROMPT[role_name])
    return render_prompt(
        template,
        {
            "doc_type_name": seed["doc_type_name"],
            "problem_generation_mode": generation["problem_generation_mode"],
            "generated_problem": generation["generated_problem"],
            "gold_short_answer": generation["gold_short_answer"],
            "gold_reference_explanation": generation["gold_reference_explanation"],
            "source_problem": seed["transformed_problem"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
        },
    )


def build_local_judge_response(seed, generation, role_name):
    generated_problem = generation["generated_problem"]
    source_text = normalized_text(
        " ".join(
            [
                seed["transformed_problem"],
                seed.get("rule_basis", ""),
                seed.get("fact_basis", ""),
                generation["gold_short_answer"],
            ]
        )
    )
    source_overlap = overlap_ratio(source_text, generated_problem)
    answer_overlap = overlap_ratio(generation["gold_short_answer"], generated_problem)
    error_tags = []

    if role_name == "Grounding":
        if source_overlap >= 0.30:
            score = 5
            reason = "문제 문장이 source와 evidence 범위를 크게 벗어나지 않습니다."
        elif source_overlap >= 0.18:
            score = 4
            reason = "문제 문장이 source에 대체로 닫히지만 일부 표현이 느슨합니다."
        else:
            score = 2
            error_tags.append("원문 외 사실 추가")
            reason = "문제 문장이 source와 evidence에 충분히 닫히지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    elif role_name == "Answerability":
        if answer_overlap >= 0.70:
            score = 2
            error_tags.append("정답 누설")
            reason = "문제 문장에 정답 표현이 과하게 드러납니다."
        elif contains_split_failure(generated_problem):
            score = 2
            error_tags.append("복수 쟁점 혼합")
            reason = "복수 질의 흔적이 남아 단일 정답형으로 보기 어렵습니다."
        elif len(tokenize(generation["gold_short_answer"])) <= 2:
            score = 4
            reason = "짧은 정답으로 답변 가능한 구조입니다."
        else:
            score = 5
            reason = "정답과 문제의 대응이 유지됩니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    else:
        sentence_count = len(split_sentences(generated_problem))
        if sentence_count > 3 or generated_problem.count("?") >= 1:
            score = 3
            error_tags.append("형식 부적합")
            reason = "설명형 문제이지만 문장 형식이 아직 질문형에 가깝습니다."
        elif not generated_problem.endswith(("설명하시오.", "서술하시오.", "밝히시오.")):
            score = 3
            error_tags.append("형식 부적합")
            reason = "설명형 서술형 문제 종결 표현이 약합니다."
        else:
            score = 5
            reason = "설명형 서술형 문제 형식에 잘 맞습니다."
        pass_or_fail = "pass" if score >= 3 else "fail"

    return {
        "score": score,
        "pass_or_fail": pass_or_fail,
        "error_tags": [tag for tag in error_tags if tag in ALLOWED_ERROR_TAGS],
        "one_sentence_reason": reason,
    }


def build_row(seed, generation, role_name, response):
    parsed = response["json"]
    return {
        "seed_sample_id": seed["seed_sample_id"],
        "candidate_id": generation["candidate_id"],
        "role_name": role_name,
        "doc_type_name": generation["doc_type_name"],
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


def evaluate_one(seed, generation, role_name, mode_config):
    attempt_count = 0
    while True:
        attempt_count += 1
        started_at = time.monotonic()
        try:
            response = call_gemini_json(
                build_prompt(seed, generation, role_name),
                response_label=f"{generation['candidate_id']}::{role_name}",
            )
            response["judge_mode"] = "gemini_api"
            response["judge_error"] = ""
            response["judge_attempt_count"] = attempt_count
            response["judge_elapsed_seconds"] = round(time.monotonic() - started_at, 3)
            row = build_row(seed, generation, role_name, response)
            time.sleep(mode_config["success_sleep_seconds"])
            return role_name, row
        except Exception as exc:  # noqa: BLE001
            if mode_config["allow_local_fallback"] and mode_config["max_attempts"] and attempt_count >= mode_config["max_attempts"]:
                response = {
                    "json": build_local_judge_response(seed, generation, role_name),
                    "model": "local_rule_fallback",
                    "judge_mode": "local_rule_fallback",
                    "judge_error": str(exc)[:300],
                    "judge_attempt_count": attempt_count,
                    "judge_elapsed_seconds": round(time.monotonic() - started_at, 3),
                }
                row = build_row(seed, generation, role_name, response)
                return role_name, row

            wait_seconds = min(60, mode_config["retry_base_seconds"] * attempt_count)
            print(
                "[problem judge retry]",
                f"mode={mode_config['mode_name']}",
                f"seed_sample_id={seed['seed_sample_id']}",
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
    write_jsonl_atomic(ANSWERABILITY_LOG_PATH, outputs["Answerability"])
    write_jsonl_atomic(TASKFIT_LOG_PATH, outputs["TaskFit"])


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
    snapshot_prompts(["judge_grounding.md", "judge_answerability.md", "judge_taskfit.md"])
    seed_map = {seed["seed_sample_id"]: seed for seed in load_jsonl(SEED_READY_PATH)}
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    outputs = load_existing_outputs()

    completed_keys = {
        (row["candidate_id"], role_name)
        for role_name, rows in outputs.items()
        for row in rows
    }

    pending_jobs = []
    for generation in generations:
        seed = seed_map[generation["seed_sample_id"]]
        for role_name in ROLE_TO_PROMPT:
            key = (generation["candidate_id"], role_name)
            if key in completed_keys:
                continue
            pending_jobs.append((seed, generation, role_name))

    if not pending_jobs:
        checkpoint_outputs(outputs)
        return outputs

    with ThreadPoolExecutor(max_workers=mode_config["max_workers"]) as executor:
        future_map = {
            executor.submit(evaluate_one, seed, generation, role_name, mode_config): (seed, generation, role_name)
            for seed, generation, role_name in pending_jobs
        }
        completed_since_checkpoint = 0
        for future in as_completed(future_map):
            role_name, row = future.result()
            outputs[role_name].append(row)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= mode_config["checkpoint_every"]:
                checkpoint_outputs(outputs)
                completed_since_checkpoint = 0

    checkpoint_outputs(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "strict_finalize"), default="main")
    args = parser.parse_args()
    main(mode=args.mode)
