import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from common import (
    call_gemini_json,
    load_jsonl,
    load_prompt,
    normalized_text,
    render_prompt,
    snapshot_prompts,
    tokenize,
    utc_now_iso,
    write_jsonl_atomic,
)
from settings import (
    ALLOWED_ERROR_TAGS,
    DISTRACTORFIT_LOG_PATH,
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
    KEYEDNESS_LOG_PATH,
    SEED_READY_PATH,
)


ROLE_TO_PROMPT = {
    "Grounding": "judge_grounding.md",
    "Keyedness": "judge_keyedness.md",
    "DistractorFit": "judge_distractorfit.md",
}

ROLE_TO_LOG_PATH = {
    "Grounding": GROUNDING_LOG_PATH,
    "Keyedness": KEYEDNESS_LOG_PATH,
    "DistractorFit": DISTRACTORFIT_LOG_PATH,
}


def overlap_ratio(base_text, compare_text):
    base_tokens = set(tokenize(base_text))
    compare_tokens = set(tokenize(compare_text))
    if not base_tokens:
        return 0.0
    return len(base_tokens & compare_tokens) / len(base_tokens)


def choices_text(generation):
    return " ".join(
        [
            generation["choice_a"],
            generation["choice_b"],
            generation["choice_c"],
            generation["choice_d"],
        ]
    )


def build_prompt(seed, generation, role_name):
    template = load_prompt(ROLE_TO_PROMPT[role_name])
    return render_prompt(
        template,
        {
            "doc_type_name": seed["doc_type_name"],
            "problem_generation_mode": generation["problem_generation_mode"],
            "generated_stem": generation["generated_stem"],
            "choice_a": generation["choice_a"],
            "choice_b": generation["choice_b"],
            "choice_c": generation["choice_c"],
            "choice_d": generation["choice_d"],
            "correct_choice": generation["correct_choice"],
            "gold_short_answer": generation["gold_short_answer"],
            "gold_reference_explanation": generation["gold_reference_explanation"],
            "source_problem": seed["transformed_problem"],
            "rule_basis": seed.get("rule_basis", ""),
            "fact_basis": seed.get("fact_basis", ""),
        },
    )


def build_local_judge_response(seed, generation, role_name):
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
    option_map = {
        "A": generation["choice_a"],
        "B": generation["choice_b"],
        "C": generation["choice_c"],
        "D": generation["choice_d"],
    }
    correct_choice = generation["correct_choice"]
    correct_text = option_map.get(correct_choice, "")
    all_choices = list(option_map.values())
    error_tags = []

    if role_name == "Grounding":
        source_overlap = overlap_ratio(source_text, normalized_text(generation["generated_stem"] + " " + choices_text(generation)))
        if source_overlap >= 0.30:
            score = 5
            reason = "문제와 선택지가 source와 evidence 범위를 크게 벗어나지 않습니다."
        elif source_overlap >= 0.18:
            score = 4
            reason = "문제와 선택지가 source에 대체로 닫히지만 일부 표현이 느슨합니다."
        else:
            score = 2
            error_tags.append("원문 외 사실 추가")
            reason = "선택지나 문제 문장이 source 범위를 충분히 따르지 않습니다."
        pass_or_fail = "pass" if score >= 4 else "fail"
    elif role_name == "Keyedness":
        duplicate_choices = len(set(normalized_text(choice) for choice in all_choices)) != 4
        if duplicate_choices:
            score = 1
            error_tags.append("선택지 중복")
            pass_or_fail = "fail"
            reason = "선택지 중복이 있어 단일정답형으로 볼 수 없습니다."
        else:
            correct_overlap = overlap_ratio(generation["gold_short_answer"], correct_text)
            competing_overlaps = [
                overlap_ratio(generation["gold_short_answer"], choice)
                for label, choice in option_map.items()
                if label != correct_choice
            ]
            if correct_overlap < 0.18:
                score = 2
                error_tags.append("정답 비유일")
                pass_or_fail = "fail"
                reason = "정답 선택지와 기준 정답의 대응이 약합니다."
            elif any(overlap >= correct_overlap - 0.05 for overlap in competing_overlaps):
                score = 2
                error_tags.append("오답이 정답 가능")
                pass_or_fail = "fail"
                reason = "오답 선택지 중 기준 정답과 지나치게 가까운 선택지가 있습니다."
            else:
                score = 5
                pass_or_fail = "pass"
                reason = "단일정답 구조가 유지됩니다."
    else:
        stem_overlap = overlap_ratio(generation["gold_short_answer"], generation["generated_stem"])
        if stem_overlap >= 0.70:
            score = 2
            error_tags.append("정답 누설")
            reason = "문제 본문에 정답이 과하게 노출됩니다."
        elif "?" in generation["generated_stem"] and generation["generated_stem"].count("?") >= 2:
            score = 2
            error_tags.append("복수 쟁점 혼합")
            reason = "문제 본문에 복수 질의가 섞여 있습니다."
        else:
            distractor_lengths = [len(tokenize(choice)) for label, choice in option_map.items() if label != correct_choice]
            if distractor_lengths and min(distractor_lengths) < 3:
                score = 3
                reason = "오답 선택지 길이가 짧아 품질이 다소 약합니다."
            else:
                score = 5
                reason = "오답 선택지가 그럴듯하지만 정답과 구분됩니다."
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
    write_jsonl_atomic(KEYEDNESS_LOG_PATH, outputs["Keyedness"])
    write_jsonl_atomic(DISTRACTORFIT_LOG_PATH, outputs["DistractorFit"])


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
    snapshot_prompts(["judge_grounding.md", "judge_keyedness.md", "judge_distractorfit.md"])
    seed_map = {seed["seed_sample_id"]: seed for seed in load_jsonl(SEED_READY_PATH)}
    generations = load_jsonl(GENERATED_PROBLEMS_PATH)
    outputs = load_existing_outputs()

    completed_keys = {
        (row["candidate_id"], role_name)
        for role_name, rows in outputs.items()
        for row in rows
    }

    futures = []
    with ThreadPoolExecutor(max_workers=mode_config["max_workers"]) as executor:
        for generation in generations:
            seed = seed_map[generation["seed_sample_id"]]
            for role_name in outputs:
                key = (generation["candidate_id"], role_name)
                if key in completed_keys:
                    continue
                futures.append(executor.submit(evaluate_one, seed, generation, role_name, mode_config))

        for completed_index, future in enumerate(as_completed(futures), start=1):
            resolved_role_name, row = future.result()
            outputs[resolved_role_name].append(row)
            if completed_index % max(1, mode_config["checkpoint_every"]) == 0:
                checkpoint_outputs(outputs)

    for role_name in outputs:
        outputs[role_name].sort(key=lambda row: (row["seed_sample_id"], row["candidate_id"]))
    checkpoint_outputs(outputs)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("main", "strict_finalize"), default="main")
    args = parser.parse_args()
    main(mode=args.mode)
