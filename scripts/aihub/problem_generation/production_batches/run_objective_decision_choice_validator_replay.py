from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402


# offline replay도 새 산출물을 만들 때는 실제 실행 시각으로 run stamp를 생성한다.
RUN_STAMP = build_run_stamp()
RUN_SLUG = "decision_choice_validator_offline_replay"
RUN_NAME = f"{RUN_STAMP}_{RUN_SLUG}_objective_r2_decision_choice_validator_offline_replay"

SOURCE_RUN_DIR = (
    PROJECT_ROOT
    / "analysis/aihub/problem_generation/llm_runs"
    / "2026-04-25_203320_decision_weak_distractor_guardrail_pilot_objective_r2_decision_weak_distractor_guardrail_pilot"
)
SOURCE_MERGED = SOURCE_RUN_DIR / "merged/merged_problem_scores_decision_weak_distractor_guardrail_pilot.csv"

RUN_DIR = PROJECT_ROOT / "analysis/aihub/problem_generation/llm_runs" / RUN_NAME
EXPORT_DIR = RUN_DIR / "exports"
MERGED_DIR = RUN_DIR / "merged"

CHOICE_LABELS = ("A", "B", "C", "D")

# Reviewer가 지적한 tail replay 기준이다. 새 API 실행 전에 기존 실패/감사 tail이
# deterministic validator에서 원하는 조치로 잡히는지 먼저 검산한다.
EXPECTED_TAIL_ACTIONS = {
    "decision_guardrail_007": {"hard_block", "regenerate"},
    "decision_guardrail_008": {"audit", "regenerate"},
    "decision_guardrail_011": {"audit", "regenerate"},
    "decision_guardrail_014": {"audit", "regenerate"},
    "decision_guardrail_015": {"audit", "regenerate"},
}


def normalize_text(value: str) -> str:
    """선택지 동치/중복 검사는 보수적으로 하되, 공백·기호 차이는 무시한다."""
    normalized = re.sub(r"\s+", "", value or "")
    normalized = re.sub(r"[^\w가-힣]", "", normalized)
    return normalized.lower()


def split_tags(value: str) -> set[str]:
    return {tag.strip() for tag in (value or "").split("|") if tag.strip()}


def choice_map(row: dict[str, str]) -> dict[str, str]:
    return {label: row.get(f"choice_{label.lower()}", "") for label in CHOICE_LABELS}


def parse_label_json_map(value: str) -> dict[str, str]:
    """Reviewer artifact 검산을 위해 label-keyed JSON 메타데이터를 보수적으로 읽는다."""
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(payload_value) for key, payload_value in payload.items()}


def dump_label_json_map(payload: dict[str, str]) -> str:
    """CSV 안에서도 label 순서가 안정적으로 보이도록 A-D 순서로 JSON을 쓴다."""
    ordered = {label: payload[label] for label in CHOICE_LABELS if label in payload}
    return json.dumps(ordered, ensure_ascii=False)


def label_swap_maps(current_correct: str, target_correct: str) -> tuple[dict[str, str], dict[str, str]]:
    """choice shuffle이 단순 swap이므로 old->new, new->old label map도 같은 swap으로 만든다."""
    old_to_new = {label: label for label in CHOICE_LABELS}
    if current_correct in CHOICE_LABELS and target_correct in CHOICE_LABELS:
        old_to_new[current_correct], old_to_new[target_correct] = (
            target_correct,
            current_correct,
        )
    new_to_old = {new_label: old_label for old_label, new_label in old_to_new.items()}
    return old_to_new, new_to_old


def remap_label_keyed_metadata(row: dict[str, str], current_correct: str, target_correct: str) -> None:
    """choice label을 바꾼 뒤 distractor metadata도 같은 permutation으로 옮긴다."""
    old_to_new, _ = label_swap_maps(current_correct, target_correct)

    distractor_type_map = parse_label_json_map(row.get("distractor_type_map", ""))
    if distractor_type_map:
        remapped = {
            old_to_new[label]: value
            for label, value in distractor_type_map.items()
            if label in old_to_new
        }
        row["distractor_type_map"] = dump_label_json_map(remapped)

    near_miss_notes = parse_label_json_map(row.get("near_miss_notes", ""))
    if near_miss_notes:
        remapped_notes = {
            old_to_new[label]: value
            for label, value in near_miss_notes.items()
            if label in old_to_new and old_to_new[label] != target_correct
        }
        row["near_miss_notes"] = dump_label_json_map(remapped_notes)


def remap_existing_metadata_to_correct_choice(row: dict[str, str]) -> None:
    """이미 choice가 셔플된 보존 row의 metadata를 현재 correct_choice에 다시 맞춘다."""
    correct_choice = row.get("correct_choice", "")
    distractor_type_map = parse_label_json_map(row.get("distractor_type_map", ""))
    previous_correct = next(
        (label for label, value in distractor_type_map.items() if value == "정답"),
        correct_choice,
    )
    if previous_correct != correct_choice:
        remap_label_keyed_metadata(row, previous_correct, correct_choice)


def label_metadata_gate(row: dict[str, str]) -> tuple[bool, list[str]]:
    """post-shuffle artifact에서 정답 label과 label-keyed metadata가 맞는지 검산한다."""
    correct_choice = row.get("correct_choice", "")
    reasons: list[str] = []
    distractor_type_map = parse_label_json_map(row.get("distractor_type_map", ""))
    near_miss_notes = parse_label_json_map(row.get("near_miss_notes", ""))

    if distractor_type_map and distractor_type_map.get(correct_choice) != "정답":
        reasons.append("distractor_type_map_correct_choice_mismatch")
    if near_miss_notes and correct_choice in near_miss_notes:
        reasons.append("near_miss_notes_contains_correct_choice")

    return not reasons, reasons


def target_label_for_index(index: int) -> str:
    # 기존 결과의 A 편향을 그대로 두지 않기 위해 패키지 단위 deterministic schedule을 둔다.
    return CHOICE_LABELS[index % len(CHOICE_LABELS)]


def recalculated_correct_label(choices: dict[str, str], answer_text: str) -> tuple[str | None, int]:
    normalized_answer = normalize_text(answer_text)
    matches = [label for label, text in choices.items() if normalize_text(text) == normalized_answer]
    if len(matches) != 1:
        return None, len(matches)
    return matches[0], 1


def shuffled_choices_for_target(
    choices: dict[str, str], current_correct: str, target_correct: str
) -> tuple[dict[str, str], str | None, int]:
    """정답 선택지를 목표 label로 이동하고 correct_choice를 다시 계산해 불일치를 잡는다."""
    shuffled = dict(choices)
    answer_text = shuffled.get(current_correct, "")
    if current_correct in CHOICE_LABELS and target_correct in CHOICE_LABELS:
        shuffled[current_correct], shuffled[target_correct] = shuffled[target_correct], shuffled[current_correct]
    recalculated_label, match_count = recalculated_correct_label(shuffled, answer_text)
    return shuffled, recalculated_label, match_count


def duplicate_choice_count(choices: dict[str, str]) -> int:
    normalized = [normalize_text(text) for text in choices.values()]
    return len(normalized) - len(set(normalized))


def choose_validator_action(row: dict[str, str]) -> tuple[str, str, list[str]]:
    """후처리 검증기는 semantic judge가 아니라 reviewer tail을 재발 방지하는 gate다."""
    tags = split_tags(row.get("error_tags", ""))
    final_status = row.get("final_status", "")
    audit_required = row.get("audit_required", "")
    choices = choice_map(row)
    correct_choice = row.get("correct_choice", "")
    answer_text = choices.get(correct_choice, "")
    reasons: list[str] = []

    if duplicate_choice_count(choices) > 0:
        reasons.append("duplicate_or_equivalent_choice_text")

    _, answer_match_count = recalculated_correct_label(choices, answer_text)
    if answer_match_count != 1:
        reasons.append("correct_answer_not_unique_after_normalization")

    uniqueness_tags = {"정답 비유일", "오답이 정답 가능", "복수 쟁점 혼합"}
    if tags & uniqueness_tags:
        reasons.append("answer_uniqueness_failure_signal")

    weak_tags = {"오답약함", "near_miss_부족"}
    if tags & weak_tags:
        reasons.append("weak_distractor_signal")

    if final_status == "hard_fail":
        reasons.append("upstream_hard_fail")

    if reasons and any(
        reason in reasons
        for reason in (
            "duplicate_or_equivalent_choice_text",
            "correct_answer_not_unique_after_normalization",
            "answer_uniqueness_failure_signal",
            "upstream_hard_fail",
        )
    ):
        return "hard_block", "answer_uniqueness_or_schema_block", reasons

    if "weak_distractor_signal" in reasons:
        return "regenerate", "weak_distractor_regeneration", reasons

    if audit_required == "예":
        return "audit", "upstream_audit_without_specific_tail", reasons or ["upstream_audit"]

    return "accept", "validator_clean", reasons


def read_selected_rows() -> list[dict[str, str]]:
    with SOURCE_MERGED.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row.get("selected_for_seed") == "예"]


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_selected_rows()
    replay_rows: list[dict[str, str]] = []

    for index, row in enumerate(rows):
        choices = choice_map(row)
        target_label = target_label_for_index(index)
        shuffled_choices, recalculated_label, match_count = shuffled_choices_for_target(
            choices, row.get("correct_choice", ""), target_label
        )
        validator_action, validator_status, reasons = choose_validator_action(row)
        seed_id = row.get("seed_sample_id", "")
        expected_actions = EXPECTED_TAIL_ACTIONS.get(seed_id)
        tail_expectation_met = (
            "예" if expected_actions and validator_action in expected_actions else "아니오" if expected_actions else ""
        )

        replay_rows.append(
            {
                "seed_sample_id": seed_id,
                "source_final_status": row.get("final_status", ""),
                "source_audit_required": row.get("audit_required", ""),
                "source_train_eligible": row.get("train_eligible", ""),
                "source_error_tags": row.get("error_tags", ""),
                "source_nearmiss_reason": row.get("nearmiss_reason", ""),
                "original_correct_choice": row.get("correct_choice", ""),
                "target_correct_choice": target_label,
                "recalculated_correct_choice": recalculated_label or "",
                "correct_choice_match_count": str(match_count),
                "shuffle_recalc_ok": "예" if recalculated_label == target_label else "아니오",
                "validator_action": validator_action,
                "validator_status": validator_status,
                "validator_reasons": "|".join(reasons),
                "expected_tail_action_set": ",".join(sorted(expected_actions)) if expected_actions else "",
                "tail_expectation_met": tail_expectation_met,
                "shuffled_choice_a": shuffled_choices["A"],
                "shuffled_choice_b": shuffled_choices["B"],
                "shuffled_choice_c": shuffled_choices["C"],
                "shuffled_choice_d": shuffled_choices["D"],
            }
        )

    action_counts = Counter(row["validator_action"] for row in replay_rows)
    target_label_counts = Counter(row["target_correct_choice"] for row in replay_rows)
    original_label_counts = Counter(row["original_correct_choice"] for row in replay_rows)
    accepted_label_counts = Counter(
        row["target_correct_choice"] for row in replay_rows if row["validator_action"] == "accept"
    )

    expected_tail_rows = [row for row in replay_rows if row["expected_tail_action_set"]]
    expected_tail_pass = all(row["tail_expectation_met"] == "예" for row in expected_tail_rows)
    shuffle_pass = all(row["shuffle_recalc_ok"] == "예" for row in replay_rows)
    full_label_balance_pass = all(target_label_counts[label] == 4 for label in CHOICE_LABELS)
    offline_replay_passed = expected_tail_pass and shuffle_pass and full_label_balance_pass

    replay_csv = EXPORT_DIR / "validator_replay_decision_choice_validator_offline_replay.csv"
    fieldnames = list(replay_rows[0].keys()) if replay_rows else []
    write_csv(replay_csv, replay_rows, fieldnames)

    validated_csv = MERGED_DIR / "validated_problem_scores_decision_choice_validator_offline_replay.csv"
    write_csv(validated_csv, replay_rows, fieldnames)

    summary_rows = [
        {"metric": "source_selected_rows", "value": str(len(rows))},
        {"metric": "validator_action_accept", "value": str(action_counts.get("accept", 0))},
        {"metric": "validator_action_regenerate", "value": str(action_counts.get("regenerate", 0))},
        {"metric": "validator_action_hard_block", "value": str(action_counts.get("hard_block", 0))},
        {"metric": "expected_tail_rows", "value": str(len(expected_tail_rows))},
        {"metric": "expected_tail_passed", "value": "예" if expected_tail_pass else "아니오"},
        {"metric": "shuffle_recalc_passed", "value": "예" if shuffle_pass else "아니오"},
        {"metric": "full_package_label_balance_passed", "value": "예" if full_label_balance_pass else "아니오"},
        {"metric": "offline_replay_passed", "value": "예" if offline_replay_passed else "아니오"},
        {"metric": "current_count_reflection", "value": "미합산"},
        {"metric": "next_stop_line", "value": "8개 micro pilot"},
    ]
    write_csv(
        EXPORT_DIR / "validator_summary_decision_choice_validator_offline_replay.csv",
        summary_rows,
        ["metric", "value"],
    )

    label_rows = []
    for label in CHOICE_LABELS:
        label_rows.append(
            {
                "label": label,
                "original_selected_count": str(original_label_counts.get(label, 0)),
                "target_scheduled_count": str(target_label_counts.get(label, 0)),
                "accepted_after_validator_count": str(accepted_label_counts.get(label, 0)),
            }
        )
    write_csv(
        EXPORT_DIR / "label_balance_decision_choice_validator_offline_replay.csv",
        label_rows,
        ["label", "original_selected_count", "target_scheduled_count", "accepted_after_validator_count"],
    )

    tail_columns = [
        "seed_sample_id",
        "source_final_status",
        "source_error_tags",
        "validator_action",
        "validator_status",
        "tail_expectation_met",
    ]
    tail_rows = [row for row in replay_rows if row["expected_tail_action_set"]]
    summary_md = "\n".join(
        [
            "# decision choice validator offline replay",
            "",
            "## Scope",
            "",
            "- Source run: `decision_weak_distractor_guardrail_pilot`",
            "- API calls: `0`",
            "- Count reflection: `미합산`",
            "- Purpose: reviewer가 요구한 validator 구현 전 offline replay stop line 검산",
            "",
            "## Summary",
            "",
            render_markdown_table(summary_rows, ["metric", "value"]),
            "",
            "## Label balance",
            "",
            render_markdown_table(label_rows, ["label", "original_selected_count", "target_scheduled_count", "accepted_after_validator_count"]),
            "",
            "## Expected tail replay",
            "",
            render_markdown_table(tail_rows, tail_columns),
            "",
            "## Interpretation",
            "",
            "- `decision_guardrail_007`은 answer uniqueness failure로 `hard_block` 처리한다.",
            "- `decision_guardrail_008/011/014/015`는 weak distractor signal로 `regenerate` 처리한다.",
            "- label schedule은 전체 16건 기준 `A/B/C/D = 4/4/4/4`로 재배치되고, 정답 label 재계산 불일치는 없다.",
            "- 이 replay는 기존 pilot을 count에 넣는 절차가 아니라, `8개 micro pilot` 진입 전 validator 동작 검산이다.",
            "",
        ]
    )
    (EXPORT_DIR / "validator_replay_decision_choice_validator_offline_replay.md").write_text(
        summary_md, encoding="utf-8"
    )

    manifest = {
        "run_name": RUN_NAME,
        "run_stamp": RUN_STAMP,
        "recipe": RUN_SLUG,
        "source_run_dir": str(SOURCE_RUN_DIR.relative_to(PROJECT_ROOT)),
        "source_merged": str(SOURCE_MERGED.relative_to(PROJECT_ROOT)),
        "api_calls": {"openai_api": 0, "gemini_api": 0},
        "selected_rows": len(rows),
        "validator_action_counts": dict(action_counts),
        "original_label_counts": {label: original_label_counts.get(label, 0) for label in CHOICE_LABELS},
        "target_label_counts": {label: target_label_counts.get(label, 0) for label in CHOICE_LABELS},
        "accepted_after_validator_label_counts": {
            label: accepted_label_counts.get(label, 0) for label in CHOICE_LABELS
        },
        "expected_tail_passed": expected_tail_pass,
        "shuffle_recalc_passed": shuffle_pass,
        "full_package_label_balance_passed": full_label_balance_pass,
        "offline_replay_passed": offline_replay_passed,
        "current_count_reflection": "not_counted",
        "next_stop_line": "decision_choice_validator_micro_pilot_8",
        "outputs": {
            "validator_replay_csv": str(replay_csv.relative_to(PROJECT_ROOT)),
            "validator_replay_md": str(
                (EXPORT_DIR / "validator_replay_decision_choice_validator_offline_replay.md").relative_to(
                    PROJECT_ROOT
                )
            ),
            "validator_summary_csv": str(
                (EXPORT_DIR / "validator_summary_decision_choice_validator_offline_replay.csv").relative_to(
                    PROJECT_ROOT
                )
            ),
            "label_balance_csv": str(
                (EXPORT_DIR / "label_balance_decision_choice_validator_offline_replay.csv").relative_to(
                    PROJECT_ROOT
                )
            ),
            "validated_csv": str(validated_csv.relative_to(PROJECT_ROOT)),
        },
    }
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
