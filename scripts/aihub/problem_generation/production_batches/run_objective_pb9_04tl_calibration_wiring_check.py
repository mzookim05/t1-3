from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

# `pb9` API run 이후 reviewer가 요구한 stop line은 추가 API 호출이 아니라
# `weak_distractor_*` 구조화 필드가 실제 판정/리포트 schema에 필수로 물리는지 확인하는 것이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

VERSION_TAG = "pb9_04tl_decision_weak_distractor_calibration_wiring_check"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}"

RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_EXPORTS_DIR = RUN_DIR / "exports"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"

VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"wiring_check_{VERSION_TAG}.md"

JUDGE_STRUCTURED_REQUIRED_FIELDS = [
    "weak_distractor_count",
    "weak_distractor_labels",
    "all_three_near_miss",
    "one_axis_perturbation_count",
]

VALIDATOR_REPORT_REQUIRED_FIELDS = [
    "seed_sample_id",
    "case_name",
    "source_subset",
    "sampling_lane",
    "nearmiss_score",
    "distractorfit_score",
    *JUDGE_STRUCTURED_REQUIRED_FIELDS,
    "validator_action",
    "validator_export_disposition",
    "validator_status",
    "validator_reasons",
    "split_allowed",
    "count_allowed",
    "target_correct_choice",
    "export_correct_choice",
    "batch_status",
    "count_reflection_status",
    "downstream_consumption_allowed",
]

TAIL_MEMO_REQUIRED_FIELDS = [
    "seed_sample_id",
    "case_name",
    "source_subset",
    "sampling_lane",
    "validator_action",
    "validator_export_disposition",
    "validator_status",
    "validator_reasons",
    "split_allowed",
    "count_allowed",
    "target_correct_choice",
    "export_correct_choice",
    "nearmiss_score",
    "distractorfit_score",
    "weak_distractor_count",
    "weak_distractor_labels",
    "tail_class",
]

BATCH_STATUS = "failed_not_counted"
COUNT_REFLECTION_STATUS = "not_counted"
DOWNSTREAM_CONSUMPTION_ALLOWED = "아니오"
VALID_CHOICE_LABELS = {"A", "B", "C", "D"}
EXPORT_DISPOSITION_BY_ACTION = {
    "accept": "export_ready",
    "audit": "audit_queue",
    "regenerate": "regenerate_required",
    "hard_block": "hard_blocked",
}


def utc_now_iso() -> str:
    # 로컬 기본 Python이 3.9라 `datetime.UTC` 대신 `timezone.utc`를 사용한다.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_int(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def parse_weak_distractor_labels(value: str) -> list[str]:
    # Judge가 여러 weak label을 `C|D`처럼 내보낼 수 있으므로, report 자동화를 위해
    # 빈 token을 버리고 label 단위로 정규화한다.
    return [label.strip() for label in str(value).split("|") if label.strip()]


def missing_required_structured_fields(row: dict[str, str]) -> list[str]:
    # `weak_distractor_labels`는 `weak_distractor_count = 0`이면 빈 값이 정상이다.
    # 따라서 key 존재와 값 필수성을 분리해, clean row가 schema check에서 막히지 않게 한다.
    missing = [field for field in JUDGE_STRUCTURED_REQUIRED_FIELDS if field not in row]
    for field in ("weak_distractor_count", "all_three_near_miss", "one_axis_perturbation_count"):
        if str(row.get(field, "")).strip() == "":
            missing.append(field)
    return missing


def weak_distractor_label_errors(row: dict[str, str], weak_count: int) -> list[str]:
    # 이번 hotfix는 `weak_distractor_count`와 label이 따로 놀아도 통과하던 구멍을 막는다.
    # count가 0이면 label 공란을 허용하되, count가 있으면 label 수/도메인/정답 제외를 모두 검산한다.
    labels = parse_weak_distractor_labels(row.get("weak_distractor_labels", ""))
    correct_choice = str(row.get("export_correct_choice") or row.get("target_correct_choice", "")).strip()
    errors: list[str] = []
    if weak_count < 0:
        errors.append("weak_count_negative")
    if weak_count == 0 and labels:
        errors.append("weak_labels_present_when_count_zero")
    if weak_count > 0 and not labels:
        errors.append("weak_labels_blank_when_count_positive")
    if weak_count > 0 and len(labels) != weak_count:
        errors.append("weak_label_count_mismatch")
    invalid_labels = [label for label in labels if label not in VALID_CHOICE_LABELS]
    if invalid_labels:
        errors.append(f"invalid_weak_label:{','.join(invalid_labels)}")
    if len(set(labels)) != len(labels):
        errors.append("duplicate_weak_label")
    if correct_choice and correct_choice in labels:
        errors.append("weak_label_includes_correct_choice")
    return errors


def disposition_for_action(action: str) -> str:
    # action과 export disposition을 분리해 남겨야 reviewer가 export/split/count 차단을 기계적으로 검산할 수 있다.
    return EXPORT_DISPOSITION_BY_ACTION[action]


def split_allowed_for_action(action: str) -> str:
    # wiring check는 실제 split을 만들지 않지만, validator 정책상 accept만 split 후보가 될 수 있음을 명시한다.
    return "예" if action == "accept" else "아니오"


def choose_calibration_action(row: dict[str, str]) -> tuple[str, str, list[str]]:
    # 이번 check의 핵심은 기존 `NearMiss` 자연어 reason에 기대지 않고, 구조화 필드가
    # 없거나 파싱되지 않으면 export-ready로 새지 않게 막는 것이다.
    missing = missing_required_structured_fields(row)
    if missing:
        return "hard_block", "judge_structured_output_missing", [f"missing:{','.join(missing)}"]

    nearmiss_score = parse_int(row.get("nearmiss_score", ""))
    weak_count = parse_int(row.get("weak_distractor_count", ""))
    one_axis_count = parse_int(row.get("one_axis_perturbation_count", ""))
    if nearmiss_score is None or weak_count is None or one_axis_count is None:
        return "hard_block", "judge_structured_output_parse_failure", ["structured_parse_failure"]

    if row.get("all_three_near_miss") not in {"예", "아니오"}:
        return "hard_block", "judge_structured_output_parse_failure", ["all_three_near_miss_parse_failure"]
    label_errors = weak_distractor_label_errors(row, weak_count)
    if label_errors:
        return "hard_block", "weak_distractor_label_consistency_failure", label_errors

    if nearmiss_score <= 2:
        return "regenerate", "nearmiss_score_le_2_regenerate", ["nearmiss_score_le_2"]
    if nearmiss_score == 3:
        return "audit", "nearmiss_score_3_audit", ["nearmiss_score_3"]
    if nearmiss_score == 4 and weak_count >= 1:
        return "regenerate", "nearmiss_4_weak_distractor_regenerate", ["weak_distractor_count_ge_1"]
    if row["all_three_near_miss"] == "아니오" or one_axis_count < 3:
        return "regenerate", "near_miss_structure_incomplete", ["all_three_or_one_axis_incomplete"]
    return "accept", "validator_clean", []


def fixture_rows() -> list[dict[str, str]]:
    base = {
        "doc_type_name": "결정례_QA",
        "source_subset": "04_TL_결정례_QA",
        "sampling_lane": "generalization_03_04",
        "distractorfit_score": "5",
        "target_correct_choice": "A",
        "export_correct_choice": "A",
    }
    fixtures = [
        ("clean_nearmiss_5", {"nearmiss_score": "5", "weak_distractor_count": "0", "weak_distractor_labels": "", "all_three_near_miss": "예", "one_axis_perturbation_count": "3"}, "accept"),
        ("nearmiss_4_no_weak", {"nearmiss_score": "4", "weak_distractor_count": "0", "weak_distractor_labels": "", "all_three_near_miss": "예", "one_axis_perturbation_count": "3"}, "accept"),
        ("nearmiss_4_one_weak", {"nearmiss_score": "4", "weak_distractor_count": "1", "weak_distractor_labels": "D", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "regenerate"),
        ("nearmiss_3", {"nearmiss_score": "3", "weak_distractor_count": "1", "weak_distractor_labels": "C", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "audit"),
        ("low_nearmiss", {"nearmiss_score": "2", "weak_distractor_count": "2", "weak_distractor_labels": "C|D", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "1"}, "regenerate"),
        ("missing_structured_field", {"nearmiss_score": "4", "weak_distractor_count": "", "weak_distractor_labels": "D", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
        ("parse_failure_structured_field", {"nearmiss_score": "4", "weak_distractor_count": "one", "weak_distractor_labels": "D", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
        ("weak_count_positive_labels_blank", {"nearmiss_score": "4", "weak_distractor_count": "1", "weak_distractor_labels": "", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
        ("weak_count_label_mismatch", {"nearmiss_score": "4", "weak_distractor_count": "2", "weak_distractor_labels": "D", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
        ("weak_label_invalid", {"nearmiss_score": "4", "weak_distractor_count": "1", "weak_distractor_labels": "E", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
        ("weak_label_equals_correct_choice", {"nearmiss_score": "4", "weak_distractor_count": "1", "weak_distractor_labels": "A", "all_three_near_miss": "아니오", "one_axis_perturbation_count": "2"}, "hard_block"),
    ]
    rows: list[dict[str, str]] = []
    for index, (case_name, updates, expected_action) in enumerate(fixtures, start=1):
        row = {
            **base,
            **updates,
            "seed_sample_id": f"wiring_fixture_{index:03d}",
            "case_name": case_name,
            "expected_action": expected_action,
        }
        rows.append(row)
    return rows


def build_report_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    report_rows = []
    for row in rows:
        action, status, reasons = choose_calibration_action(row)
        export_disposition = disposition_for_action(action)
        split_allowed = split_allowed_for_action(action)
        report_rows.append(
            {
                **{field: row.get(field, "") for field in VALIDATOR_REPORT_REQUIRED_FIELDS},
                "case_name": row["case_name"],
                "expected_action": row["expected_action"],
                "validator_action": action,
                "validator_export_disposition": export_disposition,
                "validator_status": status,
                "validator_reasons": "|".join(reasons),
                "split_allowed": split_allowed,
                "count_allowed": "아니오",
                "batch_status": BATCH_STATUS,
                "count_reflection_status": COUNT_REFLECTION_STATUS,
                "downstream_consumption_allowed": DOWNSTREAM_CONSUMPTION_ALLOWED,
                "action_passed": "예" if action == row["expected_action"] else "아니오",
            }
        )
    return report_rows


def build_tail_rows(report_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    tail_rows = []
    for row in report_rows:
        if row["validator_action"] == "accept":
            continue
        tail_class = "decision weak distractor calibration"
        if row["validator_status"].startswith("judge_structured"):
            tail_class = "structured output wiring failure"
        tail_rows.append(
            {
                **{field: row.get(field, "") for field in TAIL_MEMO_REQUIRED_FIELDS},
                "tail_class": tail_class,
            }
        )
    return tail_rows


def assert_wiring(report_rows: list[dict[str, str]], tail_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    checks = []
    checks.append(
        {
            "gate": "judge_structured_required_fields",
            "result": "pass",
            "evidence": ",".join(JUDGE_STRUCTURED_REQUIRED_FIELDS),
        }
    )
    if any(row["action_passed"] != "예" for row in report_rows):
        raise RuntimeError("calibration action fixture mismatch")
    checks.append({"gate": "validator_policy_fixture", "result": "pass", "evidence": "all expected actions matched"})

    report_missing = [field for field in VALIDATOR_REPORT_REQUIRED_FIELDS if field not in report_rows[0]]
    tail_missing = [field for field in TAIL_MEMO_REQUIRED_FIELDS if field not in tail_rows[0]]
    if report_missing or tail_missing:
        raise RuntimeError(f"report schema missing: report={report_missing}, tail={tail_missing}")
    checks.append({"gate": "validator_report_schema", "result": "pass", "evidence": ",".join(VALIDATOR_REPORT_REQUIRED_FIELDS)})
    checks.append({"gate": "tail_memo_schema", "result": "pass", "evidence": ",".join(TAIL_MEMO_REQUIRED_FIELDS)})

    leaked = [
        row for row in report_rows
        if row["validator_status"].startswith("judge_structured") and row["validator_action"] == "accept"
    ]
    if leaked:
        raise RuntimeError("structured output parse failure leaked to accept")
    checks.append({"gate": "parse_failure_export_block", "result": "pass", "evidence": "parse/missing -> hard_block"})

    label_consistency_statuses = {
        "weak_distractor_label_consistency_failure",
        "validator_clean",
        "nearmiss_4_weak_distractor_regenerate",
        "nearmiss_score_3_audit",
        "nearmiss_score_le_2_regenerate",
    }
    label_consistency_rows = [
        row for row in report_rows
        if row["validator_status"] in label_consistency_statuses
    ]
    if not label_consistency_rows:
        raise RuntimeError("weak distractor label consistency fixture missing")
    label_failure_rows = [
        row for row in report_rows
        if row["case_name"].startswith("weak_") and row["validator_status"] != "weak_distractor_label_consistency_failure"
    ]
    if label_failure_rows:
        raise RuntimeError("weak distractor label consistency failure leaked")
    checks.append(
        {
            "gate": "weak_distractor_label_consistency",
            "result": "pass",
            "evidence": "count/labels/domain/correct-label fixtures blocked",
        }
    )

    disposition_failures = [
        row for row in report_rows
        if row["validator_action"] != "accept"
        and (
            row["validator_export_disposition"] == "export_ready"
            or row["split_allowed"] != "아니오"
            or row["count_allowed"] != "아니오"
        )
    ]
    count_failures = [row for row in report_rows if row["count_allowed"] != "아니오"]
    if disposition_failures or count_failures:
        raise RuntimeError("validator disposition/split/count block failed")
    checks.append(
        {
            "gate": "validator_export_split_count_block",
            "result": "pass",
            "evidence": "non-accept rows cannot export/split/count; count_allowed all 아니오",
        }
    )

    status_failures = [
        row for row in report_rows
        if row["batch_status"] != BATCH_STATUS
        or row["count_reflection_status"] != COUNT_REFLECTION_STATUS
        or row["downstream_consumption_allowed"] != DOWNSTREAM_CONSUMPTION_ALLOWED
    ]
    if status_failures:
        raise RuntimeError("failed/not-counted status propagation failed")
    checks.append({"gate": "status_propagation", "result": "pass", "evidence": "failed_not_counted/not_counted/아니오"})
    checks.append({"gate": "api_calls", "result": "pass", "evidence": "0"})
    return checks


def write_wiring_md(check_rows: list[dict[str, str]], report_rows: list[dict[str, str]], tail_rows: list[dict[str, str]]) -> None:
    lines = [
        f"# `{VERSION_TAG}`",
        "",
        "## summary",
        "",
        "This no-API wiring check locks the structured weak-distractor fields before any `8개` calibration pilot.",
        "",
        "| gate | result | evidence |",
        "| --- | --- | --- |",
    ]
    for row in check_rows:
        lines.append(f"| `{row['gate']}` | `{row['result']}` | `{row['evidence']}` |")
    lines.extend(
        [
            "",
            "## action counts",
            "",
            f"- validator_action_counts: `{dict(Counter(row['validator_action'] for row in report_rows))}`",
            f"- validator_export_disposition_counts: `{dict(Counter(row['validator_export_disposition'] for row in report_rows))}`",
            f"- tail_rows: `{len(tail_rows)}`",
            "- count_reflection: `not_counted`",
            "- downstream_consumption_allowed: `아니오`",
        ]
    )
    write_text(WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def main() -> None:
    started_at_utc = utc_now_iso()
    rows = fixture_rows()
    report_rows = build_report_rows(rows)
    tail_rows = build_tail_rows(report_rows)
    check_rows = assert_wiring(report_rows, tail_rows)

    write_csv(VALIDATOR_REPORT_CSV_PATH, report_rows, list(report_rows[0].keys()))
    write_csv(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    write_wiring_md(check_rows, report_rows, tail_rows)

    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "run_mode": "no_api_wiring_check",
        "started_at_utc": started_at_utc,
        "finished_at_utc": utc_now_iso(),
        "api_calls": 0,
        "judge_structured_required_fields": JUDGE_STRUCTURED_REQUIRED_FIELDS,
        "validator_report_required_fields": VALIDATOR_REPORT_REQUIRED_FIELDS,
        "tail_memo_required_fields": TAIL_MEMO_REQUIRED_FIELDS,
        "validator_action_counts": dict(Counter(row["validator_action"] for row in report_rows)),
        # reviewer가 artifact만 보고 hotfix 통과 여부를 검산할 수 있도록 disposition과 allow flag 분포를 manifest에도 남긴다.
        "validator_export_disposition_counts": dict(Counter(row["validator_export_disposition"] for row in report_rows)),
        "split_allowed_counts": dict(Counter(row["split_allowed"] for row in report_rows)),
        "count_allowed_counts": dict(Counter(row["count_allowed"] for row in report_rows)),
        "weak_distractor_label_consistency_fixture_cases": [
            row["case_name"] for row in report_rows if row["validator_status"] == "weak_distractor_label_consistency_failure"
        ],
        "batch_status": BATCH_STATUS,
        "count_reflection_status": COUNT_REFLECTION_STATUS,
        "downstream_consumption_allowed": False,
        "wiring_check_passed": True,
        "next_stop_line": "reviewer sign-off before 8-seed 04TL targeted calibration pilot",
        "artifact_paths": {
            "validator_report": str(VALIDATOR_REPORT_CSV_PATH),
            "tail_memo": str(TAIL_MEMO_CSV_PATH),
            "wiring_check": str(WIRING_CHECK_MD_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)
    print(json.dumps({"run_name": RUN_NAME, "api_calls": 0, "wiring_check_passed": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
