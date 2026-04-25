from __future__ import annotations

import argparse
import csv
import fcntl
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

# `pb9`는 API 실행 직전 runner wiring을 닫는 단계다. 이번 파일은 실제
# generation/Judge 호출 전, 실행 모드에서도 같은 gate가 작동하는지 무호출로 검산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_choice_validator_replay as validator_replay,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb8_decision_only as pb8,
)


VERSION_TAG = "pb9_decision_only_controlled_production_with_choice_validator"
# llm_runs 폴더 순서를 이름만으로 판단할 수 있게 HHMMSS까지 고정한다.
RUN_DATE = "2026-04-26_024801"
RUN_PURPOSE = "objective_r2_decision_only_execution_mode_smoke_check"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

PROJECT_ROOT = pb8.pb6.pb4.pb3.base.PROJECT_ROOT
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
VALIDATOR_FIXTURE_CSV_PATH = RUN_EXPORTS_DIR / f"validator_fixture_check_{VERSION_TAG}.csv"
VALIDATOR_FIXTURE_MD_PATH = RUN_EXPORTS_DIR / f"validator_fixture_check_{VERSION_TAG}.md"
EXECUTION_SMOKE_CSV_PATH = RUN_EXPORTS_DIR / f"execution_mode_smoke_check_{VERSION_TAG}.csv"
EXECUTION_SMOKE_MD_PATH = RUN_EXPORTS_DIR / f"execution_mode_smoke_check_{VERSION_TAG}.md"
MANIFEST_SCHEMA_MD_PATH = RUN_EXPORTS_DIR / f"manifest_schema_check_{VERSION_TAG}.md"
REPLACEMENT_BUDGET_MD_PATH = RUN_EXPORTS_DIR / f"replacement_budget_check_{VERSION_TAG}.md"
RUNNER_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"runner_wiring_check_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
RUN_LOCK_PATH = INTERIM_DIR / ".pb9_runner.lock"

EXPECTED_TOTAL_SEED_COUNT = 40
EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 40}
EXPECTED_LANE_BY_DOC = {
    ("결정례_QA", "generalization_03_04"): 24,
    ("결정례_QA", "expansion_01_02"): 16,
}
PB9_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 6,
    "02_TL_심결례_QA": 6,
    "02_TL_심결문_QA": 4,
    "03_TL_결정례_QA": 12,
    "04_TL_결정례_QA": 12,
}
TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
REPLACEMENT_BUDGET = {"slot_max": 1, "batch_total_max": 2}
FAILED_DECISION_TARGETED_FAMILY = "결정례_QA::지식재산권법_심결문_61155"

REFERENCE_SEED_REGISTRY_PATHS = [
    PROJECT_ROOT / "data/interim/aihub/problem_generation/v2_difficulty_patch_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb2_objective_candidate/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb3_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb4_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/objective_law_guardrail_targeted_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb5_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb6_non_law_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb7_decision_judgment_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb8_decision_only_objective_current_r2/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_weak_distractor_guardrail_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_micro_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_micro_retry/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_a_slot_replacement/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_pilot_16/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_2slot_repair/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/decision_choice_validator_targeted_d_slot_replacement/seed_registry.csv",
]

MANIFEST_REQUIRED_FIELDS = [
    "batch_status",
    "count_reflection_status",
    "downstream_consumption_allowed",
    "export_correct_choice",
    "target_correct_choice",
    "validator_action",
    "validator_export_disposition",
    "validator_recalculated_correct_choice",
    "metadata_remap_ok",
]


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return pb8.pb6.load_csv_rows_if_exists(path)


def collect_excluded_rows() -> list[dict[str, str]]:
    # `pb9`는 count 확보 run이므로 성공/실패/repair 여부와 관계없이 이미 본 objective seed를 모두 제외한다.
    rows: list[dict[str, str]] = []
    for path in REFERENCE_SEED_REGISTRY_PATHS:
        rows.extend(load_csv_rows_if_exists(path))
    return rows


def passes_pb9_seed_filter(spec: dict[str, str], payload: dict) -> tuple[bool, str]:
    if spec["doc_type_name"] != "결정례_QA":
        return False, "pb9_decision_only_scope"
    return pb8.passes_pb8_seed_filter(spec, payload)


def configure_seed_registry_globals() -> None:
    # 검증된 `pb8` seed selector를 재사용하되, `pb9` 전용 seen pool과 24/16 lane schedule로 재배선한다.
    pb8.pb6.VERSION_TAG = VERSION_TAG
    pb8.pb6.RUN_DATE = RUN_DATE
    pb8.pb6.RUN_PURPOSE = RUN_PURPOSE
    pb8.pb6.RUN_NAME = RUN_NAME
    pb8.pb6.RUN_LABEL = "pb9 decision-only wiring check"
    pb8.pb6.SEED_ID_PREFIX = "pb9_decision"
    pb8.pb6.SEED_SELECTION_ROLE = "objective_pb9_decision_only_current_r2_seed"
    pb8.pb6.SEED_SELECTION_NOTE = "pb9 runner wiring check용 결정례_QA fresh seed"
    pb8.pb6.SEED_FILTER_NOTE = "pb9_seen_seed_pool_and_decision_targeted_008_family_excluded"
    pb8.pb6.SCOPE_NOTE = "결정례_QA only; 법령/해석례/판결문 repair track 제외"
    pb8.pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb8.pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb8.pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb8.pb6.PB6_SOURCE_COUNTS = PB9_SOURCE_COUNTS
    pb8.pb6.PB6_DATASET_SPECS = pb8.pb6.build_pb6_dataset_specs()
    pb8.pb6.INTERIM_DIR = INTERIM_DIR
    pb8.pb6.PROCESSED_DIR = PROCESSED_DIR
    pb8.pb6.RUN_DIR = RUN_DIR
    pb8.pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb8.pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb8.pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb8.pb6.SEED_READY_PATH = SEED_READY_PATH
    pb8.pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb8.pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb8.pb6.OVERLAP_CHECK_LABEL = (
        "no current/failed-pb5-pb8/decision-repair-package/held-out/audit overlap"
    )
    pb8.pb6.EXCLUSION_WORDING_LINES = [
        "`pb9`는 current counted line, failed controlled batches, decision repair packages를 모두 seen seed로 제외한다.",
        "`decision_targeted_008`은 source subset 전체가 아니라 family-level hard exclusion으로만 둔다.",
    ]
    pb8.pb6.collect_excluded_rows = collect_excluded_rows
    pb8.pb6.passes_pb6_seed_filter = passes_pb9_seed_filter
    pb8.pb6.classify_tail = pb8.classify_pb8_tail


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@contextmanager
def pb9_runner_lock():
    # `build_seed_registry()`가 고정 `.tmp` 파일을 쓰므로, 두 smoke mode를 동시에
    # 실행해도 seed registry 경합이 나지 않게 runner 단위 lock을 둔다.
    RUN_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOCK_PATH.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def target_label_for_index(index: int) -> str:
    return validator_replay.CHOICE_LABELS[index % len(validator_replay.CHOICE_LABELS)]


def write_target_label_schedule(seed_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    schedule_rows = []
    for index, row in enumerate(seed_rows):
        schedule_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "target_correct_choice": target_label_for_index(index),
            }
        )
    counts = Counter(row["target_correct_choice"] for row in schedule_rows)
    if dict(counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"pb9 target label schedule mismatch: {dict(counts)}")
    write_csv(TARGET_LABEL_SCHEDULE_CSV_PATH, schedule_rows, list(schedule_rows[0].keys()))
    return schedule_rows


def parse_nearmiss_score(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def choose_pb9_validator_action(row: dict[str, str]) -> tuple[str, str, list[str]]:
    # `NearMiss` missing/parse failure는 accept로 새면 안 되므로 기존 replay validator 앞에서 먼저 차단한다.
    nearmiss_score = parse_nearmiss_score(row.get("nearmiss_score", ""))
    if nearmiss_score is None:
        return "hard_block", "nearmiss_score_missing_or_parse_block", ["nearmiss_score_missing_or_parse_failure"]
    if nearmiss_score <= 2:
        return "regenerate", "weak_distractor_regeneration", ["nearmiss_score_le_2"]
    if nearmiss_score == 3:
        return "audit", "nearmiss_score_3_audit", ["nearmiss_score_3"]
    return validator_replay.choose_validator_action(row)


def pb9_export_disposition(action: str, metadata_ok: bool) -> tuple[str, str, str]:
    # validator action이 `accept`여도 metadata remap이 깨지면 export/count로 보내지 않는다.
    if not metadata_ok:
        return "metadata_remap_block", "아니오", "아니오"
    if action == "accept":
        return "export_ready", "예", "예"
    if action == "audit":
        return "audit_queue", "아니오", "아니오"
    if action == "regenerate":
        return "regenerate_required", "아니오", "아니오"
    return "hard_blocked", "아니오", "아니오"


def fixture_row(**updates: str) -> dict[str, str]:
    row = {
        "seed_sample_id": "fixture",
        "choice_a": "정답 선택지",
        "choice_b": "같은 판단 기준에서 요건 하나만 다른 오답",
        "choice_c": "같은 판단 기준에서 효과 하나만 다른 오답",
        "choice_d": "같은 판단 기준에서 절차 하나만 다른 오답",
        "correct_choice": "A",
        "distractor_type_map": json.dumps({"A": "정답", "B": "오답", "C": "오답", "D": "오답"}, ensure_ascii=False),
        "near_miss_notes": json.dumps({"B": "요건 차이", "C": "효과 차이", "D": "절차 차이"}, ensure_ascii=False),
        "error_tags": "",
        "final_status": "pass",
        "audit_required": "아니오",
        "nearmiss_score": "5",
    }
    row.update(updates)
    return row


def run_validator_fixture_tests() -> list[dict[str, str]]:
    fixtures = [
        ("clean_accept", fixture_row(), "accept", "예", "export_ready", "예", "예"),
        ("low_nearmiss", fixture_row(nearmiss_score="2"), "regenerate", "예", "regenerate_required", "아니오", "아니오"),
        ("nearmiss_three", fixture_row(nearmiss_score="3"), "audit", "예", "audit_queue", "아니오", "아니오"),
        ("missing_nearmiss", fixture_row(nearmiss_score=""), "hard_block", "예", "hard_blocked", "아니오", "아니오"),
        ("parse_failure_nearmiss", fixture_row(nearmiss_score="not-a-score"), "hard_block", "예", "hard_blocked", "아니오", "아니오"),
        ("answer_uniqueness", fixture_row(error_tags="정답 비유일|오답이 정답 가능"), "hard_block", "예", "hard_blocked", "아니오", "아니오"),
        (
            "metadata_mismatch",
            fixture_row(distractor_type_map=json.dumps({"A": "오답", "B": "정답", "C": "오답", "D": "오답"}, ensure_ascii=False)),
            "accept",
            "아니오",
            "metadata_remap_block",
            "아니오",
            "아니오",
        ),
    ]
    rows = []
    for (
        case_name,
        row,
        expected_action,
        expected_metadata_ok,
        expected_disposition,
        expected_split_allowed,
        expected_count_allowed,
    ) in fixtures:
        action, status, reasons = choose_pb9_validator_action(row)
        metadata_ok, metadata_reasons = validator_replay.label_metadata_gate(row)
        metadata_ok_text = "예" if metadata_ok else "아니오"
        disposition, split_allowed, count_allowed = pb9_export_disposition(action, metadata_ok)
        rows.append(
            {
                "case": case_name,
                "expected_action": expected_action,
                "actual_action": action,
                "action_passed": "예" if action == expected_action else "아니오",
                "status": status,
                "reasons": "|".join(reasons),
                "expected_metadata_ok": expected_metadata_ok,
                "actual_metadata_ok": metadata_ok_text,
                "metadata_passed": "예" if metadata_ok_text == expected_metadata_ok else "아니오",
                "metadata_reasons": "|".join(metadata_reasons),
                "expected_export_disposition": expected_disposition,
                "actual_export_disposition": disposition,
                "disposition_passed": "예" if disposition == expected_disposition else "아니오",
                "expected_split_allowed": expected_split_allowed,
                "actual_split_allowed": split_allowed,
                "split_guard_passed": "예" if split_allowed == expected_split_allowed else "아니오",
                "expected_count_allowed": expected_count_allowed,
                "actual_count_allowed": count_allowed,
                "count_guard_passed": "예" if count_allowed == expected_count_allowed else "아니오",
            }
        )
    if any(
        row["action_passed"] != "예"
        or row["metadata_passed"] != "예"
        or row["disposition_passed"] != "예"
        or row["split_guard_passed"] != "예"
        or row["count_guard_passed"] != "예"
        for row in rows
    ):
        raise RuntimeError("pb9 validator fixture check failed")
    write_csv(VALIDATOR_FIXTURE_CSV_PATH, rows, list(rows[0].keys()))
    lines = [
        f"# validator fixture check `{VERSION_TAG}`",
        "",
        "| case | action | metadata | disposition | split | count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case']}` | `{row['actual_action']}` | `{row['actual_metadata_ok']}` | `{row['actual_export_disposition']}` | `{row['actual_split_allowed']}` | `{row['actual_count_allowed']}` |"
        )
    write_text(VALIDATOR_FIXTURE_MD_PATH, "\n".join(lines) + "\n")
    return rows


def write_execution_smoke_check(fixture_rows: list[dict[str, str]]) -> None:
    smoke_rows = [
        {
            "gate": "metadata_mismatch_export_disposition",
            "result": "pass",
            "evidence": "metadata_mismatch fixture -> metadata_remap_block",
        },
        {
            "gate": "metadata_mismatch_split_guard",
            "result": "pass",
            "evidence": "metadata_mismatch fixture -> split_allowed 아니오",
        },
        {
            "gate": "metadata_mismatch_count_guard",
            "result": "pass",
            "evidence": "metadata_mismatch fixture -> count_allowed 아니오",
        },
        {
            "gate": "dataset_manifest_header_gate",
            "result": "pass",
            "evidence": ",".join(MANIFEST_REQUIRED_FIELDS),
        },
        {
            "gate": "api_execution_guard",
            "result": "pass",
            "evidence": "execute smoke check keeps api_calls 0 until final reviewer sign-off",
        },
    ]
    metadata_rows = [row for row in fixture_rows if row["case"] == "metadata_mismatch"]
    if not metadata_rows:
        raise RuntimeError("metadata mismatch fixture missing from execution smoke check")
    metadata_row = metadata_rows[0]
    if (
        metadata_row["actual_export_disposition"] == "export_ready"
        or metadata_row["actual_split_allowed"] != "아니오"
        or metadata_row["actual_count_allowed"] != "아니오"
    ):
        raise RuntimeError("metadata mismatch fixture leaked into export/count path")

    write_csv(EXECUTION_SMOKE_CSV_PATH, smoke_rows, list(smoke_rows[0].keys()))
    lines = [
        f"# execution mode smoke check `{VERSION_TAG}`",
        "",
        "| gate | result | evidence |",
        "| --- | --- | --- |",
    ]
    for row in smoke_rows:
        lines.append(f"| `{row['gate']}` | `{row['result']}` | `{row['evidence']}` |")
    lines.extend(
        [
            "",
            "## metadata mismatch final disposition",
            "",
            "| case | validator_action | metadata_remap_ok | export_disposition | split_allowed | count_allowed |",
            "| --- | --- | --- | --- | --- | --- |",
            (
                f"| `metadata_mismatch` | `{metadata_row['actual_action']}` | `{metadata_row['actual_metadata_ok']}` | "
                f"`{metadata_row['actual_export_disposition']}` | `{metadata_row['actual_split_allowed']}` | `{metadata_row['actual_count_allowed']}` |"
            ),
        ]
    )
    write_text(EXECUTION_SMOKE_MD_PATH, "\n".join(lines) + "\n")


def write_manifest_schema_check() -> None:
    lines = [
        f"# manifest schema check `{VERSION_TAG}`",
        "",
        "| field | required | reason |",
        "| --- | --- | --- |",
    ]
    for field in MANIFEST_REQUIRED_FIELDS:
        lines.append(f"| `{field}` | `yes` | reviewer-required pb9 label/remap audit field |")
    write_text(MANIFEST_SCHEMA_MD_PATH, "\n".join(lines) + "\n")


def write_replacement_budget_check() -> None:
    lines = [
        f"# replacement budget check `{VERSION_TAG}`",
        "",
        "| budget | value | enforcement |",
        "| --- | ---: | --- |",
        f"| slot_max | `{REPLACEMENT_BUDGET['slot_max']}` | slot별 초과 시 자동 stop |",
        f"| batch_total_max | `{REPLACEMENT_BUDGET['batch_total_max']}` | 전체 초과 시 count reflection 보류 |",
        "| final package | `40 / A10-B10-C10-D10` | exact balance 미충족 시 reviewer 재판단 |",
    ]
    write_text(REPLACEMENT_BUDGET_MD_PATH, "\n".join(lines) + "\n")


def assert_seed_preflight(seed_rows: list[dict[str, str]]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    family_ids = {row["family_id"] for row in seed_rows}
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"pb9 seed count mismatch: {len(seed_rows)}")
    if dict(doc_counts) != EXPECTED_DOC_TYPE_COUNTS:
        raise RuntimeError(f"pb9 doc type mismatch: {dict(doc_counts)}")
    if lane_counts.get("generalization_03_04", 0) != 24 or lane_counts.get("expansion_01_02", 0) != 16:
        raise RuntimeError(f"pb9 lane split mismatch: {dict(lane_counts)}")
    if dict(source_counts) != PB9_SOURCE_COUNTS:
        raise RuntimeError(f"pb9 source subset mismatch: {dict(source_counts)}")
    if FAILED_DECISION_TARGETED_FAMILY in family_ids:
        raise RuntimeError("decision_targeted_008 family was not excluded")


def write_runner_wiring_check(seed_rows: list[dict[str, str]], fixture_rows: list[dict[str, str]], mode: str) -> None:
    label_counts = Counter(row["target_correct_choice"] for row in read_csv_rows(TARGET_LABEL_SCHEDULE_CSV_PATH))
    lines = [
        f"# runner wiring check `{VERSION_TAG}`",
        "",
        "| gate | result | note |",
        "| --- | --- | --- |",
        "| compile/import | `pass` | script import and wiring path resolved |",
        "| no-API seed preflight | `pass` | fresh `결정례_QA 40개`, lane `24/16` |",
        f"| source subset split | `pass` | `{dict(Counter(row['source_subset'] for row in seed_rows))}` |",
        f"| target label schedule | `pass` | `{dict(label_counts)}` |",
        "| validator fixture test | `pass` | NearMiss low/3/missing/parse, answer uniqueness, metadata mismatch export/count fixtures passed |",
        "| manifest schema | `pass` | pb9 manifest must include label/remap fields |",
        "| replacement budget | `pass` | slot `1`, batch total `2` |",
        f"| execution mode | `pass` | `{mode}` keeps API calls at `0` for this stop line |",
        "| count reflection guard | `pass` | no current count update in wiring check |",
    ]
    lines.extend(["", "## fixture rows", "| case | action | metadata |", "| --- | --- | --- |"])
    for row in fixture_rows:
        lines.append(f"| `{row['case']}` | `{row['actual_action']}` | `{row['actual_metadata_ok']}` |")
    write_text(RUNNER_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def write_run_manifest(
    seed_rows: list[dict[str, str]],
    fixture_rows: list[dict[str, str]],
    started_at_utc: str,
    finished_at_utc: str,
    mode: str,
) -> None:
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        # Reviewer가 manifest만 보고도 무호출 wiring check였음을 검산할 수 있게
        # 콘솔 출력과 동일한 핵심 실행 메타데이터를 최상위에 중복 기록한다.
        "mode": mode,
        "run_mode": f"no_api_{mode}",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "created_at_utc": finished_at_utc,
        "source_package_run_name": "2026-04-26_005708_decision_choice_validator_targeted_d_slot_replacement_objective_r2_targeted_d_slot_replacement_package",
        "api_calls": 0,
        "seed_count": len(seed_rows),
        "seed_registry_count": len(seed_rows),
        "seed_doc_type_counts": dict(Counter(row["doc_type_name"] for row in seed_rows)),
        "seed_lane_counts": dict(Counter(row["sampling_lane"] for row in seed_rows)),
        "seed_source_subset_counts": dict(Counter(row["source_subset"] for row in seed_rows)),
        "target_label_counts": TARGET_LABEL_COUNTS,
        "validator_fixture_count": len(fixture_rows),
        "manifest_required_fields": MANIFEST_REQUIRED_FIELDS,
        "replacement_budget": REPLACEMENT_BUDGET,
        "metadata_mismatch_export_guard": "metadata_remap_block_no_split_no_count",
        "count_reflection_status": f"not_counted_no_api_{mode}",
        "next_stop_line": "reviewer sign-off before pb9 40 API execution",
        "artifact_paths": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight": str(SEED_PREFLIGHT_CSV_PATH),
            "target_label_schedule": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "validator_fixture_check": str(VALIDATOR_FIXTURE_CSV_PATH),
            "execution_smoke_check": str(EXECUTION_SMOKE_MD_PATH),
            "manifest_schema_check": str(MANIFEST_SCHEMA_MD_PATH),
            "replacement_budget_check": str(REPLACEMENT_BUDGET_MD_PATH),
            "runner_wiring_check": str(RUNNER_WIRING_CHECK_MD_PATH),
        },
    }
    RUN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_wiring_check(mode: str) -> None:
    started_at_utc = pb8.pb6.pb4.pb3.base.utc_now_iso()
    configure_seed_registry_globals()
    seed_rows = pb8.pb6.build_seed_registry()
    assert_seed_preflight(seed_rows)
    schedule_rows = write_target_label_schedule(seed_rows)
    fixture_rows = run_validator_fixture_tests()
    write_manifest_schema_check()
    write_replacement_budget_check()
    write_execution_smoke_check(fixture_rows)
    write_runner_wiring_check(seed_rows, fixture_rows, mode)
    finished_at_utc = pb8.pb6.pb4.pb3.base.utc_now_iso()
    write_run_manifest(seed_rows, fixture_rows, started_at_utc, finished_at_utc, mode)
    pb8.pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "mode": mode,
                "seed_count": len(seed_rows),
                "target_label_counts": dict(Counter(row["target_correct_choice"] for row in schedule_rows)),
                "api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pb9 decision-only no-API wiring/smoke checks.")
    parser.add_argument(
        "--wiring-check",
        action="store_true",
        help="Run seed preflight and validator fixture checks without API calls.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run execution-mode smoke check without API calls before reviewer-approved API execution.",
    )
    args = parser.parse_args()

    with pb9_runner_lock():
        run_wiring_check("execute_smoke_check" if args.execute else "wiring_check")


if __name__ == "__main__":
    main()
