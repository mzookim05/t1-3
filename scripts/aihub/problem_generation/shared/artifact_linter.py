from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.shared.production_batch_common import (  # noqa: E402
    ensure_dirs,
    write_csv_atomic,
    write_json_atomic,
    write_text_atomic,
)


COUNTED_BATCH_STATUS = "counted_current_production"
COUNTED_REFLECTION_STATUS = "counted"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
YES = "예"
NO = "아니오"

DEFAULT_FIXTURE_ROOT = PROJECT_ROOT / "local" / "fixtures" / "artifact_linter"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "artifact_linter"

INTERPRETATION_212921_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-26_212921_objective_interpretation_repair_dslot_final_replacement_package_objective_r2_interpretation_repair_remaining_dslot_fresh_replacement"
)
INTERPRETATION_205615_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-26_205615_objective_interpretation_repair_dslot_replacement_package_objective_r2_interpretation_repair_dslot_fresh_replacement"
)
PB9_FINAL_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-26_055123_pb9_cslot_final_replacement_package_objective_r2_pb9_remaining_cslot_salvage_package"
)
JUDGMENT_072939_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-26_072939_objective_judgment_repair_a_slot_replacement_package_objective_r2_judgment_repair_a_slot_fresh_replacement"
)


@dataclass
class Finding:
    fixture_id: str
    severity: str
    code: str
    path: str
    detail: str


@dataclass
class FixtureOutcome:
    fixture_id: str
    artifact_role: str
    fixture_mode: str
    expected_result: str
    expected_failure_code: str
    expected_failure_codes: list[str]
    expectation_matched: bool
    expected_failure_detected: bool
    unexpected_blocking_codes: list[str]
    unexpected_blocking_count: int
    expected_only_pass: bool
    fixture_passed: bool


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def add_missing_fields(rows: list[dict[str, str]], fieldnames: list[str], extra_fields: list[str]) -> list[str]:
    # fixture CSV는 실제 artifact의 일부 row만 복사하므로, stale status 재현에 필요한 필드를 보존한다.
    output_fields = list(fieldnames)
    for field in extra_fields:
        if field not in output_fields:
            output_fields.append(field)
        for row in rows:
            row.setdefault(field, "")
    return output_fields


def materialize_pre_hotfix_fixture(fixture_root: Path) -> None:
    fixture_dir = fixture_root / "212921_pre_hotfix_fail"
    ensure_dirs(fixture_dir)

    run_manifest = json.loads((INTERPRETATION_212921_RUN / "run_manifest.json").read_text(encoding="utf-8"))
    write_json_atomic(fixture_dir / "run_manifest.json", run_manifest)

    source_files = {
        "final_package.csv": INTERPRETATION_212921_RUN
        / "exports"
        / "final_package_objective_interpretation_repair_dslot_final_replacement_package.csv",
        "merged.csv": INTERPRETATION_212921_RUN
        / "merged"
        / "merged_problem_scores_objective_interpretation_repair_dslot_final_replacement_package.csv",
        "validator_report.csv": INTERPRETATION_212921_RUN
        / "exports"
        / "validator_report_objective_interpretation_repair_dslot_final_replacement_package.csv",
    }

    for output_name, source_path in source_files.items():
        rows, fieldnames = read_csv_rows(source_path)
        sample_size = 1 if output_name == "validator_report.csv" else 2
        sample_rows = [dict(row) for row in rows[:sample_size]]
        output_fields = add_missing_fields(
            sample_rows,
            fieldnames,
            [
                "batch_status",
                "count_reflection_status",
                "downstream_consumption_allowed",
                "count_disposition",
                "count_allowed",
            ],
        )
        for row in sample_rows:
            row["batch_status"] = "interpretation_repair_dslot_final_replacement_candidate_not_counted"
            row["count_reflection_status"] = "not_counted_until_reviewer_signoff"
            row["downstream_consumption_allowed"] = NO
            row["count_disposition"] = "candidate_not_counted"
            row["count_allowed"] = NO
        write_csv_atomic(fixture_dir / output_name, sample_rows, output_fields)

    write_text_atomic(
        fixture_dir / "manifest_header_gate.md",
        "\n".join(
            [
                "# pre-hotfix manifest header gate fixture",
                "",
                "| gate | result | note |",
                "| --- | --- | --- |",
                "| count reflection | `not_counted until reviewer sign-off` | stale pre-hotfix state |",
                "",
            ]
        ),
    )
    # compiler_summary markdown은 reviewer-facing count 결정 표면이라,
    # stale pre-signoff 문구가 다시 들어오면 별도 fixture로도 잡히게 한다.
    write_text_atomic(
        fixture_dir / "compiler_summary.md",
        "\n".join(
            [
                "# stale compiler summary fixture",
                "",
                "| field | value |",
                "| --- | --- |",
                "| package_role | `count_reflection_candidate_package` |",
                "| batch_status | `candidate_not_counted` |",
                "| count_reflection_status | `not_counted_until_reviewer_signoff` |",
                "| count_allowed | count_allowed = 아니오 |",
                "",
            ]
        ),
    )


def build_fixture_manifest(fixture_root: Path) -> list[dict[str, Any]]:
    pre_hotfix_dir = fixture_root / "212921_pre_hotfix_fail"
    return [
        {
            "fixture_id": "212921_pre_hotfix_fail",
            "artifact_role": "counted_final_package",
            "fixture_mode": "frozen_fixture",
            "expected_result": "fail",
            "expected_failure_code": "artifact_parity",
            "expected_failure_codes": ["artifact_parity", "stale_phrase"],
            "paths": {
                "run_manifest": repo_rel(pre_hotfix_dir / "run_manifest.json"),
                "final_package_csv": repo_rel(pre_hotfix_dir / "final_package.csv"),
                "merged_csv": repo_rel(pre_hotfix_dir / "merged.csv"),
                "validator_report_csv": repo_rel(pre_hotfix_dir / "validator_report.csv"),
                "header_gate_md": repo_rel(pre_hotfix_dir / "manifest_header_gate.md"),
                "compiler_summary_md": repo_rel(pre_hotfix_dir / "compiler_summary.md"),
            },
        },
        {
            "fixture_id": "212921_post_hotfix_pass",
            "artifact_role": "counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "paths": build_counted_paths(
                INTERPRETATION_212921_RUN,
                "objective_interpretation_repair_dslot_final_replacement_package",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "objective_interpretation_repair_dslot_final_replacement_package",
            ),
        },
        {
            "fixture_id": "212921_seed_preflight_snapshot_pass",
            "artifact_role": "pre_execution_snapshot",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "paths": {
                "snapshot_md": repo_rel(
                    INTERPRETATION_212921_RUN
                    / "exports"
                    / "seed_preflight_objective_interpretation_repair_dslot_final_replacement_package.md"
                )
            },
        },
        {
            "fixture_id": "205615_failed_package_not_counted_normal",
            "artifact_role": "failed_package_not_counted_normal",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "paths": build_failed_paths(
                INTERPRETATION_205615_RUN,
                "objective_interpretation_repair_dslot_replacement_package",
            ),
        },
        {
            "fixture_id": "pb9_cslot_final_replacement_package_pass",
            "artifact_role": "counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "paths": build_counted_paths(
                PB9_FINAL_RUN,
                "pb9_cslot_final_replacement_package",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "pb9_cslot_final_replacement_package",
            ),
        },
        {
            "fixture_id": "judgment_a_slot_replacement_package_pass",
            "artifact_role": "counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "paths": build_counted_paths(
                JUDGMENT_072939_RUN,
                "objective_judgment_repair_a_slot_replacement_package",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "objective_judgment_repair_a_slot_replacement_package",
            ),
        },
    ]


def build_counted_paths(run_dir: Path, version_tag: str, processed_dir: Path) -> dict[str, Any]:
    return {
        "run_manifest": repo_rel(run_dir / "run_manifest.json"),
        "processed_manifest": repo_rel(processed_dir / "dataset_manifest.csv"),
        "split_jsonl": [
            repo_rel(processed_dir / "train.jsonl"),
            repo_rel(processed_dir / "dev.jsonl"),
            repo_rel(processed_dir / "test.jsonl"),
        ],
        "final_package_csv": repo_rel(run_dir / "exports" / f"final_package_{version_tag}.csv"),
        "merged_csv": repo_rel(run_dir / "merged" / f"merged_problem_scores_{version_tag}.csv"),
        "validator_report_csv": repo_rel(run_dir / "exports" / f"validator_report_{version_tag}.csv"),
        "header_gate_md": repo_rel(run_dir / "exports" / f"manifest_header_gate_{version_tag}.md"),
        "final_package_md": repo_rel(run_dir / "exports" / f"final_package_{version_tag}.md"),
        "validator_report_md": repo_rel(run_dir / "exports" / f"validator_report_{version_tag}.md"),
    }


def build_failed_paths(run_dir: Path, version_tag: str) -> dict[str, Any]:
    return {
        "run_manifest": repo_rel(run_dir / "run_manifest.json"),
        "final_package_csv": repo_rel(run_dir / "exports" / f"final_package_{version_tag}.csv"),
        "merged_csv": repo_rel(run_dir / "merged" / f"merged_problem_scores_{version_tag}.csv"),
        "validator_report_csv": repo_rel(run_dir / "exports" / f"validator_report_{version_tag}.csv"),
        "header_gate_md": repo_rel(run_dir / "exports" / f"manifest_header_gate_{version_tag}.md"),
    }


def materialize_default_fixtures(fixture_root: Path) -> Path:
    ensure_dirs(fixture_root)
    materialize_pre_hotfix_fixture(fixture_root)
    manifest = {
        "fixture_version": "artifact_linter_minimal_v1",
        "description": "Count Promotion Contract v1 regression fixtures.",
        "fixtures": build_fixture_manifest(fixture_root),
    }
    manifest_path = fixture_root / "fixture_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def add_finding(findings: list[Finding], fixture_id: str, severity: str, code: str, path: Path | None, detail: str) -> None:
    findings.append(Finding(fixture_id, severity, code, repo_rel(path) if path else "", detail))


def lint_existing_path(findings: list[Finding], fixture_id: str, path: Path | None, code: str) -> bool:
    if path is None:
        add_finding(findings, fixture_id, "P2", code, None, "required artifact path is missing from fixture manifest")
        return False
    if not path.exists() or not path.is_file():
        add_finding(findings, fixture_id, "P2", code, path, "required artifact file does not exist")
        return False
    return True


def lint_manifest_counted(findings: list[Finding], fixture_id: str, path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_run_manifest"):
        return
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "batch_status": COUNTED_BATCH_STATUS,
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "downstream_consumption_allowed": YES,
    }
    for field, expected_value in expected.items():
        actual = str(payload.get(field, ""))
        if actual != expected_value:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "artifact_parity",
                path,
                f"run_manifest {field} expected {expected_value!r}, got {actual!r}",
            )


def lint_manifest_candidate(findings: list[Finding], fixture_id: str, path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_run_manifest"):
        return
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    # Candidate package는 reviewer sign-off 전 not-counted가 정상이며, 별도 compiler/promotion gate로 all-green 후보성을 표시한다.
    expected = {
        "package_role": "count_reflection_candidate_package",
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": "candidate_not_counted",
        "promotion_contract_passed": YES,
        "compiler_gate_passed": YES,
        "promotion_contract_status": "passed_not_counted",
    }
    for field, expected_value in expected.items():
        actual = str(payload.get(field, ""))
        if actual != expected_value:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "candidate_artifact_parity",
                path,
                f"run_manifest {field} expected {expected_value!r}, got {actual!r}",
            )


def lint_csv_counted(findings: list[Finding], fixture_id: str, path: Path, artifact_name: str) -> None:
    if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
        return
    rows, fieldnames = read_csv_rows(path)
    if not rows:
        add_finding(findings, fixture_id, "P2", "empty_artifact", path, f"{artifact_name} has no rows")
        return

    required_when_present = {
        "batch_status": COUNTED_BATCH_STATUS,
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "downstream_consumption_allowed": YES,
        "count_allowed": YES,
        "count_disposition": "counted",
        "validator_action": "accept",
        "validator_export_disposition": "export_ready",
        "metadata_remap_ok": YES,
    }
    contract_optional = {"count_allowed", "count_disposition"}

    for field, expected_value in required_when_present.items():
        if field not in fieldnames:
            severity = "P3" if field in contract_optional else "P2"
            add_finding(
                findings,
                fixture_id,
                severity,
                "missing_contract_field",
                path,
                f"{artifact_name} is missing {field}",
            )
            continue
        bad_values = sorted({row.get(field, "") for row in rows if row.get(field, "") != expected_value})
        if bad_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "artifact_parity",
                path,
                f"{artifact_name} {field} expected {expected_value!r}, got {bad_values[:5]!r}",
            )


def lint_csv_candidate(findings: list[Finding], fixture_id: str, path: Path, artifact_name: str) -> None:
    if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
        return
    rows, fieldnames = read_csv_rows(path)
    if not rows:
        add_finding(findings, fixture_id, "P2", "empty_artifact", path, f"{artifact_name} has no rows")
        return

    required_when_present = {
        "package_role": "count_reflection_candidate_package",
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": "candidate_not_counted",
        "promotion_contract_passed": YES,
        "compiler_gate_passed": YES,
        "final_status": "pass",
        "audit_required": NO,
        "train_eligible": YES,
        "validator_action": "accept",
        "validator_export_disposition": "export_ready",
        "metadata_remap_ok": YES,
    }

    for field, expected_value in required_when_present.items():
        if field not in fieldnames:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "missing_candidate_contract_field",
                path,
                f"{artifact_name} is missing {field}",
            )
            continue
        bad_values = sorted({row.get(field, "") for row in rows if row.get(field, "") != expected_value})
        if bad_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "candidate_artifact_parity",
                path,
                f"{artifact_name} {field} expected {expected_value!r}, got {bad_values[:5]!r}",
            )


def lint_pool_tail_split(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # package factory에서는 quota surplus가 자주 생기므로, 좋은 surplus와 실제 품질 tail을 자동으로 분리 검산한다.
    run_manifest_path = resolve_repo_path(paths.get("run_manifest"))
    rejected_path = resolve_repo_path(paths.get("rejected_pool_csv"))
    tail_path = resolve_repo_path(paths.get("tail_taxonomy_csv"))
    quota_path = resolve_repo_path(paths.get("quota_surplus_csv"))
    payload: dict[str, Any] = {}
    if lint_existing_path(findings, fixture_id, run_manifest_path, "missing_run_manifest"):
        assert run_manifest_path is not None
        payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    def required_rows(path: Path | None, code: str) -> tuple[list[dict[str, str]], list[str]]:
        if not lint_existing_path(findings, fixture_id, path, code):
            return [], []
        assert path is not None
        return read_csv_rows(path)

    rejected_rows, rejected_fields = required_rows(rejected_path, "missing_rejected_pool_csv")
    tail_rows, tail_fields = required_rows(tail_path, "missing_tail_taxonomy_csv")
    quota_rows, quota_fields = required_rows(quota_path, "missing_quota_surplus_csv")

    count_expectations = {
        "rejected_total": (len(rejected_rows), rejected_path),
        "quality_tail_total": (len(tail_rows), tail_path),
        "quota_surplus_total": (len(quota_rows), quota_path),
    }
    for field, (actual_count, path) in count_expectations.items():
        if field not in payload:
            add_finding(findings, fixture_id, "P2", "pool_tail_split", run_manifest_path, f"run_manifest is missing {field}")
            continue
        try:
            expected_count = int(payload.get(field))
        except (TypeError, ValueError):
            add_finding(findings, fixture_id, "P2", "pool_tail_split", run_manifest_path, f"run_manifest {field} is not an integer")
            continue
        if actual_count != expected_count:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "pool_tail_split",
                path,
                f"{field} expected {expected_count}, got {actual_count}",
            )

    required_rejected_fields = {"pool_class", "quality_failure", "tail_class", "not_selected_reason"}
    required_tail_fields = {"pool_class", "quality_failure", "tail_class", "not_selected_reason"}
    required_quota_fields = {
        "pool_class",
        "quality_failure",
        "tail_class",
        "not_selected_reason",
        "future_candidate_reusable",
        "candidate_reuse_policy",
    }
    for field in sorted(required_rejected_fields - set(rejected_fields)):
        add_finding(findings, fixture_id, "P2", "pool_tail_split", rejected_path, f"rejected_pool is missing {field}")
    for field in sorted(required_tail_fields - set(tail_fields)):
        add_finding(findings, fixture_id, "P2", "pool_tail_split", tail_path, f"tail_taxonomy is missing {field}")
    for field in sorted(required_quota_fields - set(quota_fields)):
        add_finding(findings, fixture_id, "P2", "pool_tail_split", quota_path, f"quota_surplus_pool is missing {field}")

    rejected_quality_count = 0
    rejected_quota_count = 0
    for row in rejected_rows:
        pool_class = row.get("pool_class", "")
        quality_failure = row.get("quality_failure", "")
        if pool_class == "quality_reject":
            rejected_quality_count += 1
            if quality_failure != YES:
                add_finding(findings, fixture_id, "P2", "pool_tail_split", rejected_path, "quality_reject row must have quality_failure = 예")
        elif pool_class == "quota_surplus":
            rejected_quota_count += 1
            if quality_failure != NO:
                add_finding(findings, fixture_id, "P2", "pool_tail_split", rejected_path, "quota_surplus row must have quality_failure = 아니오")
        else:
            add_finding(findings, fixture_id, "P2", "pool_tail_split", rejected_path, f"unsupported rejected pool_class: {pool_class!r}")

    if rejected_quality_count != len(tail_rows):
        add_finding(
            findings,
            fixture_id,
            "P2",
            "pool_tail_split",
            rejected_path,
            f"quality_reject rows in rejected_pool expected tail_taxonomy row count {len(tail_rows)}, got {rejected_quality_count}",
        )
    if rejected_quota_count != len(quota_rows):
        add_finding(
            findings,
            fixture_id,
            "P2",
            "pool_tail_split",
            rejected_path,
            f"quota_surplus rows in rejected_pool expected quota_surplus_pool row count {len(quota_rows)}, got {rejected_quota_count}",
        )

    for row in tail_rows:
        if row.get("pool_class") != "quality_reject" or row.get("quality_failure") != YES:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "pool_tail_split",
                tail_path,
                "tail_taxonomy must contain only quality_reject rows with quality_failure = 예",
            )
        if row.get("not_selected_reason") == "label_quota_filled" or row.get("tail_class") == "quota_surplus_not_quality_failure":
            add_finding(
                findings,
                fixture_id,
                "P2",
                "pool_tail_split",
                tail_path,
                "tail_taxonomy contains quota surplus leakage",
            )

    quota_expected = {
        "pool_class": "quota_surplus",
        "quality_failure": NO,
        "tail_class": "quota_surplus_not_quality_failure",
        "not_selected_reason": "label_quota_filled",
        "future_candidate_reusable": YES,
        "candidate_reuse_policy": "reuse_allowed_as_surplus_candidate",
    }
    for row in quota_rows:
        for field, expected_value in quota_expected.items():
            if row.get(field, "") != expected_value:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "pool_tail_split",
                    quota_path,
                    f"quota_surplus_pool {field} expected {expected_value!r}, got {row.get(field, '')!r}",
                )


def lint_jsonl_counted(findings: list[Finding], fixture_id: str, path: Path) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_split_jsonl"):
        return
    rows = read_jsonl_rows(path)
    for field, expected_value in {
        "batch_status": COUNTED_BATCH_STATUS,
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "downstream_consumption_allowed": YES,
        "count_allowed": YES,
        "count_disposition": "counted",
    }.items():
        if not rows:
            continue
        if field not in rows[0]:
            add_finding(findings, fixture_id, "P3", "missing_contract_field", path, f"split JSONL is missing {field}")
            continue
        bad_values = sorted({str(row.get(field, "")) for row in rows if str(row.get(field, "")) != expected_value})
        if bad_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "artifact_parity",
                path,
                f"split JSONL {field} expected {expected_value!r}, got {bad_values[:5]!r}",
            )


def lint_jsonl_candidate(findings: list[Finding], fixture_id: str, path: Path) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_split_jsonl"):
        return
    rows = read_jsonl_rows(path)
    for field, expected_value in {
        "package_role": "count_reflection_candidate_package",
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": "candidate_not_counted",
        "promotion_contract_passed": YES,
        "compiler_gate_passed": YES,
    }.items():
        if not rows:
            continue
        if field not in rows[0]:
            add_finding(findings, fixture_id, "P2", "missing_candidate_contract_field", path, f"split JSONL is missing {field}")
            continue
        bad_values = sorted({str(row.get(field, "")) for row in rows if str(row.get(field, "")) != expected_value})
        if bad_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "candidate_artifact_parity",
                path,
                f"split JSONL {field} expected {expected_value!r}, got {bad_values[:5]!r}",
            )


def lint_markdown_counted(findings: list[Finding], fixture_id: str, path: Path, artifact_name: str) -> None:
    if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
        return
    text = path.read_text(encoding="utf-8")
    stale_phrases = [
        "candidate_not_counted",
        "not_counted_until_reviewer_signoff",
        "not_counted until reviewer sign-off",
        "`not_counted`",
        "count_allowed = 아니오",
        "downstream_consumption_allowed = 아니오",
    ]
    hits = [phrase for phrase in stale_phrases if phrase in text]
    if hits:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "stale_phrase",
            path,
            f"{artifact_name} contains stale count-facing phrase(s): {hits}",
        )


def lint_markdown_candidate(findings: list[Finding], fixture_id: str, path: Path, artifact_name: str) -> None:
    if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
        return
    text = path.read_text(encoding="utf-8")
    required_phrases = [
        "count_reflection_candidate_package",
        CANDIDATE_REFLECTION_STATUS,
        "candidate_not_counted",
        "compiler_gate_passed",
        "promotion_contract_passed",
        "promotion_contract_status",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "candidate_markdown_contract",
            path,
            f"{artifact_name} is missing candidate contract phrase(s): {missing}",
        )
    counted_leaks = [phrase for phrase in [COUNTED_BATCH_STATUS, "`counted`", "count_allowed = 예"] if phrase in text]
    if counted_leaks:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "candidate_counted_leak",
            path,
            f"{artifact_name} contains counted-state phrase(s) before sign-off: {counted_leaks}",
        )


def lint_validator_wiring_check_md(findings: list[Finding], fixture_id: str, path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_validator_wiring_check_md"):
        return
    assert path is not None
    text = path.read_text(encoding="utf-8")
    # Overgeneration run은 candidate seed 수와 final package 수가 다르므로,
    # wiring artifact에서 두 숫자가 섞이면 source-preflight provenance가 깨진다.
    stale_phrases = ["no-API preflight 16개", "`A/B/C/D = 4/4/4/4` target 적용"]
    hits = [phrase for phrase in stale_phrases if phrase in text]
    if hits:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "validator_wiring_stale_scope",
            path,
            f"validator_wiring_check_md contains stale scope phrase(s): {hits}",
        )
    required_phrases = [
        "no-API preflight 28개와 같은 seed registry 사용",
        "candidate target `A/B/C/D = 7/7/7/7`",
        "final export `A/B/C/D = 4/4/4/4`",
    ]
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "validator_wiring_scope_missing",
            path,
            f"validator_wiring_check_md is missing scope phrase(s): {missing}",
        )


def lint_counted_final_package(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []

    lint_manifest_counted(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    for key, artifact_name in [
        ("processed_manifest", "processed_manifest"),
        ("final_package_csv", "final_package_csv"),
        ("merged_csv", "merged_csv"),
        ("validator_report_csv", "validator_report_csv"),
    ]:
        path = resolve_repo_path(paths.get(key))
        if path is not None:
            lint_csv_counted(findings, fixture_id, path, artifact_name)
        elif key != "processed_manifest":
            add_finding(findings, fixture_id, "P2", f"missing_{artifact_name}", None, f"{key} is required")

    for split_path_text in paths.get("split_jsonl", []) or []:
        path = resolve_repo_path(split_path_text)
        if path is not None:
            lint_jsonl_counted(findings, fixture_id, path)

    # compiler summary도 final package의 count-facing 요약이므로 stale counted/candidate 문구를 함께 차단한다.
    for key, artifact_name in [
        ("header_gate_md", "header_gate_md"),
        ("final_package_md", "final_package_md"),
        ("validator_report_md", "validator_report_md"),
        ("compiler_summary_md", "compiler_summary_md"),
    ]:
        path = resolve_repo_path(paths.get(key))
        if path is not None:
            lint_markdown_counted(findings, fixture_id, path, artifact_name)
    return findings


def lint_count_reflection_candidate_package(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []

    lint_manifest_candidate(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    for key, artifact_name in [
        ("processed_manifest", "processed_manifest"),
        ("final_package_csv", "final_package_csv"),
        ("merged_csv", "merged_csv"),
        ("validator_report_csv", "validator_report_csv"),
    ]:
        path = resolve_repo_path(paths.get(key))
        if path is not None:
            lint_csv_candidate(findings, fixture_id, path, artifact_name)
        else:
            add_finding(findings, fixture_id, "P2", f"missing_{artifact_name}", None, f"{key} is required")

    for split_path_text in paths.get("split_jsonl", []) or []:
        path = resolve_repo_path(split_path_text)
        if path is not None:
            lint_jsonl_candidate(findings, fixture_id, path)

    # candidate package도 compiler summary가 있으면 pre-signoff contract 문구를 검산한다.
    for key, artifact_name in [
        ("header_gate_md", "header_gate_md"),
        ("final_package_md", "final_package_md"),
        ("validator_report_md", "validator_report_md"),
        ("compiler_summary_md", "compiler_summary_md"),
    ]:
        path = resolve_repo_path(paths.get(key))
        if path is not None:
            lint_markdown_candidate(findings, fixture_id, path, artifact_name)
    lint_validator_wiring_check_md(findings, fixture_id, resolve_repo_path(paths.get("validator_wiring_check_md")))
    lint_pool_tail_split(findings, fixture_id, paths)
    return findings


def lint_snapshot_fixture(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []
    snapshot_path = resolve_repo_path(paths.get("snapshot_md"))
    if lint_existing_path(findings, fixture_id, snapshot_path, "missing_snapshot_artifact"):
        add_finding(
            findings,
            fixture_id,
            "INFO",
            "snapshot_exception",
            snapshot_path,
            "pre-execution snapshot is excluded from final count-facing stale phrase checks",
        )
    return findings


def lint_failed_package_fixture(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []
    for key in ["run_manifest", "final_package_csv", "merged_csv", "validator_report_csv", "header_gate_md"]:
        path = resolve_repo_path(paths.get(key))
        lint_existing_path(findings, fixture_id, path, f"missing_{key}")
    add_finding(
        findings,
        fixture_id,
        "INFO",
        "not_counted_normal",
        None,
        "failed/candidate package may legitimately remain not_counted outside final counted package gate",
    )
    return findings


def infer_fixture_mode(fixture: dict[str, Any]) -> str:
    # 예전 manifest도 읽을 수 있게 하되, 새 report에서는 frozen/live 구분을 항상 드러낸다.
    explicit_mode = str(fixture.get("fixture_mode", "")).strip()
    if explicit_mode:
        return explicit_mode
    paths = fixture.get("paths", {})
    flattened = json.dumps(paths, ensure_ascii=False)
    if "local/fixtures/" in flattened:
        return "frozen_fixture"
    return "live_artifact_check"


def lint_fixture(fixture: dict[str, Any]) -> tuple[list[Finding], FixtureOutcome]:
    role = fixture.get("artifact_role", "")
    fixture_id = fixture.get("fixture_id", "unknown")
    if role == "counted_final_package":
        findings = lint_counted_final_package(fixture)
    elif role == "count_reflection_candidate_package":
        findings = lint_count_reflection_candidate_package(fixture)
    elif role == "pre_execution_snapshot":
        findings = lint_snapshot_fixture(fixture)
    elif role == "failed_package_not_counted_normal":
        findings = lint_failed_package_fixture(fixture)
    else:
        findings = [
            Finding(
                fixture.get("fixture_id", "unknown"),
                "P2",
                "unknown_artifact_role",
                "",
                f"unsupported artifact role: {role}",
            )
        ]

    blocking_codes = {finding.code for finding in findings if finding.severity in {"P1", "P2"}}
    expected_result = fixture.get("expected_result", "pass")
    expected_failure_code = fixture.get("expected_failure_code", "")
    expected_failure_codes = [str(code) for code in fixture.get("expected_failure_codes", []) if str(code)]
    if expected_failure_code and str(expected_failure_code) not in expected_failure_codes:
        expected_failure_codes.insert(0, str(expected_failure_code))
    expected_failure_detected = bool(expected_failure_codes and set(expected_failure_codes).issubset(blocking_codes))
    if expected_result == "fail":
        passed = expected_failure_detected
    else:
        passed = not blocking_codes
    unexpected_blocking_codes = sorted(blocking_codes - set(expected_failure_codes))
    expected_only_pass = passed and not unexpected_blocking_codes
    outcome = FixtureOutcome(
        fixture_id=fixture_id,
        artifact_role=role,
        fixture_mode=infer_fixture_mode(fixture),
        expected_result=str(expected_result),
        expected_failure_code=str(expected_failure_code),
        expected_failure_codes=expected_failure_codes,
        expectation_matched=passed,
        expected_failure_detected=expected_failure_detected,
        unexpected_blocking_codes=unexpected_blocking_codes,
        unexpected_blocking_count=len(unexpected_blocking_codes),
        expected_only_pass=expected_only_pass,
        fixture_passed=passed,
    )
    return findings, outcome


def flatten_findings(
    findings_by_fixture: dict[str, list[Finding]],
    outcomes_by_fixture: dict[str, FixtureOutcome],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for fixture_id, findings in findings_by_fixture.items():
        outcome = outcomes_by_fixture[fixture_id]
        base = {
            "fixture_id": fixture_id,
            "artifact_role": outcome.artifact_role,
            "fixture_mode": outcome.fixture_mode,
            "expected_result": outcome.expected_result,
            "expected_failure_code": outcome.expected_failure_code,
            "expected_failure_codes": ";".join(outcome.expected_failure_codes),
            "expected_failure_detected": YES if outcome.expected_failure_detected else NO,
            "expectation_matched": YES if outcome.expectation_matched else NO,
            "unexpected_blocking_codes": ";".join(outcome.unexpected_blocking_codes),
            "unexpected_blocking_count": str(outcome.unexpected_blocking_count),
            "expected_only_pass": YES if outcome.expected_only_pass else NO,
            "fixture_passed": str(outcome.fixture_passed).lower(),
        }
        if not findings:
            rows.append({**base, "severity": "INFO", "code": "no_findings", "path": "", "detail": "no findings"})
            continue
        for finding in findings:
            rows.append(
                {
                    **base,
                    "severity": finding.severity,
                    "code": finding.code,
                    "path": finding.path,
                    "detail": finding.detail,
                }
            )
    return rows


def render_markdown_report(
    manifest_path: Path,
    output_dir: Path,
    findings_by_fixture: dict[str, list[Finding]],
    outcomes_by_fixture: dict[str, FixtureOutcome],
) -> str:
    total = len(outcomes_by_fixture)
    passed = sum(1 for outcome in outcomes_by_fixture.values() if outcome.fixture_passed)
    failed = total - passed
    lines = [
        "# artifact linter minimal dry-run report",
        "",
        f"- fixture_manifest: `{repo_rel(manifest_path)}`",
        f"- output_dir: `{repo_rel(output_dir)}`",
        f"- fixture_total: `{total}`",
        f"- fixture_passed: `{passed}`",
        f"- fixture_failed: `{failed}`",
        "",
        "## fixture summary",
        "",
        "| fixture | mode | expected | expected codes | expected failure detected | expectation matched | unexpected blocking | expected-only pass | passed | blocking findings | p3 findings | info findings |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for fixture_id, findings in findings_by_fixture.items():
        outcome = outcomes_by_fixture[fixture_id]
        blocking = sum(1 for finding in findings if finding.severity in {"P1", "P2"})
        p3 = sum(1 for finding in findings if finding.severity == "P3")
        info = sum(1 for finding in findings if finding.severity == "INFO")
        lines.append(
            "| "
            f"`{fixture_id}` | `{outcome.fixture_mode}` | `{outcome.expected_result}` | `{';'.join(outcome.expected_failure_codes)}` | "
            f"`{YES if outcome.expected_failure_detected else NO}` | `{YES if outcome.expectation_matched else NO}` | "
            f"`{outcome.unexpected_blocking_count}` | `{YES if outcome.expected_only_pass else NO}` | "
            f"`{str(outcome.fixture_passed).lower()}` | `{blocking}` | `{p3}` | `{info}` |"
        )

    lines.extend(
        [
            "",
            "## findings",
            "",
            "| fixture | mode | expected | expectation matched | severity | code | path | detail |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for fixture_id, findings in findings_by_fixture.items():
        outcome = outcomes_by_fixture[fixture_id]
        expectation = f"{outcome.expected_result}/{';'.join(outcome.expected_failure_codes) or '-'}"
        matched = YES if outcome.expectation_matched else NO
        if not findings:
            lines.append(
                f"| `{fixture_id}` | `{outcome.fixture_mode}` | `{expectation}` | `{matched}` | `INFO` | `no_findings` |  | no findings |"
            )
            continue
        for finding in findings:
            safe_detail = finding.detail.replace("|", "\\|")
            lines.append(
                f"| `{fixture_id}` | `{outcome.fixture_mode}` | `{expectation}` | `{matched}` | "
                f"`{finding.severity}` | `{finding.code}` | `{finding.path}` | {safe_detail} |"
            )
    lines.append("")
    return "\n".join(lines)


def run_linter(manifest_path: Path, output_dir: Path) -> bool:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures", [])
    findings_by_fixture: dict[str, list[Finding]] = {}
    outcomes_by_fixture: dict[str, FixtureOutcome] = {}
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id", "unknown")
        findings, outcome = lint_fixture(fixture)
        findings_by_fixture[fixture_id] = findings
        outcomes_by_fixture[fixture_id] = outcome

    ensure_dirs(output_dir)
    flat_rows = flatten_findings(findings_by_fixture, outcomes_by_fixture)
    write_csv_atomic(
        output_dir / "artifact_linter_report.csv",
        flat_rows,
        [
            "fixture_id",
            "artifact_role",
            "fixture_mode",
            "expected_result",
            "expected_failure_code",
            "expected_failure_codes",
            "expected_failure_detected",
            "expectation_matched",
            "unexpected_blocking_codes",
            "unexpected_blocking_count",
            "expected_only_pass",
            "fixture_passed",
            "severity",
            "code",
            "path",
            "detail",
        ],
    )
    # Summary JSON은 evidence card가 바로 읽을 수 있도록 fixture expectation 메타데이터를 함께 둔다.
    fixture_summary = {
        fixture_id: {
            "artifact_role": outcome.artifact_role,
            "fixture_mode": outcome.fixture_mode,
            "expected_result": outcome.expected_result,
            "expected_failure_code": outcome.expected_failure_code,
            "expected_failure_codes": outcome.expected_failure_codes,
            "expected_failure_detected": outcome.expected_failure_detected,
            "expectation_matched": outcome.expectation_matched,
            "unexpected_blocking_codes": outcome.unexpected_blocking_codes,
            "unexpected_blocking_count": outcome.unexpected_blocking_count,
            "expected_only_pass": outcome.expected_only_pass,
            "passed": outcome.fixture_passed,
        }
        for fixture_id, outcome in outcomes_by_fixture.items()
    }
    unexpected_blocking_count_total = sum(outcome.unexpected_blocking_count for outcome in outcomes_by_fixture.values())
    write_json_atomic(
        output_dir / "artifact_linter_summary.json",
        {
            "fixture_manifest": repo_rel(manifest_path),
            "fixture_total": len(outcomes_by_fixture),
            "fixture_passed": sum(1 for outcome in outcomes_by_fixture.values() if outcome.fixture_passed),
            "fixture_failed": sum(1 for outcome in outcomes_by_fixture.values() if not outcome.fixture_passed),
            "blocking_finding_count": sum(
                1 for findings in findings_by_fixture.values() for finding in findings if finding.severity in {"P1", "P2"}
            ),
            "p3_finding_count": sum(1 for findings in findings_by_fixture.values() for finding in findings if finding.severity == "P3"),
            # Expected-fail fixtures may intentionally carry P1/P2 findings, so this alias exposes only unexpected blockers.
            "unexpected_blocking_count_total": unexpected_blocking_count_total,
            "passed": all(outcome.fixture_passed for outcome in outcomes_by_fixture.values()),
            "fixtures": fixture_summary,
        },
    )
    write_text_atomic(
        output_dir / "artifact_linter_report.md",
        render_markdown_report(manifest_path, output_dir, findings_by_fixture, outcomes_by_fixture),
    )
    return all(outcome.fixture_passed for outcome in outcomes_by_fixture.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AI Hub problem-generation artifact linter.")
    parser.add_argument("--fixture-root", type=Path, default=DEFAULT_FIXTURE_ROOT)
    parser.add_argument("--fixture-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--materialize-default-fixtures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixture_manifest = args.fixture_manifest
    if args.materialize_default_fixtures:
        fixture_manifest = materialize_default_fixtures(args.fixture_root)
    if fixture_manifest is None:
        fixture_manifest = args.fixture_root / "fixture_manifest.json"
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{build_run_stamp()}_artifact_linter_minimal_dry_run"
    passed = run_linter(fixture_manifest, output_dir)
    print(f"artifact_linter_report={output_dir / 'artifact_linter_report.md'}")
    print(f"artifact_linter_passed={str(passed).lower()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
