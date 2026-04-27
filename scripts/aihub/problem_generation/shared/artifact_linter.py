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
DECISION_MEDIUM_062200_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_062200_objective_decision_medium_overgeneration_pilot_objective_r2_decision_target40_candidate64_api_execution"
)
DECISION_ADDON_071841_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_071841_objective_decision_addon_overgeneration_pilot_objective_r2_decision_target24_candidate40_api_execution"
)
DESCRIPTIVE_WAVE_090909_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_090909_descriptive_wave_v2_constrained_descriptive_v3_split_lock_candidate34_primary24_fallback20_api_execution"
)
DESCRIPTIVE_FOLLOWUP_095340_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_095340_descriptive_wave_v2_followup_constrained_descriptive_v3_split_lock_followup_target20_candidate34_api_execution"
)
DESCRIPTIVE_SECOND_FOLLOWUP_105251_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_105251_descriptive_wave_v2_second_followup_constrained_descriptive_v3_split_lock_second_followup_target24_candidate34_40_api_execution"
)
DESCRIPTIVE_HOTFIX_NEXT_API_113712_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_113712_descriptive_wave_v2_split_lock_hotfix_next_api_descriptive_v3_split_lock_eval_hotfix_medium_primary_constrained_fallback_api_execution"
)
DESCRIPTIVE_TAIL_SYNC_FOLLOWUP_122242_RUN = (
    PROJECT_ROOT
    / "analysis"
    / "aihub"
    / "problem_generation"
    / "llm_runs"
    / "2026-04-27_122242_descriptive_tail_manifest_sync_constrained_followup_descriptive_v3_tail_manifest_sync_constrained_followup_api_execution"
)


def latest_llm_run_dir(name_fragment: str) -> Path:
    # 새 API package는 실행 시각이 매번 달라지므로 fixture가 날짜/시간 prefix에 묶이지 않게 최신 run dir를 찾는다.
    candidates = sorted((PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs").glob(f"*{name_fragment}*"))
    if not candidates:
        return PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / f"MISSING_{name_fragment}"
    return candidates[-1]


DESCRIPTIVE_MEDIUM_SYNC_RUN = latest_llm_run_dir(
    "descriptive_inventory_linter_pointer_sync_medium_primary_descriptive_v3_availability_map_medium_primary_api_execution"
)
DESCRIPTIVE_MEDIUM_REPEAT_RUN = latest_llm_run_dir(
    "descriptive_medium_repeat_availability_aware_descriptive_v3_medium_repeat_target40_candidate56_64_api_execution"
)
DESCRIPTIVE_EMERGENCY_154712_RUN = latest_llm_run_dir(
    "descriptive_emergency_candidate128_final80_descriptive_v3_emergency_candidate128_final80_judge16_with_candidate96_64_fallback_api_execution"
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
        {
            "fixture_id": "decision_medium_overgeneration_counted_package_pass",
            "artifact_role": "counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "validator_wiring_expectations": {
                "stale_phrases": ["reviewer sign-off 전 core current count 미변경"],
                "required_phrases": [
                    "`2026-04-27_062129_objective_decision_medium_overgeneration_pilot_preflight_objective_r2_decision_target40_candidate64_seed_spec_wiring_check` seed registry 사용",
                    "candidate target `A/B/C/D = 16/16/16/16`",
                    "final export `A/B/C/D = 10/10/10/10`",
                    "reviewer sign-off 이후 no-API/API-first count reflection 반영 완료",
                ],
            },
            "paths": build_package_factory_counted_paths(
                DECISION_MEDIUM_062200_RUN,
                "objective_decision_medium_overgeneration_pilot",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "objective_decision_medium_overgeneration_pilot",
                seed_preflight=True,
                evidence=True,
            ),
        },
        {
            "fixture_id": "decision_addon_overgeneration_counted_package_pass",
            "artifact_role": "counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "validator_wiring_expectations": {
                "stale_phrases": ["reviewer sign-off 전 core current count 미변경"],
                "required_phrases": [
                    "`2026-04-27_071829_objective_decision_addon_overgeneration_preflight_objective_r2_decision_target24_candidate40_seed_spec_wiring_check` seed registry 사용",
                    "candidate target `A/B/C/D = 10/10/10/10`",
                    "final export `A/B/C/D = 6/6/6/6`",
                    "reviewer sign-off 이후 no-API/API-first count reflection 반영 완료",
                ],
            },
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "same 16 seed registry",
                    "source split is 01/02/03/04 each 4",
                    "lane split is 8/8",
                    "target label schedule is A/B/C/D = 4/4/4/4",
                ],
                "required_phrases": [
                    "same 40 seed registry",
                    "8/6/6/10/10",
                    "lane split is 20/20",
                    "target label schedule is A/B/C/D = 10/10/10/10",
                ],
            },
            "paths": build_package_factory_counted_paths(
                DECISION_ADDON_071841_RUN,
                "objective_decision_addon_overgeneration_pilot",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "objective_decision_addon_overgeneration_pilot",
                seed_preflight=True,
                evidence=True,
            ),
        },
        {
            "fixture_id": "descriptive_wave_v2_constrained_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "2026-04-27_091842_descriptive_wave_v2_constrained",
                    "2026-04-27_091814_descriptive_wave_v2_constrained",
                    "2026-04-27_091755_descriptive_wave_v2_constrained",
                ],
                "required_phrases": [
                    "candidate_count: `34`",
                    "fallback_final_target_count: `20`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_WAVE_090909_RUN,
                "descriptive_wave_v2_constrained",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_wave_v2_constrained",
            ),
        },
        {
            "fixture_id": "descriptive_wave_v2_followup_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "2026-04-27_091842_descriptive_wave_v2_constrained",
                    "2026-04-27_090909_descriptive_wave_v2_constrained",
                ],
                "required_phrases": [
                    "candidate_count: `34`",
                    "fallback_final_target_count: `20`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_FOLLOWUP_095340_RUN,
                "descriptive_wave_v2_followup_constrained",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_wave_v2_followup_constrained",
            ),
        },
        {
            "fixture_id": "descriptive_wave_v2_second_followup_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "2026-04-27_091842_descriptive_wave_v2_constrained",
                    "2026-04-27_090909_descriptive_wave_v2_constrained",
                    "2026-04-27_095340_descriptive_wave_v2_followup_constrained",
                ],
                "required_phrases": [
                    "candidate_count: `34`",
                    "final_target_count: `24`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_SECOND_FOLLOWUP_105251_RUN,
                "descriptive_wave_v2_second_followup_constrained",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_wave_v2_second_followup_constrained",
            ),
        },
        {
            "fixture_id": "descriptive_wave_v2_split_lock_hotfix_next_api_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "descriptive_v3_split_lock_second_followup_target24_candidate34_40_api_execution",
                    "descriptive_wave_v2_second_followup_constrained_api_execution",
                ],
                "required_phrases": [
                    "candidate_count: `34`",
                    "final_target_count: `24`",
                    "fallback_final_target_count: `20`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_HOTFIX_NEXT_API_113712_RUN,
                "descriptive_wave_v2_split_lock_hotfix_next_api",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_wave_v2_split_lock_hotfix_next_api",
            ),
        },
        {
            "fixture_id": "descriptive_tail_manifest_sync_constrained_followup_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "primary_target40_candidate56",
                    "fallback_target24_candidate36",
                ],
                "required_phrases": [
                    "candidate_count: `34`",
                    "final_target_count: `24`",
                    "fallback_final_target_count: `20`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_TAIL_SYNC_FOLLOWUP_122242_RUN,
                "descriptive_tail_manifest_sync_constrained_followup",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_tail_manifest_sync_constrained_followup",
            ),
        },
        {
            "fixture_id": "descriptive_inventory_linter_pointer_sync_medium_primary_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "medium route: `보류`",
                    "constrained_candidate34_primary24_fallback20",
                ],
                "required_phrases": [
                    "candidate_count: `56`",
                    "final_target_count: `40`",
                    "source_balance_relaxation",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_MEDIUM_SYNC_RUN,
                "descriptive_inventory_linter_pointer_sync_medium_primary",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_inventory_linter_pointer_sync_medium_primary",
            ),
        },
        {
            "fixture_id": "descriptive_medium_repeat_availability_aware_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "constrained_candidate34_primary24_fallback20",
                    "candidate_count: `56`",
                ],
                "required_phrases": [
                    "candidate_count: `64`",
                    "final_target_count: `40`",
                    "medium_source_relaxed_candidate64_final40",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_MEDIUM_REPEAT_RUN,
                "descriptive_medium_repeat_availability_aware",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_medium_repeat_availability_aware",
            ),
        },
        {
            "fixture_id": "descriptive_emergency_candidate128_final80_counted_package_pass",
            "artifact_role": "descriptive_counted_final_package",
            "fixture_mode": "live_artifact_check",
            "expected_result": "pass",
            "expected_failure_code": "",
            "expected_failure_codes": [],
            "seed_preflight_expectations": {
                "stale_phrases": [
                    "candidate_count: `64`",
                    "final_target_count: `40`",
                ],
                "required_phrases": [
                    "candidate_count: `128`",
                    "final_target_count: `80`",
                    "final_split_target: `train 64 / dev 8 / test 8`",
                    "Tier 0 fresh 또는 Tier 2 train-only split-lock만 허용",
                ],
            },
            "paths": build_descriptive_factory_counted_paths(
                DESCRIPTIVE_EMERGENCY_154712_RUN,
                "descriptive_emergency_candidate128_final80",
                processed_dir=PROJECT_ROOT
                / "data"
                / "processed"
                / "aihub"
                / "problem_generation"
                / "production_batches"
                / "descriptive_emergency_candidate128_final80",
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


def build_package_factory_counted_paths(
    run_dir: Path,
    version_tag: str,
    processed_dir: Path,
    *,
    seed_preflight: bool = False,
    evidence: bool = False,
) -> dict[str, Any]:
    # Package factory run은 final package 외에도 pool/tail/surplus와 reviewer-facing gate가 핵심 증거라 함께 검산한다.
    paths = build_counted_paths(run_dir, version_tag, processed_dir)
    paths.update(
        {
            "candidate_pool_csv": repo_rel(run_dir / "candidate_pool.csv"),
            "accepted_pool_csv": repo_rel(run_dir / "accepted_pool.csv"),
            "rejected_pool_csv": repo_rel(run_dir / "rejected_pool.csv"),
            "tail_taxonomy_csv": repo_rel(run_dir / "tail_taxonomy.csv"),
            "quota_surplus_csv": repo_rel(run_dir / "quota_surplus_pool.csv"),
            "validator_wiring_check_md": repo_rel(run_dir / "exports" / f"validator_wiring_check_{version_tag}.md"),
            "compiler_summary_md": repo_rel(run_dir / "exports" / f"compiler_summary_{version_tag}.md"),
        }
    )
    if seed_preflight:
        paths["seed_preflight_md"] = repo_rel(run_dir / "exports" / f"seed_preflight_{version_tag}.md")
    if evidence:
        paths["evidence_summary_json"] = repo_rel(run_dir / "evidence_card" / "evidence_card_summary.json")
        paths["evidence_card_package_manifest"] = repo_rel(run_dir / "evidence_card_package_manifest.json")
    return paths


def build_descriptive_factory_counted_paths(
    run_dir: Path,
    version_tag: str,
    processed_dir: Path,
) -> dict[str, Any]:
    # 서술형 package factory도 objective와 같은 pool/evidence surface를 검산해
    # strict final package 밖 row가 count로 새지 않도록 막는다.
    paths = {
        "run_manifest": repo_rel(run_dir / "run_manifest.json"),
        "processed_manifest": repo_rel(processed_dir / "dataset_manifest.csv"),
        "split_jsonl": [
            repo_rel(processed_dir / "train.jsonl"),
            repo_rel(processed_dir / "dev.jsonl"),
            repo_rel(processed_dir / "test.jsonl"),
        ],
        "final_package_csv": repo_rel(run_dir / "exports" / f"final_package_{version_tag}.csv"),
        "merged_csv": repo_rel(run_dir / "merged" / f"merged_problem_scores_{version_tag}.csv"),
        "candidate_pool_csv": repo_rel(run_dir / "candidate_pool.csv"),
        "accepted_pool_csv": repo_rel(run_dir / "accepted_pool.csv"),
        "rejected_pool_csv": repo_rel(run_dir / "rejected_pool.csv"),
        "tail_taxonomy_csv": repo_rel(run_dir / "tail_taxonomy.csv"),
        "quota_surplus_csv": repo_rel(run_dir / "quota_surplus_pool.csv"),
        "compiler_manifest_json": repo_rel(run_dir / "compiler_manifest.json"),
        "seed_registry_csv": repo_rel(
            PROJECT_ROOT
            / "data"
            / "interim"
            / "aihub"
            / "problem_generation"
            / "production_batches"
            / version_tag
            / "seed_registry.csv"
        ),
        "seed_preflight_md": repo_rel(run_dir / "exports" / f"seed_preflight_{version_tag}.md"),
    }
    # evidence card는 linter output을 입력으로 삼으므로 최초 linter pass 전에는 없을 수 있다.
    # 파일이 생긴 뒤 다시 materialize하면 pointer parity까지 P2 gate로 검산한다.
    evidence_summary_path = run_dir / "evidence_card" / "evidence_card_summary.json"
    evidence_manifest_path = run_dir / "evidence_card_package_manifest.json"
    if evidence_summary_path.exists():
        paths["evidence_summary_json"] = repo_rel(evidence_summary_path)
    if evidence_manifest_path.exists():
        paths["evidence_card_package_manifest"] = repo_rel(evidence_manifest_path)
    return paths


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


def lint_manifest_evidence_alias(
    findings: list[Finding],
    fixture_id: str,
    run_manifest_path: Path | None,
    evidence_summary_path: Path | None,
) -> None:
    if not lint_existing_path(findings, fixture_id, run_manifest_path, "missing_run_manifest"):
        return
    if not lint_existing_path(findings, fixture_id, evidence_summary_path, "missing_evidence_summary_json"):
        return
    assert run_manifest_path is not None
    assert evidence_summary_path is not None
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_summary_path.read_text(encoding="utf-8"))
    package_total = int(evidence.get("package_total", 0) or 0)
    all_green_total = int(evidence.get("all_green_total", 0) or 0)
    expected_all_green = package_total > 0 and package_total == all_green_total
    packages = evidence.get("packages", []) or []
    expected_linter_passed = bool(packages) and all(bool(package.get("linter_passed")) for package in packages)

    # Reviewer-facing manifest alias는 evidence card와 linter의 짧은 handoff 표면이라 stale 값이면 count 상태를 오해하게 된다.
    for field, expected_value in {
        "artifact_linter_passed": expected_linter_passed,
        "evidence_card_passed": expected_all_green,
        "evidence_card_all_green": expected_all_green,
    }.items():
        if field not in manifest:
            add_finding(
                findings,
                fixture_id,
                "P3",
                "missing_manifest_alias",
                run_manifest_path,
                f"run_manifest is missing top-level {field}",
            )
            continue
        if bool(manifest.get(field)) != expected_value:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "evidence_alias_mismatch",
                run_manifest_path,
                f"run_manifest {field} expected {expected_value!r} from evidence summary, got {manifest.get(field)!r}",
            )


def canonical_linter_report_dir(run_manifest: dict[str, Any]) -> str:
    # run manifest의 linter report 파일을 canonical proof로 삼고, evidence 쪽은 같은 디렉토리를 가리켜야 한다.
    artifact_paths = run_manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, dict):
        return ""
    linter_report = artifact_paths.get("linter_report")
    if not isinstance(linter_report, str) or not linter_report:
        return ""
    return repo_rel(resolve_repo_path(linter_report).parent)


def lint_evidence_linter_pointer_parity(
    findings: list[Finding],
    fixture_id: str,
    run_manifest_path: Path | None,
    evidence_summary_path: Path | None,
    evidence_package_manifest_path: Path | None,
) -> None:
    if not lint_existing_path(findings, fixture_id, run_manifest_path, "missing_run_manifest"):
        return
    assert run_manifest_path is not None
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    expected_dir = canonical_linter_report_dir(run_manifest)
    if not expected_dir:
        return

    # evidence card는 reviewer handoff의 표면이라, run manifest와 다른 linter proof를 가리키면 P2로 막는다.
    for artifact_name, path in [
        ("evidence_summary_json", evidence_summary_path),
        ("evidence_card_package_manifest", evidence_package_manifest_path),
    ]:
        if path is None:
            continue
        if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        packages = payload.get("packages", [])
        if not isinstance(packages, list):
            continue
        for index, package in enumerate(packages):
            if not isinstance(package, dict):
                continue
            actual_dir = str(package.get("linter_report_dir", "") or "")
            if actual_dir and actual_dir != expected_dir:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "evidence_linter_pointer_parity",
                    path,
                    f"{artifact_name} package[{index}].linter_report_dir expected {expected_dir!r}, got {actual_dir!r}",
                )


def lint_manifest_artifact_paths(findings: list[Finding], fixture_id: str, run_manifest_path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, run_manifest_path, "missing_run_manifest"):
        return
    assert run_manifest_path is not None
    payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    artifact_paths = payload.get("artifact_paths", {})
    if not isinstance(artifact_paths, dict):
        add_finding(findings, fixture_id, "P2", "manifest_artifact_paths", run_manifest_path, "run_manifest artifact_paths must be an object")
        return
    for key, value in artifact_paths.items():
        if isinstance(value, list):
            values = value
        else:
            values = [value]
        for item in values:
            if not isinstance(item, str) or not item:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "manifest_artifact_paths",
                    run_manifest_path,
                    f"artifact_paths.{key} must be a non-empty path string",
                )
                continue
            artifact_path = resolve_repo_path(item)
            if not artifact_path.exists():
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "manifest_artifact_paths",
                    artifact_path,
                    f"run_manifest artifact_paths.{key} points to a missing file",
                )


def lint_evidence_package_manifest(findings: list[Finding], fixture_id: str, path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_evidence_card_package_manifest"):
        return
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("packages", [])
    if not isinstance(packages, list) or not packages:
        add_finding(findings, fixture_id, "P2", "evidence_package_manifest", path, "evidence card package manifest has no packages")
        return
    required = {"package_id", "run_name", "version_tag", "package_role", "run_dir", "processed_package_dir", "linter_fixture_id", "linter_report_dir"}
    for index, package in enumerate(packages):
        if not isinstance(package, dict):
            add_finding(findings, fixture_id, "P2", "evidence_package_manifest", path, f"package entry {index} is not an object")
            continue
        missing = sorted(required - set(package))
        if missing:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "evidence_package_manifest",
                path,
                f"package entry {index} is missing field(s): {missing}",
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
        "promotion_contract_status",
    }
    for field in sorted(required_rejected_fields - set(rejected_fields)):
        add_finding(findings, fixture_id, "P2", "pool_tail_split", rejected_path, f"rejected_pool is missing {field}")
    # tail이 0건인 all-green package는 header만 `empty`일 수 있으므로, row가 있을 때만 tail field contract를 강제한다.
    if tail_rows:
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
        "future_candidate_reusable": YES,
        "promotion_contract_status": "not_promoted_quota_surplus",
    }
    allowed_not_selected_reasons = {"label_quota_filled", "source_quota_filled"}
    allowed_reuse_policies = {"reuse_allowed_as_surplus_candidate", "reuse_allowed_as_surplus_candidate_after_dedup"}

    def lint_quota_row(row: dict[str, str], path: Path | None, artifact_name: str) -> None:
        for field, expected_value in quota_expected.items():
            if row.get(field, "") != expected_value:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "pool_tail_split",
                    path,
                    f"{artifact_name} {field} expected {expected_value!r}, got {row.get(field, '')!r}",
                )
        if row.get("not_selected_reason", "") not in allowed_not_selected_reasons:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "pool_tail_split",
                path,
                f"{artifact_name} not_selected_reason expected one of {sorted(allowed_not_selected_reasons)!r}, got {row.get('not_selected_reason', '')!r}",
            )
        if row.get("candidate_reuse_policy", "") not in allowed_reuse_policies:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "pool_tail_split",
                path,
                f"{artifact_name} candidate_reuse_policy expected one of {sorted(allowed_reuse_policies)!r}, got {row.get('candidate_reuse_policy', '')!r}",
            )

    for row in quota_rows:
        lint_quota_row(row, quota_path, "quota_surplus_pool")
    for row in rejected_rows:
        if row.get("pool_class") == "quota_surplus":
            lint_quota_row(row, rejected_path, "rejected_pool quota_surplus row")


def lint_candidate_pool_neutral_status(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # raw candidate pool은 final package가 아니므로, 품질 실패처럼 보이는 상태값이 남으면 재사용 정책을 오해할 수 있다.
    candidate_pool_path = resolve_repo_path(paths.get("candidate_pool_csv"))
    if not lint_existing_path(findings, fixture_id, candidate_pool_path, "missing_candidate_pool_csv"):
        return
    assert candidate_pool_path is not None
    rows, fields = read_csv_rows(candidate_pool_path)
    required_fields = {
        "pool_class",
        "quality_failure",
        "promotion_contract_status",
        "count_reflection_status",
        "count_allowed",
    }
    for field in sorted(required_fields - set(fields)):
        add_finding(findings, fixture_id, "P2", "candidate_pool_neutral_status", candidate_pool_path, f"candidate_pool is missing {field}")
    expected_values = {
        "pool_class": "candidate_pool",
        "quality_failure": "대상아님",
        "promotion_contract_status": "candidate_pool_not_promoted",
        "count_reflection_status": "not_counted_until_reviewer_signoff",
        "count_allowed": NO,
    }
    for row in rows:
        for field, expected_value in expected_values.items():
            if row.get(field, "") != expected_value:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "candidate_pool_neutral_status",
                    candidate_pool_path,
                    f"candidate_pool {field} expected {expected_value!r}, got {row.get(field, '')!r}",
                )


def lint_accepted_pool_final_integrity(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # accepted_pool에는 final selected와 reusable surplus가 함께 있으므로,
    # counted 상태는 final_package_selected row에만 허용해야 한다.
    accepted_path = resolve_repo_path(paths.get("accepted_pool_csv"))
    final_path = resolve_repo_path(paths.get("final_package_csv"))
    if not lint_existing_path(findings, fixture_id, accepted_path, "missing_accepted_pool_csv"):
        return
    if not lint_existing_path(findings, fixture_id, final_path, "missing_final_package_csv"):
        return
    assert accepted_path is not None
    assert final_path is not None
    accepted_rows, accepted_fields = read_csv_rows(accepted_path)
    final_rows, _ = read_csv_rows(final_path)
    if "pool_class" not in accepted_fields:
        add_finding(findings, fixture_id, "P2", "accepted_pool_final_integrity", accepted_path, "accepted_pool is missing pool_class")
        return
    counted_rows = [
        row
        for row in accepted_rows
        if row.get("count_reflection_status") == COUNTED_REFLECTION_STATUS or row.get("count_allowed") == YES
    ]
    final_selected_rows = [row for row in accepted_rows if row.get("pool_class") == "final_package_selected"]
    if len(counted_rows) != len(final_rows):
        add_finding(
            findings,
            fixture_id,
            "P2",
            "accepted_pool_final_integrity",
            accepted_path,
            f"accepted_pool counted rows expected final package row count {len(final_rows)}, got {len(counted_rows)}",
        )
    if len(final_selected_rows) != len(final_rows):
        add_finding(
            findings,
            fixture_id,
            "P2",
            "accepted_pool_final_integrity",
            accepted_path,
            f"accepted_pool final_package_selected rows expected final package row count {len(final_rows)}, got {len(final_selected_rows)}",
        )
    leaked = [row.get("candidate_id", "") for row in counted_rows if row.get("pool_class") != "final_package_selected"]
    if leaked:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "accepted_pool_final_integrity",
            accepted_path,
            f"accepted_pool non-final rows are counted: {leaked[:5]!r}",
        )
    final_ids = {row.get("candidate_id", "") for row in final_rows}
    counted_ids = {row.get("candidate_id", "") for row in counted_rows}
    if final_ids != counted_ids:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "accepted_pool_final_integrity",
            accepted_path,
            "accepted_pool counted candidate_id set does not match final_package candidate_id set",
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


def lint_descriptive_csv_counted(findings: list[Finding], fixture_id: str, path: Path | None, artifact_name: str) -> None:
    if not lint_existing_path(findings, fixture_id, path, f"missing_{artifact_name}"):
        return
    assert path is not None
    rows, fieldnames = read_csv_rows(path)
    if not rows:
        add_finding(findings, fixture_id, "P2", "empty_artifact", path, f"{artifact_name} has no rows")
        return
    required = {
        "package_role": COUNTED_BATCH_STATUS,
        "batch_status": COUNTED_BATCH_STATUS,
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "downstream_consumption_allowed": YES,
        "count_allowed": YES,
        "count_disposition": "counted",
        "train_eligible": YES,
        "audit_required": NO,
    }
    allowed_values = {
        # 예전 package는 reviewer sign-off 뒤 counted, 최신 API-first package는 strict final success 즉시 counted다.
        "promotion_contract_status": {
            "counted_after_reviewer_signoff",
            "counted_under_api_first_contract",
        }
    }
    if "promotion_contract_status" not in fieldnames:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "missing_contract_field",
            path,
            f"{artifact_name} is missing promotion_contract_status",
        )
    else:
        bad_promotion_values = sorted(
            {row.get("promotion_contract_status", "") for row in rows if row.get("promotion_contract_status", "") not in allowed_values["promotion_contract_status"]}
        )
        if bad_promotion_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "bad_contract_value",
                path,
                f"{artifact_name} has promotion_contract_status values {bad_promotion_values}",
            )
    for field, expected_value in required.items():
        if field not in fieldnames:
            add_finding(findings, fixture_id, "P2", "missing_contract_field", path, f"{artifact_name} is missing {field}")
            continue
        bad_values = sorted({row.get(field, "") for row in rows if row.get(field, "") != expected_value})
        if bad_values:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "descriptive_artifact_parity",
                path,
                f"{artifact_name} {field} expected {expected_value!r}, got {bad_values[:5]!r}",
            )
    if "final_status" in fieldnames:
        bad_status = sorted({row.get("final_status", "") for row in rows if row.get("final_status", "") != "pass"})
        if bad_status:
            add_finding(findings, fixture_id, "P2", "descriptive_quality_gate", path, f"{artifact_name} final_status got {bad_status[:5]!r}")


def lint_descriptive_merged_counted(findings: list[Finding], fixture_id: str, path: Path | None) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_merged_csv"):
        return
    assert path is not None
    rows, fieldnames = read_csv_rows(path)
    if not rows:
        add_finding(findings, fixture_id, "P2", "empty_artifact", path, "merged_csv has no rows")
        return
    selected_rows = [row for row in rows if row.get("final_package_selected") == YES]
    if not selected_rows:
        add_finding(findings, fixture_id, "P2", "descriptive_final_selection_missing", path, "merged_csv has no final_package_selected rows")
        return
    # merged에는 rejected candidate도 함께 남기되, counted 상태는 final selected rows에만 허용한다.
    for row in selected_rows:
        for field, expected_value in {
            "batch_status": COUNTED_BATCH_STATUS,
            "count_reflection_status": COUNTED_REFLECTION_STATUS,
            "downstream_consumption_allowed": YES,
            "count_allowed": YES,
            "count_disposition": "counted",
            "final_status": "pass",
            "audit_required": NO,
            "train_eligible": YES,
        }.items():
            if row.get(field, "") != expected_value:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "descriptive_merged_selected_parity",
                    path,
                    f"selected merged row {row.get('candidate_id', '')} {field} expected {expected_value!r}, got {row.get(field, '')!r}",
                )
    leaked = [
        row.get("candidate_id", "")
        for row in rows
        if row.get("final_package_selected") != YES and row.get("count_allowed") == YES
    ]
    if leaked:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "descriptive_nonfinal_count_leak",
            path,
            f"non-final merged rows have count_allowed = 예: {leaked[:5]!r}",
        )


def lint_descriptive_split_lock_reuse(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # Tier 2 cross-task reuse는 train split에만 허용된다. final/processed artifact에 metadata가 빠져도 seed_registry와 조인해 평가 split 누수를 잡는다.
    seed_path = resolve_repo_path(paths.get("seed_registry_csv"))
    final_path = resolve_repo_path(paths.get("final_package_csv"))
    processed_path = resolve_repo_path(paths.get("processed_manifest"))
    if not lint_existing_path(findings, fixture_id, seed_path, "missing_seed_registry_csv"):
        return
    assert seed_path is not None
    seed_rows, _ = read_csv_rows(seed_path)
    seed_meta_by_id = {row.get("seed_sample_id", ""): row for row in seed_rows}
    check_targets = [
        ("final_package_csv", final_path, "candidate_id"),
        ("processed_manifest", processed_path, "problem_id"),
    ]
    for artifact_name, artifact_path, id_field in check_targets:
        if not lint_existing_path(findings, fixture_id, artifact_path, f"missing_{artifact_name}_for_split_lock"):
            continue
        assert artifact_path is not None
        rows, fieldnames = read_csv_rows(artifact_path)
        for row in rows:
            seed_id = row.get("seed_sample_id", "")
            seed_meta = seed_meta_by_id.get(seed_id, {})
            reuse_tier = row.get("reuse_tier") or seed_meta.get("reuse_tier", "")
            source_split = row.get("source_split") or seed_meta.get("source_split", "")
            locked_split = row.get("locked_split") or seed_meta.get("locked_split", "")
            final_split = row.get("split") or row.get("dataset_disposition", "")
            if reuse_tier.startswith("Tier 2"):
                if source_split != "train" or locked_split != "train" or final_split != "train":
                    add_finding(
                        findings,
                        fixture_id,
                        "P2",
                        "descriptive_split_lock_eval_leakage",
                        artifact_path,
                        (
                            f"{artifact_name} {row.get(id_field, seed_id)} Tier 2 row must remain train-only; "
                            f"source_split={source_split!r}, locked_split={locked_split!r}, split={final_split!r}"
                        ),
                    )
            elif final_split in {"dev", "test"} and reuse_tier and not reuse_tier.startswith("Tier 0"):
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "descriptive_eval_reuse_policy",
                    artifact_path,
                    f"{artifact_name} {row.get(id_field, seed_id)} eval row must be Tier 0 fresh-only; got {reuse_tier!r}",
                )


def lint_descriptive_compiler_manifest_counted(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # compiler_manifest는 package factory의 최상위 audit trail이라 run/processed manifest가 counted여도 따로 parity를 본다.
    compiler_path = resolve_repo_path(paths.get("compiler_manifest_json"))
    if not lint_existing_path(findings, fixture_id, compiler_path, "missing_compiler_manifest_json"):
        return
    assert compiler_path is not None
    payload = json.loads(compiler_path.read_text(encoding="utf-8"))

    role = str(payload.get("package_role", ""))
    if role not in {"counted_current_production", "counted_current_production_package"}:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "compiler_manifest_counted_parity",
            compiler_path,
            f"compiler_manifest package_role expected counted role, got {role!r}",
        )
    for field, expected_value in {
        "count_reflection_status": COUNTED_REFLECTION_STATUS,
        "count_allowed": YES,
    }.items():
        actual = str(payload.get(field, ""))
        if actual != expected_value:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "compiler_manifest_counted_parity",
                compiler_path,
                f"compiler_manifest {field} expected {expected_value!r}, got {actual!r}",
            )

    final_path = resolve_repo_path(paths.get("final_package_csv"))
    candidate_path = resolve_repo_path(paths.get("candidate_pool_csv"))
    accepted_path = resolve_repo_path(paths.get("accepted_pool_csv"))
    tail_path = resolve_repo_path(paths.get("tail_taxonomy_csv"))
    quota_path = resolve_repo_path(paths.get("quota_surplus_csv"))
    row_expectations = {
        "candidate_total": candidate_path,
        "accepted_total": accepted_path,
        "final_package_total": final_path,
        "quality_tail_total": tail_path,
        "quota_surplus_total": quota_path,
    }
    row_counts = payload.get("row_counts", {})
    if not isinstance(row_counts, dict):
        add_finding(findings, fixture_id, "P2", "compiler_manifest_counted_parity", compiler_path, "compiler_manifest row_counts must be an object")
        return
    for field, artifact_path in row_expectations.items():
        if not lint_existing_path(findings, fixture_id, artifact_path, f"missing_{field}_artifact"):
            continue
        assert artifact_path is not None
        artifact_rows, _ = read_csv_rows(artifact_path)
        try:
            expected_count = int(row_counts.get(field))
        except (TypeError, ValueError):
            add_finding(
                findings,
                fixture_id,
                "P2",
                "compiler_manifest_counted_parity",
                compiler_path,
                f"compiler_manifest row_counts.{field} is not an integer",
            )
            continue
        if expected_count != len(artifact_rows):
            add_finding(
                findings,
                fixture_id,
                "P2",
                "compiler_manifest_counted_parity",
                artifact_path,
                f"compiler_manifest row_counts.{field} expected {expected_count}, got artifact rows {len(artifact_rows)}",
            )

    success_result = payload.get("success_result")
    if isinstance(success_result, dict) and success_result.get("passed") is not True:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "compiler_manifest_counted_parity",
            compiler_path,
            "compiler_manifest success_result.passed must be true for counted descriptive package",
        )


def lint_descriptive_handoff_alias_parity(findings: list[Finding], fixture_id: str, paths: dict[str, Any]) -> None:
    # API-first handoff에서는 run/compiler manifest의 짧은 alias가 cost audit과 reviewer handoff의 표면이다.
    # 값이 빠지면 strict final package는 맞아도 다음 seed planning이나 비용 검산에서 상태를 오해할 수 있다.
    run_path = resolve_repo_path(paths.get("run_manifest"))
    compiler_path = resolve_repo_path(paths.get("compiler_manifest_json"))
    if not lint_existing_path(findings, fixture_id, run_path, "missing_run_manifest_for_handoff_alias"):
        return
    if not lint_existing_path(findings, fixture_id, compiler_path, "missing_compiler_manifest_for_handoff_alias"):
        return
    assert run_path is not None
    assert compiler_path is not None

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    compiler_payload = json.loads(compiler_path.read_text(encoding="utf-8"))
    run_api_summary = run_payload.get("api_call_summary")
    if not isinstance(run_api_summary, dict):
        add_finding(findings, fixture_id, "P2", "descriptive_handoff_alias_parity", run_path, "run_manifest api_call_summary must be an object")
        return
    expected_total = run_api_summary.get("total_api_calls")
    for payload, path, manifest_name in [
        (run_payload, run_path, "run_manifest"),
        (compiler_payload, compiler_path, "compiler_manifest"),
    ]:
        if payload.get("total_api_calls") != expected_total:
            add_finding(
                findings,
                fixture_id,
                "P2",
                "descriptive_handoff_alias_parity",
                path,
                f"{manifest_name} total_api_calls expected {expected_total!r}, got {payload.get('total_api_calls')!r}",
            )
        for field, expected_value in {
            "split_lock_eval_hotfix_status": "passed",
            "artifact_linter_passed": True,
            "evidence_card_passed": True,
            "evidence_card_all_green": True,
            "count_reflection_requires_reviewer_signoff": False,
            "downstream_consumption_allowed": YES,
        }.items():
            if payload.get(field) != expected_value:
                add_finding(
                    findings,
                    fixture_id,
                    "P2",
                    "descriptive_handoff_alias_parity",
                    path,
                    f"{manifest_name} {field} expected {expected_value!r}, got {payload.get(field)!r}",
                )

    compiler_api_summary = compiler_payload.get("api_call_summary")
    if compiler_api_summary != run_api_summary:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "descriptive_handoff_alias_parity",
            compiler_path,
            "compiler_manifest api_call_summary must match run_manifest api_call_summary",
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


def lint_validator_wiring_check_md(
    findings: list[Finding],
    fixture_id: str,
    path: Path | None,
    expectations: dict[str, Any] | None = None,
) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_validator_wiring_check_md"):
        return
    assert path is not None
    text = path.read_text(encoding="utf-8")
    # Overgeneration run은 candidate seed 수와 final package 수가 다르므로,
    # fixture별 기대 문구를 받아 다음 package factory 비율에서도 같은 검산기를 재사용한다.
    expectations = expectations or {}
    stale_phrases = expectations.get(
        "stale_phrases",
        ["no-API preflight 16개", "`A/B/C/D = 4/4/4/4` target 적용"],
    )
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
    required_phrases = expectations.get(
        "required_phrases",
        [
            "no-API preflight 28개와 같은 seed registry 사용",
            "candidate target `A/B/C/D = 7/7/7/7`",
            "final export `A/B/C/D = 4/4/4/4`",
        ],
    )
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


def lint_seed_preflight_md(
    findings: list[Finding],
    fixture_id: str,
    path: Path | None,
    expectations: dict[str, Any] | None = None,
) -> None:
    if not lint_existing_path(findings, fixture_id, path, "missing_seed_preflight_md"):
        return
    assert path is not None
    expectations = expectations or {}
    text = path.read_text(encoding="utf-8")
    # Seed preflight markdown은 reviewer가 API 실행 규모를 판단하는 표면이라 이전 package 규모 문구를 강하게 차단한다.
    stale_phrases = expectations.get("stale_phrases", [])
    hits = [phrase for phrase in stale_phrases if phrase in text]
    if hits:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "seed_preflight_stale_scope",
            path,
            f"seed_preflight_md contains stale scope phrase(s): {hits}",
        )
    required_phrases = expectations.get("required_phrases", [])
    missing = [phrase for phrase in required_phrases if phrase not in text]
    if missing:
        add_finding(
            findings,
            fixture_id,
            "P2",
            "seed_preflight_scope_missing",
            path,
            f"seed_preflight_md is missing scope phrase(s): {missing}",
        )


def lint_counted_final_package(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []

    lint_manifest_counted(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    lint_manifest_artifact_paths(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    if paths.get("evidence_summary_json") is not None:
        lint_manifest_evidence_alias(
            findings,
            fixture_id,
            resolve_repo_path(paths.get("run_manifest")),
            resolve_repo_path(paths.get("evidence_summary_json")),
        )
    if paths.get("evidence_card_package_manifest") is not None:
        lint_evidence_package_manifest(findings, fixture_id, resolve_repo_path(paths.get("evidence_card_package_manifest")))
    lint_evidence_linter_pointer_parity(
        findings,
        fixture_id,
        resolve_repo_path(paths.get("run_manifest")),
        resolve_repo_path(paths.get("evidence_summary_json")),
        resolve_repo_path(paths.get("evidence_card_package_manifest")),
    )
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
    # counted package도 validator wiring markdown이 있으면 함께 본다.
    # count 반영 뒤 pre-signoff 문구가 남으면 다음 seed/API 계획에서 상태를 오해할 수 있다.
    if paths.get("validator_wiring_check_md") is not None:
        lint_validator_wiring_check_md(
            findings,
            fixture_id,
            resolve_repo_path(paths.get("validator_wiring_check_md")),
            fixture.get("validator_wiring_expectations"),
        )
    if paths.get("seed_preflight_md") is not None:
        lint_seed_preflight_md(
            findings,
            fixture_id,
            resolve_repo_path(paths.get("seed_preflight_md")),
            fixture.get("seed_preflight_expectations"),
        )
    if any(paths.get(key) is not None for key in ["rejected_pool_csv", "tail_taxonomy_csv", "quota_surplus_csv"]):
        lint_pool_tail_split(findings, fixture_id, paths)
    if paths.get("accepted_pool_csv") is not None:
        lint_accepted_pool_final_integrity(findings, fixture_id, paths)
    if paths.get("candidate_pool_csv") is not None:
        lint_candidate_pool_neutral_status(findings, fixture_id, paths)
    return findings


def lint_count_reflection_candidate_package(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []

    lint_manifest_candidate(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    lint_manifest_artifact_paths(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
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
    lint_validator_wiring_check_md(
        findings,
        fixture_id,
        resolve_repo_path(paths.get("validator_wiring_check_md")),
        fixture.get("validator_wiring_expectations"),
    )
    lint_pool_tail_split(findings, fixture_id, paths)
    if paths.get("accepted_pool_csv") is not None:
        lint_accepted_pool_final_integrity(findings, fixture_id, paths)
    if paths.get("candidate_pool_csv") is not None:
        lint_candidate_pool_neutral_status(findings, fixture_id, paths)
    return findings


def lint_descriptive_counted_final_package(fixture: dict[str, Any]) -> list[Finding]:
    fixture_id = fixture["fixture_id"]
    paths = fixture.get("paths", {})
    findings: list[Finding] = []
    lint_manifest_counted(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    lint_manifest_artifact_paths(findings, fixture_id, resolve_repo_path(paths.get("run_manifest")))
    if paths.get("evidence_summary_json") is not None:
        lint_manifest_evidence_alias(
            findings,
            fixture_id,
            resolve_repo_path(paths.get("run_manifest")),
            resolve_repo_path(paths.get("evidence_summary_json")),
        )
    if paths.get("evidence_card_package_manifest") is not None:
        lint_evidence_package_manifest(findings, fixture_id, resolve_repo_path(paths.get("evidence_card_package_manifest")))
    for key, artifact_name in [
        ("processed_manifest", "processed_manifest"),
        ("final_package_csv", "final_package_csv"),
    ]:
        lint_descriptive_csv_counted(findings, fixture_id, resolve_repo_path(paths.get(key)), artifact_name)
    lint_descriptive_merged_counted(findings, fixture_id, resolve_repo_path(paths.get("merged_csv")))
    lint_descriptive_split_lock_reuse(findings, fixture_id, paths)
    if paths.get("compiler_manifest_json") is not None:
        lint_descriptive_compiler_manifest_counted(findings, fixture_id, paths)
        lint_descriptive_handoff_alias_parity(findings, fixture_id, paths)
    for split_path_text in paths.get("split_jsonl", []) or []:
        path = resolve_repo_path(split_path_text)
        if path is not None:
            lint_jsonl_counted(findings, fixture_id, path)
    if paths.get("seed_preflight_md") is not None:
        lint_seed_preflight_md(
            findings,
            fixture_id,
            resolve_repo_path(paths.get("seed_preflight_md")),
            fixture.get("seed_preflight_expectations"),
        )
    if any(paths.get(key) is not None for key in ["rejected_pool_csv", "tail_taxonomy_csv", "quota_surplus_csv"]):
        lint_pool_tail_split(findings, fixture_id, paths)
    if paths.get("accepted_pool_csv") is not None:
        lint_accepted_pool_final_integrity(findings, fixture_id, paths)
    if paths.get("candidate_pool_csv") is not None:
        lint_candidate_pool_neutral_status(findings, fixture_id, paths)
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
    elif role == "descriptive_counted_final_package":
        findings = lint_descriptive_counted_final_package(fixture)
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
    fixtures: list[dict[str, Any]],
) -> str:
    total = len(outcomes_by_fixture)
    passed = sum(1 for outcome in outcomes_by_fixture.values() if outcome.fixture_passed)
    failed = total - passed
    unexpected_blocking_count_total = sum(outcome.unexpected_blocking_count for outcome in outcomes_by_fixture.values())
    blocking_finding_count_including_expected = sum(
        1 for findings in findings_by_fixture.values() for finding in findings if finding.severity in {"P1", "P2"}
    )
    lines = [
        "# artifact linter minimal dry-run report",
        "",
        f"- fixture_manifest: `{repo_rel(manifest_path)}`",
        f"- output_dir: `{repo_rel(output_dir)}`",
        f"- fixture_total: `{total}`",
        f"- fixture_passed: `{passed}`",
        f"- fixture_failed: `{failed}`",
        f"- unexpected_blocking_count_total: `{unexpected_blocking_count_total}`",
        f"- blocking_finding_count_including_expected_failures: `{blocking_finding_count_including_expected}`",
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
            "## coverage summary",
            "",
            "| fixture | covered path keys |",
            "| --- | --- |",
        ]
    )
    for fixture in fixtures:
        fixture_id = fixture.get("fixture_id", "unknown")
        # Reviewer handoff에서 linter가 어떤 artifact surface를 실제로 본 것인지 드러낸다.
        # 특히 validator_wiring_check_md처럼 finding이 없으면 report에서 사라지는 파일의 coverage를 보존한다.
        path_keys = sorted((fixture.get("paths") or {}).keys())
        lines.append(f"| `{fixture_id}` | `{', '.join(path_keys)}` |")

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
            "covered_path_keys": sorted((next((f.get("paths") for f in fixtures if f.get("fixture_id") == fixture_id), {}) or {}).keys()),
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
            # Reviewer-facing summary는 expected-fail fixture의 P2보다 unexpected blocker를 먼저 보게 한다.
            "unexpected_blocking_count_total": unexpected_blocking_count_total,
            "blocking_finding_count": sum(
                1 for findings in findings_by_fixture.values() for finding in findings if finding.severity in {"P1", "P2"}
            ),
            "p3_finding_count": sum(1 for findings in findings_by_fixture.values() for finding in findings if finding.severity == "P3"),
            "blocking_finding_count_including_expected_failures": sum(
                1 for findings in findings_by_fixture.values() for finding in findings if finding.severity in {"P1", "P2"}
            ),
            "passed": all(outcome.fixture_passed for outcome in outcomes_by_fixture.values()),
            "fixtures": fixture_summary,
        },
    )
    write_text_atomic(
        output_dir / "artifact_linter_report.md",
        render_markdown_report(manifest_path, output_dir, findings_by_fixture, outcomes_by_fixture, fixtures),
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
