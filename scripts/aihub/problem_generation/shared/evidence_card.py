from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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


DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "local" / "fixtures" / "evidence_card" / "package_manifest.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "evidence_cards"
DEFAULT_LINTER_REPORT_ROOT = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "artifact_linter"


@dataclass
class PackageSpec:
    package_id: str
    run_name: str
    version_tag: str
    package_role: str
    run_dir: Path
    processed_package_dir: Path
    linter_fixture_id: str
    linter_report_dir: Path
    source_chain: str
    count_context: dict[str, Any]


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_repo_path(path_text: str | None) -> Path:
    if not path_text:
        raise ValueError("path text is required")
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        return list(reader), list(reader.fieldnames or [])


def read_csv_rows_optional(path: Path) -> list[dict[str, str]]:
    # counted package에는 pool artifact가 없을 수 있으므로, candidate package에서만 보이는 pool health를 선택적으로 읽는다.
    if not path.exists() or not path.is_file():
        return []
    rows, _ = read_csv_rows(path)
    return rows


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row.get(field, "") or "(blank)") for row in rows)
    return dict(sorted(counter.items()))


def count_yes(rows: list[dict[str, Any]], field: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")) == "예")


def count_value(rows: list[dict[str, Any]], field: str, expected: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")) == expected)


def duplicate_count(rows: list[dict[str, Any]], field: str) -> int:
    values = [str(row.get(field, "")) for row in rows if str(row.get(field, ""))]
    counter = Counter(values)
    return sum(count - 1 for count in counter.values() if count > 1)


def find_latest_linter_report_dir(root: Path = DEFAULT_LINTER_REPORT_ROOT) -> Path:
    # request/review 시점마다 run stamp가 달라지므로, evidence card manifest는 최신 green linter report를 자동 참조한다.
    candidates = [
        path
        for path in root.glob("*_artifact_linter_minimal_dry_run")
        if path.is_dir() and (path / "artifact_linter_summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"artifact linter report directory not found under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def build_default_package_manifest(
    manifest_path: Path,
    linter_report_dir: Path | None = None,
    count_context: dict[str, Any] | None = None,
) -> Path:
    # evidence card는 linter fixture 이름과 실제 package 이름이 다를 수 있으므로 매핑 manifest를 별도로 둔다.
    ensure_dirs(manifest_path.parent)
    linter_report_dir = linter_report_dir or find_latest_linter_report_dir()
    count_context = count_context or {}
    packages = [
        {
            "package_id": "objective_interpretation_repair_dslot_final_replacement_package",
            "run_name": "2026-04-26_212921_objective_interpretation_repair_dslot_final_replacement_package_objective_r2_interpretation_repair_remaining_dslot_fresh_replacement",
            "version_tag": "objective_interpretation_repair_dslot_final_replacement_package",
            "package_role": "counted_current_production",
            "run_dir": "analysis/aihub/problem_generation/llm_runs/2026-04-26_212921_objective_interpretation_repair_dslot_final_replacement_package_objective_r2_interpretation_repair_remaining_dslot_fresh_replacement",
            "processed_package_dir": "data/processed/aihub/problem_generation/production_batches/objective_interpretation_repair_dslot_final_replacement_package",
            "linter_fixture_id": "212921_post_hotfix_pass",
            "linter_report_dir": repo_rel(linter_report_dir),
            "source_chain": "091859 original pilot -> 205615 replacement source -> 212921 final package",
        },
        {
            "package_id": "pb9_cslot_final_replacement_package",
            "run_name": "2026-04-26_055123_pb9_cslot_final_replacement_package_objective_r2_pb9_remaining_cslot_salvage_package",
            "version_tag": "pb9_cslot_final_replacement_package",
            "package_role": "counted_current_production",
            "run_dir": "analysis/aihub/problem_generation/llm_runs/2026-04-26_055123_pb9_cslot_final_replacement_package_objective_r2_pb9_remaining_cslot_salvage_package",
            "processed_package_dir": "data/processed/aihub/problem_generation/production_batches/pb9_cslot_final_replacement_package",
            "linter_fixture_id": "pb9_cslot_final_replacement_package_pass",
            "linter_report_dir": repo_rel(linter_report_dir),
            "source_chain": "pb9 decision-only source -> C-slot final replacement package",
        },
        {
            "package_id": "objective_judgment_repair_a_slot_replacement_package",
            "run_name": "2026-04-26_072939_objective_judgment_repair_a_slot_replacement_package_objective_r2_judgment_repair_a_slot_fresh_replacement",
            "version_tag": "objective_judgment_repair_a_slot_replacement_package",
            "package_role": "counted_current_production",
            "run_dir": "analysis/aihub/problem_generation/llm_runs/2026-04-26_072939_objective_judgment_repair_a_slot_replacement_package_objective_r2_judgment_repair_a_slot_fresh_replacement",
            "processed_package_dir": "data/processed/aihub/problem_generation/production_batches/objective_judgment_repair_a_slot_replacement_package",
            "linter_fixture_id": "judgment_a_slot_replacement_package_pass",
            "linter_report_dir": repo_rel(linter_report_dir),
            "source_chain": "judgment repair pilot -> A-slot fresh replacement package",
        },
    ]
    write_json_atomic(
        manifest_path,
        {
            "manifest_version": "evidence_card_dry_run_v1",
            "description": "First reviewer-facing evidence card dry-run package mapping.",
            "count_context": count_context,
            "packages": packages,
        },
    )
    return manifest_path


def load_package_specs(manifest_path: Path) -> list[PackageSpec]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    count_context = payload.get("count_context", {})
    specs: list[PackageSpec] = []
    for item in payload.get("packages", []):
        specs.append(
            PackageSpec(
                package_id=item["package_id"],
                run_name=item["run_name"],
                version_tag=item["version_tag"],
                package_role=item.get("package_role", ""),
                run_dir=resolve_repo_path(item["run_dir"]),
                processed_package_dir=resolve_repo_path(item["processed_package_dir"]),
                linter_fixture_id=item["linter_fixture_id"],
                linter_report_dir=resolve_repo_path(item["linter_report_dir"]),
                source_chain=item.get("source_chain", ""),
                count_context=count_context,
            )
        )
    return specs


def load_linter_fixture_summary(linter_report_dir: Path, fixture_id: str) -> dict[str, Any]:
    summary_path = linter_report_dir / "artifact_linter_summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    fixture_payload = payload.get("fixtures", {}).get(fixture_id)
    if not fixture_payload:
        raise KeyError(f"linter fixture not found: {fixture_id}")
    return fixture_payload


def load_linter_findings(linter_report_dir: Path, fixture_id: str) -> list[dict[str, str]]:
    report_path = linter_report_dir / "artifact_linter_report.csv"
    rows, _ = read_csv_rows(report_path)
    return [row for row in rows if row.get("fixture_id") == fixture_id]


def summarize_package(spec: PackageSpec) -> dict[str, Any]:
    run_manifest = json.loads((spec.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    manifest_rows, _ = read_csv_rows(spec.processed_package_dir / "dataset_manifest.csv")
    final_rows, _ = read_csv_rows(spec.run_dir / "exports" / f"final_package_{spec.version_tag}.csv")
    merged_rows, _ = read_csv_rows(spec.run_dir / "merged" / f"merged_problem_scores_{spec.version_tag}.csv")
    # 서술형 package factory처럼 별도 validator_report를 만들지 않는 run도 있어,
    # evidence card는 manifest/final rows를 기준으로 gate를 계산하고 validator artifact는 선택 입력으로만 둔다.
    validator_rows = read_csv_rows_optional(spec.run_dir / "exports" / f"validator_report_{spec.version_tag}.csv")
    candidate_pool_rows = read_csv_rows_optional(spec.run_dir / "candidate_pool.csv")
    accepted_pool_rows = read_csv_rows_optional(spec.run_dir / "accepted_pool.csv")
    rejected_pool_rows = read_csv_rows_optional(spec.run_dir / "rejected_pool.csv")
    tail_taxonomy_rows = read_csv_rows_optional(spec.run_dir / "tail_taxonomy.csv")
    quota_surplus_rows = read_csv_rows_optional(spec.run_dir / "quota_surplus_pool.csv")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split_name in ["train", "dev", "test"]:
        split_rows[split_name] = read_jsonl_rows(spec.processed_package_dir / f"{split_name}.jsonl")

    all_split_rows = [row for rows in split_rows.values() for row in rows]
    linter_summary = load_linter_fixture_summary(spec.linter_report_dir, spec.linter_fixture_id)
    linter_findings = load_linter_findings(spec.linter_report_dir, spec.linter_fixture_id)
    blocking_count = sum(1 for row in linter_findings if row.get("severity") in {"P1", "P2"})
    p3_count = sum(1 for row in linter_findings if row.get("severity") == "P3")

    usable = count_value(manifest_rows, "train_eligible", "예")
    audit = count_value(manifest_rows, "audit_required", "예")
    # Package factory는 rejected/tail candidate를 merged에 함께 보존하므로,
    # final package evidence에서는 strict final rows만 품질 gate의 분모로 사용한다.
    quality_rows = final_rows or merged_rows
    hard_fail = count_value(quality_rows, "final_status", "hard_fail")
    soft_fail = count_value(quality_rows, "final_status", "soft_fail")
    pass_count = count_value(quality_rows, "final_status", "pass")
    train = len(split_rows["train"])
    dev = len(split_rows["dev"])
    test = len(split_rows["test"])
    # 서술형 package에는 objective choice validator field가 없을 수 있어,
    # 필드가 없는 경우에는 별도 validator gate 미적용으로 해석한다.
    manifest_fields = set(manifest_rows[0].keys()) if manifest_rows else set()
    validator_applicable = bool(validator_rows) or bool({"validator_action", "validator_export_disposition"} & manifest_fields)
    descriptive_rows = manifest_rows if "problem_task_type" in manifest_fields else all_split_rows
    validator_not_applicable_for_descriptive = (
        not validator_applicable
        and bool(descriptive_rows)
        and all("descriptive" in str(row.get("problem_task_type", "")) for row in descriptive_rows)
    )
    non_accept = 0 if "validator_action" not in manifest_fields else len([row for row in manifest_rows if row.get("validator_action") != "accept"])
    non_export_ready = (
        0
        if "validator_export_disposition" not in manifest_fields
        else len([row for row in manifest_rows if row.get("validator_export_disposition") != "export_ready"])
    )
    metadata_mismatch = 0 if "metadata_remap_ok" not in manifest_fields else len([row for row in manifest_rows if row.get("metadata_remap_ok") != "예"])
    shuffle_mismatch = len(
        [
            row
            for row in manifest_rows
            if row.get("export_correct_choice")
            and row.get("validator_recalculated_correct_choice")
            and row.get("export_correct_choice") != row.get("validator_recalculated_correct_choice")
        ]
    )
    duplicate_seed = duplicate_count(manifest_rows, "seed_sample_id")
    duplicate_family = duplicate_count(manifest_rows, "family_id")
    duplicate_problem = duplicate_count(manifest_rows, "problem_id")
    downstream_allowed = len([row for row in all_split_rows if row.get("downstream_consumption_allowed") == "예"])
    downstream_not_allowed = len([row for row in all_split_rows if row.get("downstream_consumption_allowed") != "예"])
    candidate_package = (
        spec.package_role == "count_reflection_candidate_package"
        or run_manifest.get("package_role") == "count_reflection_candidate_package"
        or run_manifest.get("count_reflection_status") == "not_counted_until_reviewer_signoff"
    )
    already_counted = (
        spec.package_role == "counted_current_production"
        or run_manifest.get("count_reflection_status") == "counted"
    )
    # Candidate package는 pre-signoff라 downstream/count를 막아야 정상이므로 counted package와 gate 의미를 분리한다.
    candidate_status_ok = (
        candidate_package
        and run_manifest.get("promotion_contract_passed") == "예"
        and run_manifest.get("compiler_gate_passed") == "예"
        and run_manifest.get("count_allowed") == "아니오"
        and run_manifest.get("count_disposition") == "candidate_not_counted"
        and downstream_allowed == 0
        and downstream_not_allowed == len(all_split_rows)
    )
    counted_status_ok = already_counted and downstream_not_allowed == 0
    # already-counted package에서는 candidate gate가 적용 대상이 아니므로,
    # reviewer-facing card에는 실패처럼 보이는 false 대신 N/A를 노출한다.
    candidate_status_ok_display = "N/A" if already_counted else str(candidate_status_ok).lower()
    counted_status_ok_display = "N/A" if candidate_package and not already_counted else str(counted_status_ok).lower()

    all_green = (
        bool(linter_summary.get("passed"))
        and blocking_count == 0
        and usable == len(manifest_rows)
        and audit == 0
        and hard_fail == 0
        and soft_fail == 0
        and non_accept == 0
        and non_export_ready == 0
        and metadata_mismatch == 0
        and shuffle_mismatch == 0
        and duplicate_seed == 0
        and duplicate_family == 0
        and duplicate_problem == 0
        and (counted_status_ok or candidate_status_ok)
    )
    decision_context = "already_counted_validation" if already_counted else "new_count_reflection_candidate"
    count_delta_applies = "아니오" if already_counted else "예"
    proposed_usable_delta = 0 if already_counted else usable
    proposed_train_delta = 0 if already_counted else train
    proposed_eval_delta = 0 if already_counted else dev + test
    current_count_before = {
        "usable": spec.count_context.get("current_usable", ""),
        "train": spec.count_context.get("current_train", ""),
        "eval": spec.count_context.get("current_eval", ""),
    }
    if all_green and already_counted:
        count_reflection_action = "no_count_change"
        reviewer_decision = "counted package all-green validation"
    elif all_green:
        count_reflection_action = "reviewer_signoff_needed"
        reviewer_decision = "count reflection candidate"
    else:
        count_reflection_action = "blocked"
        reviewer_decision = "needs no-API sync"

    def add_if_number(value: Any, delta: int) -> Any:
        try:
            return int(value) + delta
        except (TypeError, ValueError):
            return ""

    current_count_after = {
        "usable": add_if_number(current_count_before["usable"], proposed_usable_delta),
        "train": add_if_number(current_count_before["train"], proposed_train_delta),
        "eval": add_if_number(current_count_before["eval"], proposed_eval_delta),
    }

    return {
        "package_id": spec.package_id,
        "run_name": spec.run_name,
        "version_tag": spec.version_tag,
        "package_role": spec.package_role,
        "decision_context": decision_context,
        "count_reflection_action": count_reflection_action,
        "source_chain": spec.source_chain,
        "run_manifest_status": run_manifest.get("batch_status", ""),
        "count_reflection_status": run_manifest.get("count_reflection_status", ""),
        "downstream_consumption_allowed": run_manifest.get("downstream_consumption_allowed", ""),
        "run_manifest_count_allowed": run_manifest.get("count_allowed", ""),
        "run_manifest_count_disposition": run_manifest.get("count_disposition", ""),
        "promotion_contract_passed": run_manifest.get("promotion_contract_passed", ""),
        "compiler_gate_passed": run_manifest.get("compiler_gate_passed", ""),
        "promotion_contract_status": run_manifest.get("promotion_contract_status", ""),
        "candidate_total": run_manifest.get("candidate_total", len(candidate_pool_rows)),
        "accepted_total": run_manifest.get("accepted_total", len(accepted_pool_rows)),
        "final_package_total": run_manifest.get("final_package_total", len(final_rows)),
        "rejected_total": run_manifest.get("rejected_total", len(rejected_pool_rows)),
        "quality_tail_total": run_manifest.get("quality_tail_total", len(tail_taxonomy_rows)),
        "quota_surplus_total": run_manifest.get("quota_surplus_total", len(quota_surplus_rows)),
        "actual_candidate_pool_rows": len(candidate_pool_rows),
        "actual_accepted_pool_rows": len(accepted_pool_rows),
        "actual_rejected_pool_rows": len(rejected_pool_rows),
        "actual_quality_tail_rows": len(tail_taxonomy_rows),
        "actual_quota_surplus_rows": len(quota_surplus_rows),
        "quality_tail_by_class": count_values(tail_taxonomy_rows, "tail_class"),
        "quality_tail_by_reason": count_values(tail_taxonomy_rows, "not_selected_reason"),
        "quota_surplus_by_label": count_values(quota_surplus_rows, "export_correct_choice"),
        "quota_surplus_by_source": count_values(quota_surplus_rows, "source_subset"),
        "quota_surplus_by_lane": count_values(quota_surplus_rows, "sampling_lane"),
        "candidate_status_ok": candidate_status_ok,
        "counted_status_ok": counted_status_ok,
        "candidate_status_ok_display": candidate_status_ok_display,
        "counted_status_ok_display": counted_status_ok_display,
        "linter_report_dir": repo_rel(spec.linter_report_dir),
        "linter_fixture_id": spec.linter_fixture_id,
        "linter_passed": bool(linter_summary.get("passed")),
        "linter_blocking_finding_count": blocking_count,
        "linter_p3_finding_count": p3_count,
        "row_count": len(manifest_rows),
        "usable": usable,
        "train": train,
        "dev": dev,
        "test": test,
        "audit": audit,
        "pass": pass_count,
        "hard_fail": hard_fail,
        "soft_fail": soft_fail,
        "validator_applicable": validator_applicable,
        "validator_not_applicable_for_descriptive": validator_not_applicable_for_descriptive,
        "validator_accept": count_value(manifest_rows, "validator_action", "accept"),
        "validator_export_ready": count_value(manifest_rows, "validator_export_disposition", "export_ready"),
        "validator_non_accept": non_accept,
        "validator_non_export_ready": non_export_ready,
        "metadata_remap_mismatch": metadata_mismatch,
        "shuffle_mismatch": shuffle_mismatch,
        "target_label_distribution": count_values(manifest_rows, "target_correct_choice"),
        "export_label_distribution": count_values(manifest_rows, "export_correct_choice"),
        "doc_type_distribution": count_values(manifest_rows, "doc_type_name"),
        "source_subset_distribution": count_values(manifest_rows, "source_subset"),
        "lane_distribution": count_values(manifest_rows, "sampling_lane"),
        "duplicate_seed_sample_id": duplicate_seed,
        "duplicate_family_id": duplicate_family,
        "duplicate_problem_id": duplicate_problem,
        "downstream_allowed": downstream_allowed,
        "downstream_not_allowed": downstream_not_allowed,
        "current_count_before": current_count_before,
        "proposed_usable_delta": proposed_usable_delta,
        "proposed_train_delta": proposed_train_delta,
        "proposed_eval_delta": proposed_eval_delta,
        "current_count_after": current_count_after,
        "count_delta_applies": count_delta_applies,
        "exception_summary": {
            "non_green_rows": 0 if all_green else len(manifest_rows) - usable + audit + hard_fail + soft_fail,
            "expected_negative_fixture": "not_applicable",
            "p3_debt": p3_count,
        },
        "explanation_status": "문제-선택지-정답 package 기준. 해설 attachment는 별도 stop line.",
        "reviewer_decision": reviewer_decision,
        "all_green": all_green,
    }


def as_count_text(distribution: dict[str, int]) -> str:
    return " / ".join(f"{key} {value}" for key, value in distribution.items()) if distribution else "(empty)"


def render_package_card(summary: dict[str, Any]) -> str:
    lines = [
        f"# evidence card: `{summary['package_id']}`",
        "",
        "## identity",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| package_id | `{summary['package_id']}` |",
        f"| run_name | `{summary['run_name']}` |",
        f"| version_tag | `{summary['version_tag']}` |",
        f"| package_role | `{summary['package_role']}` |",
        f"| decision_context | `{summary['decision_context']}` |",
        f"| count_reflection_action | `{summary['count_reflection_action']}` |",
        f"| run_manifest_count_allowed | `{summary['run_manifest_count_allowed']}` |",
        f"| run_manifest_count_disposition | `{summary['run_manifest_count_disposition']}` |",
        f"| promotion_contract_passed | `{summary['promotion_contract_passed']}` |",
        f"| compiler_gate_passed | `{summary['compiler_gate_passed']}` |",
        f"| promotion_contract_status | `{summary['promotion_contract_status']}` |",
        f"| candidate_status_ok | `{summary['candidate_status_ok_display']}` |",
        f"| counted_status_ok | `{summary['counted_status_ok_display']}` |",
        f"| source_chain | {summary['source_chain']} |",
        "",
        "## candidate pool health",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| candidate_total | `{summary['candidate_total']}` |",
        f"| accepted_total | `{summary['accepted_total']}` |",
        f"| rejected_total | `{summary['rejected_total']}` |",
        f"| quality_tail_total | `{summary['quality_tail_total']}` |",
        f"| quota_surplus_total | `{summary['quota_surplus_total']}` |",
        f"| actual_candidate_pool_rows | `{summary['actual_candidate_pool_rows']}` |",
        f"| actual_accepted_pool_rows | `{summary['actual_accepted_pool_rows']}` |",
        f"| actual_rejected_pool_rows | `{summary['actual_rejected_pool_rows']}` |",
        f"| actual_quality_tail_rows | `{summary['actual_quality_tail_rows']}` |",
        f"| actual_quota_surplus_rows | `{summary['actual_quota_surplus_rows']}` |",
        f"| quality_tail_by_class | `{as_count_text(summary['quality_tail_by_class'])}` |",
        f"| quality_tail_by_reason | `{as_count_text(summary['quality_tail_by_reason'])}` |",
        f"| quota_surplus_by_label | `{as_count_text(summary['quota_surplus_by_label'])}` |",
        f"| quota_surplus_by_source | `{as_count_text(summary['quota_surplus_by_source'])}` |",
        f"| quota_surplus_by_lane | `{as_count_text(summary['quota_surplus_by_lane'])}` |",
        "",
        "## count summary",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| row_count | `{summary['row_count']}` |",
        f"| usable | `{summary['usable']}` |",
        f"| train | `{summary['train']}` |",
        f"| dev | `{summary['dev']}` |",
        f"| test | `{summary['test']}` |",
        f"| audit | `{summary['audit']}` |",
        f"| hard_fail | `{summary['hard_fail']}` |",
        f"| soft_fail | `{summary['soft_fail']}` |",
        f"| count_delta_applies | `{summary['count_delta_applies']}` |",
        f"| current_count_before | `{json.dumps(summary['current_count_before'], ensure_ascii=False, sort_keys=True)}` |",
        f"| proposed_usable_delta | `{summary['proposed_usable_delta']}` |",
        f"| proposed_train_delta | `{summary['proposed_train_delta']}` |",
        f"| proposed_eval_delta | `{summary['proposed_eval_delta']}` |",
        f"| current_count_after | `{json.dumps(summary['current_count_after'], ensure_ascii=False, sort_keys=True)}` |",
        "",
        "## linter and validator",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| linter_fixture_id | `{summary['linter_fixture_id']}` |",
        f"| linter_passed | `{str(summary['linter_passed']).lower()}` |",
        f"| linter_blocking_finding_count | `{summary['linter_blocking_finding_count']}` |",
        f"| linter_p3_finding_count | `{summary['linter_p3_finding_count']}` |",
        f"| validator_applicable | `{str(summary['validator_applicable']).lower()}` |",
        f"| validator_not_applicable_for_descriptive | `{str(summary['validator_not_applicable_for_descriptive']).lower()}` |",
        f"| validator_accept | `{summary['validator_accept']}` |",
        f"| validator_export_ready | `{summary['validator_export_ready']}` |",
        f"| metadata_remap_mismatch | `{summary['metadata_remap_mismatch']}` |",
        f"| shuffle_mismatch | `{summary['shuffle_mismatch']}` |",
        "",
        "## balance",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| target_label_distribution | `{as_count_text(summary['target_label_distribution'])}` |",
        f"| export_label_distribution | `{as_count_text(summary['export_label_distribution'])}` |",
        f"| doc_type_distribution | `{as_count_text(summary['doc_type_distribution'])}` |",
        f"| source_subset_distribution | `{as_count_text(summary['source_subset_distribution'])}` |",
        f"| lane_distribution | `{as_count_text(summary['lane_distribution'])}` |",
        "",
        "## overlap and decision",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| duplicate_seed_sample_id | `{summary['duplicate_seed_sample_id']}` |",
        f"| duplicate_family_id | `{summary['duplicate_family_id']}` |",
        f"| duplicate_problem_id | `{summary['duplicate_problem_id']}` |",
        f"| downstream_allowed | `{summary['downstream_allowed']}` |",
        f"| downstream_not_allowed | `{summary['downstream_not_allowed']}` |",
        f"| explanation_status | {summary['explanation_status']} |",
        f"| all_green | `{str(summary['all_green']).lower()}` |",
        f"| reviewer_decision | `{summary['reviewer_decision']}` |",
        "",
    ]
    return "\n".join(lines)


def render_summary_report(summaries: list[dict[str, Any]], manifest_path: Path, output_dir: Path) -> str:
    lines = [
        "# objective evidence card dry-run summary",
        "",
        f"- package_manifest: `{repo_rel(manifest_path)}`",
        f"- output_dir: `{repo_rel(output_dir)}`",
        f"- package_total: `{len(summaries)}`",
        f"- all_green_total: `{sum(1 for item in summaries if item['all_green'])}`",
        "",
        "| package_id | context | action | linter_fixture_id | candidate | accepted | final | quality_tail | quota_surplus | usable | delta applies | split | audit | hard_fail | soft_fail | linter | all_green | decision |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for item in summaries:
        split_text = f"train {item['train']} / dev {item['dev']} / test {item['test']}"
        lines.append(
            f"| `{item['package_id']}` | `{item['decision_context']}` | `{item['count_reflection_action']}` | "
            f"`{item['linter_fixture_id']}` | `{item['candidate_total']}` | `{item['accepted_total']}` | "
            f"`{item['final_package_total'] if 'final_package_total' in item else item['row_count']}` | "
            f"`{item['quality_tail_total']}` | `{item['quota_surplus_total']}` | "
            f"`{item['usable']}` | `{item['count_delta_applies']}` | {split_text} | "
            f"`{item['audit']}` | `{item['hard_fail']}` | `{item['soft_fail']}` | "
            f"`{str(item['linter_passed']).lower()}` | `{str(item['all_green']).lower()}` | `{item['reviewer_decision']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def flatten_summary(summary: dict[str, Any]) -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            flat[key] = str(value)
    return flat


def run_evidence_card(manifest_path: Path, output_dir: Path) -> bool:
    specs = load_package_specs(manifest_path)
    ensure_dirs(output_dir / "cards")
    summaries = [summarize_package(spec) for spec in specs]
    summary_fields = list(flatten_summary(summaries[0]).keys()) if summaries else []
    write_csv_atomic(output_dir / "evidence_card_summary.csv", [flatten_summary(item) for item in summaries], summary_fields)
    write_json_atomic(
        output_dir / "evidence_card_summary.json",
        {
            "package_manifest": repo_rel(manifest_path),
            "package_total": len(summaries),
            "all_green_total": sum(1 for item in summaries if item["all_green"]),
            "packages": summaries,
        },
    )
    write_text_atomic(output_dir / "evidence_card_summary.md", render_summary_report(summaries, manifest_path, output_dir))
    for summary in summaries:
        write_text_atomic(output_dir / "cards" / f"{summary['package_id']}.md", render_package_card(summary))
    return all(item["all_green"] for item in summaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing evidence cards for counted packages.")
    parser.add_argument("--package-manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--materialize-default-manifest", action="store_true")
    parser.add_argument("--linter-report-dir", type=Path)
    parser.add_argument("--current-usable", type=int)
    parser.add_argument("--current-train", type=int)
    parser.add_argument("--current-eval", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_manifest = args.package_manifest
    if args.materialize_default_manifest:
        # current count는 drift가 잦으므로 코드에 박지 않고 실행 인자로 manifest에 materialize한다.
        count_context = {
            "current_usable": args.current_usable if args.current_usable is not None else "",
            "current_train": args.current_train if args.current_train is not None else "",
            "current_eval": args.current_eval if args.current_eval is not None else "",
        }
        package_manifest = build_default_package_manifest(package_manifest, args.linter_report_dir, count_context)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{build_run_stamp()}_objective_evidence_card_dry_run"
    passed = run_evidence_card(package_manifest, output_dir)
    print(f"evidence_card_summary={output_dir / 'evidence_card_summary.md'}")
    print(f"evidence_card_passed={str(passed).lower()}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
