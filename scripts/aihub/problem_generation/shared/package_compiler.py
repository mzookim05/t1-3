from __future__ import annotations

import argparse
import json
import sys
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
    write_jsonl_atomic,
    write_text_atomic,
)


YES = "예"
NO = "아니오"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
PACKAGE_ROLE = "count_reflection_candidate_package"
VERSION_TAG = "objective_package_compiler_minimal_dry_run"
COMPILER_SEED = "fixture_stable_sort_v1"
COUNT_DISPOSITION = "candidate_not_counted"
PROMOTION_CONTRACT_STATUS = "passed_not_counted"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "package_compiler"


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def base_candidate(candidate_id: str, label: str, source: str, lane: str, risk_score: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "seed_sample_id": f"compiler_seed_{candidate_id[-3:]}",
        "family_id": f"compiler_family_{candidate_id[-3:]}",
        "problem_id": f"compiler_problem_{candidate_id[-3:]}",
        "doc_type_name": "결정례_QA",
        "source_subset": source,
        "sampling_lane": lane,
        "target_correct_choice": label,
        "export_correct_choice": label,
        "validator_recalculated_correct_choice": label,
        "risk_score": str(risk_score),
        "generated_stem": f"{candidate_id}의 쟁점에 관한 설명으로 옳은 것은?",
        "choice_a": "선택지 A",
        "choice_b": "선택지 B",
        "choice_c": "선택지 C",
        "choice_d": "선택지 D",
        "correct_choice": label,
        "distractor_type_map": "hard_negative",
        "near_miss_notes": "one_axis_perturbation",
        "gold_short_answer": "컴파일러 dry-run 정답",
        "gold_reference_explanation": "fixture 기반 검산용 설명",
        "metadata_remap_ok": YES,
        "validator_shuffle_recalc_ok": YES,
        "validator_action": "accept",
        "validator_export_disposition": "export_ready",
        "validator_reason_short": "compiler_accept",
        "final_status": "pass",
        "audit_required": NO,
        "audit_reason": "",
        "train_eligible": YES,
        "compiler_gate_passed": YES,
        "promotion_contract_passed": YES,
        "package_role": PACKAGE_ROLE,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
        "pool_class": "",
        "quality_failure": "",
        "tail_class": "",
        "future_candidate_reusable": "",
        "candidate_reuse_policy": "",
    }


def build_fixture_candidates() -> list[dict[str, Any]]:
    # Minimal dry-run은 API 없이 selection/rejection shape만 검증하므로, tail 유형을 의도적으로 섞은 fixture pool을 사용한다.
    rows = [
        base_candidate("compiler_candidate_001", "A", "03_TL_결정례_QA", "generalization_03_04", 1),
        base_candidate("compiler_candidate_002", "B", "03_TL_결정례_QA", "generalization_03_04", 1),
        base_candidate("compiler_candidate_003", "C", "04_TL_결정례_QA", "generalization_03_04", 1),
        base_candidate("compiler_candidate_004", "D", "04_TL_결정례_QA", "generalization_03_04", 1),
        base_candidate("compiler_candidate_005", "A", "04_TL_결정례_QA", "generalization_03_04", 3),
        base_candidate("compiler_candidate_006", "B", "02_TL_결정례_QA", "expansion_01_02", 5),
        base_candidate("compiler_candidate_007", "C", "02_TL_결정례_QA", "expansion_01_02", 4),
        base_candidate("compiler_candidate_008", "D", "01_TL_결정례_QA", "expansion_01_02", 4),
    ]
    rows[5].update(
        {
            "final_status": "hard_fail",
            "train_eligible": NO,
            "compiler_gate_passed": NO,
            "promotion_contract_passed": NO,
            "validator_action": "hard_block",
            "validator_export_disposition": "hard_blocked",
            "validator_reason_short": "answer_uniqueness_failure",
        }
    )
    rows[6].update(
        {
            "audit_required": YES,
            "train_eligible": NO,
            "compiler_gate_passed": NO,
            "promotion_contract_passed": NO,
            "validator_action": "audit",
            "validator_export_disposition": "audit_queue",
            "validator_reason_short": "weak_distractor_audit",
        }
    )
    rows[7].update(
        {
            "metadata_remap_ok": NO,
            "train_eligible": NO,
            "compiler_gate_passed": NO,
            "promotion_contract_passed": NO,
            "validator_reason_short": "metadata_mismatch",
        }
    )
    return rows


def rejection_metadata(reason: str) -> dict[str, str]:
    # quota surplus는 좋은 후보가 quota 때문에 밀린 상태라 quality tail과 분리해야 다음 seed 재사용 판단이 흐려지지 않는다.
    if reason == "label_quota_filled":
        return {
            "pool_class": "quota_surplus",
            "quality_failure": NO,
            "tail_class": "quota_surplus_not_quality_failure",
            "future_candidate_reusable": YES,
            "candidate_reuse_policy": "reuse_allowed_as_surplus_candidate",
        }
    tail_class_by_reason = {
        "hard_or_soft_fail": "final_status_failure",
        "audit_required": "audit_tail",
        "validator_not_export_ready": "validator_failure",
        "metadata_mismatch": "metadata_failure",
    }
    return {
        "pool_class": "quality_reject",
        "quality_failure": YES,
        "tail_class": tail_class_by_reason.get(reason, "quality_or_artifact_failure"),
        "future_candidate_reusable": NO,
        "candidate_reuse_policy": "do_not_reuse_without_repair_review",
    }


def compile_package(candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strict_accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    label_quota = {"A": 1, "B": 1, "C": 1, "D": 1}
    selected_by_label = {label: 0 for label in label_quota}
    sort_key = lambda item: (int(item["risk_score"]), item["candidate_id"], item["seed_sample_id"])

    # strict gate와 package quota selection을 분리해야 accepted_pool과 final_package 의미가 섞이지 않는다.
    for row in sorted(candidate_rows, key=sort_key):
        reject_reason = ""
        if row["final_status"] != "pass":
            reject_reason = "hard_or_soft_fail"
        elif row["audit_required"] == YES:
            reject_reason = "audit_required"
        elif row["validator_action"] != "accept" or row["validator_export_disposition"] != "export_ready":
            reject_reason = "validator_not_export_ready"
        elif row["metadata_remap_ok"] != YES:
            reject_reason = "metadata_mismatch"

        if reject_reason:
            rejected.append(
                {
                    **row,
                    **rejection_metadata(reject_reason),
                    "not_selected_reason": reject_reason,
                    "selection_rank": "",
                    "selection_reason": "",
                }
            )
            continue

        strict_accepted.append(
            {
                **row,
                "pool_class": "strict_accepted",
                "quality_failure": NO,
                "tail_class": "",
                "future_candidate_reusable": "",
                "candidate_reuse_policy": "eligible_for_package_selection",
                "selection_rank": "",
                "selection_reason": "strict_gate_accepted",
                "not_selected_reason": "",
            }
        )

    final_rows: list[dict[str, Any]] = []
    accepted_with_selection: list[dict[str, Any]] = []
    for row in sorted(strict_accepted, key=sort_key):
        label = row["export_correct_choice"]
        if selected_by_label[label] >= label_quota[label]:
            not_selected = {
                **row,
                **rejection_metadata("label_quota_filled"),
                "not_selected_reason": "label_quota_filled",
                "selection_reason": "strict_gate_accepted_not_selected",
            }
            accepted_with_selection.append(not_selected)
            rejected.append(not_selected)
            continue
        selected_by_label[label] += 1
        selected = {
            **row,
            "pool_class": "final_package_selected",
            "quality_failure": NO,
            "tail_class": "",
            "future_candidate_reusable": "",
            "candidate_reuse_policy": "selected_final_package",
            "selection_rank": str(len(final_rows) + 1),
            "selection_reason": f"quota_selected_by_{COMPILER_SEED}",
            "not_selected_reason": "",
        }
        accepted_with_selection.append(selected)
        final_rows.append(selected)

    return accepted_with_selection, rejected, final_rows


def add_split(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_by_rank = {1: "train", 2: "train", 3: "dev", 4: "test"}
    output: list[dict[str, Any]] = []
    for row in rows:
        rank = int(row["selection_rank"])
        output.append({**row, "split": split_by_rank.get(rank, "train")})
    return output


def union_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def render_candidate_markdown(title: str, package_id: str, row_count: int) -> str:
    return "\n".join(
        [
            f"# {title}",
            "",
            "| field | value |",
            "| --- | --- |",
            f"| package_role | `{PACKAGE_ROLE}` |",
            f"| batch_status | `{CANDIDATE_BATCH_STATUS}` |",
            f"| count_reflection_status | `{CANDIDATE_REFLECTION_STATUS}` |",
            "| downstream_consumption_allowed | `아니오` |",
            "| count_allowed | `아니오` |",
            f"| count_disposition | `{COUNT_DISPOSITION}` |",
            "| compiler_gate_passed | `예` |",
            "| promotion_contract_passed | `예` |",
            f"| promotion_contract_status | `{PROMOTION_CONTRACT_STATUS}` |",
            f"| package_id | `{package_id}` |",
            f"| row_count | `{row_count}` |",
            "",
        ]
    )


def write_outputs(output_dir: Path, current_usable: int, current_train: int, current_eval: int) -> dict[str, Any]:
    ensure_dirs(
        output_dir / "exports",
        output_dir / "merged",
        output_dir / "processed",
        output_dir / "linter",
        output_dir / "evidence_card",
        output_dir / "mutation_fixtures",
    )
    candidate_rows = build_fixture_candidates()
    accepted_rows, rejected_rows, final_rows = compile_package(candidate_rows)
    final_rows = add_split(final_rows)
    quality_tail_rows = [row for row in rejected_rows if row.get("quality_failure") == YES]
    quota_surplus_rows = [row for row in rejected_rows if row.get("pool_class") == "quota_surplus"]
    package_id = VERSION_TAG

    write_csv_atomic(output_dir / "candidate_pool.csv", candidate_rows, union_fields(candidate_rows))
    write_csv_atomic(output_dir / "accepted_pool.csv", accepted_rows, union_fields(accepted_rows))
    write_csv_atomic(output_dir / "rejected_pool.csv", rejected_rows, union_fields(rejected_rows))
    write_csv_atomic(output_dir / "quota_surplus_pool.csv", quota_surplus_rows, union_fields(quota_surplus_rows))
    write_csv_atomic(output_dir / "tail_taxonomy.csv", quality_tail_rows, union_fields(quality_tail_rows))
    write_csv_atomic(output_dir / "exports" / f"final_package_{VERSION_TAG}.csv", final_rows, union_fields(final_rows))
    write_csv_atomic(output_dir / "merged" / f"merged_problem_scores_{VERSION_TAG}.csv", final_rows, union_fields(final_rows))
    write_csv_atomic(output_dir / "exports" / f"validator_report_{VERSION_TAG}.csv", final_rows, union_fields(final_rows))
    write_csv_atomic(output_dir / "processed" / "dataset_manifest.csv", final_rows, union_fields(final_rows))

    # linter regression fixture는 실제 pool/tail artifact를 일부러 망가뜨려 자동 gate가 분리를 잡는지 검증한다.
    quota_mutation_rows = [dict(row) for row in quota_surplus_rows]
    for row in quota_mutation_rows:
        row["quality_failure"] = YES
    tail_mutation_rows = [*quality_tail_rows, *quota_surplus_rows]
    mutation_dir = output_dir / "mutation_fixtures"
    write_csv_atomic(mutation_dir / "quota_surplus_quality_failure.csv", quota_mutation_rows, union_fields(quota_mutation_rows))
    write_csv_atomic(mutation_dir / "tail_taxonomy_quota_leak.csv", tail_mutation_rows, union_fields(tail_mutation_rows))

    for split_name in ["train", "dev", "test"]:
        write_jsonl_atomic(output_dir / "processed" / f"{split_name}.jsonl", [row for row in final_rows if row["split"] == split_name])

    for filename, title in [
        (f"manifest_header_gate_{VERSION_TAG}.md", "candidate manifest header gate"),
        (f"final_package_{VERSION_TAG}.md", "candidate final package"),
        (f"validator_report_{VERSION_TAG}.md", "candidate validator report"),
    ]:
        write_text_atomic(output_dir / "exports" / filename, render_candidate_markdown(title, package_id, len(final_rows)))

    run_manifest = {
        "run_name": output_dir.name,
        "package_id": package_id,
        "version_tag": VERSION_TAG,
        "package_role": PACKAGE_ROLE,
        "compiler_seed": COMPILER_SEED,
        "batch_status": CANDIDATE_BATCH_STATUS,
        "count_reflection_status": CANDIDATE_REFLECTION_STATUS,
        "downstream_consumption_allowed": NO,
        "count_allowed": NO,
        "count_disposition": COUNT_DISPOSITION,
        "promotion_contract_passed": YES,
        "compiler_gate_passed": YES,
        "promotion_contract_status": PROMOTION_CONTRACT_STATUS,
        "candidate_total": len(candidate_rows),
        "accepted_total": len(accepted_rows),
        "final_package_total": len(final_rows),
        "rejected_total": len(rejected_rows),
        "quality_tail_total": len(quality_tail_rows),
        "quota_surplus_total": len(quota_surplus_rows),
        "count_reflection_action": "reviewer_signoff_needed",
        # Post-compile validation은 별도 linter/evidence step에서 닫히지만,
        # handoff 시 run manifest만 열어도 다음 검증 파일 위치를 바로 찾도록 alias를 둔다.
        "post_compile_validation_status": "pending_artifact_linter_and_evidence_card",
        "artifact_linter_report_path": repo_rel(output_dir / "linter" / "artifact_linter_report.md"),
        "evidence_card_summary_path": repo_rel(output_dir / "evidence_card" / "evidence_card_summary.md"),
        "api_calls": 0,
    }
    write_json_atomic(output_dir / "run_manifest.json", run_manifest)

    common_linter_paths = {
        "run_manifest": repo_rel(output_dir / "run_manifest.json"),
        "processed_manifest": repo_rel(output_dir / "processed" / "dataset_manifest.csv"),
        "split_jsonl": [
            repo_rel(output_dir / "processed" / "train.jsonl"),
            repo_rel(output_dir / "processed" / "dev.jsonl"),
            repo_rel(output_dir / "processed" / "test.jsonl"),
        ],
        "final_package_csv": repo_rel(output_dir / "exports" / f"final_package_{VERSION_TAG}.csv"),
        "merged_csv": repo_rel(output_dir / "merged" / f"merged_problem_scores_{VERSION_TAG}.csv"),
        "validator_report_csv": repo_rel(output_dir / "exports" / f"validator_report_{VERSION_TAG}.csv"),
        "rejected_pool_csv": repo_rel(output_dir / "rejected_pool.csv"),
        "tail_taxonomy_csv": repo_rel(output_dir / "tail_taxonomy.csv"),
        "quota_surplus_csv": repo_rel(output_dir / "quota_surplus_pool.csv"),
        "header_gate_md": repo_rel(output_dir / "exports" / f"manifest_header_gate_{VERSION_TAG}.md"),
        "final_package_md": repo_rel(output_dir / "exports" / f"final_package_{VERSION_TAG}.md"),
        "validator_report_md": repo_rel(output_dir / "exports" / f"validator_report_{VERSION_TAG}.md"),
    }
    fixture_manifest = {
        "fixture_version": "package_compiler_minimal_v1",
        "description": "Candidate role fixture for package compiler dry-run.",
        "fixtures": [
            {
                "fixture_id": "package_compiler_candidate_package_pass",
                "artifact_role": PACKAGE_ROLE,
                "fixture_mode": "live_artifact_check",
                "expected_result": "pass",
                "expected_failure_code": "",
                "expected_failure_codes": [],
                "paths": common_linter_paths,
            },
            {
                "fixture_id": "package_compiler_quota_surplus_quality_failure_fail",
                "artifact_role": PACKAGE_ROLE,
                "fixture_mode": "mutation_fixture",
                "expected_result": "fail",
                "expected_failure_code": "pool_tail_split",
                "expected_failure_codes": ["pool_tail_split"],
                "paths": {
                    **common_linter_paths,
                    "quota_surplus_csv": repo_rel(mutation_dir / "quota_surplus_quality_failure.csv"),
                },
            },
            {
                "fixture_id": "package_compiler_tail_taxonomy_quota_leak_fail",
                "artifact_role": PACKAGE_ROLE,
                "fixture_mode": "mutation_fixture",
                "expected_result": "fail",
                "expected_failure_code": "pool_tail_split",
                "expected_failure_codes": ["pool_tail_split"],
                "paths": {
                    **common_linter_paths,
                    "tail_taxonomy_csv": repo_rel(mutation_dir / "tail_taxonomy_quota_leak.csv"),
                },
            },
        ],
    }
    fixture_manifest_path = output_dir / "artifact_linter_fixture_manifest.json"
    write_json_atomic(fixture_manifest_path, fixture_manifest)

    evidence_manifest = {
        "manifest_version": "evidence_card_candidate_v1",
        "description": "Candidate package evidence card input for compiler minimal dry-run.",
        "count_context": {
            "current_usable": current_usable,
            "current_train": current_train,
            "current_eval": current_eval,
        },
        "packages": [
            {
                "package_id": package_id,
                "run_name": output_dir.name,
                "version_tag": VERSION_TAG,
                "package_role": PACKAGE_ROLE,
                "run_dir": repo_rel(output_dir),
                "processed_package_dir": repo_rel(output_dir / "processed"),
                "linter_fixture_id": "package_compiler_candidate_package_pass",
                "linter_report_dir": repo_rel(output_dir / "linter"),
                "source_chain": "fixture candidate pool -> strict gate -> compiled candidate package",
            }
        ],
    }
    evidence_manifest_path = output_dir / "evidence_card_package_manifest.json"
    write_json_atomic(evidence_manifest_path, evidence_manifest)

    compiler_manifest = {
        "compiler_manifest_version": "package_compiler_minimal_v1",
        "package_role": PACKAGE_ROLE,
        "compiler_seed": COMPILER_SEED,
        "selection_policy": "hard/soft/audit/validator/metadata reject, then quota by label with stable risk-score sort",
        "deterministic_tie_breaker": ["risk_score", "candidate_id", "seed_sample_id"],
        "pool_class_policy": {
            "quality_reject": "failed quality, audit, validator, or metadata gate; appears in tail_taxonomy",
            "quota_surplus": "strict gate accepted but not selected due package quota; not a quality failure",
        },
        "row_counts": {
            "candidate_total": len(candidate_rows),
            "accepted_total": len(accepted_rows),
            "final_package_total": len(final_rows),
            "rejected_total": len(rejected_rows),
            "quality_tail_total": len(quality_tail_rows),
            "quota_surplus_total": len(quota_surplus_rows),
        },
        "current_count_before": {"usable": current_usable, "train": current_train, "eval": current_eval},
        "proposed_delta": {
            "usable": len(final_rows),
            "train": len([row for row in final_rows if row["split"] == "train"]),
            "eval": len([row for row in final_rows if row["split"] in {"dev", "test"}]),
        },
        "artifacts": {
            "run_manifest": repo_rel(output_dir / "run_manifest.json"),
            "fixture_manifest": repo_rel(fixture_manifest_path),
            "evidence_manifest": repo_rel(evidence_manifest_path),
            "candidate_pool_path": repo_rel(output_dir / "candidate_pool.csv"),
            "accepted_pool_path": repo_rel(output_dir / "accepted_pool.csv"),
            "rejected_pool_path": repo_rel(output_dir / "rejected_pool.csv"),
            "tail_taxonomy_path": repo_rel(output_dir / "tail_taxonomy.csv"),
            "quota_surplus_pool_path": repo_rel(output_dir / "quota_surplus_pool.csv"),
            "final_package_path": repo_rel(output_dir / "exports" / f"final_package_{VERSION_TAG}.csv"),
            "processed_manifest_path": repo_rel(output_dir / "processed" / "dataset_manifest.csv"),
            "mutation_quota_surplus_quality_failure_path": repo_rel(mutation_dir / "quota_surplus_quality_failure.csv"),
            "mutation_tail_taxonomy_quota_leak_path": repo_rel(mutation_dir / "tail_taxonomy_quota_leak.csv"),
        },
    }
    write_json_atomic(output_dir / "compiler_manifest.json", compiler_manifest)
    write_text_atomic(
        output_dir / "compiler_summary.md",
        "\n".join(
            [
                "# objective package compiler minimal dry-run",
                "",
                f"- package_role: `{PACKAGE_ROLE}`",
                f"- candidate_total: `{len(candidate_rows)}`",
                f"- accepted_total: `{len(accepted_rows)}`",
                f"- final_package_total: `{len(final_rows)}`",
                f"- rejected_total: `{len(rejected_rows)}`",
                f"- quality_tail_total: `{len(quality_tail_rows)}`",
                f"- quota_surplus_total: `{len(quota_surplus_rows)}`",
                "- api_calls: `0`",
                "- count_reflection_status: `not_counted_until_reviewer_signoff`",
                "- downstream_consumption_allowed: `아니오`",
                "- count_allowed: `아니오`",
                f"- count_disposition: `{COUNT_DISPOSITION}`",
                f"- compiler_seed: `{COMPILER_SEED}`",
                "- compiler_gate_passed: `예`",
                "- promotion_contract_passed: `예`",
                f"- promotion_contract_status: `{PROMOTION_CONTRACT_STATUS}`",
                "",
            ]
        ),
    )
    return {
        "output_dir": output_dir,
        "fixture_manifest": fixture_manifest_path,
        "evidence_manifest": evidence_manifest_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a no-API minimal package compiler dry-run.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--current-usable", type=int, required=True)
    parser.add_argument("--current-train", type=int, required=True)
    parser.add_argument("--current-eval", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / f"{build_run_stamp()}_objective_package_compiler_minimal_dry_run"
    result = write_outputs(output_dir, args.current_usable, args.current_train, args.current_eval)
    print(f"package_compiler_output={result['output_dir']}")
    print(f"artifact_linter_fixture_manifest={result['fixture_manifest']}")
    print(f"evidence_card_package_manifest={result['evidence_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
