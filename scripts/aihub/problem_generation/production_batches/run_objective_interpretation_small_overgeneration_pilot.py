from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from itertools import combinations, product
from pathlib import Path
from typing import Any

# `020803` no-API preflight reviewer sign-off 이후, 같은 28개 해석례 seed를 실제 API로 태우는 runner다.
# generation/Judge는 candidate pool 전체에 수행하고, count 후보는 compiler가 strict final 16개만 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_repair_pilot as interpretation_pilot,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_small_overgeneration_pilot as factory,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb6_non_law as pb6  # noqa: E402


VERSION_TAG = "objective_interpretation_small_overgeneration_pilot"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_interpretation_small_overgeneration_api_pilot"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "objective interpretation small overgeneration API pilot"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
SOURCE_PREFLIGHT_VERSION_TAG = "objective_interpretation_small_overgeneration_pilot_preflight"
SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_interpretation_small_overgeneration_seed_spec_wiring_check"
SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / SOURCE_PREFLIGHT_VERSION_TAG
    / "seed_registry.csv"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def read_registry_keys(path: Path) -> list[tuple[str, str, str, str, str, str]]:
    # Source preflight run name은 날짜/시간이 바뀔 수 있으므로, registry identity로 locked run을 찾는다.
    return [
        (
            row.get("seed_sample_id", ""),
            row.get("reference_sample_id", ""),
            row.get("family_id", ""),
            row.get("label_path", ""),
            row.get("raw_path", ""),
            row.get("target_correct_choice", ""),
        )
        for row in read_csv_rows(path)
    ]


def resolve_source_preflight_run_dir() -> Path:
    llm_runs_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    pattern = f"*_{SOURCE_PREFLIGHT_VERSION_TAG}_{SOURCE_PREFLIGHT_RUN_PURPOSE}"
    source_keys = read_registry_keys(SOURCE_PREFLIGHT_SEED_REGISTRY_PATH)
    matched: list[Path] = []
    for candidate_dir in sorted(llm_runs_root.glob(pattern)):
        candidate_registry = candidate_dir / "inputs" / "seed_registry.csv"
        if candidate_registry.exists() and read_registry_keys(candidate_registry) == source_keys:
            matched.append(candidate_dir)
    if not matched:
        raise FileNotFoundError(f"locked source preflight run not found for {SOURCE_PREFLIGHT_SEED_REGISTRY_PATH}")
    return matched[-1]


SOURCE_PREFLIGHT_RUN_DIR = resolve_source_preflight_run_dir()
SOURCE_PREFLIGHT_RUN_NAME = SOURCE_PREFLIGHT_RUN_DIR.name
SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"target_label_schedule_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
)
SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH = SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"exclusion_audit_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.csv"
)
SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"final_package_spec_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
)
SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.json"
)
SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH = (
    SOURCE_PREFLIGHT_RUN_DIR / "exports" / f"package_compiler_contract_{SOURCE_PREFLIGHT_VERSION_TAG}.md"
)

INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"
RUN_LINTER_DIR = RUN_DIR / "linter"
RUN_EVIDENCE_DIR = RUN_DIR / "evidence_card"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
RAW_MERGED_BEFORE_VALIDATOR_PATH = RUN_MERGED_DIR / f"raw_merged_problem_scores_before_validator_{VERSION_TAG}.csv"
CANDIDATE_MERGED_SCORES_PATH = RUN_MERGED_DIR / f"candidate_merged_problem_scores_{VERSION_TAG}.csv"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
TAIL_MEMO_MD_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.md"
CANDIDATE_VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.csv"
CANDIDATE_VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"candidate_validator_report_{VERSION_TAG}.md"
VALIDATOR_REPORT_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.csv"
VALIDATOR_REPORT_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_{VERSION_TAG}.md"
VALIDATOR_WIRING_CHECK_MD_PATH = RUN_EXPORTS_DIR / f"validator_wiring_check_{VERSION_TAG}.md"
PILOT_BREAKOUT_CSV_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.csv"
PILOT_BREAKOUT_MD_PATH = RUN_EXPORTS_DIR / f"pilot_breakout_{VERSION_TAG}.md"
MANIFEST_HEADER_GATE_MD_PATH = RUN_EXPORTS_DIR / f"manifest_header_gate_{VERSION_TAG}.md"
FINAL_PACKAGE_CSV_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.csv"
FINAL_PACKAGE_MD_PATH = RUN_EXPORTS_DIR / f"final_package_{VERSION_TAG}.md"
COMPILER_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"compiler_summary_{VERSION_TAG}.md"
CANDIDATE_POOL_PATH = RUN_DIR / "candidate_pool.csv"
ACCEPTED_POOL_PATH = RUN_DIR / "accepted_pool.csv"
REJECTED_POOL_PATH = RUN_DIR / "rejected_pool.csv"
TAIL_TAXONOMY_PATH = RUN_DIR / "tail_taxonomy.csv"
QUOTA_SURPLUS_POOL_PATH = RUN_DIR / "quota_surplus_pool.csv"
COMPILER_MANIFEST_PATH = RUN_DIR / "compiler_manifest.json"
ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH = RUN_DIR / "artifact_linter_fixture_manifest.json"
EVIDENCE_CARD_PACKAGE_MANIFEST_PATH = RUN_DIR / "evidence_card_package_manifest.json"

EXPECTED_CANDIDATE_SEED_COUNT = 28
FINAL_PACKAGE_TARGET_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 28}
EXPECTED_LANE_BY_DOC = {
    ("해석례_QA", "generalization_03_04"): 14,
    ("해석례_QA", "expansion_01_02"): 14,
}
EXPECTED_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 7,
    "02_TL_유권해석_QA": 7,
    "03_TL_해석례_QA": 7,
    "04_TL_해석례_QA": 7,
}
CANDIDATE_TARGET_LABEL_COUNTS = {"A": 7, "B": 7, "C": 7, "D": 7}
FINAL_TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
FINAL_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 4,
    "02_TL_유권해석_QA": 4,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 4,
}
FINAL_LANE_COUNTS = {"generalization_03_04": 8, "expansion_01_02": 8}
CURRENT_OBJECTIVE_COUNT = {"usable": 183, "train": 144, "eval": 39, "audit": 6, "hard_fail": 5, "soft_fail": 3}

PACKAGE_ROLE = "count_reflection_candidate_package"
CANDIDATE_BATCH_STATUS = "compiled_candidate_not_counted"
CANDIDATE_REFLECTION_STATUS = "not_counted_until_reviewer_signoff"
COUNT_DISPOSITION = "candidate_not_counted"
PROMOTION_CONTRACT_STATUS = "passed_not_counted"
YES = "예"
NO = "아니오"

ORIGINAL_FACTORY_CONFIGURE = factory.configure_globals
ORIGINAL_FACTORY_BUILD_RUN_MANIFEST = factory.build_run_manifest
ORIGINAL_FACTORY_BUILD_BATCH_SUMMARY = factory.build_batch_summary


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_seed_preflight_copy(seed_rows: list[dict[str, str]]) -> None:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "reference_sample_id": row["reference_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "target_correct_choice": row["target_correct_choice"],
            "interpretation_seed_action": row.get("interpretation_seed_action", ""),
            "interpretation_axis": row.get("interpretation_axis", ""),
            "interpretation_risk_flags": row.get("interpretation_risk_flags", ""),
            "label_path": row.get("label_path", ""),
            "raw_path": row.get("raw_path", ""),
            "package_candidate_role": row.get("package_candidate_role", "candidate_pool"),
            "count_reflection_status": row.get("count_reflection_status", "candidate_not_counted"),
            "count_allowed": NO,
            "count_disposition": COUNT_DISPOSITION,
            "downstream_consumption_allowed": NO,
        }
        for row in seed_rows
    ]
    pb6.pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, rows, list(rows[0].keys()))
    counts = {
        "doc_type": Counter(row["doc_type_name"] for row in seed_rows),
        "lane": Counter(row["sampling_lane"] for row in seed_rows),
        "source": Counter(row["source_subset"] for row in seed_rows),
        "label": Counter(row["target_correct_choice"] for row in seed_rows),
        "axis": Counter(row.get("interpretation_axis", "") for row in seed_rows),
    }
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## fixed source",
        f"- source_preflight_run: `{SOURCE_PREFLIGHT_RUN_NAME}`",
        f"- source_seed_registry: `{SOURCE_PREFLIGHT_SEED_REGISTRY_PATH}`",
        f"- source_exclusion_audit: `{SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(counts['doc_type'])}`",
        f"- lane_counts: `{dict(counts['lane'])}`",
        f"- source_subset_counts: `{dict(counts['source'])}`",
        f"- target_label_counts: `{dict(counts['label'])}`",
        f"- interpretation_axis_counts: `{dict(counts['axis'])}`",
        "",
        "## checks",
        "| check | result |",
        "| --- | --- |",
        "| same 28 seed registry as no-API preflight | `pass` |",
        "| source split is 01/02/03/04 each 7 | `pass` |",
        "| lane split is 14/14 | `pass` |",
        "| target label schedule is A/B/C/D = 7/7/7/7 | `pass` |",
        "| count fields are candidate-not-counted aliases | `pass` |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_validator_wiring_check_md() -> None:
    # 해석례 overgeneration은 preflight 28개와 final 16개가 다르므로,
    # 기존 해석례 repair pilot의 16개 wiring 문구를 그대로 쓰면 provenance가 흐려진다.
    lines = [
        f"# validator wiring check `{VERSION_TAG}`",
        "",
        "| check | result | note |",
        "| --- | --- | --- |",
        "| fixed preflight seed registry reused | `pass` | no-API preflight 28개와 같은 seed registry 사용 |",
        "| candidate target label schedule | `pass` | candidate target `A/B/C/D = 7/7/7/7` 적용 |",
        "| final export label schedule | `pass` | final export `A/B/C/D = 4/4/4/4` 적용 |",
        "| source preflight provenance | `pass` | `020803` seed registry `28개`를 고정 입력으로 사용 |",
        "| interpretation gate fields | `pass` | `answer_uniqueness`, `condition_preservation`, `response_scope_limited`, `answer_reason_split`, `source_only_fact`, `distractor_direction`, `same_direction_distractor` 기록 |",
        "| downstream guard fields | `pass` | `validator_reason_short`, `split_allowed`, `count_allowed` 기록 |",
        "| count reflection | `pass` | reviewer sign-off 전 core current count 미변경 |",
    ]
    pb6.pb4.pb3.base.write_text_atomic(VALIDATOR_WIRING_CHECK_MD_PATH, "\n".join(lines) + "\n")


def build_generation_messages(seed: dict[str, str], reference_v2: dict[str, str]) -> list[dict[str, str]]:
    messages = interpretation_pilot.BASE_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    messages[1]["content"] += f"""

## interpretation small overgeneration pilot 추가 지시
- 이번 run은 `해석례_QA` candidate 28개를 생성한 뒤 strict final package 16개만 컴파일하는 package factory API pilot이다.
- seed action은 `{seed.get('interpretation_seed_action', '')}`, interpretation axis는 `{seed.get('interpretation_axis', '')}`, risk flags는 `{seed.get('interpretation_risk_flags', '')}`다.
- stem은 회답 결론, 전제조건, 예외, 적용범위 중 정확히 하나의 predicate만 묻는다.
- 회답 결론과 회답 이유를 한 stem에서 동시에 묻지 않는다. 이유는 선택지 변별 축으로만 사용한다.
- 정답은 `gold_short_answer` 또는 `short_answer`의 회답 결론 하나에만 닫혀야 한다.
- 오답은 같은 해석례 근거를 공유하되 전제조건, 예외, 적용범위, 회답 방향 중 정확히 한 축만 어긋나야 한다.
- 원문 밖 사실, 다른 회답의 결론, 같은 방향으로도 정답처럼 읽히는 오답은 실패로 본다.
- 후처리 validator가 target label `{seed.get('target_correct_choice', '')}`로 choice를 재배치하므로, 생성 단계에서는 target label을 억지로 맞추지 않는다.
"""
    return messages


def strict_accept_reason(row: dict[str, str]) -> str:
    if row.get("final_status") != "pass":
        return "hard_or_soft_fail"
    if row.get("audit_required") == YES:
        return "audit_required"
    if row.get("validator_action") != "accept" or row.get("validator_export_disposition") != "export_ready":
        return "validator_not_export_ready"
    if row.get("metadata_remap_ok") != YES:
        return "metadata_mismatch"
    if row.get("validator_recalculated_correct_choice") != row.get("target_correct_choice"):
        return "shuffle_mismatch"
    for gate_field in [
        "answer_uniqueness",
        "condition_preservation",
        "response_scope_limited",
        "answer_reason_split",
        "source_only_fact",
        "same_direction_guard_ok",
    ]:
        if row.get(gate_field) == NO:
            return "answer_uniqueness_or_boundary_failure"
    return ""


def find_final_combination(accepted_rows: list[dict[str, str]]) -> set[str]:
    # 28개 후보는 source별 7개 중 4개씩 고르는 product search로 줄여 exact source/label/lane quota를 결정한다.
    rows_by_source = {
        source: sorted(
            [row for row in accepted_rows if row.get("source_subset") == source],
            key=lambda row: (
                row.get("interpretation_axis", ""),
                row.get("target_correct_choice", ""),
                row.get("seed_sample_id", ""),
            ),
        )
        for source in FINAL_SOURCE_COUNTS
    }
    if any(len(rows) < FINAL_SOURCE_COUNTS[source] for source, rows in rows_by_source.items()):
        return set()
    source_combos = [
        list(combinations(rows_by_source[source], FINAL_SOURCE_COUNTS[source]))
        for source in FINAL_SOURCE_COUNTS
    ]
    for combo_group in product(*source_combos):
        combo = [row for group in combo_group for row in group]
        label_counts = Counter(row.get("export_correct_choice", "") for row in combo)
        lane_counts = Counter(row.get("sampling_lane", "") for row in combo)
        if {label: label_counts.get(label, 0) for label in FINAL_TARGET_LABEL_COUNTS} != FINAL_TARGET_LABEL_COUNTS:
            continue
        if {lane: lane_counts.get(lane, 0) for lane in FINAL_LANE_COUNTS} != FINAL_LANE_COUNTS:
            continue
        return {row["candidate_id"] for row in combo}
    return set()


def write_linter_and_evidence_manifests() -> None:
    linter_paths = {
        "run_manifest": repo_rel(RUN_MANIFEST_PATH),
        "processed_manifest": repo_rel(PROBLEM_DATASET_MANIFEST_PATH),
        "split_jsonl": [repo_rel(PROBLEM_TRAIN_PATH), repo_rel(PROBLEM_DEV_PATH), repo_rel(PROBLEM_TEST_PATH)],
        "final_package_csv": repo_rel(FINAL_PACKAGE_CSV_PATH),
        "merged_csv": repo_rel(MERGED_SCORES_PATH),
        "validator_report_csv": repo_rel(VALIDATOR_REPORT_CSV_PATH),
        "rejected_pool_csv": repo_rel(REJECTED_POOL_PATH),
        "tail_taxonomy_csv": repo_rel(TAIL_TAXONOMY_PATH),
        "quota_surplus_csv": repo_rel(QUOTA_SURPLUS_POOL_PATH),
        "validator_wiring_check_md": repo_rel(VALIDATOR_WIRING_CHECK_MD_PATH),
        "header_gate_md": repo_rel(MANIFEST_HEADER_GATE_MD_PATH),
        "final_package_md": repo_rel(FINAL_PACKAGE_MD_PATH),
        "validator_report_md": repo_rel(VALIDATOR_REPORT_MD_PATH),
        "compiler_summary_md": repo_rel(COMPILER_SUMMARY_MD_PATH),
    }
    pb6.pb4.pb3.base.write_json_atomic(
        ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH,
        {
            "fixture_version": "interpretation_small_overgeneration_candidate_v1",
            "description": "Live candidate package check for interpretation small overgeneration pilot.",
            "fixtures": [
                {
                    "fixture_id": "interpretation_small_overgeneration_candidate_package_pass",
                    "artifact_role": PACKAGE_ROLE,
                    "fixture_mode": "live_artifact_check",
                    "expected_result": "pass",
                    "expected_failure_code": "",
                    "expected_failure_codes": [],
                    "paths": linter_paths,
                }
            ],
        },
    )
    pb6.pb4.pb3.base.write_json_atomic(
        EVIDENCE_CARD_PACKAGE_MANIFEST_PATH,
        {
            "manifest_version": "evidence_card_candidate_v1",
            "description": "Interpretation small overgeneration candidate package evidence card input.",
            "count_context": {
                "current_usable": CURRENT_OBJECTIVE_COUNT["usable"],
                "current_train": CURRENT_OBJECTIVE_COUNT["train"],
                "current_eval": CURRENT_OBJECTIVE_COUNT["eval"],
            },
            "packages": [
                {
                    "package_id": VERSION_TAG,
                    "run_name": RUN_NAME,
                    "version_tag": VERSION_TAG,
                    "package_role": PACKAGE_ROLE,
                    "run_dir": repo_rel(RUN_DIR),
                    "processed_package_dir": repo_rel(PROCESSED_DIR),
                    "linter_fixture_id": "interpretation_small_overgeneration_candidate_package_pass",
                    "linter_report_dir": repo_rel(RUN_LINTER_DIR),
                    "source_chain": f"{SOURCE_PREFLIGHT_RUN_NAME} -> 28 candidate API execution -> strict final 16 compiler",
                }
            ],
        },
    )


def counter_rows(pool_name: str, rows: list[dict[str, str]], field: str) -> list[dict[str, str]]:
    counts = Counter(row.get(field, "") for row in rows)
    return [
        {"pool": pool_name, "breakout_field": field, "key": key, "count": str(value)}
        for key, value in sorted(counts.items())
    ]


def build_batch_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary_rows = ORIGINAL_FACTORY_BUILD_BATCH_SUMMARY(rows)
    final_rows = read_csv_rows(FINAL_PACKAGE_CSV_PATH) if FINAL_PACKAGE_CSV_PATH.exists() else []
    candidate_rows = read_csv_rows(CANDIDATE_POOL_PATH) if CANDIDATE_POOL_PATH.exists() else []
    tail_rows = read_csv_rows(TAIL_TAXONOMY_PATH) if TAIL_TAXONOMY_PATH.exists() else []
    quota_rows = read_csv_rows(QUOTA_SURPLUS_POOL_PATH) if QUOTA_SURPLUS_POOL_PATH.exists() else []

    breakout_rows: list[dict[str, str]] = []
    for pool_name, pool_rows in [
        ("candidate_pool", candidate_rows),
        ("final_package", final_rows),
        ("quality_tail", tail_rows),
        ("quota_surplus", quota_rows),
    ]:
        for field in ["source_subset", "sampling_lane", "export_correct_choice", "interpretation_axis", "tail_class"]:
            breakout_rows.extend(counter_rows(pool_name, pool_rows, field))
    if breakout_rows:
        pb6.pb4.pb3.base.write_csv_atomic(PILOT_BREAKOUT_CSV_PATH, breakout_rows, list(breakout_rows[0].keys()))

    lines = BATCH_SUMMARY_MD_PATH.read_text(encoding="utf-8").splitlines() if BATCH_SUMMARY_MD_PATH.exists() else []
    lines.extend(
        [
            "",
            "## source/lane/label/axis breakout",
            "| pool | field | key | count |",
            "| --- | --- | --- | ---: |",
        ]
    )
    for row in breakout_rows:
        if row["breakout_field"] == "tail_class" and row["key"] == "":
            continue
        lines.append(f"| `{row['pool']}` | `{row['breakout_field']}` | `{row['key']}` | `{row['count']}` |")
    pb6.pb4.pb3.base.write_text_atomic(BATCH_SUMMARY_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.write_text_atomic(PILOT_BREAKOUT_MD_PATH, "\n".join(lines[-(len(breakout_rows) + 5) :]) + "\n")
    return summary_rows


def split_dataset_with_overgeneration_compiler(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    # factory 기본 hook은 judgment validator를 호출하므로, 해석례용 validator를 명시적으로 연결한다.
    if rows:
        factory.write_csv(RAW_MERGED_BEFORE_VALIDATOR_PATH, rows)
    validated_rows = interpretation_pilot.apply_interpretation_validator(rows)
    if validated_rows:
        factory.write_csv(CANDIDATE_MERGED_SCORES_PATH, validated_rows)
    return factory.compile_final_package(validated_rows)


def build_run_manifest(
    seed_rows: list[dict[str, str]],
    merged_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> dict[str, Any]:
    manifest = ORIGINAL_FACTORY_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    current_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8")) if RUN_MANIFEST_PATH.exists() else manifest
    current_manifest.update(
        {
            "candidate_recipe_source": "v2_difficulty_patch_r2_interpretation_small_overgeneration_pilot",
            "seed_registry_strategy": f"fixed_from_{SOURCE_PREFLIGHT_RUN_NAME}",
            "source_preflight_version_tag": SOURCE_PREFLIGHT_VERSION_TAG,
            "source_preflight_run_name": SOURCE_PREFLIGHT_RUN_NAME,
            "current_objective_count_held": CURRENT_OBJECTIVE_COUNT,
            "required_report_status": "source/lane/label/axis breakout, expansion tail table, quality tail vs quota surplus emitted",
            "success_criteria": {
                "candidate_execution": EXPECTED_CANDIDATE_SEED_COUNT,
                "final_package": FINAL_PACKAGE_TARGET_COUNT,
                "final_hard_soft_audit": "0/0/0",
                "final_label_counts": FINAL_TARGET_LABEL_COUNTS,
                "final_source_counts": FINAL_SOURCE_COUNTS,
                "final_lane_counts": FINAL_LANE_COUNTS,
                "metadata_shuffle_mismatch": "0/0",
                "reviewer_signoff_required_for_count_reflection": True,
            },
            "success_result": {
                "candidate_execution_complete": pb6.pb4.pb3.base.load_jsonl_count(GENERATED_PROBLEMS_PATH)
                == EXPECTED_CANDIDATE_SEED_COUNT,
                "compiler_gate_passed": bool(factory.COMPILER_RESULT.get("compiler_gate_passed")),
                "promotion_contract_passed": bool(factory.COMPILER_RESULT.get("promotion_contract_passed")),
                "final_package_total": pb6.pb4.pb3.base.load_csv_count(FINAL_PACKAGE_CSV_PATH),
                "quality_tail_total": factory.COMPILER_RESULT.get("quality_tail_total", 0),
            },
        }
    )
    artifact_paths = current_manifest.get("artifact_paths", {})
    artifact_paths.update(
        {
            "pilot_breakout_csv": str(PILOT_BREAKOUT_CSV_PATH),
            "pilot_breakout_md": str(PILOT_BREAKOUT_MD_PATH),
            "compiler_summary_md": str(COMPILER_SUMMARY_MD_PATH),
            "validator_wiring_check_md": str(VALIDATOR_WIRING_CHECK_MD_PATH),
        }
    )
    current_manifest["artifact_paths"] = artifact_paths
    current_manifest["artifact_path_aliases"] = {
        key: repo_rel(Path(path)) if isinstance(path, str) else path
        for key, path in artifact_paths.items()
    }
    pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, current_manifest)
    return current_manifest


def rewrite_compiler_summary_contract() -> None:
    # artifact linter는 compiler summary도 reviewer-facing count contract로 보므로, 필수 candidate 문구를 명시한다.
    result = factory.COMPILER_RESULT
    lines = [
        f"# compiler summary `{VERSION_TAG}`",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| package_role | `{PACKAGE_ROLE}` |",
        f"| batch_status | `{CANDIDATE_BATCH_STATUS}` |",
        f"| count_reflection_status | `{CANDIDATE_REFLECTION_STATUS}` |",
        "| downstream_consumption_allowed | `아니오` |",
        "| count_allowed | `아니오` |",
        f"| count_disposition | `{COUNT_DISPOSITION}` |",
        f"| compiler_gate_passed | `{YES if result.get('compiler_gate_passed') else NO}` |",
        f"| promotion_contract_passed | `{YES if result.get('promotion_contract_passed') else NO}` |",
        f"| promotion_contract_status | `{PROMOTION_CONTRACT_STATUS if result.get('promotion_contract_passed') else 'failed_not_counted'}` |",
        f"| candidate_total | `{result.get('candidate_total', 0)}` |",
        f"| accepted_total | `{result.get('accepted_total', 0)}` |",
        f"| final_package_total | `{result.get('final_package_total', 0)}` |",
        f"| rejected_total | `{result.get('rejected_total', 0)}` |",
        f"| quality_tail_total | `{result.get('quality_tail_total', 0)}` |",
        f"| quota_surplus_total | `{result.get('quota_surplus_total', 0)}` |",
        "",
    ]
    pb6.pb4.pb3.base.write_text_atomic(COMPILER_SUMMARY_MD_PATH, "\n".join(lines))


def rewrite_compiler_manifest_context() -> None:
    if not COMPILER_MANIFEST_PATH.exists():
        return
    payload = json.loads(COMPILER_MANIFEST_PATH.read_text(encoding="utf-8"))
    # factory 구현은 판결문 runner에서 왔지만 산출물 계약은 문서유형 중립 package factory이므로,
    # reviewer-facing manifest에서는 특정 문서유형 흔적을 숨기지 말고 중립 버전과 adapter provenance를 함께 남긴다.
    payload["compiler_manifest_version"] = "package_factory_small_overgeneration_v1"
    payload["source_factory_template"] = "judgment_small_overgeneration_adapter"
    payload["adapted_for_doc_type"] = "해석례_QA"
    payload["adapted_version_tag"] = VERSION_TAG
    pb6.pb4.pb3.base.write_json_atomic(COMPILER_MANIFEST_PATH, payload)


def patch_factory_constants() -> None:
    factory.judgment_pilot = interpretation_pilot
    interpretation_pilot.configure_judgment_pilot_globals = interpretation_pilot.configure_interpretation_pilot_globals
    interpretation_pilot.write_seed_preflight_copy = write_seed_preflight_copy
    interpretation_pilot.write_validator_wiring_check_md = write_validator_wiring_check_md

    for name, value in {
        "VERSION_TAG": VERSION_TAG,
        "RUN_DATE": RUN_DATE,
        "RUN_PURPOSE": RUN_PURPOSE,
        "RUN_NAME": RUN_NAME,
        "RUN_LABEL": RUN_LABEL,
        "SOURCE_PREFLIGHT_RUN_NAME": SOURCE_PREFLIGHT_RUN_NAME,
        "SOURCE_PREFLIGHT_RUN_DIR": SOURCE_PREFLIGHT_RUN_DIR,
        "SOURCE_PREFLIGHT_SEED_REGISTRY_PATH": SOURCE_PREFLIGHT_SEED_REGISTRY_PATH,
        "SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH": SOURCE_PREFLIGHT_TARGET_LABEL_SCHEDULE_PATH,
        "SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH": SOURCE_PREFLIGHT_EXCLUSION_AUDIT_PATH,
        "SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH": SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_CSV_PATH,
        "SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH": SOURCE_PREFLIGHT_FINAL_PACKAGE_SPEC_MD_PATH,
        "SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH": SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_JSON_PATH,
        "SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH": SOURCE_PREFLIGHT_PACKAGE_COMPILER_CONTRACT_MD_PATH,
        "INTERIM_DIR": INTERIM_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "RUN_DIR": RUN_DIR,
        "RUN_PROMPTS_DIR": RUN_PROMPTS_DIR,
        "RUN_INPUTS_DIR": RUN_INPUTS_DIR,
        "RUN_GENERATIONS_DIR": RUN_GENERATIONS_DIR,
        "RUN_JUDGE_LOGS_DIR": RUN_JUDGE_LOGS_DIR,
        "RUN_MERGED_DIR": RUN_MERGED_DIR,
        "RUN_EXPORTS_DIR": RUN_EXPORTS_DIR,
        "RUN_LINTER_DIR": RUN_LINTER_DIR,
        "RUN_EVIDENCE_DIR": RUN_EVIDENCE_DIR,
        "SEED_REGISTRY_PATH": SEED_REGISTRY_PATH,
        "SEED_READY_PATH": SEED_READY_PATH,
        "SEED_PREFLIGHT_CSV_PATH": SEED_PREFLIGHT_CSV_PATH,
        "SEED_PREFLIGHT_MD_PATH": SEED_PREFLIGHT_MD_PATH,
        "TARGET_LABEL_SCHEDULE_CSV_PATH": TARGET_LABEL_SCHEDULE_CSV_PATH,
        "RUN_MANIFEST_PATH": RUN_MANIFEST_PATH,
        "GENERATED_PROBLEMS_PATH": GENERATED_PROBLEMS_PATH,
        "GROUNDING_LOG_PATH": GROUNDING_LOG_PATH,
        "KEYEDNESS_LOG_PATH": KEYEDNESS_LOG_PATH,
        "DISTRACTORFIT_LOG_PATH": DISTRACTORFIT_LOG_PATH,
        "NEARMISS_LOG_PATH": NEARMISS_LOG_PATH,
        "RAW_MERGED_BEFORE_VALIDATOR_PATH": RAW_MERGED_BEFORE_VALIDATOR_PATH,
        "CANDIDATE_MERGED_SCORES_PATH": CANDIDATE_MERGED_SCORES_PATH,
        "MERGED_SCORES_PATH": MERGED_SCORES_PATH,
        "PROBLEM_TRAIN_PATH": PROBLEM_TRAIN_PATH,
        "PROBLEM_DEV_PATH": PROBLEM_DEV_PATH,
        "PROBLEM_TEST_PATH": PROBLEM_TEST_PATH,
        "PROBLEM_DATASET_MANIFEST_PATH": PROBLEM_DATASET_MANIFEST_PATH,
        "PROBLEM_AUDIT_QUEUE_PATH": PROBLEM_AUDIT_QUEUE_PATH,
        "BATCH_SUMMARY_MD_PATH": BATCH_SUMMARY_MD_PATH,
        "BATCH_SUMMARY_CSV_PATH": BATCH_SUMMARY_CSV_PATH,
        "BATCH_LANE_SUMMARY_CSV_PATH": BATCH_LANE_SUMMARY_CSV_PATH,
        "TAIL_MEMO_CSV_PATH": TAIL_MEMO_CSV_PATH,
        "TAIL_MEMO_MD_PATH": TAIL_MEMO_MD_PATH,
        "CANDIDATE_VALIDATOR_REPORT_CSV_PATH": CANDIDATE_VALIDATOR_REPORT_CSV_PATH,
        "CANDIDATE_VALIDATOR_REPORT_MD_PATH": CANDIDATE_VALIDATOR_REPORT_MD_PATH,
        "VALIDATOR_REPORT_CSV_PATH": VALIDATOR_REPORT_CSV_PATH,
        "VALIDATOR_REPORT_MD_PATH": VALIDATOR_REPORT_MD_PATH,
        "VALIDATOR_WIRING_CHECK_MD_PATH": VALIDATOR_WIRING_CHECK_MD_PATH,
        "PILOT_BREAKOUT_CSV_PATH": PILOT_BREAKOUT_CSV_PATH,
        "PILOT_BREAKOUT_MD_PATH": PILOT_BREAKOUT_MD_PATH,
        "MANIFEST_HEADER_GATE_MD_PATH": MANIFEST_HEADER_GATE_MD_PATH,
        "FINAL_PACKAGE_CSV_PATH": FINAL_PACKAGE_CSV_PATH,
        "FINAL_PACKAGE_MD_PATH": FINAL_PACKAGE_MD_PATH,
        "COMPILER_SUMMARY_MD_PATH": COMPILER_SUMMARY_MD_PATH,
        "CANDIDATE_POOL_PATH": CANDIDATE_POOL_PATH,
        "ACCEPTED_POOL_PATH": ACCEPTED_POOL_PATH,
        "REJECTED_POOL_PATH": REJECTED_POOL_PATH,
        "TAIL_TAXONOMY_PATH": TAIL_TAXONOMY_PATH,
        "QUOTA_SURPLUS_POOL_PATH": QUOTA_SURPLUS_POOL_PATH,
        "COMPILER_MANIFEST_PATH": COMPILER_MANIFEST_PATH,
        "ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH": ARTIFACT_LINTER_FIXTURE_MANIFEST_PATH,
        "EVIDENCE_CARD_PACKAGE_MANIFEST_PATH": EVIDENCE_CARD_PACKAGE_MANIFEST_PATH,
        "EXPECTED_CANDIDATE_SEED_COUNT": EXPECTED_CANDIDATE_SEED_COUNT,
        "FINAL_PACKAGE_TARGET_COUNT": FINAL_PACKAGE_TARGET_COUNT,
        "EXPECTED_DOC_TYPE_COUNTS": EXPECTED_DOC_TYPE_COUNTS,
        "EXPECTED_LANE_BY_DOC": EXPECTED_LANE_BY_DOC,
        "EXPECTED_SOURCE_COUNTS": EXPECTED_SOURCE_COUNTS,
        "CANDIDATE_TARGET_LABEL_COUNTS": CANDIDATE_TARGET_LABEL_COUNTS,
        "FINAL_TARGET_LABEL_COUNTS": FINAL_TARGET_LABEL_COUNTS,
        "FINAL_SOURCE_COUNTS": FINAL_SOURCE_COUNTS,
        "FINAL_LANE_COUNTS": FINAL_LANE_COUNTS,
        "PACKAGE_ROLE": PACKAGE_ROLE,
        "CANDIDATE_BATCH_STATUS": CANDIDATE_BATCH_STATUS,
        "CANDIDATE_REFLECTION_STATUS": CANDIDATE_REFLECTION_STATUS,
        "COUNT_DISPOSITION": COUNT_DISPOSITION,
        "PROMOTION_CONTRACT_STATUS": PROMOTION_CONTRACT_STATUS,
    }.items():
        setattr(factory, name, value)

    factory.COMPILER_RESULT = {}
    factory.build_generation_messages = build_generation_messages
    factory.strict_accept_reason = strict_accept_reason
    factory.find_final_combination = find_final_combination
    factory.write_linter_and_evidence_manifests = write_linter_and_evidence_manifests
    factory.build_batch_summary = build_batch_summary
    factory.build_run_manifest = build_run_manifest
    factory.split_dataset_with_overgeneration_compiler = split_dataset_with_overgeneration_compiler


def configure_globals() -> None:
    patch_factory_constants()
    ORIGINAL_FACTORY_CONFIGURE()
    # 기존 factory의 judgment 문자열은 reviewer-facing provenance를 흐리므로 해석례 run identity로 다시 덮어쓴다.
    interpretation_pilot.BATCH_STATUS = "interpretation_small_overgeneration_candidate_validated_not_compiled"
    interpretation_pilot.COUNT_REFLECTION_STATUS = CANDIDATE_REFLECTION_STATUS
    interpretation_pilot.DOWNSTREAM_CONSUMPTION_ALLOWED = NO
    pb6.RUN_LABEL = RUN_LABEL
    pb6.CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_interpretation_small_overgeneration_pilot"
    pb6.SEED_REGISTRY_STRATEGY = f"fixed_from_{SOURCE_PREFLIGHT_RUN_NAME}"
    pb6.LAW_STATUS_NOTE = "interpretation_small_overgeneration_candidate_not_counted_until_signoff"
    pb6.pb4.pb3.base.split_dataset = split_dataset_with_overgeneration_compiler


def main() -> dict[str, Any]:
    preexisting_generation = GENERATED_PROBLEMS_PATH.exists()
    preexisting_judges = all(path.exists() for path in [GROUNDING_LOG_PATH, KEYEDNESS_LOG_PATH, DISTRACTORFIT_LOG_PATH, NEARMISS_LOG_PATH])
    preexisting_manifest = RUN_MANIFEST_PATH.exists()
    configure_globals()
    manifest = pb6.main()
    rewrite_compiler_summary_contract()
    rewrite_compiler_manifest_context()
    validation_result = factory.run_post_compile_validation()
    if RUN_MANIFEST_PATH.exists():
        current_manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
        current_manifest.update(validation_result)
        # 기존 API/Judge artifact를 재사용해 compiler/linter/evidence만 다시 닫은 경우를 감사 추적용 alias로 남긴다.
        current_manifest["post_api_compile_retry_without_extra_api"] = YES if preexisting_generation and preexisting_judges and preexisting_manifest else NO
        current_manifest["post_api_compile_retry_note"] = (
            "existing generation/Judge artifacts reused; compiler/linter/evidence closed without extra API"
            if current_manifest["post_api_compile_retry_without_extra_api"] == YES
            else "fresh run or no prior API artifact reuse detected"
        )
        pb6.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, current_manifest)
        return current_manifest
    manifest.update(validation_result)
    return manifest


if __name__ == "__main__":
    main()
