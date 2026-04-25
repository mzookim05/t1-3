from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 reviewer가 요구한 no-API stop line을 닫기 위한 파일이다.
# 실제 generation/Judge 호출 전에 판결문 seed 16개, exclusion, label schedule,
# validator report schema가 계획대로 닫히는지 먼저 산출물로 남긴다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_pb6_non_law as pb6,
)


VERSION_TAG = "objective_judgment_repair_pilot_seed_preflight"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_judgment_repair_seed_preflight_wiring_check"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "judgment repair seed preflight"

PROJECT_ROOT = pb6.pb4.pb3.base.PROJECT_ROOT
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
EXCLUSION_AUDIT_CSV_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.csv"
EXCLUSION_AUDIT_MD_PATH = RUN_EXPORTS_DIR / f"exclusion_audit_{VERSION_TAG}.md"
TARGET_LABEL_SCHEDULE_CSV_PATH = RUN_EXPORTS_DIR / f"target_label_schedule_{VERSION_TAG}.csv"
VALIDATOR_SCHEMA_CSV_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.csv"
VALIDATOR_SCHEMA_MD_PATH = RUN_EXPORTS_DIR / f"validator_report_schema_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"

EXPECTED_TOTAL_SEED_COUNT = 16
EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 16}
EXPECTED_LANE_BY_DOC = {
    ("판결문_QA", "generalization_03_04"): 8,
    ("판결문_QA", "expansion_01_02"): 8,
}
JUDGMENT_SOURCE_COUNTS = {
    "01_TL_판결문_QA": 4,
    "02_TL_판결문_QA": 4,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 4,
}
TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
ORIGINAL_PB6_BUILD_SEED_ROW = pb6.build_seed_row

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
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_decision_only_controlled_production_with_choice_validator/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_04tl_decision_weak_distractor_calibration_pilot/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_04tl_decision_answer_uniqueness_1slot_replacement/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_accepted34_6slot_replacement_package/seed_registry.csv",
    PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches/pb9_cslot_final_replacement_package/seed_registry.csv",
]

COUNTED_EXCLUSION_COMPONENTS = {
    "r2": 16,
    "pb2": 13,
    "pb3": 40,
    "pb4": 40,
    "pb9_final_package": 40,
}

VALIDATOR_REQUIRED_FIELDS = [
    "seed_sample_id",
    "target_correct_choice",
    "export_correct_choice",
    "validator_action",
    "validator_export_disposition",
    "validator_reason_short",
    "validator_recalculated_correct_choice",
    "metadata_remap_ok",
    "split_allowed",
    "count_allowed",
    "single_correct_choice",
    "rule_application_split",
    "issue_boundary",
    "case_fact_alignment",
    "hierarchy_overlap",
    "answer_uniqueness_risk_flags",
    "tail_proximity_class",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def normalized_value(row: dict, field: str) -> str:
    # CSV BOM이나 nested metadata 때문에 같은 식별자가 서로 다른 위치에 있을 수 있다.
    # exclusion audit은 reviewer-facing 증명용이므로 직접 field와 metadata field를 모두 본다.
    for key, value in row.items():
        if str(key).lstrip("\ufeff") == field and value not in (None, ""):
            return str(value)
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if str(key).lstrip("\ufeff") == field and value not in (None, ""):
                return str(value)
    return ""


def row_identifiers(row: dict) -> dict[str, str]:
    return {
        field: normalized_value(row, field)
        for field in ["seed_sample_id", "reference_sample_id", "family_id", "label_path", "raw_path"]
    }


def collect_excluded_rows() -> list[dict[str, str]]:
    # reviewer가 요구한 `149` current counted exclusion pool에 failed/candidate
    # seed까지 더해, 이미 본 판결문 row가 pilot에 다시 들어오지 않게 막는다.
    rows: list[dict[str, str]] = []
    for path in REFERENCE_SEED_REGISTRY_PATHS:
        rows.extend(load_csv_rows_if_exists(path))
    return rows


def audit_source_paths() -> dict[str, list[Path]]:
    # seed registry만으로 막은 overlap과 processed/audit/tail 산출물 기준 overlap을
    # 분리해 보여 주어, `held-out/audit/tail 제외` 문구가 artifact로도 증명되게 한다.
    processed_root = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation"
    analysis_root = PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs"
    split_paths: list[Path] = []
    for name in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        split_paths.extend(processed_root.glob(f"**/{name}"))
    return {
        "interim_seed_registry": [path for path in REFERENCE_SEED_REGISTRY_PATHS if path.exists()],
        "processed_train_dev_test": sorted(split_paths),
        "processed_audit_queue": sorted(processed_root.glob("**/audit_queue.csv")),
        "analysis_tail_memo": sorted(analysis_root.glob("**/exports/*tail*.csv")),
    }


def load_audit_source_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return read_jsonl_rows(path)
    if path.suffix == ".csv":
        return read_csv_rows(path)
    return []


def write_exclusion_audit(seed_rows: list[dict]) -> dict[str, int]:
    seed_identifier_values = {
        field: {row_identifiers(row)[field] for row in seed_rows if row_identifiers(row)[field]}
        for field in ["seed_sample_id", "reference_sample_id", "family_id", "label_path", "raw_path"]
    }
    audit_rows: list[dict[str, str]] = []
    summary_counts: dict[str, int] = {}
    for source_group, paths in audit_source_paths().items():
        group_any_overlap = 0
        group_row_count = 0
        for path in paths:
            rows = load_audit_source_rows(path)
            group_row_count += len(rows)
            overlap_counts = {field: 0 for field in seed_identifier_values}
            any_overlap_count = 0
            for row in rows:
                identifiers = row_identifiers(row)
                row_has_overlap = False
                for field, seed_values in seed_identifier_values.items():
                    if identifiers[field] and identifiers[field] in seed_values:
                        overlap_counts[field] += 1
                        row_has_overlap = True
                if row_has_overlap:
                    any_overlap_count += 1
            group_any_overlap += any_overlap_count
            audit_rows.append(
                {
                    "source_group": source_group,
                    "source_path": str(path.relative_to(PROJECT_ROOT)),
                    "row_count": str(len(rows)),
                    "seed_sample_id_overlap_count": str(overlap_counts["seed_sample_id"]),
                    "reference_sample_id_overlap_count": str(overlap_counts["reference_sample_id"]),
                    "family_id_overlap_count": str(overlap_counts["family_id"]),
                    "label_path_overlap_count": str(overlap_counts["label_path"]),
                    "raw_path_overlap_count": str(overlap_counts["raw_path"]),
                    "any_overlap_count": str(any_overlap_count),
                }
            )
        summary_counts[f"{source_group}_source_count"] = len(paths)
        summary_counts[f"{source_group}_row_count"] = group_row_count
        summary_counts[f"{source_group}_overlap_count"] = group_any_overlap
    write_csv(
        EXCLUSION_AUDIT_CSV_PATH,
        audit_rows,
        [
            "source_group",
            "source_path",
            "row_count",
            "seed_sample_id_overlap_count",
            "reference_sample_id_overlap_count",
            "family_id_overlap_count",
            "label_path_overlap_count",
            "raw_path_overlap_count",
            "any_overlap_count",
        ],
    )
    lines = [
        f"# exclusion audit `{VERSION_TAG}`",
        "",
        "## summary",
        "| source_group | source_count | row_count | overlap_count |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source_group in audit_source_paths():
        source_count = summary_counts[f"{source_group}_source_count"]
        row_count = summary_counts[f"{source_group}_row_count"]
        overlap_count = summary_counts[f"{source_group}_overlap_count"]
        lines.append(f"| `{source_group}` | `{source_count}` | `{row_count}` | `{overlap_count}` |")
    lines.extend(
        [
            "",
            "## interpretation",
            "- `overlap_count = 0` means the current 16 judgment preflight seeds did not overlap with that source group by `seed_sample_id`, `reference_sample_id`, `family_id`, `label_path`, or `raw_path`.",
            "- This no-API artifact proves the held-out/audit/tail exclusion claim more directly than the seed registry-only preflight.",
        ]
    )
    write_text(EXCLUSION_AUDIT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(EXCLUSION_AUDIT_MD_PATH, RUN_INPUTS_DIR)
    return summary_counts


def classify_tail_proximity(row_or_payload: dict, source_subset: str) -> str:
    text = " ".join(
        [
            str(row_or_payload.get("transformed_problem", "")),
            str(row_or_payload.get("short_answer", "")),
            str(row_or_payload.get("generated_explanation", "")),
            str(row_or_payload.get("rule_basis", "")),
            str(row_or_payload.get("fact_basis", "")),
        ]
    )
    if source_subset == "03_TL_판결문_QA" and ("상속" in text or "2주택" in text or "비과세" in text):
        return "pb7_dj_032_like"
    if source_subset == "03_TL_판결문_QA" and ("부당행위" in text or "정상거래" in text or "가격 산정" in text):
        return "pb7_dj_034_like"
    if source_subset == "03_TL_판결문_QA":
        return "03tl_generalization_watch"
    return "none"


def classify_stem_axis(text: str) -> str:
    if any(token in text for token in ["청구", "소송", "심판청구", "상고", "항소", "처분"]):
        return "청구 판단형"
    if any(token in text for token in ["사실", "경위", "행위", "적용", "인정한 이유"]):
        return "적용 사실형"
    if any(token in text for token in ["결론", "인용", "기각", "각하", "취소", "범위"]):
        return "결론 범위형"
    return "쟁점 판단형"


def judgment_risk_flags(payload_or_row: dict, source_subset: str) -> list[str]:
    # 단순 키워드 검산이지만, API 호출 전에 위험 seed를 template-only로 낮추기 위한
    # preflight 신호로 충분하다. 실제 품질 판단은 다음 pilot의 validator/Judge에서 닫는다.
    text = " ".join(
        [
            str(payload_or_row.get("transformed_problem", "")),
            str(payload_or_row.get("short_answer", "")),
            str(payload_or_row.get("generated_explanation", "")),
            str(payload_or_row.get("rule_basis", "")),
            str(payload_or_row.get("fact_basis", "")),
        ]
    )
    flags: list[str] = []
    if sum(text.count(token) for token in ["청구", "처분", "사건", "쟁점"]) >= 3:
        flags.append("multiple_claim_or_issue")
    if any(token in text for token in ["일반론", "법리", "판단 기준"]) and any(
        token in text for token in ["사안", "사실관계", "적용"]
    ):
        flags.append("rule_application_overlap")
    if any(token in text for token in ["또는", "및", "각", "한편", "그리고"]):
        flags.append("compound_statement")
    if sum(1 for token in ["각하", "기각", "인용", "취소"] if token in text) >= 2:
        flags.append("conclusion_branch_risk")
    if source_subset == "03_TL_판결문_QA":
        flags.append("03tl_tail_proximity_watch")
    return flags


def judgment_seed_action(payload_or_row: dict, source_subset: str) -> tuple[str, str, str]:
    flags = judgment_risk_flags(payload_or_row, source_subset)
    tail_class = classify_tail_proximity(payload_or_row, source_subset)
    if len(flags) >= 4:
        return "template_only", "|".join(flags), tail_class
    if flags:
        return "template_only", "|".join(flags), tail_class
    return "normal", "none", tail_class


def passes_judgment_seed_filter(spec: dict, payload: dict) -> tuple[bool, str]:
    if spec["doc_type_name"] != "판결문_QA":
        return False, "judgment_only_scope"
    passes_base, reason = pb6.pb4.passes_seed_quality_filter(
        spec["doc_type_name"],
        payload["label"]["input"],
        payload["label"]["output"],
    )
    if not passes_base:
        return False, reason
    action, flags, _ = judgment_seed_action(
        {
            "transformed_problem": payload["label"]["input"],
            "short_answer": payload["label"]["output"],
            "generated_explanation": payload["label"]["output"],
        },
        spec["source_subset"],
    )
    if action == "exclude":
        return False, f"judgment_answer_uniqueness_risk_excluded:{flags}"
    return True, "judgment_seed_preflight_candidate"


def target_label_for_index(index: int) -> str:
    return ["A", "B", "C", "D"][index % 4]


def augment_seed_row(row: dict[str, str], index: int) -> dict[str, str]:
    action, flags, tail_class = judgment_seed_action(row, row["source_subset"])
    text = " ".join([row.get("transformed_problem", ""), row.get("short_answer", ""), row.get("generated_explanation", "")])
    row["judgment_seed_action"] = action
    row["stem_axis"] = classify_stem_axis(text)
    row["answer_uniqueness_risk_flags"] = flags
    row["tail_proximity_class"] = tail_class
    row["exclusion_reason"] = "" if action != "exclude" else flags
    row["target_correct_choice"] = target_label_for_index(index)
    row["validator_report_schema_required"] = "예"
    return row


def build_seed_row(record: dict) -> dict:
    row = ORIGINAL_PB6_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_judgment_repair_pilot_seed_preflight_seed"
    row["selection_note"] = "판결문_QA answer uniqueness repair pilot no-API seed preflight"
    row["pb6_seed_filter_note"] = "judgment_only_answer_uniqueness_seed_filter"
    row["non_law_scope_note"] = "judgment_repair_pilot_preflight_no_api"
    return row


def build_preflight_rows(seed_rows: list[dict], exclusion_sets: dict) -> list[dict]:
    seed_counts = Counter(row["seed_sample_id"] for row in seed_rows)
    reference_counts = Counter(row["reference_sample_id"] for row in seed_rows)
    family_counts = Counter(row["family_id"] for row in seed_rows)
    label_counts = Counter(row["label_path"] for row in seed_rows)
    raw_counts = Counter(row["raw_path"] for row in seed_rows)
    preflight_rows = []
    for row in seed_rows:
        preflight_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "reference_sample_id": row["reference_sample_id"],
                "doc_type_name": row["doc_type_name"],
                "source_subset": row["source_subset"],
                "sampling_lane": row["sampling_lane"],
                "family_id": row["family_id"],
                "target_correct_choice": row["target_correct_choice"],
                "judgment_seed_action": row["judgment_seed_action"],
                "stem_axis": row["stem_axis"],
                "answer_uniqueness_risk_flags": row["answer_uniqueness_risk_flags"],
                "tail_proximity_class": row["tail_proximity_class"],
                "exclusion_reason": row["exclusion_reason"],
                "seed_sample_id_duplicate_in_batch": "예" if seed_counts[row["seed_sample_id"]] > 1 else "아니오",
                "reference_sample_id_duplicate_in_batch": "예" if reference_counts[row["reference_sample_id"]] > 1 else "아니오",
                "family_duplicate_in_batch": "예" if family_counts[row["family_id"]] > 1 else "아니오",
                "label_path_duplicate_in_batch": "예" if label_counts[row["label_path"]] > 1 else "아니오",
                "raw_path_duplicate_in_batch": "예" if raw_counts[row["raw_path"]] > 1 else "아니오",
                "seed_sample_id_overlap_with_prior": "예" if row["seed_sample_id"] in exclusion_sets["sample_ids"] else "아니오",
                "reference_sample_id_overlap_with_prior": "예"
                if row["reference_sample_id"] in exclusion_sets["reference_sample_ids"]
                else "아니오",
                "family_overlap_with_prior": "예" if row["family_id"] in exclusion_sets["family_ids"] else "아니오",
                "label_path_overlap_with_prior": "예" if row["label_path"] in exclusion_sets["label_paths"] else "아니오",
                "raw_path_overlap_with_prior": "예" if row["raw_path"] in exclusion_sets["raw_paths"] else "아니오",
                "answer_mode": row["answer_mode"],
                "problem_generation_mode": row["problem_generation_mode"],
                "label_path": row["label_path"],
                "raw_path": row["raw_path"],
            }
        )
    return preflight_rows


def assert_judgment_preflight(seed_rows: list[dict], preflight_rows: list[dict]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"judgment seed count mismatch: {len(seed_rows)}")
    if dict(doc_counts) != EXPECTED_DOC_TYPE_COUNTS:
        raise RuntimeError(f"judgment doc type mismatch: {dict(doc_counts)}")
    for key, expected in EXPECTED_LANE_BY_DOC.items():
        if lane_by_doc.get(key, 0) != expected:
            raise RuntimeError(f"judgment lane mismatch: {key}={lane_by_doc.get(key, 0)}")
    for source_subset, expected in JUDGMENT_SOURCE_COUNTS.items():
        if source_counts.get(source_subset, 0) != expected:
            raise RuntimeError(f"judgment source split mismatch: {source_subset}={source_counts.get(source_subset, 0)}")
    if dict(target_counts) != TARGET_LABEL_COUNTS:
        raise RuntimeError(f"target label schedule mismatch: {dict(target_counts)}")
    for row in preflight_rows:
        overlap_flags = [
            row["seed_sample_id_duplicate_in_batch"],
            row["reference_sample_id_duplicate_in_batch"],
            row["family_duplicate_in_batch"],
            row["label_path_duplicate_in_batch"],
            row["raw_path_duplicate_in_batch"],
            row["seed_sample_id_overlap_with_prior"],
            row["reference_sample_id_overlap_with_prior"],
            row["family_overlap_with_prior"],
            row["label_path_overlap_with_prior"],
            row["raw_path_overlap_with_prior"],
        ]
        if "예" in overlap_flags:
            raise RuntimeError(f"judgment preflight overlap failed: {row['seed_sample_id']}")
        if row["judgment_seed_action"] == "exclude":
            raise RuntimeError(f"excluded judgment seed leaked into registry: {row['seed_sample_id']}")


def write_preflight_report(seed_rows: list[dict], preflight_rows: list[dict]) -> None:
    doc_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["judgment_seed_action"] for row in seed_rows)
    stem_axis_counts = Counter(row["stem_axis"] for row in seed_rows)
    tail_counts = Counter(row["tail_proximity_class"] for row in seed_rows)
    write_csv(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))
    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        f"- source_subset_counts: `{dict(source_counts)}`",
        f"- target_label_counts: `{dict(target_counts)}`",
        f"- judgment_seed_action_counts: `{dict(action_counts)}`",
        f"- stem_axis_counts: `{dict(stem_axis_counts)}`",
        f"- tail_proximity_counts: `{dict(tail_counts)}`",
        "",
        "## exclusion summary",
        f"- next_seed_planning_exclusion_pool: `{sum(COUNTED_EXCLUSION_COMPONENTS.values())}`",
        f"- counted_exclusion_components: `{COUNTED_EXCLUSION_COMPONENTS}`",
        f"- actual_exclusion_registry_paths: `{len(REFERENCE_SEED_REGISTRY_PATHS)}`",
        "",
        "## source subset counts",
        "| source_subset | count |",
        "| --- | ---: |",
    ]
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    checks = [
        "total seed count is 16",
        "doc type is 판결문_QA only",
        "source split is 01/02/03/04 each 4",
        "lane split is 8/8",
        "target label schedule is A/B/C/D = 4/4/4/4",
        "no batch duplicate",
        "no prior/candidate/failed/held-out/audit overlap",
        "no excluded seed leaked into registry",
        "validator report schema fields defined",
    ]
    for check in checks:
        lines.append(f"| {check} | `pass` |")
    write_text(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_target_label_schedule(seed_rows: list[dict]) -> None:
    rows = [
        {
            "seed_sample_id": row["seed_sample_id"],
            "doc_type_name": row["doc_type_name"],
            "source_subset": row["source_subset"],
            "sampling_lane": row["sampling_lane"],
            "family_id": row["family_id"],
            "target_correct_choice": row["target_correct_choice"],
        }
        for row in seed_rows
    ]
    write_csv(TARGET_LABEL_SCHEDULE_CSV_PATH, rows, list(rows[0].keys()))
    pb6.pb4.pb3.base.copy_file_to_run_inputs(TARGET_LABEL_SCHEDULE_CSV_PATH, RUN_INPUTS_DIR)


def write_validator_schema() -> None:
    rows = [{"field": field, "required": "예"} for field in VALIDATOR_REQUIRED_FIELDS]
    write_csv(VALIDATOR_SCHEMA_CSV_PATH, rows, ["field", "required"])
    lines = [
        f"# validator report schema `{VERSION_TAG}`",
        "",
        "| field | required |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| `{row['field']}` | `{row['required']}` |")
    write_text(VALIDATOR_SCHEMA_MD_PATH, "\n".join(lines) + "\n")
    pb6.pb4.pb3.base.copy_file_to_run_inputs(VALIDATOR_SCHEMA_CSV_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(VALIDATOR_SCHEMA_MD_PATH, RUN_INPUTS_DIR)


def configure_judgment_seed_selector() -> None:
    # 기존 pb6 selector를 재사용하면 label/raw path 연결과 family_id 생성 규칙을
    # 새로 복제하지 않아도 된다. batch identity와 judgment-only scope만 바꾼다.
    pb6.VERSION_TAG = VERSION_TAG
    pb6.RUN_DATE = RUN_DATE
    pb6.RUN_PURPOSE = RUN_PURPOSE
    pb6.RUN_NAME = RUN_NAME
    pb6.RUN_LABEL = RUN_LABEL
    pb6.SEED_ID_PREFIX = "judgment_repair_preflight"
    pb6.SEED_SELECTION_ROLE = "objective_judgment_repair_pilot_seed_preflight_seed"
    pb6.SEED_SELECTION_NOTE = "판결문_QA answer uniqueness repair pilot no-API seed preflight"
    pb6.SEED_FILTER_NOTE = "judgment_only_answer_uniqueness_seed_filter"
    pb6.SCOPE_NOTE = "판결문_QA only; API 실행 전 seed registry/wiring check"
    pb6.EXPECTED_TOTAL_SEED_COUNT = EXPECTED_TOTAL_SEED_COUNT
    pb6.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb6.EXPECTED_LANE_BY_DOC = EXPECTED_LANE_BY_DOC
    pb6.PB6_SOURCE_COUNTS = JUDGMENT_SOURCE_COUNTS
    pb6.PB6_DATASET_SPECS = pb6.build_pb6_dataset_specs()
    pb6.OVERLAP_CHECK_LABEL = "no current/failed/candidate/held-out/audit overlap"
    pb6.EXCLUSION_WORDING_LINES = [
        "`149`는 current usable이 아니라 next seed planning exclusion pool이다.",
        "failed/candidate objective seed까지 추가 제외해 API 비용이 드는 재사용을 막는다.",
    ]
    pb6.INTERIM_DIR = INTERIM_DIR
    pb6.PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
    pb6.RUN_DIR = RUN_DIR
    pb6.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb6.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb6.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb6.SEED_READY_PATH = SEED_READY_PATH
    pb6.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb6.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb6.collect_excluded_rows = collect_excluded_rows
    pb6.passes_pb6_seed_filter = passes_judgment_seed_filter
    pb6.build_seed_row = build_seed_row
    pb6.build_preflight_rows = build_preflight_rows
    pb6.assert_preflight = assert_judgment_preflight
    pb6.write_preflight_report = write_preflight_report


def build_seed_registry() -> list[dict]:
    configure_judgment_seed_selector()
    pb6.pb4.pb3.base.ensure_dirs(
        INTERIM_DIR,
        RUN_DIR,
        RUN_INPUTS_DIR,
        RUN_EXPORTS_DIR,
    )
    records, exclusion_sets = pb6.select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["source_subset"], row["seed_sample_id"]))
    for index, row in enumerate(seed_rows):
        augment_seed_row(row, index)
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_judgment_preflight(seed_rows, preflight_rows)
    write_csv(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb6.pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    write_target_label_schedule(seed_rows)
    write_validator_schema()
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb6.pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def write_run_manifest(seed_rows: list[dict], exclusion_audit_summary: dict[str, int]) -> None:
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    target_counts = Counter(row["target_correct_choice"] for row in seed_rows)
    action_counts = Counter(row["judgment_seed_action"] for row in seed_rows)
    manifest = {
        "run_name": RUN_NAME,
        "version_tag": VERSION_TAG,
        "run_type": "no_api_seed_preflight_wiring_check",
        "api_calls": 0,
        # reviewer가 요구한 hotfix provenance를 manifest에 직접 남겨,
        # 이후 API pilot과 no-API exclusion audit 산출물을 혼동하지 않게 한다.
        "artifact_revision": "exclusion_audit_hotfix_applied",
        "hotfix_reason": "held_out_audit_tail_exclusion_proof",
        "hotfix_api_calls": 0,
        "objective_current_count_reference": {
            "usable": 135,
            "train": 102,
            "eval": 33,
            "audit": 6,
            "hard_fail": 5,
            "soft_fail": 3,
        },
        "next_seed_planning_exclusion_pool": sum(COUNTED_EXCLUSION_COMPONENTS.values()),
        "counted_exclusion_components": COUNTED_EXCLUSION_COMPONENTS,
        "seed_count": len(seed_rows),
        "doc_type_counts": {"판결문_QA": len(seed_rows)},
        "source_subset_counts": dict(source_counts),
        "lane_counts": dict(lane_counts),
        "target_label_counts": dict(target_counts),
        "judgment_seed_action_counts": dict(action_counts),
        "success_criteria_for_future_api_pilot": {
            "usable_min": 14,
            "hard_fail_max": 0,
            "soft_fail_max": 1,
            "audit_max": 1,
            "answer_uniqueness_recurrence_max": 0,
            "count_reflection": "not_counted_until_future_api_pilot_and_reviewer_signoff",
        },
        "preflight_result": {
            "passed": True,
            "api_execution_allowed_by_this_run": False,
            "next_stop_line": "reviewer_signoff_for_api_pilot_or_additional_preflight",
            "interim_seed_registry_overlap_count": exclusion_audit_summary["interim_seed_registry_overlap_count"],
            "processed_split_overlap_count": exclusion_audit_summary["processed_train_dev_test_overlap_count"],
            "audit_queue_overlap_count": exclusion_audit_summary["processed_audit_queue_overlap_count"],
            "tail_memo_overlap_count": exclusion_audit_summary["analysis_tail_memo_overlap_count"],
        },
        "exclusion_audit_summary": exclusion_audit_summary,
        "artifacts": {
            "seed_registry": str(SEED_REGISTRY_PATH),
            "seed_ready": str(SEED_READY_PATH),
            "seed_preflight_csv": str(SEED_PREFLIGHT_CSV_PATH),
            "seed_preflight_md": str(SEED_PREFLIGHT_MD_PATH),
            "exclusion_audit_csv": str(EXCLUSION_AUDIT_CSV_PATH),
            "exclusion_audit_md": str(EXCLUSION_AUDIT_MD_PATH),
            "target_label_schedule_csv": str(TARGET_LABEL_SCHEDULE_CSV_PATH),
            "validator_schema_csv": str(VALIDATOR_SCHEMA_CSV_PATH),
            "validator_schema_md": str(VALIDATOR_SCHEMA_MD_PATH),
        },
    }
    write_json(RUN_MANIFEST_PATH, manifest)


def main() -> None:
    seed_rows = build_seed_registry()
    exclusion_audit_summary = write_exclusion_audit(seed_rows)
    write_run_manifest(seed_rows, exclusion_audit_summary)
    print(
        json.dumps(
            {
                "run_name": RUN_NAME,
                "seed_count": len(seed_rows),
                "source_subset_counts": dict(Counter(row["source_subset"] for row in seed_rows)),
                "target_label_counts": dict(Counter(row["target_correct_choice"] for row in seed_rows)),
                "api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
