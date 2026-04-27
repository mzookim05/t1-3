import json
import sys
from collections import Counter
from pathlib import Path

# 이 runner는 reviewer가 요구한 `objective_law_repair_minispec`를 API 없이 먼저 검산한다.
# 법령_QA는 `pb5` 실패 이력이 커서, candidate64/final40 API 실행 전에
# source quota, seed exclusion, law repair guardrail contract를 별도 P2 preflight로 잠가야 한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_law_guardrail_targeted_pilot as law_base,
)


VERSION_TAG = "objective_law_repair_minispec_preflight"
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_law_repair_candidate64_final40_minispec_preflight"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
ORIGINAL_TARGET_LAW_SPECS = [dict(spec) for spec in law_base.TARGET_LAW_SPECS]
EXPECTED_SEED_COUNT = 64
EXPECTED_SOURCE_COUNT = 16
EXPECTED_LANE_COUNT = 32
TARGET_FINAL_PACKAGE_COUNT = 40
FALLBACK_FINAL_PACKAGE_COUNT = 24

PROJECT_ROOT = law_base.pb4.pb3.base.PROJECT_ROOT
INTERIM_DIR = PROJECT_ROOT / "data/interim/aihub/problem_generation/production_batches" / VERSION_TAG
PROCESSED_DIR = PROJECT_ROOT / "data/processed/aihub/problem_generation/production_batches" / VERSION_TAG
RUN_DIR = PROJECT_ROOT / "analysis/aihub/problem_generation/llm_runs" / RUN_NAME
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"


def configure_minispec_paths(sample_count):
    # 기존 law targeted pilot selector를 재사용하되, 산출물 경로는 minispec 전용으로 분리한다.
    law_base.VERSION_TAG = VERSION_TAG
    law_base.RUN_NAME = RUN_NAME
    law_base.RUN_PURPOSE = RUN_PURPOSE
    law_base.INTERIM_DIR = INTERIM_DIR
    law_base.PROCESSED_DIR = PROCESSED_DIR
    law_base.RUN_DIR = RUN_DIR
    law_base.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    law_base.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    law_base.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    law_base.SEED_READY_PATH = SEED_READY_PATH
    law_base.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    law_base.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    law_base.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    # 기존 16-seed pilot quota를 reviewer가 요청한 primary/fallback candidate quota로 확장한다.
    law_base.TARGET_LAW_SPECS = [{**spec, "sample_count": sample_count} for spec in ORIGINAL_TARGET_LAW_SPECS]


def assert_minispec_preflight(seed_rows, preflight_rows):
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)

    if len(seed_rows) != EXPECTED_SEED_COUNT:
        raise RuntimeError(f"law repair minispec seed 수가 {EXPECTED_SEED_COUNT}개가 아닙니다: {len(seed_rows)}")
    if doc_type_counts != {"법령_QA": EXPECTED_SEED_COUNT}:
        raise RuntimeError(f"법령_QA only 조건 실패: {dict(doc_type_counts)}")
    if lane_counts.get("generalization_03_04", 0) != EXPECTED_LANE_COUNT or lane_counts.get("expansion_01_02", 0) != EXPECTED_LANE_COUNT:
        raise RuntimeError(f"lane 32/32 조건 실패: {dict(lane_counts)}")
    if any(count != EXPECTED_SOURCE_COUNT for count in source_counts.values()) or len(source_counts) != 4:
        raise RuntimeError(f"source {EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT} 조건 실패: {dict(source_counts)}")

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
            raise RuntimeError(f"law repair minispec seed preflight 중복/누수 실패: {row['seed_sample_id']}")


def write_minispec_preflight_report(seed_rows, preflight_rows):
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    category_counts = Counter()
    for row in seed_rows:
        category_counts.update(row.get("law_guardrail_categories", "").split("|"))

    law_base.pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## purpose",
        "- reviewer가 요청한 `objective_law_repair_minispec`를 API 없이 먼저 검산한다.",
        "- 이 preflight는 count를 바꾸지 않으며, 통과하더라도 API execution은 reviewer sign-off 후에만 연다.",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- target_final_package: `{TARGET_FINAL_PACKAGE_COUNT}`",
        f"- fallback_final_package: `{FALLBACK_FINAL_PACKAGE_COUNT}`",
        f"- doc_type_counts: `{{'법령_QA': {EXPECTED_SEED_COUNT}}}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        "",
        "## law guardrail category counts",
        "| category | count |",
        "| --- | ---: |",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| `{category}` | `{count}` |")

    lines.extend(["", "## source subset counts", "| source_subset | count |", "| --- | ---: |"])
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")

    lines.extend(
        [
            "",
            "## repair contract gates",
            "| gate | status |",
            "| --- | --- |",
            f"| 법령_QA only candidate{EXPECTED_SEED_COUNT} | `pass` |",
            f"| source split {EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT}/{EXPECTED_SOURCE_COUNT} | `pass` |",
            f"| lane split {EXPECTED_LANE_COUNT}/{EXPECTED_LANE_COUNT} | `pass` |",
            "| no batch seed/reference/family/label/raw duplicate | `pass` |",
            "| no prior current/held-out/audit/tail overlap | `pass` |",
            "| seed filter categories recorded | `pass` |",
            "| high-risk numeric boundary / purpose-only / institution-only seed filter applied | `pass` |",
            "| stem constructor axis contract | `요건/효과/절차/적용범위 중 1개 축만 허용` |",
            "| choice validator contract | `정답 유일성/의미 중복/상하위 포함/직복사/single predicate/ending 단일화` |",
            "",
            "## next action",
            "- reviewer가 이 minispec preflight를 승인하면 `objective_law_repair_minispec` API execution으로 넘어간다.",
            "- 승인 전에는 current objective count와 downstream status를 바꾸지 않는다.",
        ]
    )
    law_base.pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    law_base.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    law_base.pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def write_blocker_manifest(blockers):
    law_base.pb4.pb3.base.ensure_dirs(RUN_DIR, RUN_EXPORTS_DIR)
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "run_mode": "no_api_law_repair_minispec_preflight_blocker",
        "api_call_count": 0,
        "count_reflection_status": "no_api_preflight_blocked_not_counted",
        "count_allowed": "아니오",
        "downstream_consumption_allowed": "아니오",
        "blockers": blockers,
        "next_step": "reviewer_should_decide_law_source_relaxation_or_smaller_repair_pilot",
    }
    law_base.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    lines = [
        f"# law repair minispec blocker `{VERSION_TAG}`",
        "",
        "이 run은 API를 호출하지 않은 법령 repair preflight blocker다.",
        "",
        "| attempt | target | result | blocker |",
        "| --- | --- | --- | --- |",
    ]
    for blocker in blockers:
        lines.append(
            f"| `{blocker['attempt']}` | `{blocker['candidate_seed_count']} -> final {blocker['target_final_package_count']}` | `blocked` | `{blocker['error']}` |"
        )
    lines.extend(
        [
            "",
            "## interpretation",
            "- 법령_QA는 reviewer 회신대로 API 전 P2가 실제로 남아 있다.",
            "- 현재 source-balanced law repair route는 high-risk seed filter 적용 뒤 `01_TL_법령_QA` 쪽에서 막힌다.",
            "- 다음 판단은 source relaxation, smaller law-only pilot, 또는 law repair spec 재조정 중 하나다.",
        ]
    )
    blocker_md_path = RUN_EXPORTS_DIR / f"seed_preflight_blocker_{VERSION_TAG}.md"
    law_base.pb4.pb3.base.write_text_atomic(blocker_md_path, "\n".join(lines) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def run_attempt(attempt_name, sample_count, target_final, fallback_final):
    global EXPECTED_SEED_COUNT, EXPECTED_SOURCE_COUNT, EXPECTED_LANE_COUNT
    global TARGET_FINAL_PACKAGE_COUNT, FALLBACK_FINAL_PACKAGE_COUNT

    EXPECTED_SOURCE_COUNT = sample_count
    EXPECTED_SEED_COUNT = sample_count * 4
    EXPECTED_LANE_COUNT = sample_count * 2
    TARGET_FINAL_PACKAGE_COUNT = target_final
    FALLBACK_FINAL_PACKAGE_COUNT = fallback_final
    configure_minispec_paths(sample_count)
    law_base.assert_preflight = assert_minispec_preflight
    law_base.write_preflight_report = write_minispec_preflight_report
    law_base.pb4.pb3.base.ensure_dirs(INTERIM_DIR, PROCESSED_DIR, RUN_DIR, RUN_INPUTS_DIR, RUN_EXPORTS_DIR)
    seed_rows = law_base.build_seed_registry()
    manifest = {
        "version_tag": VERSION_TAG,
        "run_name": RUN_NAME,
        "attempt": attempt_name,
        "run_mode": "no_api_law_repair_minispec_preflight",
        "api_call_count": 0,
        "candidate_seed_count": len(seed_rows),
        "target_final_package_count": target_final,
        "fallback_final_package_count": fallback_final,
        "doc_type_name": "법령_QA",
        "source_counts": dict(Counter(row["source_subset"] for row in seed_rows)),
        "lane_counts": dict(Counter(row["sampling_lane"] for row in seed_rows)),
        "count_reflection_status": "no_api_preflight_not_counted",
        "count_allowed": "아니오",
        "downstream_consumption_allowed": "아니오",
        "next_step": "reviewer_signoff_before_law_api_execution",
        "seed_registry_path": str(SEED_REGISTRY_PATH),
        "seed_preflight_md_path": str(SEED_PREFLIGHT_MD_PATH),
        "seed_preflight_csv_path": str(SEED_PREFLIGHT_CSV_PATH),
    }
    law_base.pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    blockers = []
    for attempt_name, sample_count, target_final, fallback_final in [
        ("primary_candidate64_final40", 16, 40, 24),
        ("fallback_candidate40_final24", 10, 24, 0),
    ]:
        try:
            return run_attempt(attempt_name, sample_count, target_final, fallback_final)
        except RuntimeError as exc:
            blockers.append(
                {
                    "attempt": attempt_name,
                    "candidate_seed_count": sample_count * 4,
                    "target_final_package_count": target_final,
                    "error": str(exc),
                }
            )
    return write_blocker_manifest(blockers)


if __name__ == "__main__":
    main()
