from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

# full availability map은 API 실행 없이 seed pool 병목을 설명하는 증거물이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_descriptive_production_wave as wave  # noqa: E402
from scripts.aihub.problem_generation.shared.production_batch_common import write_text_atomic  # noqa: E402


EXPLANATION_DIR = wave.PROJECT_ROOT / "scripts" / "aihub" / "problem_generation" / "explanation_generation"
if str(EXPLANATION_DIR) not in sys.path:
    sys.path.insert(0, str(EXPLANATION_DIR))

import common as explanation_common  # noqa: E402


VERSION_TAG = wave.VERSION_TAG
RUN_FRAGMENT = os.environ.get(
    "DESCRIPTIVE_WAVE_RUN_FRAGMENT",
    f"{wave.VERSION_TAG}_{wave.RUN_PURPOSE}",
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as input_file:
        return list(csv.DictReader(input_file))


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(wave.PROJECT_ROOT))
    except ValueError:
        return str(path)


def latest_run_dir() -> Path:
    # 기본 검색 조각도 현재 wave 설정을 따라가게 해서, 새 run을 만들 때마다 스크립트 본문을 고치지 않도록 한다.
    candidates = sorted(
        (wave.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs").glob(f"*_{RUN_FRAGMENT}")
    )
    if not candidates:
        raise FileNotFoundError(f"latest run dir not found for fragment: {RUN_FRAGMENT}")
    return candidates[-1]


def build_raw_path_lookup(raw_paths: list[Path]) -> tuple[dict[str, Path], dict[str, Path]]:
    # 기존 locate_raw_path는 label마다 raw list를 선형 탐색해 전수 map에서 매우 느렸다.
    by_stem: dict[str, Path] = {}
    by_id_tail: dict[str, Path] = {}
    for path in raw_paths:
        by_stem.setdefault(path.stem, path)
        by_id_tail.setdefault(path.stem.split("_")[-1], path)
    return by_stem, by_id_tail


def locate_raw_path_fast(
    raw_lookup: tuple[dict[str, Path], dict[str, Path]],
    doc_type_name: str,
    info: dict,
) -> Path:
    by_stem, by_id_tail = raw_lookup
    raw_match_stem = explanation_common.text_or_blank(info.get("raw_match_stem"))
    if raw_match_stem and raw_match_stem in by_stem:
        return by_stem[raw_match_stem]

    wanted_id = str(info[explanation_common.ID_FIELD_BY_DOC_TYPE[doc_type_name]]).strip()
    if wanted_id in by_id_tail:
        return by_id_tail[wanted_id]

    raise FileNotFoundError(f"{doc_type_name} raw 파일을 찾지 못했습니다: {wanted_id}")


def availability_key(
    task_axis: str,
    spec: dict,
    status: str,
    reuse_tier_or_reason: str,
    split_scope: str,
    source_task: str = "",
    source_split: str = "",
    locked_split: str = "",
) -> tuple[str, str, str, str, str, str, str, str, str]:
    return (
        task_axis,
        spec["doc_type_name"],
        spec["source_subset"],
        spec.get("sampling_lane", ""),
        status,
        reuse_tier_or_reason,
        split_scope,
        source_task,
        locked_split or source_split,
    )


def collect_full_availability(exclusion_sets: dict[str, set[str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    descriptive_available_by_source: dict[str, int] = {}
    for spec in wave.DATASET_SPECS:
        label_paths = explanation_common.list_label_files(spec["label_glob"])
        raw_lookup = build_raw_path_lookup(explanation_common.list_raw_files(spec["raw_glob"]))
        counters: Counter[tuple[str, str, str, str, str, str, str, str, str]] = Counter()

        for label_path in label_paths:
            payload = explanation_common.normalize_label_payload(
                label_path,
                explanation_common.load_json(label_path),
                spec["doc_type_name"],
            )
            try:
                raw_path = locate_raw_path_fast(raw_lookup, spec["doc_type_name"], payload["info"])
            except FileNotFoundError:
                counters[availability_key("descriptive", spec, "unavailable", "raw_path_missing", "")] += 1
                counters[availability_key("objective", spec, "unavailable", "raw_path_missing", "")] += 1
                continue

            family_id = explanation_common.make_family_id(spec["doc_type_name"], payload["info"])
            descriptive_reuse, descriptive_skip = wave.classify_reuse_policy(
                family_id,
                str(label_path),
                str(raw_path),
                exclusion_sets,
            )
            if descriptive_reuse is None:
                counters[availability_key("descriptive", spec, "unavailable", descriptive_skip, "")] += 1
            else:
                reuse_tier = descriptive_reuse["reuse_tier"]
                split_scope = "train_only" if reuse_tier.startswith("Tier 2") else "train_dev_test"
                counters[
                    availability_key(
                        "descriptive",
                        spec,
                        "available",
                        reuse_tier,
                        split_scope,
                        descriptive_reuse.get("source_task", ""),
                        descriptive_reuse.get("source_split", ""),
                        descriptive_reuse.get("locked_split", ""),
                    )
                ] += 1
                descriptive_available_by_source[spec["source_subset"]] = descriptive_available_by_source.get(spec["source_subset"], 0) + 1

            objective_tier, objective_skip = wave.objective_fresh_reuse_policy(
                family_id,
                str(label_path),
                str(raw_path),
                exclusion_sets,
            )
            if objective_tier:
                counters[availability_key("objective", spec, "available", objective_tier, "train_dev_test")] += 1
            else:
                counters[availability_key("objective", spec, "unavailable", objective_skip, "")] += 1

        for (
            task_axis,
            doc_type_name,
            source_subset,
            sampling_lane,
            status,
            reuse_tier_or_reason,
            split_scope,
            source_task,
            locked_split,
        ), count in sorted(counters.items()):
            rows.append(
                {
                    "task_axis": task_axis,
                    "doc_type_name": doc_type_name,
                    "source_subset": source_subset,
                    "sampling_lane": sampling_lane,
                    "availability_status": status,
                    "reuse_tier_or_reason": reuse_tier_or_reason,
                    "split_scope": split_scope,
                    "source_task": source_task,
                    "locked_split": locked_split,
                    "available_count": str(count if status == "available" else 0),
                    "blocked_count": str(count if status == "unavailable" else 0),
                }
            )
    return rows, descriptive_available_by_source


def scaled_source_counts(base_counts: dict[str, int], target_total: int) -> dict[str, int]:
    # medium route 이상의 feasibility는 source 비율을 유지한 대략적 quota로 비교한다.
    current_total = sum(base_counts.values())
    floors: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for source_subset, count in base_counts.items():
        scaled = count * target_total / current_total
        floor_value = int(scaled)
        floors[source_subset] = floor_value
        remainders.append((scaled - floor_value, source_subset))
    missing = target_total - sum(floors.values())
    for _, source_subset in sorted(remainders, reverse=True)[:missing]:
        floors[source_subset] += 1
    return floors


def build_route_feasibility(available_by_source: dict[str, int]) -> list[dict[str, str]]:
    route_specs = [
        ("target20_candidate34_constrained", wave.CONSTRAINED_SOURCE_COUNTS),
        ("target24_candidate40_scaled", scaled_source_counts(wave.MEDIUM_RELAXED_SOURCE_COUNTS, 40)),
        ("target40_candidate56_source_relaxed", wave.MEDIUM_RELAXED_SOURCE_COUNTS),
        ("target64_candidate96_scaled", scaled_source_counts(wave.MEDIUM_RELAXED_SOURCE_COUNTS, 96)),
    ]
    rows: list[dict[str, str]] = []
    for route_label, source_counts in route_specs:
        missing_parts = []
        for source_subset, required in sorted(source_counts.items()):
            available = available_by_source.get(source_subset, 0)
            if available < required:
                missing_parts.append(f"{source_subset}:{available}/{required}")
        rows.append(
            {
                "route_label": route_label,
                "candidate_required": str(sum(source_counts.values())),
                "candidate_available_for_required_sources": str(
                    sum(min(available_by_source.get(source_subset, 0), required) for source_subset, required in source_counts.items())
                ),
                "feasible": wave.YES if not missing_parts else wave.NO,
                "missing_source_quota": "; ".join(missing_parts),
            }
        )
    return rows


def collect_quota_surplus_reuse(exclusion_sets: dict[str, set[str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((wave.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs").glob("**/quota_surplus_pool.csv")):
        for row in read_csv_rows(path):
            family_id = wave.row_value(row, "family_id")
            label_path = wave.row_value(row, "label_path")
            raw_path = wave.row_value(row, "raw_path")
            if family_id in exclusion_sets["quality_tail_family_id"] or label_path in exclusion_sets["quality_tail_label_path"] or raw_path in exclusion_sets["quality_tail_raw_path"]:
                gate = "blocked_quality_tail_overlap"
            elif family_id in exclusion_sets["global_eval_family_id"] or label_path in exclusion_sets["global_eval_label_path"] or raw_path in exclusion_sets["global_eval_raw_path"]:
                gate = "blocked_global_eval_overlap"
            else:
                gate = "dedup_stale_context_review_required"
            rows.append(
                {
                    "source_run": path.parent.name,
                    "task_axis": row.get("task_axis") or row.get("problem_task_type", ""),
                    "doc_type_name": row.get("doc_type_name", ""),
                    "source_subset": row.get("source_subset", ""),
                    "sampling_lane": row.get("sampling_lane", ""),
                    "family_id": family_id,
                    "reuse_gate": gate,
                    "count_reflection_status": row.get("count_reflection_status", ""),
                    "quality_failure": row.get("quality_failure", ""),
                }
            )
    return rows


def render_markdown(
    availability_rows: list[dict[str, str]],
    route_rows: list[dict[str, str]],
    quota_rows: list[dict[str, str]],
) -> str:
    lines = [
        f"# full seed availability map `{VERSION_TAG}`",
        "",
        "이 파일은 route-level probe가 아니라 현재 exclusion rule 기준의 full source availability summary다.",
        "",
        "## route feasibility",
        "| route_label | candidate_required | candidate_available_for_required_sources | feasible | missing_source_quota |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in route_rows:
        lines.append(
            f"| `{row['route_label']}` | `{row['candidate_required']}` | `{row['candidate_available_for_required_sources']}` | `{row['feasible']}` | {row['missing_source_quota'] or '-'} |"
        )
    lines.extend(
        [
            "",
            "## availability summary",
            "| task_axis | doc_type_name | source_subset | lane | status | reuse_tier_or_reason | split_scope | source_task | locked_split | available | blocked |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in availability_rows:
        lines.append(
            f"| `{row['task_axis']}` | `{row['doc_type_name']}` | `{row['source_subset']}` | `{row['sampling_lane']}` | "
            f"`{row['availability_status']}` | `{row['reuse_tier_or_reason']}` | `{row['split_scope']}` | "
            f"`{row['source_task']}` | `{row['locked_split']}` | `{row['available_count']}` | `{row['blocked_count']}` |"
        )
    quota_counter = Counter(row["reuse_gate"] for row in quota_rows)
    lines.extend(["", "## quota surplus reuse gate summary", "| reuse_gate | count |", "| --- | ---: |"])
    for gate, count in sorted(quota_counter.items()):
        lines.append(f"| `{gate}` | `{count}` |")
    return "\n".join(lines) + "\n"


def update_run_manifest(run_dir: Path, outputs: dict[str, Path]) -> None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = payload.setdefault("artifact_paths", {})
    for key, path in outputs.items():
        artifact_paths[key] = repo_rel(path)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)

    exclusion_sets = wave.build_exclusion_sets(wave.collect_exclusion_rows())
    availability_rows, descriptive_available_by_source = collect_full_availability(exclusion_sets)
    route_rows = build_route_feasibility(descriptive_available_by_source)
    quota_rows = collect_quota_surplus_reuse(exclusion_sets)

    availability_csv = exports_dir / f"full_seed_availability_map_{VERSION_TAG}.csv"
    availability_md = exports_dir / f"full_seed_availability_map_{VERSION_TAG}.md"
    route_csv = exports_dir / f"full_route_feasibility_{VERSION_TAG}.csv"
    quota_csv = exports_dir / f"quota_surplus_reuse_availability_{VERSION_TAG}.csv"

    write_csv_atomic(availability_csv, availability_rows, list(availability_rows[0].keys()) if availability_rows else [])
    write_csv_atomic(route_csv, route_rows, list(route_rows[0].keys()) if route_rows else [])
    write_csv_atomic(quota_csv, quota_rows, list(quota_rows[0].keys()) if quota_rows else [])
    write_text_atomic(availability_md, render_markdown(availability_rows, route_rows, quota_rows))
    update_run_manifest(
        run_dir,
        {
            "full_seed_availability_map": availability_csv,
            "full_seed_availability_map_md": availability_md,
            "full_route_feasibility": route_csv,
            "quota_surplus_reuse_availability": quota_csv,
        },
    )
    print(f"full_seed_availability_map={availability_md}")
    print(f"full_route_feasibility={route_csv}")
    print(f"quota_surplus_reuse_availability={quota_csv}")


if __name__ == "__main__":
    main()
