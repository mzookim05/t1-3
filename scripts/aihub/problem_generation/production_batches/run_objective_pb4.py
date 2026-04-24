import json
import sys
from collections import Counter
from pathlib import Path

# `pb4`는 `pb3` 실행 흐름을 재사용하되, seed preflight만 reviewer sign-off 기준으로 더 좁힌다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb3 as pb3


VERSION_TAG = "pb4_objective_current_r2"
RUN_DATE = "2026-04-24"
RUN_PURPOSE = "objective_r2_controlled_production_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

INTERIM_DIR = pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
RUN_PROMPTS_DIR = RUN_DIR / "prompts"
RUN_INPUTS_DIR = RUN_DIR / "inputs"
RUN_GENERATIONS_DIR = RUN_DIR / "generations"
RUN_JUDGE_LOGS_DIR = RUN_DIR / "judge_logs"
RUN_MERGED_DIR = RUN_DIR / "merged"
RUN_EXPORTS_DIR = RUN_DIR / "exports"

SEED_REGISTRY_PATH = INTERIM_DIR / "seed_registry.csv"
SEED_READY_PATH = INTERIM_DIR / "seed_ready.jsonl"
SEED_PREFLIGHT_CSV_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.csv"
SEED_PREFLIGHT_MD_PATH = RUN_EXPORTS_DIR / f"seed_preflight_{VERSION_TAG}.md"
RUN_MANIFEST_PATH = RUN_DIR / "run_manifest.json"
GENERATED_PROBLEMS_PATH = RUN_GENERATIONS_DIR / f"generated_problems_{VERSION_TAG}.jsonl"
GROUNDING_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_grounding_{VERSION_TAG}.jsonl"
KEYEDNESS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_keyedness_{VERSION_TAG}.jsonl"
DISTRACTORFIT_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_distractorfit_{VERSION_TAG}.jsonl"
NEARMISS_LOG_PATH = RUN_JUDGE_LOGS_DIR / f"judge_nearmiss_{VERSION_TAG}.jsonl"
MERGED_SCORES_PATH = RUN_MERGED_DIR / f"merged_problem_scores_{VERSION_TAG}.csv"

PROBLEM_TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
PROBLEM_DEV_PATH = PROCESSED_DIR / "dev.jsonl"
PROBLEM_TEST_PATH = PROCESSED_DIR / "test.jsonl"
PROBLEM_DATASET_MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
PROBLEM_AUDIT_QUEUE_PATH = PROCESSED_DIR / "audit_queue.csv"

BATCH_SUMMARY_MD_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.md"
BATCH_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_summary_{VERSION_TAG}.csv"
BATCH_LANE_SUMMARY_CSV_PATH = RUN_EXPORTS_DIR / f"batch_lane_summary_{VERSION_TAG}.csv"

REFERENCE_PB3_SEED_REGISTRY_PATH = (
    pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb3_objective_current_r2"
    / "seed_registry.csv"
)

ORIGINAL_BUILD_SEED_ROW = pb3.build_seed_row


def configure_pb4_paths():
    # `pb3` module globals를 `pb4` 경로로 재배선해 검증된 runner 본체를 그대로 쓴다.
    pb3.VERSION_TAG = VERSION_TAG
    pb3.RUN_PURPOSE = RUN_PURPOSE
    pb3.RUN_NAME = RUN_NAME
    pb3.INTERIM_DIR = INTERIM_DIR
    pb3.PROCESSED_DIR = PROCESSED_DIR
    pb3.RUN_DIR = RUN_DIR
    pb3.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb3.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb3.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb3.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb3.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb3.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb3.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb3.SEED_READY_PATH = SEED_READY_PATH
    pb3.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb3.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb3.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb3.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb3.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb3.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb3.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb3.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb3.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb3.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb3.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb3.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb3.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb3.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb3.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb3.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb3.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH


def normalized(text):
    return " ".join((text or "").split())


def has_judgment_signal(text):
    signals = [
        "판단",
        "결정",
        "기각",
        "각하",
        "인용",
        "인정",
        "보았",
        "보아",
        "판시",
        "부적법",
        "합헌",
        "위헌",
        "이유",
        "받아들",
    ]
    return any(signal in text for signal in signals)


def passes_seed_quality_filter(doc_type_name, label_input, label_output):
    # `pb3_007`류는 정답이 실제 정의를 닫지 못하고 질문을 되풀이하므로 법령형에서 먼저 차단한다.
    input_text = normalized(label_input)
    output_text = normalized(label_output)
    if doc_type_name == "법령_QA":
        weak_definition_patterns = ["무엇인가란", "뜻은 다음과 같다", "용어의 뜻은 다음과 같다"]
        if any(pattern in output_text for pattern in weak_definition_patterns):
            return False, "law_answer_incomplete_definition"
        if input_text and input_text.rstrip("?") in output_text and len(output_text) < len(input_text) + 50:
            return False, "law_answer_repeats_question"

    # `pb3_026`류는 판단 기준을 묻는데 당사자 주장만 답하므로 결정례/판결문에서 target mismatch를 차단한다.
    asks_judgment = any(pattern in input_text for pattern in ["판단 기준", "받아들이지", "이유", "결론"])
    answer_is_only_claim = "주장" in output_text and not has_judgment_signal(output_text)
    if doc_type_name in {"결정례_QA", "판결문_QA"} and asks_judgment and answer_is_only_claim:
        return False, "answer_target_mismatch_party_claim_only"

    return True, ""


def collect_excluded_rows():
    # `pb4`는 current objective seed 69개 전체와 held-out/audit row를 동시에 보호한다.
    rows = pb3.collect_excluded_rows()
    rows.extend(pb3.load_csv_rows_if_exists(REFERENCE_PB3_SEED_REGISTRY_PATH))
    return rows


def build_exclusion_sets(rows):
    return {
        "sample_ids": {
            row.get("sample_id", "") or row.get("seed_sample_id", "")
            for row in rows
            if row.get("sample_id", "") or row.get("seed_sample_id", "")
        },
        "reference_sample_ids": {row.get("reference_sample_id", "") for row in rows if row.get("reference_sample_id", "")},
        "family_ids": {row.get("family_id", "") for row in rows if row.get("family_id", "")},
        "label_paths": {row.get("label_path", "") for row in rows if row.get("label_path", "")},
        "raw_paths": {row.get("raw_path", "") for row in rows if row.get("raw_path", "")},
    }


def select_fresh_registry_records():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)
    records = []
    sample_order = 1
    batch_family_ids = set()
    batch_label_paths = set()
    batch_raw_paths = set()

    for spec in pb3.DATASET_SPECS:
        label_paths = pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            skip_reason = ""
            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = pb3.explanation_common.normalize_label_payload(
                    candidate_label_path,
                    pb3.explanation_common.load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                passes_filter, skip_reason = passes_seed_quality_filter(
                    spec["doc_type_name"],
                    candidate_payload["label"]["input"],
                    candidate_payload["label"]["output"],
                )
                if not passes_filter:
                    continue
                try:
                    candidate_raw_path = pb3.explanation_common.locate_raw_path(
                        raw_paths,
                        spec["doc_type_name"],
                        candidate_payload["info"],
                    )
                except FileNotFoundError:
                    continue
                candidate_family_id = pb3.explanation_common.make_family_id(spec["doc_type_name"], candidate_payload["info"])
                candidate_label_path_text = str(candidate_label_path)
                candidate_raw_path_text = str(candidate_raw_path)

                if candidate_family_id in exclusion_sets["family_ids"] or candidate_family_id in batch_family_ids:
                    continue
                if candidate_label_path_text in exclusion_sets["label_paths"] or candidate_label_path_text in batch_label_paths:
                    continue
                if candidate_raw_path_text in exclusion_sets["raw_paths"] or candidate_raw_path_text in batch_raw_paths:
                    continue
                chosen = (
                    candidate_index,
                    candidate_label_path,
                    candidate_payload,
                    candidate_raw_path,
                    candidate_family_id,
                )
                break

            if chosen is None:
                raise RuntimeError(f"{spec['source_subset']}에서 fresh pb4 seed를 충분히 찾지 못했습니다. last_skip={skip_reason}")

            used_indices.add(chosen[0])
            label_path, payload, raw_path, family_id = chosen[1], chosen[2], chosen[3], chosen[4]
            info = payload["info"]
            label = payload["label"]
            sample_id = f"pb4_{sample_order:03d}"

            record = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb3.explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": local_order,
                "selected_index": chosen[0],
                "selection_note": "pb4 controlled objective seed: current objective 69개와 held-out/audit/tail row 제외 후 신규 선택",
            }
            records.append(record)
            batch_family_ids.add(family_id)
            batch_label_paths.add(str(label_path))
            batch_raw_paths.add(str(raw_path))
            sample_order += 1

    return records, exclusion_sets


def build_seed_row(record):
    row = ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_pb4_current_r2_seed"
    row["selection_note"] = "pb3 tail memo를 반영한 current r2 recipe의 두 번째 controlled production seed"
    row["pb4_seed_filter_note"] = "answer_completeness_target_alignment_and_overlap_preflight"
    return row


def build_preflight_rows(seed_rows, exclusion_sets):
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


def assert_preflight(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    if len(seed_rows) != 40:
        raise RuntimeError(f"pb4 seed 수가 40개가 아닙니다: {len(seed_rows)}")
    for doc_type_name, expected_count in pb3.EXPECTED_DOC_TYPE_COUNTS.items():
        actual_count = doc_type_counts.get(doc_type_name, 0)
        if actual_count != expected_count:
            raise RuntimeError(f"{doc_type_name} seed 수가 {expected_count}개가 아닙니다: {actual_count}")
        if lane_by_doc.get((doc_type_name, "generalization_03_04"), 0) != 6:
            raise RuntimeError(f"{doc_type_name} generalization_03_04 seed 수가 6개가 아닙니다.")
        if lane_by_doc.get((doc_type_name, "expansion_01_02"), 0) != 4:
            raise RuntimeError(f"{doc_type_name} expansion_01_02 seed 수가 4개가 아닙니다.")

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
            raise RuntimeError(f"pb4 seed preflight 중복/누수 실패: {row['seed_sample_id']}")


def write_preflight_report(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)

    pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_type_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        "",
        "## doc type lane split",
        "| doc_type | generalization_03_04 | expansion_01_02 |",
        "| --- | ---: | ---: |",
    ]
    for doc_type_name in sorted(pb3.EXPECTED_DOC_TYPE_COUNTS):
        lines.append(
            f"| `{doc_type_name}` | `{lane_by_doc.get((doc_type_name, 'generalization_03_04'), 0)}` | `{lane_by_doc.get((doc_type_name, 'expansion_01_02'), 0)}` |"
        )
    lines.extend(["", "## source subset counts", "| source_subset | count |", "| --- | ---: |"])
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    lines.extend(
        [
            "| total seed count is 40 | `pass` |",
            "| doc type count is 10 each | `pass` |",
            "| doc type lane split is 6/4 | `pass` |",
            "| no batch seed_sample_id/reference_sample_id duplicate | `pass` |",
            "| no batch family_id/label_path/raw_path duplicate | `pass` |",
            "| no prior seed/reference/family/label/raw overlap | `pass` |",
            "| answer completeness and target alignment filter applied | `pass` |",
        ]
    )
    pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry():
    pb3.base.ensure_dirs(
        INTERIM_DIR,
        PROCESSED_DIR,
        RUN_DIR,
        RUN_PROMPTS_DIR,
        RUN_INPUTS_DIR,
        RUN_GENERATIONS_DIR,
        RUN_JUDGE_LOGS_DIR,
        RUN_MERGED_DIR,
        RUN_EXPORTS_DIR,
    )
    records, exclusion_sets = select_fresh_registry_records()
    seed_rows = [build_seed_row(record) for record in records]
    seed_rows.sort(key=lambda row: (row["doc_type_name"], row["sampling_lane"], row["seed_sample_id"]))
    preflight_rows = build_preflight_rows(seed_rows, exclusion_sets)
    assert_preflight(seed_rows, preflight_rows)

    pb3.base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


ORIGINAL_BUILD_RUN_MANIFEST = pb3.build_run_manifest


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    manifest.update(
        {
            "seed_registry_strategy": "fresh_aihub_qa_training_pool_excluding_r2_pb2_pb3_heldout_audit_tail_rows",
            "pb4_success_criterion": "same quantitative thresholds as pb3 plus no repeated pb3_007/pb3_010/pb3_026 failure class",
            "seed_preflight_required_checks": [
                "seed_count_40",
                "doc_type_10_each",
                "doc_type_lane_split_6_4",
                "no_seed_sample_id_or_reference_sample_id_duplicate",
                "no_family_label_raw_duplicate",
                "no_prior_current_or_heldout_overlap",
                "answer_completeness_filter",
                "question_answer_target_alignment_filter",
            ],
        }
    )
    pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def main():
    configure_pb4_paths()
    pb3.select_fresh_registry_records = select_fresh_registry_records
    pb3.build_seed_row = build_seed_row
    pb3.build_preflight_rows = build_preflight_rows
    pb3.assert_preflight = assert_preflight
    pb3.write_preflight_report = write_preflight_report
    pb3.build_seed_registry = build_seed_registry
    pb3.build_run_manifest = build_run_manifest
    return pb3.main()


if __name__ == "__main__":
    main()
