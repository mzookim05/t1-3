import csv
import json
import sys
from collections import Counter
from pathlib import Path

# `pb6`는 법령형 repair가 잠길 때까지 objective 생산을 멈추지 않기 위한
# 비법령 전용 controlled batch다. 기존 `pb5` runner를 직접 이어받지 않고,
# current recipe(`r2`)는 유지하되 `법령_QA` row를 seed 단계에서 완전히 제외한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.production_batches import run_objective_pb4 as pb4
from scripts.aihub.problem_generation.production_batches import run_objective_pb5 as pb5


VERSION_TAG = "pb6_non_law_objective_current_r2"
# llm_runs 폴더 정렬을 위해 최초 생성 시각의 HHMMSS까지 run stamp에 고정한다.
RUN_DATE = "2026-04-25_051819"
RUN_PURPOSE = "objective_r2_non_law_controlled_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"
RUN_LABEL = "pb6 non-law"
SEED_ID_PREFIX = "pb6_nonlaw"
SEED_SELECTION_ROLE = "objective_pb6_non_law_current_r2_seed"
SEED_SELECTION_NOTE = "법령_QA temporary restricted 상태에서 비법령 controlled production만 이어가는 seed"
SEED_FILTER_NOTE = "law_doc_type_excluded_and_seen_seed_pool_excluded"
SCOPE_NOTE = "해석례_QA/결정례_QA/판결문_QA only; 법령_QA는 law repair line에서 별도 처리"
EXPECTED_TOTAL_SEED_COUNT = 45
SUCCESS_USABLE_MIN = 40
SUCCESS_HARD_FAIL_MAX = 0
SUCCESS_SOFT_FAIL_MAX = 1
SUCCESS_AUDIT_MAX = 5
SUCCESS_LAW_ROW_COUNT = 0
CANDIDATE_RECIPE_SOURCE = "v2_difficulty_patch_r2_non_law_current"
SEED_REGISTRY_STRATEGY = "fresh_aihub_qa_training_non_law_pool_excluding_current_law_targeted_failed_pb5_heldout_audit_rows"
LAW_STATUS_NOTE = "temporary_restricted_and_excluded_from_pb6"
EXCLUSION_WORDING_LINES = [
    "`current counted-line attempted seed registry 109개`는 usable count가 아니라 `r2 + pb2 + pb3 + pb4`에 실제 투입된 seed registry 규모를 뜻한다.",
    "`law targeted pilot 16개`와 failed `pb5 40개`까지 더해, 이번 fresh 비법령 batch에서는 seen objective seed `165개`를 제외 대상으로 본다.",
]
OVERLAP_CHECK_LABEL = "no current/law-targeted/failed-pb5/held-out/audit overlap"

INTERIM_DIR = pb4.pb3.base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
PROCESSED_DIR = pb4.pb3.base.PROJECT_ROOT / "data" / "processed" / "aihub" / "problem_generation" / "production_batches" / VERSION_TAG
RUN_DIR = pb4.pb3.base.PROJECT_ROOT / "analysis" / "aihub" / "problem_generation" / "llm_runs" / RUN_NAME
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
TAIL_MEMO_CSV_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.csv"
TAIL_MEMO_MD_PATH = RUN_EXPORTS_DIR / f"tail_memo_{VERSION_TAG}.md"

REFERENCE_PB5_SEED_REGISTRY_PATH = (
    pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb5_objective_current_r2"
    / "seed_registry.csv"
)

EXPECTED_DOC_TYPE_COUNTS = {
    "해석례_QA": 15,
    "결정례_QA": 15,
    "판결문_QA": 15,
}

EXPECTED_LANE_BY_DOC = {
    ("해석례_QA", "generalization_03_04"): 9,
    ("해석례_QA", "expansion_01_02"): 6,
    ("결정례_QA", "generalization_03_04"): 9,
    ("결정례_QA", "expansion_01_02"): 6,
    ("판결문_QA", "generalization_03_04"): 9,
    ("판결문_QA", "expansion_01_02"): 6,
}

# 기존 source subset 구조를 그대로 쓰되, 비법령 `45개` stop line에 맞춰
# subset별 sample_count만 명시적으로 재배분한다.
PB6_SOURCE_COUNTS = {
    "01_TL_유권해석_QA": 3,
    "02_TL_유권해석_QA": 3,
    "03_TL_해석례_QA": 4,
    "04_TL_해석례_QA": 5,
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 2,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 4,
    "04_TL_결정례_QA": 5,
    "01_TL_판결문_QA": 3,
    "02_TL_판결문_QA": 3,
    "03_TL_판결문_QA": 4,
    "04_TL_판결문_QA": 5,
}

ORIGINAL_BUILD_RUN_MANIFEST = pb4.pb3.build_run_manifest
ORIGINAL_BUILD_BATCH_SUMMARY = pb4.pb3.build_batch_summary
ORIGINAL_BUILD_GENERATION_MESSAGES = pb4.pb3.r2.build_generation_messages
ORIGINAL_POSTPROCESS_PROBLEM = pb4.pb3.r2.postprocess_problem
ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM = pb4.pb3.r2.build_local_fallback_problem


def build_pb6_dataset_specs():
    specs_by_subset = {spec["source_subset"]: spec for spec in pb4.pb3.DATASET_SPECS}
    pb6_specs = []
    for source_subset, sample_count in PB6_SOURCE_COUNTS.items():
        spec = dict(specs_by_subset[source_subset])
        spec["sample_count"] = sample_count
        pb6_specs.append(spec)
    return pb6_specs


PB6_DATASET_SPECS = build_pb6_dataset_specs()


def load_csv_rows_if_exists(path):
    if not path.exists():
        return []
    return pb4.pb3.base.load_csv_rows(path)


def collect_excluded_rows():
    # `pb5` 실패 seed도 이미 generation/Judge를 본 seed이므로 fresh 비법령 생산에서는 제외한다.
    rows = pb5.collect_excluded_rows()
    rows.extend(load_csv_rows_if_exists(REFERENCE_PB5_SEED_REGISTRY_PATH))
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


def passes_pb6_seed_filter(spec, payload):
    if spec["doc_type_name"] == "법령_QA":
        return False, "law_doc_type_temporary_restricted"
    return pb4.passes_seed_quality_filter(
        spec["doc_type_name"],
        payload["label"]["input"],
        payload["label"]["output"],
    )


def select_fresh_registry_records():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)
    records = []
    sample_order = 1
    batch_family_ids = set()
    batch_label_paths = set()
    batch_raw_paths = set()

    for spec in PB6_DATASET_SPECS:
        label_paths = pb4.pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb4.pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb4.pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            skip_reason = ""
            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = pb4.pb3.explanation_common.normalize_label_payload(
                    candidate_label_path,
                    pb4.pb3.explanation_common.load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                passes_filter, skip_reason = passes_pb6_seed_filter(spec, candidate_payload)
                if not passes_filter:
                    continue
                try:
                    candidate_raw_path = pb4.pb3.explanation_common.locate_raw_path(
                        raw_paths,
                        spec["doc_type_name"],
                        candidate_payload["info"],
                    )
                except FileNotFoundError:
                    continue
                candidate_family_id = pb4.pb3.explanation_common.make_family_id(spec["doc_type_name"], candidate_payload["info"])
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
                raise RuntimeError(f"{spec['source_subset']}에서 fresh pb6 non-law seed를 충분히 찾지 못했습니다. last_skip={skip_reason}")

            used_indices.add(chosen[0])
            label_path, payload, raw_path, family_id = chosen[1], chosen[2], chosen[3], chosen[4]
            info = payload["info"]
            label = payload["label"]
            sample_id = f"{SEED_ID_PREFIX}_{sample_order:03d}"

            record = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": pb4.pb3.explanation_common.build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": local_order,
                "selected_index": chosen[0],
                "selection_note": f"{RUN_LABEL} controlled objective seed: current counted-line, law targeted, failed batch seed 제외",
            }
            records.append(record)
            batch_family_ids.add(family_id)
            batch_label_paths.add(str(label_path))
            batch_raw_paths.add(str(raw_path))
            sample_order += 1

    return records, exclusion_sets


def build_seed_row(record):
    row = pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = SEED_SELECTION_ROLE
    row["selection_note"] = SEED_SELECTION_NOTE
    row["pb6_seed_filter_note"] = SEED_FILTER_NOTE
    row["non_law_scope_note"] = SCOPE_NOTE
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
    if len(seed_rows) != EXPECTED_TOTAL_SEED_COUNT:
        raise RuntimeError(f"{RUN_LABEL} seed 수가 {EXPECTED_TOTAL_SEED_COUNT}개가 아닙니다: {len(seed_rows)}")
    if doc_type_counts.get("법령_QA", 0) != 0:
        raise RuntimeError("pb6 non-law seed에 법령_QA가 포함되어 있습니다.")
    for doc_type_name, expected_count in EXPECTED_DOC_TYPE_COUNTS.items():
        actual_count = doc_type_counts.get(doc_type_name, 0)
        if actual_count != expected_count:
            raise RuntimeError(f"{doc_type_name} seed 수가 {expected_count}개가 아닙니다: {actual_count}")
    for key, expected_count in EXPECTED_LANE_BY_DOC.items():
        actual_count = lane_by_doc.get(key, 0)
        if actual_count != expected_count:
            raise RuntimeError(f"{key} lane seed 수가 {expected_count}개가 아닙니다: {actual_count}")

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
            raise RuntimeError(f"{RUN_LABEL} seed preflight 중복/누수 실패: {row['seed_sample_id']}")


def write_preflight_report(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)

    pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

    lines = [
        f"# seed preflight `{VERSION_TAG}`",
        "",
        "## summary",
        f"- seed_count: `{len(seed_rows)}`",
        f"- doc_type_counts: `{dict(doc_type_counts)}`",
        f"- lane_counts: `{dict(lane_counts)}`",
        "- law_doc_type_count: `0`",
        "",
        "## exclusion wording",
        "",
        *EXCLUSION_WORDING_LINES,
        "",
        "## doc type lane split",
        "| doc_type | generalization_03_04 | expansion_01_02 |",
        "| --- | ---: | ---: |",
    ]
    for doc_type_name in sorted(EXPECTED_DOC_TYPE_COUNTS):
        lines.append(
            f"| `{doc_type_name}` | `{lane_by_doc.get((doc_type_name, 'generalization_03_04'), 0)}` | `{lane_by_doc.get((doc_type_name, 'expansion_01_02'), 0)}` |"
        )
    lines.extend(["", "## source subset counts", "| source_subset | count |", "| --- | ---: |"])
    for source_subset, count in sorted(source_counts.items()):
        lines.append(f"| `{source_subset}` | `{count}` |")
    lines.extend(["", "## checks", "| check | result |", "| --- | --- |"])
    # 이 runner는 pb7/pb8처럼 lane split이 달라지는 wrapper가 재사용하므로
    # check label은 특정 숫자 대신 configured target 충족 여부로 남긴다.
    lines.extend(
        [
            f"| total seed count is {EXPECTED_TOTAL_SEED_COUNT} | `pass` |",
            "| doc type count matches configured target | `pass` |",
            "| law doc type count is 0 | `pass` |",
            "| doc type lane split matches configured target | `pass` |",
            "| no batch seed_sample_id/reference_sample_id duplicate | `pass` |",
            "| no batch family_id/label_path/raw_path duplicate | `pass` |",
            f"| {OVERLAP_CHECK_LABEL} | `pass` |",
        ]
    )
    pb4.pb3.base.write_text_atomic(SEED_PREFLIGHT_MD_PATH, "\n".join(lines) + "\n")
    pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_CSV_PATH, RUN_INPUTS_DIR)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_PREFLIGHT_MD_PATH, RUN_INPUTS_DIR)


def build_seed_registry():
    pb4.pb3.base.ensure_dirs(
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

    pb4.pb3.base.write_csv_atomic(SEED_REGISTRY_PATH, seed_rows, list(seed_rows[0].keys()))
    pb4.pb3.base.write_jsonl_atomic(SEED_READY_PATH, seed_rows)
    write_preflight_report(seed_rows, preflight_rows)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_REGISTRY_PATH, RUN_INPUTS_DIR)
    pb4.pb3.base.copy_file_to_run_inputs(SEED_READY_PATH, RUN_INPUTS_DIR)
    return seed_rows


def classify_tail(row):
    tags = row.get("error_tags", "")
    # 비법령 실패가 다시 보이면 다음 batch를 전체로 키우지 않고,
    # 문서유형별 repair 축으로 바로 분리할 수 있도록 tail class를 더 구체화한다.
    if "정답 비유일" in tags or "오답이 정답 가능" in tags:
        return "non-law answer uniqueness failure"
    if "원문 외 사실 추가" in tags:
        return "non-law grounding/additional-fact failure"
    if "형식 부적합" in tags:
        return "non-law ending/form audit"
    if "오답약함" in tags or "near_miss_부족" in tags:
        return "non-law weak distractor"
    if row.get("final_status") == "hard_fail":
        return "non-law hard fail"
    if row.get("final_status") == "soft_fail":
        return "non-law soft fail"
    return "non-law audit tail"


def build_tail_memo(merged_rows):
    tail_rows = []
    for row in merged_rows:
        if row.get("selected_for_seed") != "예" or row.get("train_eligible") == "예":
            continue
        tail_rows.append(
            {
                "seed_sample_id": row["seed_sample_id"],
                "doc_type_name": row.get("doc_type_name", ""),
                "sampling_lane": row.get("sampling_lane", ""),
                "final_status": row.get("final_status", ""),
                "audit_required": row.get("audit_required", ""),
                "error_tags": row.get("error_tags", ""),
                "tail_class": classify_tail(row),
                "generated_stem": row.get("generated_stem", ""),
            }
        )

    if not tail_rows:
        tail_rows = [
            {
                "seed_sample_id": "",
                "doc_type_name": "",
                "sampling_lane": "",
                "final_status": "",
                "audit_required": "",
                "error_tags": "",
                "tail_class": "tail 없음",
                "generated_stem": "",
            }
        ]

    pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | doc_type | lane | status | audit | error_tags | tail_class |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['doc_type_name']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['error_tags']}` | `{row['tail_class']}` |"
        )
    pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def build_batch_summary(pb6_rows):
    summary_rows = ORIGINAL_BUILD_BATCH_SUMMARY(pb6_rows)
    selected_rows = [row for row in pb6_rows if row["selected_for_seed"] == "예"]
    summary = pb4.pb3.summarize_rows(pb6_rows)

    with BATCH_SUMMARY_MD_PATH.open("a", encoding="utf-8") as f:
        f.write(f"\n## {RUN_LABEL} success criteria\n")
        f.write("| criterion | target | result |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| usable | `>= {SUCCESS_USABLE_MIN} / {EXPECTED_TOTAL_SEED_COUNT}` | `{summary['selected_train_eligible_count']}` |\n")
        f.write(f"| hard_fail | `<= {SUCCESS_HARD_FAIL_MAX}` | `{summary['selected_hard_fail_count']}` |\n")
        f.write(f"| soft_fail | `<= {SUCCESS_SOFT_FAIL_MAX}` | `{summary['selected_soft_fail_count']}` |\n")
        f.write(f"| audit | `<= {SUCCESS_AUDIT_MAX}` | `{summary['selected_audit_required_count']}` |\n")
        f.write(f"| law row | `{SUCCESS_LAW_ROW_COUNT}` | `{sum(1 for row in selected_rows if row['doc_type_name'] == '법령_QA')}` |\n")
    return summary_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    tail_rows = build_tail_memo(merged_rows)
    summary = pb4.pb3.summarize_rows(merged_rows)
    law_row_count = sum(1 for row in seed_rows if row.get("doc_type_name") == "법령_QA")
    success = (
        summary["selected_train_eligible_count"] >= SUCCESS_USABLE_MIN
        and summary["selected_hard_fail_count"] <= SUCCESS_HARD_FAIL_MAX
        and summary["selected_soft_fail_count"] <= SUCCESS_SOFT_FAIL_MAX
        and summary["selected_audit_required_count"] <= SUCCESS_AUDIT_MAX
        and law_row_count == SUCCESS_LAW_ROW_COUNT
    )
    manifest.update(
        {
            "version_tag": VERSION_TAG,
            "run_name": RUN_NAME,
            "candidate_recipe_source": CANDIDATE_RECIPE_SOURCE,
            "seed_registry_strategy": SEED_REGISTRY_STRATEGY,
            "law_status": LAW_STATUS_NOTE,
            "current_count_decision": "count_train_eligible_if_success_criteria_pass" if success else "not_counted_due_to_failed_success_criteria",
            "success_criteria": {
                "usable_min": SUCCESS_USABLE_MIN,
                "hard_fail_max": SUCCESS_HARD_FAIL_MAX,
                "soft_fail_max": SUCCESS_SOFT_FAIL_MAX,
                "audit_max": SUCCESS_AUDIT_MAX,
                "law_row_count": SUCCESS_LAW_ROW_COUNT,
            },
            "success_result": {
                "usable": summary["selected_train_eligible_count"],
                "hard_fail": summary["selected_hard_fail_count"],
                "soft_fail": summary["selected_soft_fail_count"],
                "audit": summary["selected_audit_required_count"],
                "law_row_count": law_row_count,
                "passed": success,
            },
            "tail_memo_csv_path": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md_path": str(TAIL_MEMO_MD_PATH),
            "tail_memo_count": len([row for row in tail_rows if row.get("seed_sample_id")]),
        }
    )
    pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_pb6_paths():
    # `pb4`와 `pb3` module globals를 함께 재배선해야 base runner가 모든 파일을 pb6 폴더에 쓴다.
    pb4.VERSION_TAG = VERSION_TAG
    pb4.RUN_PURPOSE = RUN_PURPOSE
    pb4.RUN_NAME = RUN_NAME
    pb4.INTERIM_DIR = INTERIM_DIR
    pb4.PROCESSED_DIR = PROCESSED_DIR
    pb4.RUN_DIR = RUN_DIR
    pb4.RUN_PROMPTS_DIR = RUN_PROMPTS_DIR
    pb4.RUN_INPUTS_DIR = RUN_INPUTS_DIR
    pb4.RUN_GENERATIONS_DIR = RUN_GENERATIONS_DIR
    pb4.RUN_JUDGE_LOGS_DIR = RUN_JUDGE_LOGS_DIR
    pb4.RUN_MERGED_DIR = RUN_MERGED_DIR
    pb4.RUN_EXPORTS_DIR = RUN_EXPORTS_DIR
    pb4.SEED_REGISTRY_PATH = SEED_REGISTRY_PATH
    pb4.SEED_READY_PATH = SEED_READY_PATH
    pb4.SEED_PREFLIGHT_CSV_PATH = SEED_PREFLIGHT_CSV_PATH
    pb4.SEED_PREFLIGHT_MD_PATH = SEED_PREFLIGHT_MD_PATH
    pb4.RUN_MANIFEST_PATH = RUN_MANIFEST_PATH
    pb4.GENERATED_PROBLEMS_PATH = GENERATED_PROBLEMS_PATH
    pb4.GROUNDING_LOG_PATH = GROUNDING_LOG_PATH
    pb4.KEYEDNESS_LOG_PATH = KEYEDNESS_LOG_PATH
    pb4.DISTRACTORFIT_LOG_PATH = DISTRACTORFIT_LOG_PATH
    pb4.NEARMISS_LOG_PATH = NEARMISS_LOG_PATH
    pb4.MERGED_SCORES_PATH = MERGED_SCORES_PATH
    pb4.PROBLEM_TRAIN_PATH = PROBLEM_TRAIN_PATH
    pb4.PROBLEM_DEV_PATH = PROBLEM_DEV_PATH
    pb4.PROBLEM_TEST_PATH = PROBLEM_TEST_PATH
    pb4.PROBLEM_DATASET_MANIFEST_PATH = PROBLEM_DATASET_MANIFEST_PATH
    pb4.PROBLEM_AUDIT_QUEUE_PATH = PROBLEM_AUDIT_QUEUE_PATH
    pb4.BATCH_SUMMARY_MD_PATH = BATCH_SUMMARY_MD_PATH
    pb4.BATCH_SUMMARY_CSV_PATH = BATCH_SUMMARY_CSV_PATH
    pb4.BATCH_LANE_SUMMARY_CSV_PATH = BATCH_LANE_SUMMARY_CSV_PATH
    pb4.configure_pb4_paths()


def main():
    configure_pb6_paths()
    pb4.pb3.EXPECTED_DOC_TYPE_COUNTS = EXPECTED_DOC_TYPE_COUNTS
    pb4.pb3.select_fresh_registry_records = select_fresh_registry_records
    pb4.pb3.build_seed_row = build_seed_row
    pb4.pb3.build_preflight_rows = build_preflight_rows
    pb4.pb3.assert_preflight = assert_preflight
    pb4.pb3.write_preflight_report = write_preflight_report
    pb4.pb3.build_seed_registry = build_seed_registry
    pb4.pb3.build_batch_summary = build_batch_summary
    pb4.pb3.build_run_manifest = build_run_manifest
    pb4.pb3.r2.build_generation_messages = ORIGINAL_BUILD_GENERATION_MESSAGES
    pb4.pb3.r2.postprocess_problem = ORIGINAL_POSTPROCESS_PROBLEM
    pb4.pb3.r2.build_local_fallback_problem = ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM
    return pb4.pb3.main()


if __name__ == "__main__":
    main()
