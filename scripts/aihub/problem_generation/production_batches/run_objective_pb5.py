import json
import re
import sys
from collections import Counter
from pathlib import Path

# `pb5`는 새 subtype이 아니라, reviewer sign-off를 받은 law guardrail을
# current objective recipe(`v2_difficulty_patch_r2`)에 흡수해 다시 40개 controlled batch로 돌아가는 runner다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402

from scripts.aihub.problem_generation.production_batches import run_objective_law_guardrail_targeted_pilot as law_guardrail
from scripts.aihub.problem_generation.production_batches import run_objective_pb4 as pb4


VERSION_TAG = "pb5_objective_current_r2"
# llm_runs 이름은 실제 실행 시각과 맞아야 하므로 run stamp를 자동 생성한다.
RUN_DATE = build_run_stamp()
RUN_PURPOSE = "objective_r2_law_guardrail_absorbed_controlled_batch"
RUN_NAME = f"{RUN_DATE}_{VERSION_TAG}_{RUN_PURPOSE}"

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

REFERENCE_PB4_SEED_REGISTRY_PATH = (
    pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "pb4_objective_current_r2"
    / "seed_registry.csv"
)
REFERENCE_LAW_GUARDRAIL_SEED_REGISTRY_PATH = (
    pb4.pb3.base.PROJECT_ROOT
    / "data"
    / "interim"
    / "aihub"
    / "problem_generation"
    / "production_batches"
    / "objective_law_guardrail_targeted_pilot"
    / "seed_registry.csv"
)

ORIGINAL_BUILD_RUN_MANIFEST = pb4.pb3.build_run_manifest
ORIGINAL_BUILD_BATCH_SUMMARY = pb4.pb3.build_batch_summary
ORIGINAL_BUILD_GENERATION_MESSAGES = pb4.pb3.r2.build_generation_messages
ORIGINAL_POSTPROCESS_PROBLEM = pb4.pb3.r2.postprocess_problem
ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM = pb4.pb3.r2.build_local_fallback_problem


def normalized(text):
    return " ".join((text or "").split())


def collect_excluded_rows():
    # `pb5`는 current objective 95개와 law targeted pilot seed까지 모두 보호해야 같은 seed 재사용 착시를 피한다.
    rows = pb4.collect_excluded_rows()
    rows.extend(pb4.pb3.load_csv_rows_if_exists(REFERENCE_PB4_SEED_REGISTRY_PATH))
    rows.extend(pb4.pb3.load_csv_rows_if_exists(REFERENCE_LAW_GUARDRAIL_SEED_REGISTRY_PATH))
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


def classify_record(record):
    if record["doc_type_name"] != "법령_QA":
        return ""
    return "|".join(law_guardrail.classify_law_seed(record["label_input"], record["label_output"]))


def passes_pb5_seed_filter(spec, payload):
    # 법령형은 targeted pilot에서 검증한 high-risk class filter를 production batch에도 흡수한다.
    categories = []
    if spec["doc_type_name"] == "법령_QA":
        categories = law_guardrail.classify_law_seed(payload["label"]["input"], payload["label"]["output"])
        should_skip, skip_reason = law_guardrail.should_skip_law_seed(categories)
        if should_skip:
            return False, skip_reason, categories

    passes_filter, skip_reason = pb4.passes_seed_quality_filter(
        spec["doc_type_name"],
        payload["label"]["input"],
        payload["label"]["output"],
    )
    return passes_filter, skip_reason, categories


def select_fresh_registry_records():
    excluded_rows = collect_excluded_rows()
    exclusion_sets = build_exclusion_sets(excluded_rows)
    records = []
    sample_order = 1
    batch_family_ids = set()
    batch_label_paths = set()
    batch_raw_paths = set()

    for spec in pb4.pb3.DATASET_SPECS:
        label_paths = pb4.pb3.explanation_common.list_label_files(spec["label_glob"])
        raw_paths = pb4.pb3.explanation_common.list_raw_files(spec["raw_glob"])
        selected_indices = pb4.pb3.explanation_common.build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            chosen = None
            last_skip_reason = ""
            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = pb4.pb3.explanation_common.normalize_label_payload(
                    candidate_label_path,
                    pb4.pb3.explanation_common.load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                passes_filter, last_skip_reason, categories = passes_pb5_seed_filter(spec, candidate_payload)
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
                    categories,
                )
                break

            if chosen is None:
                raise RuntimeError(f"{spec['source_subset']}에서 fresh pb5 seed를 충분히 찾지 못했습니다. last_skip={last_skip_reason}")

            used_indices.add(chosen[0])
            label_path, payload, raw_path, family_id, categories = chosen[1], chosen[2], chosen[3], chosen[4], chosen[5]
            info = payload["info"]
            label = payload["label"]
            sample_id = f"pb5_{sample_order:03d}"

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
                "law_guardrail_categories": "|".join(categories),
                "local_selection_order": local_order,
                "selected_index": chosen[0],
                "selection_note": "pb5 controlled objective seed: current objective와 law targeted pilot seed 제외 후 law guardrail 흡수",
            }
            records.append(record)
            batch_family_ids.add(family_id)
            batch_label_paths.add(str(label_path))
            batch_raw_paths.add(str(raw_path))
            sample_order += 1

    return records, exclusion_sets


def build_seed_row(record):
    row = pb4.ORIGINAL_BUILD_SEED_ROW(record)
    row["selection_role"] = "objective_pb5_current_r2_law_guardrail_seed"
    row["selection_note"] = "law guardrail absorption memo를 반영한 current r2 controlled production seed"
    row["pb5_seed_filter_note"] = "pb4_and_law_targeted_seed_excluded_with_law_guardrail_filter"
    row["law_guardrail_categories"] = classify_record(record)
    row["law_guardrail_absorption_note"] = (
        "법령_QA에는 정답 유일성, 선택지 의미 중복 차단, stem 단일 predicate, stem ending 단일화 guardrail을 적용"
        if record["doc_type_name"] == "법령_QA"
        else ""
    )
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
                "law_guardrail_categories": row.get("law_guardrail_categories", ""),
                "law_guardrail_absorption_note": row.get("law_guardrail_absorption_note", ""),
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
        raise RuntimeError(f"pb5 seed 수가 40개가 아닙니다: {len(seed_rows)}")
    for doc_type_name, expected_count in pb4.pb3.EXPECTED_DOC_TYPE_COUNTS.items():
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
            raise RuntimeError(f"pb5 seed preflight 중복/누수 실패: {row['seed_sample_id']}")


def write_preflight_report(seed_rows, preflight_rows):
    doc_type_counts = Counter(row["doc_type_name"] for row in seed_rows)
    lane_counts = Counter(row["sampling_lane"] for row in seed_rows)
    source_counts = Counter(row["source_subset"] for row in seed_rows)
    lane_by_doc = Counter((row["doc_type_name"], row["sampling_lane"]) for row in seed_rows)
    law_category_counts = Counter()
    for row in seed_rows:
        if row.get("law_guardrail_categories"):
            law_category_counts.update(row["law_guardrail_categories"].split("|"))

    pb4.pb3.base.write_csv_atomic(SEED_PREFLIGHT_CSV_PATH, preflight_rows, list(preflight_rows[0].keys()))

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
    for doc_type_name in sorted(pb4.pb3.EXPECTED_DOC_TYPE_COUNTS):
        lines.append(
            f"| `{doc_type_name}` | `{lane_by_doc.get((doc_type_name, 'generalization_03_04'), 0)}` | `{lane_by_doc.get((doc_type_name, 'expansion_01_02'), 0)}` |"
        )
    # Law guardrail categories can be multi-label, so the exploded count is a
    # preflight construction signal rather than a post-generation quality gate.
    lines.extend(
        [
            "",
            "## law guardrail category counts",
            "",
            "이 count는 multi-label row를 `|` 기준으로 펼친 구조 조건 집계이므로, 합계가 법령 seed row 수 `10`과 다를 수 있다.",
            "또한 아래 `pass`는 seed construction check 통과를 뜻하며, 생성 품질 통과를 보장하지 않는다.",
            "",
            "| category | count |",
            "| --- | ---: |",
        ]
    )
    for category, count in sorted(law_category_counts.items()):
        lines.append(f"| `{category}` | `{count}` |")
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
            "| no prior current/law-targeted/held-out overlap | `pass` |",
            "| 법령_QA guardrail category and high-risk seed filter applied | `pass` |",
            "| stem single predicate and ending guardrail will be enforced at generation/postprocess | `pass` |",
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


def build_generation_messages(seed, reference_v2):
    messages = ORIGINAL_BUILD_GENERATION_MESSAGES(seed, reference_v2)
    # targeted pilot의 성공 신호는 유지하고, `law_guardrail_002`의 stem widening만 추가로 막는다.
    messages[1]["content"] += """

## pb5 law guardrail absorption memo
- `법령_QA`에서는 정답 유일성, 선택지 의미 중복 차단, stem 단일 predicate, stem ending 단일화를 모두 지킨다.
- stem은 하나의 predicate만 묻는다.
- `판단 근거`와 `적용 범위`, `요건`과 `효과`, `우선순위`와 `절차`, `기준`과 `정함 방식`을 한 stem 안에서 동시에 묻지 말 것.
- 문항 끝맺음은 `가장 옳은 것은?` 또는 `옳은 설명은?` 중 하나로만 닫고, `올바른 적용 범위를 고르시오` 같은 추가 지시구를 덧붙이지 말 것.
- 숫자·기간·나이 표현이 실질적으로 같은 범위라면 정답과 오답으로 동시에 두지 말 것.
- 오답끼리도 의미가 중복되지 않게 각각 다른 축을 하나씩만 비틀 것.
"""
    return messages


def has_forbidden_pair(stem, left_terms, right_terms):
    return any(left in stem for left in left_terms) and any(right in stem for right in right_terms)


def violates_law_stem_single_predicate(stem):
    stem = normalized(stem)
    forbidden_pairs = [
        (["판단 근거", "판단의 근거", "근거"], ["적용 범위", "범위"]),
        (["요건"], ["효과"]),
        (["우선순위", "우선 순위"], ["절차"]),
        (["기준"], ["정함 방식", "정하는 방식"]),
    ]
    if any(has_forbidden_pair(stem, left_terms, right_terms) for left_terms, right_terms in forbidden_pairs):
        return True
    if "가장 옳은 것은" in stem and "고르시오" in stem:
        return True
    if "무엇인가" in stem and "고르시오" in stem:
        return True
    if "옳은 설명" in stem and "가장 옳은 것은" in stem:
        return True
    return False


def build_local_fallback_problem(seed):
    payload = ORIGINAL_BUILD_LOCAL_FALLBACK_PROBLEM(seed)
    payload["near_miss_notes"] = payload.get("near_miss_notes", "") + " / pb5 law guardrail absorption fallback"
    if seed["doc_type_name"] == "법령_QA" and violates_law_stem_single_predicate(payload["generated_stem"]):
        # fallback 자체가 넓어질 때는 r2의 source-core 기반 단일 ending stem으로 한 번 더 접는다.
        payload["generated_stem"] = pb4.pb3.r2.normalize_stem_ending(
            pb4.pb3.r2.normalize_seed_stem_core(seed),
            seed["problem_generation_mode"],
        )
    return payload


def postprocess_problem(seed, payload):
    payload = ORIGINAL_POSTPROCESS_PROBLEM(seed, payload)
    if seed["doc_type_name"] != "법령_QA":
        return payload
    if law_guardrail.choices_are_duplicate_or_equivalent(payload):
        return build_local_fallback_problem(seed)
    if violates_law_stem_single_predicate(payload["generated_stem"]):
        return build_local_fallback_problem(seed)
    return payload


def classify_tail(row):
    tags = row.get("error_tags", "")
    if any(tag in tags for tag in ["정답 비유일", "오답이 정답 가능", "선택지 중복"]):
        return "정답 비유일/선택지 중복 재발"
    if "복수 쟁점 혼합" in tags:
        return "stem 단일 predicate 재발"
    return "기타 tail"


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
                "recurrence_class": classify_tail(row),
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
                "recurrence_class": "tail 없음",
                "generated_stem": "",
            }
        ]

    pb4.pb3.base.write_csv_atomic(TAIL_MEMO_CSV_PATH, tail_rows, list(tail_rows[0].keys()))
    lines = [
        f"# tail memo `{VERSION_TAG}`",
        "",
        "| seed | doc_type | lane | status | audit | error_tags | recurrence_class |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in tail_rows:
        lines.append(
            f"| `{row['seed_sample_id']}` | `{row['doc_type_name']}` | `{row['sampling_lane']}` | `{row['final_status']}` | `{row['audit_required']}` | `{row['error_tags']}` | `{row['recurrence_class']}` |"
        )
    pb4.pb3.base.write_text_atomic(TAIL_MEMO_MD_PATH, "\n".join(lines) + "\n")
    return tail_rows


def build_batch_summary(pb5_rows):
    selected_rows = [row for row in pb5_rows if row["selected_for_seed"] == "예"]
    selected_rows.sort(key=lambda row: (row["doc_type_name"], row["seed_sample_id"]))
    summary_rows = ORIGINAL_BUILD_BATCH_SUMMARY(pb5_rows)
    summary = pb4.pb3.summarize_rows(pb5_rows)
    tail_rows = [row for row in selected_rows if row.get("train_eligible") != "예"]
    unique_answer_recurrence_count = sum(1 for row in tail_rows if classify_tail(row) == "정답 비유일/선택지 중복 재발")
    stem_predicate_recurrence_count = sum(1 for row in tail_rows if classify_tail(row) == "stem 단일 predicate 재발")

    with BATCH_SUMMARY_MD_PATH.open("a", encoding="utf-8") as f:
        f.write("\n## pb5 success criteria\n")
        f.write("| criterion | target | result |\n")
        f.write("| --- | --- | --- |\n")
        f.write(f"| usable | `>= 35` | `{summary['selected_train_eligible_count']}` |\n")
        f.write(f"| hard_fail | `<= 3` | `{summary['selected_hard_fail_count']}` |\n")
        f.write(f"| audit | `<= 1` | `{summary['selected_audit_required_count']}` |\n")
        f.write(f"| soft_fail | `<= 1` | `{summary['selected_soft_fail_count']}` |\n")
        f.write(f"| 정답 비유일·선택지 중복 재발 | `0` | `{unique_answer_recurrence_count}` |\n")
        f.write(f"| stem 단일 predicate 재발 | `0` | `{stem_predicate_recurrence_count}` |\n")
    return summary_rows


def build_run_manifest(seed_rows, merged_rows, manifest_rows, summary_rows):
    manifest = ORIGINAL_BUILD_RUN_MANIFEST(seed_rows, merged_rows, manifest_rows, summary_rows)
    tail_rows = build_tail_memo(merged_rows)
    summary = pb4.pb3.summarize_rows(merged_rows)
    unique_answer_recurrence_count = sum(
        1 for row in tail_rows if row.get("recurrence_class") == "정답 비유일/선택지 중복 재발"
    )
    stem_predicate_recurrence_count = sum(1 for row in tail_rows if row.get("recurrence_class") == "stem 단일 predicate 재발")
    manifest.update(
        {
            "version_tag": VERSION_TAG,
            "run_name": RUN_NAME,
            "candidate_recipe_source": "v2_difficulty_patch_r2_plus_absorbed_law_guardrail",
            "seed_registry_strategy": "fresh_aihub_qa_training_pool_excluding_current_objective_and_law_targeted_pilot_seed",
            "count_reflection": "success_required_before_inventory_count",
            "law_guardrail_absorption": [
                "unique_answer_preflight",
                "choice_semantic_duplicate_block",
                "stem_single_predicate_guardrail",
                "stem_ending_single_instruction",
            ],
            "success_criteria": {
                "usable_min": 35,
                "hard_fail_max": 3,
                "audit_max": 1,
                "soft_fail_max": 1,
                "unique_answer_recurrence_max": 0,
                "stem_single_predicate_recurrence_max": 0,
            },
            "success_result": {
                "usable": summary["selected_train_eligible_count"],
                "hard_fail": summary["selected_hard_fail_count"],
                "audit": summary["selected_audit_required_count"],
                "soft_fail": summary["selected_soft_fail_count"],
                "unique_answer_recurrence": unique_answer_recurrence_count,
                "stem_single_predicate_recurrence": stem_predicate_recurrence_count,
            },
            "tail_memo_csv_path": str(TAIL_MEMO_CSV_PATH),
            "tail_memo_md_path": str(TAIL_MEMO_MD_PATH),
        }
    )
    pb4.pb3.base.write_json_atomic(RUN_MANIFEST_PATH, manifest)
    return manifest


def configure_pb5_paths():
    # `pb4` module globals도 함께 바꿔 두면 pb4 helper가 생성하는 report path가 pb5로 일관된다.
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
    configure_pb5_paths()
    pb4.pb3.select_fresh_registry_records = select_fresh_registry_records
    pb4.pb3.build_seed_row = build_seed_row
    pb4.pb3.build_preflight_rows = build_preflight_rows
    pb4.pb3.assert_preflight = assert_preflight
    pb4.pb3.write_preflight_report = write_preflight_report
    pb4.pb3.build_seed_registry = build_seed_registry
    pb4.pb3.build_batch_summary = build_batch_summary
    pb4.pb3.build_run_manifest = build_run_manifest
    pb4.pb3.r2.build_generation_messages = build_generation_messages
    pb4.pb3.r2.postprocess_problem = postprocess_problem
    pb4.pb3.r2.build_local_fallback_problem = build_local_fallback_problem
    return pb4.pb3.main()


if __name__ == "__main__":
    main()
