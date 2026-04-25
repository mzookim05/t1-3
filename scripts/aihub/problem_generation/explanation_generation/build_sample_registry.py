import json
from pathlib import Path

from common import (
    build_run_manifest,
    build_sample_indices,
    build_title,
    copy_file_to_run_inputs,
    ensure_run_dirs,
    list_label_files,
    list_raw_files,
    load_json,
    locate_raw_path,
    load_csv_rows,
    make_family_id,
    normalize_label_payload,
    write_csv_atomic,
    write_json_atomic,
)
from settings import (
    DATASET_SPECS,
    REFERENCE_SAMPLE_REGISTRY_PATH,
    RUN_MANIFEST_PATH,
    RUN_SELECTED_SAMPLES_PATH,
    SAMPLE_REGISTRY_PATH,
    VERSION_TAG,
)


def load_previous_registry_state():
    previous_paths = sorted(
        path
        for path in SAMPLE_REGISTRY_PATH.parent.glob("sample_registry_v*.csv")
        if path.name != SAMPLE_REGISTRY_PATH.name
    )
    used_family_ids = set()
    used_label_paths = set()
    used_raw_paths = set()

    for path in previous_paths:
        for row in load_csv_rows(path):
            used_family_ids.add(row["family_id"])
            used_label_paths.add(row["label_path"])
            used_raw_paths.add(row["raw_path"])

    return previous_paths, used_family_ids, used_label_paths, used_raw_paths


def build_rows_from_reference(reference_path):
    reference_rows = load_csv_rows(reference_path)
    rows = []

    for sample_order, reference_row in enumerate(reference_rows, start=1):
        label_path = Path(reference_row["label_path"])
        raw_path = Path(reference_row["raw_path"])
        payload = normalize_label_payload(
            label_path,
            load_json(label_path),
            reference_row["doc_type_name"],
        )
        info = payload["info"]
        label = payload["label"]
        family_id = make_family_id(reference_row["doc_type_name"], info)

        rows.append(
            {
                "sample_id": f"{VERSION_TAG}_{sample_order:03d}",
                "sample_order": sample_order,
                "source_subset": reference_row["source_subset"],
                "domain": reference_row["domain"],
                "doc_type_name": reference_row["doc_type_name"],
                "sampling_lane": reference_row.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": build_title({"info": info, "doc_type_name": reference_row["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": reference_row.get("local_selection_order", ""),
                "selected_index": reference_row.get("selected_index", ""),
                "selection_note": f"{reference_path.name} 표본 재사용 + 현재 라벨 정규화 재적용",
                "reference_sample_id": reference_row["sample_id"],
            }
        )

    return rows, [str(reference_path)]


def build_rows():
    if REFERENCE_SAMPLE_REGISTRY_PATH.exists():
        return build_rows_from_reference(REFERENCE_SAMPLE_REGISTRY_PATH)

    rows = []
    sample_order = 1
    previous_paths, used_family_ids, used_label_paths, used_raw_paths = load_previous_registry_state()

    for spec in DATASET_SPECS:
        label_paths = list_label_files(spec["label_glob"])
        raw_paths = list_raw_files(spec["raw_glob"])
        selected_indices = build_sample_indices(len(label_paths), spec["sample_count"])
        used_indices = set()

        for local_order, selected_index in enumerate(selected_indices, start=1):
            candidate_indices = list(range(selected_index, len(label_paths))) + list(range(0, selected_index))
            label_path = None
            payload = None
            raw_path = None

            for candidate_index in candidate_indices:
                if candidate_index in used_indices:
                    continue
                candidate_label_path = label_paths[candidate_index]
                candidate_payload = normalize_label_payload(
                    candidate_label_path,
                    load_json(candidate_label_path),
                    spec["doc_type_name"],
                )
                try:
                    candidate_raw_path = locate_raw_path(raw_paths, spec["doc_type_name"], candidate_payload["info"])
                except FileNotFoundError:
                    continue
                candidate_family_id = make_family_id(spec["doc_type_name"], candidate_payload["info"])
                if candidate_family_id in used_family_ids:
                    continue
                if str(candidate_label_path) in used_label_paths:
                    continue
                if str(candidate_raw_path) in used_raw_paths:
                    continue
                used_indices.add(candidate_index)
                label_path = candidate_label_path
                payload = candidate_payload
                raw_path = candidate_raw_path
                selected_index = candidate_index
                break

            if label_path is None or payload is None or raw_path is None:
                raise RuntimeError(f"{spec['source_subset']}에서 raw와 매칭되는 샘플을 충분히 찾지 못했습니다.")

            info = payload["info"]
            label = payload["label"]
            family_id = make_family_id(spec["doc_type_name"], info)
            # 실행 버전과 샘플 식별자를 맞춰 두면 review와 후속 비교에서
            # 어떤 런의 산출물인지 파일명만으로 바로 구분할 수 있다.
            sample_id = f"{VERSION_TAG}_{sample_order:03d}"

            row = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "sampling_lane": spec.get("sampling_lane", "generalization_03_04"),
                "source_schema": info.get("source_schema", ""),
                "family_id": family_id,
                "title": build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": local_order,
                "selected_index": selected_index,
                "selection_note": "prior family_id/label_path/raw_path 제외 후 신규 generalization set 구성",
            }
            rows.append(row)
            used_family_ids.add(family_id)
            used_label_paths.add(str(label_path))
            used_raw_paths.add(str(raw_path))
            sample_order += 1

    return rows, [str(path) for path in previous_paths]


def main():
    ensure_run_dirs()
    rows, previous_paths = build_rows()
    fieldnames = list(rows[0].keys())
    write_csv_atomic(SAMPLE_REGISTRY_PATH, rows, fieldnames)
    write_csv_atomic(RUN_SELECTED_SAMPLES_PATH, rows, fieldnames)
    run_manifest = build_run_manifest()
    run_manifest["previous_registry_paths"] = previous_paths
    if REFERENCE_SAMPLE_REGISTRY_PATH.exists():
        run_manifest["sampling_strategy"] = f"reference registry reuse: {REFERENCE_SAMPLE_REGISTRY_PATH.name}"
    else:
        run_manifest["sampling_strategy"] = "prior registry exclusion + evenly spaced fresh families"
    write_json_atomic(RUN_MANIFEST_PATH, run_manifest)
    copy_file_to_run_inputs(SAMPLE_REGISTRY_PATH)
    return rows


if __name__ == "__main__":
    main()
