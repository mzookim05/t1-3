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
    make_family_id,
    write_csv_atomic,
    write_json_atomic,
)
from settings import DATASET_SPECS, RUN_MANIFEST_PATH, RUN_SELECTED_SAMPLES_PATH, SAMPLE_REGISTRY_PATH


def build_rows():
    rows = []
    sample_order = 1

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
                candidate_payload = load_json(candidate_label_path)
                try:
                    candidate_raw_path = locate_raw_path(raw_paths, spec["doc_type_name"], candidate_payload["info"])
                except FileNotFoundError:
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
            sample_id = f"v3_{sample_order:03d}"

            row = {
                "sample_id": sample_id,
                "sample_order": sample_order,
                "source_subset": spec["source_subset"],
                "domain": spec["domain"],
                "doc_type_name": spec["doc_type_name"],
                "family_id": family_id,
                "title": build_title({"info": info, "doc_type_name": spec["doc_type_name"]}),
                "info_json": json.dumps(info, ensure_ascii=False),
                "label_path": str(label_path),
                "raw_path": str(raw_path),
                "label_input": label["input"],
                "label_output": label["output"],
                "local_selection_order": local_order,
                "selected_index": selected_index,
            }
            rows.append(row)
            sample_order += 1

    return rows


def main():
    ensure_run_dirs()
    rows = build_rows()
    fieldnames = list(rows[0].keys())
    write_csv_atomic(SAMPLE_REGISTRY_PATH, rows, fieldnames)
    write_csv_atomic(RUN_SELECTED_SAMPLES_PATH, rows, fieldnames)
    write_json_atomic(RUN_MANIFEST_PATH, build_run_manifest())
    copy_file_to_run_inputs(SAMPLE_REGISTRY_PATH)
    return rows


if __name__ == "__main__":
    main()
