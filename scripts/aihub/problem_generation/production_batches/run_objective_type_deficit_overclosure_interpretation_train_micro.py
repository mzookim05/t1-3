from __future__ import annotations

import sys
from pathlib import Path

# preflight가 고정한 해석례_QA final 8 train-only seed를 API로 실행한다.
# reviewer sign-off 전에는 candidate package로 남겨 count integrity를 유지한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_source_relaxed_repeat as repeat,
)


def configure_type_deficit_overclosure_interpretation_train_micro() -> None:
    repeat.configure_interpretation_source_relaxed_repeat()
    base = repeat.relaxed.base

    base.VERSION_TAG = "objective_type_deficit_overclosure_interpretation_train_micro"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_train_micro_candidate16_final8_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective type-deficit overclosure interpretation train micro API execution"
    base.LINTER_FIXTURE_ID = "objective_type_deficit_overclosure_interpretation_train_micro_candidate_package_pass"
    base.GATE_FIELD_LABEL = "interpretation train micro gate fields"

    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_type_deficit_overclosure_interpretation_train_micro_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_interpretation_train_micro_candidate16_final8_preflight"
    base.SOURCE_PREFLIGHT_SEED_REGISTRY_PATH = (
        base.PROJECT_ROOT
        / "data"
        / "interim"
        / "aihub"
        / "problem_generation"
        / "production_batches"
        / base.SOURCE_PREFLIGHT_VERSION_TAG
        / "seed_registry.csv"
    )

    base.EXPECTED_CANDIDATE_SEED_COUNT = 16
    base.FINAL_PACKAGE_TARGET_COUNT = 8
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 16}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "expansion_01_02"): 8,
        ("해석례_QA", "generalization_03_04"): 8,
    }
    base.EXPECTED_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 8,
        "03_TL_해석례_QA": 8,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 4, "B": 4, "C": 4, "D": 4}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 2, "B": 2, "C": 2, "D": 2}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 4,
        "03_TL_해석례_QA": 4,
    }
    base.FINAL_LANE_COUNTS = {"expansion_01_02": 4, "generalization_03_04": 4}
    base.FINAL_DEV_COUNT = 0
    base.FINAL_TEST_COUNT = 0
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 683,
        "train": 524,
        "eval": 159,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.refresh_paths()
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_type_deficit_overclosure_interpretation_train_micro()
    return repeat.relaxed.base.main()


if __name__ == "__main__":
    main()
