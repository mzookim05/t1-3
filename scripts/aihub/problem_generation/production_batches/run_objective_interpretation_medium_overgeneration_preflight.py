from __future__ import annotations

import sys
from pathlib import Path

# deadline-constrained 운영에서는 해석례_QA도 small package 반복 대신 target 40 / candidate 64로 확장한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_small_overgeneration_preflight as base,
)


def configure_medium_overgeneration() -> None:
    # 기존 small 해석례 runner의 검증 공정을 유지하고, quota/path/current-count만 medium route에 맞춘다.
    base.VERSION_TAG = "objective_interpretation_medium_overgeneration_pilot_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_target40_candidate64_seed_spec_wiring_check"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "interpretation medium overgeneration preflight"
    base.SEED_ID_PREFIX = "interpretation_medium_overgen_preflight"
    base.SEED_SELECTION_ROLE = "objective_interpretation_medium_overgeneration_candidate_seed"
    base.SEED_SELECTION_NOTE = "해석례_QA package factory no-API candidate seed for target 40 / candidate 64"
    base.SEED_FILTER_NOTE = "interpretation_only_medium_overgeneration_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "interpretation_medium_overgeneration_preflight_no_api_candidate_not_counted"

    # candidate 전체가 아니라 strict final package만 count 후보가 되도록 source/lane/label quota를 분리한다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "generalization_03_04"): 32,
        ("해석례_QA", "expansion_01_02"): 32,
    }
    base.INTERPRETATION_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 16,
        "02_TL_유권해석_QA": 16,
        "03_TL_해석례_QA": 16,
        "04_TL_해석례_QA": 16,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 10,
        "02_TL_유권해석_QA": 10,
        "03_TL_해석례_QA": 10,
        "04_TL_해석례_QA": 10,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}

    # reviewer가 승인한 040231 count reflection 이후의 current objective count를 seed planning 기준으로 고정한다.
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 239,
        "train": 196,
        "eval": 43,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "r2": 16,
        "pb2": 13,
        "pb3": 40,
        "pb4": 40,
        "pb9_final_package": 40,
        "judgment_a_slot_final_package": 16,
        "interpretation_dslot_final_package": 16,
        "judgment_small_overgeneration_final_package": 16,
        "interpretation_small_overgeneration_final_package": 16,
        "judgment_medium_overgeneration_final_package": 40,
    }
    base.refresh_paths()


def main() -> None:
    configure_medium_overgeneration()
    base.main()


if __name__ == "__main__":
    main()
