from __future__ import annotations

import sys
from pathlib import Path

# target 40 / candidate 64는 기존 small-overgeneration 공정을 재사용하되 quota와 path만 바꾸는 확대 preflight다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_small_overgeneration_preflight as base,
)


def configure_medium_overgeneration() -> None:
    # 기존 16개 package 반복 병목을 줄이기 위해 판결문_QA를 target 40 / candidate 64로 확대한다.
    base.VERSION_TAG = "objective_judgment_medium_overgeneration_pilot_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_judgment_target40_candidate64_seed_spec_wiring_check"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "judgment medium overgeneration preflight"
    base.SEED_ID_PREFIX = "judgment_medium_overgen_preflight"
    base.SEED_SELECTION_ROLE = "objective_judgment_medium_overgeneration_candidate_seed"
    base.SEED_SELECTION_NOTE = "판결문_QA package factory no-API candidate seed for target 40 / candidate 64"
    base.SEED_FILTER_NOTE = "judgment_only_medium_overgeneration_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "judgment_medium_overgeneration_preflight_no_api_candidate_not_counted"

    # candidate pool과 final package quota를 분리해 실패 row를 count에 섞지 않는 계약을 먼저 검산한다.
    base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.EXPECTED_DOC_TYPE_COUNTS = {"판결문_QA": 64}
    base.EXPECTED_LANE_BY_DOC = {
        ("판결문_QA", "generalization_03_04"): 32,
        ("판결문_QA", "expansion_01_02"): 32,
    }
    base.JUDGMENT_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 16,
        "02_TL_판결문_QA": 16,
        "03_TL_판결문_QA": 16,
        "04_TL_판결문_QA": 16,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_판결문_QA": 10,
        "02_TL_판결문_QA": 10,
        "03_TL_판결문_QA": 10,
        "04_TL_판결문_QA": 10,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}

    # current count는 r2+pb2+pb3+pb4+validated final packages만 반영하고, failed/candidate seen seed는 별도 제외한다.
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
    }
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 199,
        "train": 158,
        "eval": 41,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }

    # failed/candidate run도 이미 모델과 Judge에 노출된 seed이므로 fresh pool에서는 전부 제외한다.
    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_medium_overgeneration()
    base.main()


if __name__ == "__main__":
    main()
