from __future__ import annotations

import sys
from pathlib import Path

# 해석례_QA는 source-balanced candidate 64가 막히는 경우가 있어,
# type-deficit overclosure에서는 source-relaxed final 8 train-only micro route로 좁힌다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_source_relaxed_repeat_preflight as repeat,
)


def configure_type_deficit_overclosure_interpretation_train_micro_preflight() -> None:
    repeat.configure_interpretation_source_relaxed_repeat_preflight()
    base = repeat.relaxed.medium.base

    base.VERSION_TAG = "objective_type_deficit_overclosure_interpretation_train_micro_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_train_micro_candidate16_final8_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective type-deficit overclosure interpretation train micro preflight"
    base.SEED_ID_PREFIX = "type_deficit_interpretation_train_micro_preflight"
    base.SEED_SELECTION_ROLE = "objective_type_deficit_overclosure_interpretation_train_micro_candidate_seed"
    base.SEED_SELECTION_NOTE = "해석례_QA train 균형 보강용 source-relaxed candidate 16 / final 8 micro seed"
    base.NON_LAW_SCOPE_NOTE = "type_deficit_interpretation_train_micro_preflight_no_api_candidate_not_counted"

    base.EXPECTED_CANDIDATE_SEED_COUNT = 16
    base.FINAL_PACKAGE_TARGET_COUNT = 8
    base.EXPECTED_DOC_TYPE_COUNTS = {"해석례_QA": 16}
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "expansion_01_02"): 8,
        ("해석례_QA", "generalization_03_04"): 8,
    }
    base.INTERPRETATION_SOURCE_COUNTS = {
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
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 683,
        "train": 524,
        "eval": 159,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.CURRENT_OBJECTIVE_COUNT = dict(base.OBJECTIVE_CURRENT_COUNT_REFERENCE)

    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_type_deficit_overclosure_interpretation_train_micro_preflight()
    repeat.relaxed.medium.base.main()


if __name__ == "__main__":
    main()
