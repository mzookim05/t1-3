from __future__ import annotations

import sys
from pathlib import Path

# 해석례_QA는 02/04 source가 고갈됐으므로 01/03 source-relaxed quota로 candidate 64 가능성을 재검산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_medium_overgeneration_preflight as medium,
)


def configure_balanced_deficit_recovery_interpretation_relaxed_preflight() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base
    base.VERSION_TAG = "objective_balanced_deficit_recovery_interpretation_relaxed_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_interpretation_source_relaxed_candidate64_final40_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery interpretation source-relaxed preflight"
    base.SEED_ID_PREFIX = "balanced_recovery_interpretation_relaxed_preflight"
    base.SEED_SELECTION_ROLE = "objective_balanced_deficit_recovery_interpretation_relaxed_candidate_seed"
    base.SEED_SELECTION_NOTE = "해석례_QA 02/04 source blocker를 우회하기 위한 01/03 source-relaxed candidate 64 seed"
    base.SEED_FILTER_NOTE = "interpretation_source_relaxed_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "interpretation_source_relaxed_preflight_no_api_candidate_not_counted"
    base.INTERPRETATION_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 32,
        "03_TL_해석례_QA": 32,
    }
    base.EXPECTED_LANE_BY_DOC = {
        ("해석례_QA", "expansion_01_02"): 32,
        ("해석례_QA", "generalization_03_04"): 32,
    }
    base.FINAL_SOURCE_COUNTS = {
        "01_TL_유권해석_QA": 20,
        "03_TL_해석례_QA": 20,
    }
    base.FINAL_LANE_COUNTS = {"expansion_01_02": 20, "generalization_03_04": 20}
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 407,
        "train": 338,
        "eval": 69,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "current_objective_counted_seed_pool": 421,
    }
    base.refresh_paths()


def main() -> None:
    configure_balanced_deficit_recovery_interpretation_relaxed_preflight()
    medium.base.main()


if __name__ == "__main__":
    main()
