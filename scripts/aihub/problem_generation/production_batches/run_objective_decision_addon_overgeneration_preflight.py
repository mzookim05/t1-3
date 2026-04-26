from __future__ import annotations

import sys
from pathlib import Path

# descriptive split-lock wave가 seed availability에서 막힌 경우, reviewer가 지정한 decision add-on fallback을 먼저 검산한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_medium_overgeneration_preflight as medium,
)


DECISION_ADDON_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 8,
    "02_TL_심결례_QA": 6,
    "02_TL_심결문_QA": 6,
    "03_TL_결정례_QA": 10,
    "04_TL_결정례_QA": 10,
}
DECISION_ADDON_FINAL_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 5,
    "02_TL_심결례_QA": 4,
    "02_TL_심결문_QA": 3,
    "03_TL_결정례_QA": 6,
    "04_TL_결정례_QA": 6,
}


def configure_decision_addon_overgeneration() -> None:
    medium.configure_decision_medium_overgeneration()
    medium.base.base.VERSION_TAG = "objective_decision_addon_overgeneration_preflight"
    medium.base.base.RUN_DATE = build_run_stamp()
    medium.base.base.RUN_PURPOSE = "objective_r2_decision_target24_candidate40_seed_spec_wiring_check"
    medium.base.base.RUN_NAME = f"{medium.base.base.RUN_DATE}_{medium.base.base.VERSION_TAG}_{medium.base.base.RUN_PURPOSE}"
    medium.base.base.RUN_LABEL = "decision add-on overgeneration preflight"
    medium.base.base.SEED_ID_PREFIX = "decision_addon_overgen_preflight"
    medium.base.base.SEED_SELECTION_ROLE = "objective_decision_addon_overgeneration_candidate_seed"
    medium.base.base.SEED_SELECTION_NOTE = "결정례_QA package factory fallback seed for target 24 / candidate 40"
    medium.base.base.SEED_FILTER_NOTE = "decision_only_addon_overgeneration_seed_filter"
    medium.base.base.NON_LAW_SCOPE_NOTE = "decision_addon_overgeneration_preflight_no_api_candidate_not_counted"
    medium.base.base.EXPECTED_CANDIDATE_SEED_COUNT = 40
    medium.base.base.FINAL_PACKAGE_TARGET_COUNT = 24
    medium.base.base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 40}
    medium.base.base.EXPECTED_LANE_BY_DOC = {
        ("결정례_QA", "generalization_03_04"): 20,
        ("결정례_QA", "expansion_01_02"): 20,
    }
    medium.base.base.JUDGMENT_SOURCE_COUNTS = DECISION_ADDON_SOURCE_COUNTS
    medium.base.base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    medium.base.base.FINAL_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    medium.base.base.FINAL_SOURCE_COUNTS = DECISION_ADDON_FINAL_SOURCE_COUNTS
    medium.base.base.FINAL_LANE_COUNTS = {"generalization_03_04": 12, "expansion_01_02": 12}
    medium.base.base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 303,
        "train": 256,
        "eval": 47,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    medium.base.base.refresh_paths()


def main() -> None:
    configure_decision_addon_overgeneration()
    medium.base.base.main()


if __name__ == "__main__":
    main()
