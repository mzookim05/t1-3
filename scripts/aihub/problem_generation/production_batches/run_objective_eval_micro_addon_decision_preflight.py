from __future__ import annotations

import sys
from pathlib import Path

# law/interpretation recovery 이후 objective eval만 소량 남는 경우를 대비한 micro add-on preflight다.
# 큰 wave를 다시 열지 않고 결정례_QA candidate 24에서 final 12 eval package만 조립한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_balanced_deficit_recovery_decision_eval_addon_preflight as addon,
)


MICRO_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 5,
    "02_TL_심결례_QA": 4,
    "02_TL_심결문_QA": 4,
    "03_TL_결정례_QA": 5,
    "04_TL_결정례_QA": 6,
}
MICRO_FINAL_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 3,
    "02_TL_심결례_QA": 2,
    "02_TL_심결문_QA": 1,
    "03_TL_결정례_QA": 3,
    "04_TL_결정례_QA": 3,
}


def configure_eval_micro_addon_decision_preflight() -> None:
    addon.configure_balanced_deficit_recovery_decision_eval_addon_preflight()
    base = addon.addon.medium.base.base
    base.VERSION_TAG = "objective_eval_micro_addon_decision_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_decision_candidate24_final12_eval_micro_addon_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective eval micro add-on decision preflight"
    base.SEED_ID_PREFIX = "objective_eval_micro_addon_decision_preflight"
    base.SEED_SELECTION_ROLE = "objective_eval_micro_addon_decision_candidate_seed"
    base.SEED_SELECTION_NOTE = "objective eval target closure를 위한 결정례_QA candidate 24 / final 12 micro add-on seed"
    base.EXPECTED_CANDIDATE_SEED_COUNT = 24
    base.FINAL_PACKAGE_TARGET_COUNT = 12
    # 기존 decision add-on은 candidate 40을 가정하므로 micro add-on의 doc/lane quota를 함께 낮춰야 한다.
    base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 24}
    base.EXPECTED_LANE_BY_DOC = {
        # micro source schedule은 01/02/02심결문 expansion 13건, 03/04 generalization 11건이다.
        ("결정례_QA", "expansion_01_02"): 13,
        ("결정례_QA", "generalization_03_04"): 11,
    }
    base.JUDGMENT_SOURCE_COUNTS = MICRO_SOURCE_COUNTS
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 6, "B": 6, "C": 6, "D": 6}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 3, "B": 3, "C": 3, "D": 3}
    base.FINAL_SOURCE_COUNTS = MICRO_FINAL_SOURCE_COUNTS
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 6, "expansion_01_02": 6}
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 511,
        "train": 410,
        "eval": 101,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    production_seed_root = base.PROJECT_ROOT / "data" / "interim" / "aihub" / "problem_generation" / "production_batches"
    base.REFERENCE_SEED_REGISTRY_PATHS = [
        path for path in sorted(production_seed_root.glob("*/seed_registry.csv")) if path.parent.name != base.VERSION_TAG
    ]
    base.refresh_paths()


def main() -> None:
    configure_eval_micro_addon_decision_preflight()
    addon.addon.medium.base.base.main()


if __name__ == "__main__":
    main()
