from __future__ import annotations

import sys
from pathlib import Path

# 결정례_QA는 train이 초과됐으므로 final 24 중 eval 12를 확보하는 eval add-on seed contract를 먼저 고정한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_addon_overgeneration_preflight as addon,
)


def configure_balanced_deficit_recovery_decision_eval_addon_preflight() -> None:
    # 기존 add-on preflight 함수명을 그대로 호출해 source/label quota 기본값을 먼저 맞춘다.
    addon.configure_decision_addon_overgeneration()
    base = addon.medium.base.base
    base.VERSION_TAG = "objective_balanced_deficit_recovery_decision_eval_addon_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_decision_candidate40_final24_eval_addon_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery decision eval add-on preflight"
    base.SEED_ID_PREFIX = "balanced_recovery_decision_eval_addon_preflight"
    base.SEED_SELECTION_ROLE = "objective_balanced_deficit_recovery_decision_eval_addon_candidate_seed"
    base.SEED_SELECTION_NOTE = "결정례_QA train 초과를 억제하고 eval 부족을 보충하기 위한 candidate 40 / final 24 seed"
    base.SEED_FILTER_NOTE = "decision_eval_addon_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "decision_eval_addon_preflight_no_api_candidate_not_counted"
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 407,
        "train": 338,
        "eval": 69,
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
    configure_balanced_deficit_recovery_decision_eval_addon_preflight()
    addon.medium.base.base.main()


if __name__ == "__main__":
    main()
