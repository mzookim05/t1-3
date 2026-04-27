from __future__ import annotations

import sys
from pathlib import Path

# 결정례_QA eval add-on은 final 24 중 dev/test 6/6을 확보해 eval shortage를 직접 줄인다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_addon_overgeneration_pilot as addon,
)


def configure_balanced_deficit_recovery_decision_eval_addon() -> None:
    addon.configure_decision_addon_overgeneration()
    base = addon.medium.base
    base.VERSION_TAG = "objective_balanced_deficit_recovery_decision_eval_addon"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_decision_candidate40_final24_eval_addon_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective balanced deficit recovery decision eval add-on API execution"
    base.LINTER_FIXTURE_ID = "balanced_deficit_recovery_decision_eval_addon_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_balanced_deficit_recovery_decision_eval_addon_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_decision_candidate40_final24_eval_addon_preflight"
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
    base.FINAL_DEV_COUNT = 6
    base.FINAL_TEST_COUNT = 6
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 407,
        "train": 338,
        "eval": 69,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS = 16
    base.pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS = 16
    base.refresh_paths()
    # add-on 신규 run의 postprocess 중 자기 merged 파일을 찾는 stale reference를 막는다.
    base.pb6.pb4.pb3.r2.load_reference_patch_rows = lambda: {}


def main() -> dict:
    configure_balanced_deficit_recovery_decision_eval_addon()
    return addon.medium.base.main()


if __name__ == "__main__":
    main()
