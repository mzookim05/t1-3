from __future__ import annotations

import sys
from pathlib import Path

# 해석례_QA medium source quota가 막힐 때 사용하는 candidate 40 / final 24 eval-aware API fallback이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_constrained_overgeneration_pilot as constrained,
)


def configure_eval_aware_superwave_interpretation_constrained() -> None:
    constrained.configure_constrained_overgeneration()
    base = constrained.base
    base.VERSION_TAG = "objective_non_law_eval_aware_superwave_interpretation_constrained"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_non_law_interpretation_candidate40_final24_eval_aware_api_execution"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "non-law eval-aware superwave interpretation constrained API execution"
    base.LINTER_FIXTURE_ID = "non_law_eval_aware_interpretation_constrained_candidate_package_pass"
    base.SOURCE_PREFLIGHT_VERSION_TAG = "objective_non_law_eval_aware_superwave_interpretation_constrained_preflight"
    base.SOURCE_PREFLIGHT_RUN_PURPOSE = "objective_r2_non_law_interpretation_candidate40_final24_eval_aware_preflight"
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
    base.FINAL_DEV_COUNT = 3
    base.FINAL_TEST_COUNT = 3
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 327,
        "train": 278,
        "eval": 49,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.pb6.pb4.pb3.base.GENERATOR_MAIN_MAX_WORKERS = 16
    base.pb6.pb4.pb3.base.JUDGE_MAIN_MAX_WORKERS = 16
    base.refresh_paths()


def main() -> dict:
    configure_eval_aware_superwave_interpretation_constrained()
    return constrained.base.main()


if __name__ == "__main__":
    main()
