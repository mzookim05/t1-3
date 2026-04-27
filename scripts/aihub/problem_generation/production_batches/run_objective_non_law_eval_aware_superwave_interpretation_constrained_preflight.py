from __future__ import annotations

import sys
from pathlib import Path

# candidate 64가 막힐 때 같은 stop line 안에서 해석례_QA를 candidate 40 / final 24로 낮추는 안전 fallback이다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_constrained_overgeneration_preflight as constrained,
)


def configure_eval_aware_superwave_interpretation_constrained() -> None:
    constrained.configure_constrained_overgeneration()
    base = constrained.base
    base.VERSION_TAG = "objective_non_law_eval_aware_superwave_interpretation_constrained_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_non_law_interpretation_candidate40_final24_eval_aware_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "non-law eval-aware superwave interpretation constrained preflight"
    base.SEED_ID_PREFIX = "nonlaw_eval_interpretation_constrained_preflight"
    base.SEED_SELECTION_NOTE = "해석례_QA non-law eval-aware fallback seed for candidate 40 / final 24"
    base.NON_LAW_SCOPE_NOTE = "interpretation_eval_aware_constrained_preflight_no_api_candidate_not_counted"
    base.CURRENT_OBJECTIVE_COUNT = {
        "usable": 327,
        "train": 278,
        "eval": 49,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    base.refresh_paths()


def main() -> None:
    configure_eval_aware_superwave_interpretation_constrained()
    constrained.base.main()


if __name__ == "__main__":
    main()
