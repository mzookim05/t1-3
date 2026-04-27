from __future__ import annotations

import sys
from pathlib import Path

# 해석례_QA는 source availability 병목이 반복됐기 때문에 먼저 candidate 64 가능성을 새 exclusion set에서 재확인한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_interpretation_medium_overgeneration_preflight as medium,
)


def configure_eval_aware_superwave_interpretation() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base
    base.VERSION_TAG = "objective_non_law_eval_aware_superwave_interpretation_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_non_law_interpretation_candidate64_final40_eval_aware_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "non-law eval-aware superwave interpretation preflight"
    base.SEED_ID_PREFIX = "nonlaw_eval_interpretation_preflight"
    base.SEED_SELECTION_NOTE = "해석례_QA non-law eval-aware superwave seed for candidate 64 / final 40"
    base.NON_LAW_SCOPE_NOTE = "interpretation_eval_aware_superwave_preflight_no_api_candidate_not_counted"
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
    configure_eval_aware_superwave_interpretation()
    medium.base.main()


if __name__ == "__main__":
    main()
