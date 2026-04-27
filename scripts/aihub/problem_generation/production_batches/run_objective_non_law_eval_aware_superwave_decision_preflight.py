from __future__ import annotations

import sys
from pathlib import Path

# 결정례_QA도 새 superwave tag를 사용해 기존 counted/candidate seed registry를 모두 fresh exclusion에 포함한다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_decision_medium_overgeneration_preflight as medium,
)


def configure_eval_aware_superwave_decision() -> None:
    medium.configure_decision_medium_overgeneration()
    base = medium.base.base
    base.VERSION_TAG = "objective_non_law_eval_aware_superwave_decision_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_non_law_decision_candidate64_final40_eval_aware_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "non-law eval-aware superwave decision preflight"
    base.SEED_ID_PREFIX = "nonlaw_eval_decision_preflight"
    base.SEED_SELECTION_NOTE = "결정례_QA non-law eval-aware superwave seed for candidate 64 / final 40"
    base.NON_LAW_SCOPE_NOTE = "decision_eval_aware_superwave_preflight_no_api_candidate_not_counted"
    base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 327,
        "train": 278,
        "eval": 49,
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
    configure_eval_aware_superwave_decision()
    medium.base.base.main()


if __name__ == "__main__":
    main()
