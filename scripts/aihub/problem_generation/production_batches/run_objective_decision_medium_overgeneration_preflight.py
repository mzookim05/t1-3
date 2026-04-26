from __future__ import annotations

import sys
from pathlib import Path

# descriptive wave가 seed availability P2로 막힐 때 쓰는 결정례_QA package factory fallback preflight다.
# 기존 판결문 medium overgeneration runner의 quota/compiler 구조를 재사용하되, doc/source/filter를 결정례 전용으로 바꾼다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_preflight as base,
)
from scripts.aihub.problem_generation.production_batches import run_objective_pb8_decision_only as decision_base  # noqa: E402


DECISION_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 12,
    "02_TL_심결례_QA": 10,
    "02_TL_심결문_QA": 10,
    "03_TL_결정례_QA": 16,
    "04_TL_결정례_QA": 16,
}
DECISION_FINAL_SOURCE_COUNTS = {
    "01_TL_심결례_QA": 8,
    "02_TL_심결례_QA": 6,
    "02_TL_심결문_QA": 6,
    "03_TL_결정례_QA": 10,
    "04_TL_결정례_QA": 10,
}


def passes_decision_seed_filter(spec: dict, payload: dict) -> tuple[bool, str]:
    # `결정례_QA` fallback은 이미 pb8/pb9에서 검증한 기본 seed quality filter를 유지한다.
    if spec["doc_type_name"] != "결정례_QA":
        return False, "decision_only_scope"
    return decision_base.passes_pb8_seed_filter(spec, payload)


def configure_decision_medium_overgeneration() -> None:
    base.configure_medium_overgeneration()
    base.base.VERSION_TAG = "objective_decision_medium_overgeneration_pilot_preflight"
    base.base.RUN_DATE = build_run_stamp()
    base.base.RUN_PURPOSE = "objective_r2_decision_target40_candidate64_seed_spec_wiring_check"
    base.base.RUN_NAME = f"{base.base.RUN_DATE}_{base.base.VERSION_TAG}_{base.base.RUN_PURPOSE}"
    base.base.RUN_LABEL = "decision medium overgeneration preflight"
    base.base.SEED_ID_PREFIX = "decision_medium_overgen_preflight"
    base.base.SEED_SELECTION_ROLE = "objective_decision_medium_overgeneration_candidate_seed"
    base.base.SEED_SELECTION_NOTE = "결정례_QA package factory no-API candidate seed for target 40 / candidate 64"
    base.base.SEED_FILTER_NOTE = "decision_only_medium_overgeneration_seed_filter"
    base.base.NON_LAW_SCOPE_NOTE = "decision_medium_overgeneration_preflight_no_api_candidate_not_counted"
    base.base.EXPECTED_CANDIDATE_SEED_COUNT = 64
    base.base.FINAL_PACKAGE_TARGET_COUNT = 40
    base.base.EXPECTED_DOC_TYPE_COUNTS = {"결정례_QA": 64}
    base.base.EXPECTED_LANE_BY_DOC = {
        ("결정례_QA", "generalization_03_04"): 32,
        ("결정례_QA", "expansion_01_02"): 32,
    }
    base.base.JUDGMENT_SOURCE_COUNTS = DECISION_SOURCE_COUNTS
    base.base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 16, "B": 16, "C": 16, "D": 16}
    base.base.FINAL_TARGET_LABEL_COUNTS = {"A": 10, "B": 10, "C": 10, "D": 10}
    base.base.FINAL_SOURCE_COUNTS = DECISION_FINAL_SOURCE_COUNTS
    base.base.FINAL_LANE_COUNTS = {"generalization_03_04": 20, "expansion_01_02": 20}
    base.base.OBJECTIVE_CURRENT_COUNT_REFERENCE = {
        "usable": 263,
        "train": 218,
        "eval": 45,
        "audit": 6,
        "hard_fail": 5,
        "soft_fail": 3,
    }
    # 기존 함수명은 judgment지만, 여기서는 결정례 scope만 통과시키도록 교체한다.
    base.base.judgment_base.passes_judgment_seed_filter = passes_decision_seed_filter
    base.base.refresh_paths()


def main() -> None:
    configure_decision_medium_overgeneration()
    base.base.main()


if __name__ == "__main__":
    main()
