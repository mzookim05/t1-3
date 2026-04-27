from __future__ import annotations

import sys
from pathlib import Path

# 법령형은 01/02 균등 quota가 이미 blocked 되었으므로, 같은 package-factory 계약을
# 03/04 source-relaxed 대형 route로 재배선해 API 전에 availability와 quota를 먼저 잠근다.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_FOR_IMPORT = SCRIPT_DIR.parents[3]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from scripts.aihub.problem_generation.run_stamp import build_run_stamp  # noqa: E402
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_judgment_medium_overgeneration_preflight as medium,
)
from scripts.aihub.problem_generation.production_batches import (  # noqa: E402
    run_objective_law_guardrail_targeted_pilot as law_guardrail,
)


def configure_law_source_relaxed_repair_superwave() -> None:
    medium.configure_medium_overgeneration()
    base = medium.base

    base.VERSION_TAG = "objective_law_source_relaxed_repair_superwave_preflight"
    base.RUN_DATE = build_run_stamp()
    base.RUN_PURPOSE = "objective_r2_law_03_04_candidate192_final120_preflight"
    base.RUN_NAME = f"{base.RUN_DATE}_{base.VERSION_TAG}_{base.RUN_PURPOSE}"
    base.RUN_LABEL = "objective law source-relaxed repair superwave preflight"
    base.SEED_ID_PREFIX = "law_source_relaxed_preflight"
    base.SEED_SELECTION_ROLE = "objective_law_source_relaxed_repair_superwave_candidate_seed"
    base.SEED_SELECTION_NOTE = "법령_QA 01/02 blocker 이후 03/04 중심 candidate 192 / final 120 source-relaxed seed"
    base.SEED_FILTER_NOTE = "law_source_relaxed_03_04_high_risk_seed_filter"
    base.NON_LAW_SCOPE_NOTE = "law_source_relaxed_preflight_no_api_candidate_not_counted"

    base.EXPECTED_CANDIDATE_SEED_COUNT = 192
    base.FINAL_PACKAGE_TARGET_COUNT = 120
    base.EXPECTED_DOC_TYPE_COUNTS = {"법령_QA": 192}
    base.EXPECTED_LANE_BY_DOC = {("법령_QA", "generalization_03_04"): 192}
    base.JUDGMENT_SOURCE_COUNTS = {
        "03_TL_법령_QA": 96,
        "04_TL_법령_QA": 96,
    }
    base.CANDIDATE_TARGET_LABEL_COUNTS = {"A": 48, "B": 48, "C": 48, "D": 48}
    base.FINAL_TARGET_LABEL_COUNTS = {"A": 30, "B": 30, "C": 30, "D": 30}
    base.FINAL_SOURCE_COUNTS = {
        "03_TL_법령_QA": 60,
        "04_TL_법령_QA": 60,
    }
    base.FINAL_LANE_COUNTS = {"generalization_03_04": 120}
    base.COUNTED_EXCLUSION_COMPONENTS = {
        "current_objective_counted_seed_pool_after_balanced_recovery_signoff": 525,
    }
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

    original_configure_selector = base.configure_judgment_seed_selector

    def passes_law_seed_filter(spec: dict, payload: dict) -> tuple[bool, str]:
        if spec["doc_type_name"] != "법령_QA":
            return False, "law_only_scope"
        categories = law_guardrail.classify_law_seed(payload["label"]["input"], payload["label"]["output"])
        should_skip, skip_reason = law_guardrail.should_skip_law_seed(categories)
        if should_skip:
            return False, skip_reason
        passes_base, reason = law_guardrail.pb4.passes_seed_quality_filter(
            spec["doc_type_name"],
            payload["label"]["input"],
            payload["label"]["output"],
        )
        if not passes_base:
            return False, reason
        return True, "law_source_relaxed_candidate"

    def augment_law_seed_row(row: dict[str, str], index: int) -> dict[str, str]:
        categories = law_guardrail.classify_law_seed(row.get("transformed_problem", ""), row.get("short_answer", ""))
        row["judgment_seed_action"] = "template_only" if "plain_law_requirement" not in categories else "normal"
        row["stem_axis"] = "law_single_predicate"
        row["answer_uniqueness_risk_flags"] = "|".join(categories)
        row["tail_proximity_class"] = "law_source_relaxed_repair_candidate"
        row["exclusion_reason"] = ""
        row["target_correct_choice"] = ["A", "B", "C", "D"][index % 4]
        row["validator_report_schema_required"] = "예"
        row["law_guardrail_categories"] = "|".join(categories)
        return row

    def configure_law_seed_selector() -> None:
        original_configure_selector()
        base.pb6.passes_pb6_seed_filter = passes_law_seed_filter
        base.judgment_base.augment_seed_row = augment_law_seed_row

    base.configure_judgment_seed_selector = configure_law_seed_selector
    base.refresh_paths()


def main() -> None:
    configure_law_source_relaxed_repair_superwave()
    medium.base.main()


if __name__ == "__main__":
    main()
