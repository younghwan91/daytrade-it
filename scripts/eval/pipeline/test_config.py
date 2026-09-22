"""The freeze is enforced in code, not by convention."""

import pytest
from config import HoldoutViolation, assert_no_holdout_labels, drop_holdout, holdout_days, load


def test_holdout_is_the_31_day_d55dd90_window() -> None:
    days = sorted(holdout_days())
    assert len(days) == 31
    assert (days[0], days[-1]) == ("2026-08-03", "2026-09-15")


def test_stage1_train_range_ends_before_the_holdout() -> None:
    assert load()["stage1"]["train_range"][1] < min(holdout_days())


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_stages_before_4_cannot_load_holdout_outcomes(stage: int) -> None:
    with pytest.raises(HoldoutViolation):
        assert_no_holdout_labels(["2026-07-31", "2026-08-03"], stage=stage)


def test_stage_4_may() -> None:
    assert_no_holdout_labels(["2026-08-03"], stage=4)


def test_drop_holdout_removes_exactly_the_frozen_days() -> None:
    rows = [{"day": "2026-07-31"}, {"day": "2026-08-03"}, {"day": "2026-09-16"}]
    assert [r["day"] for r in drop_holdout(rows, stage=1)] == ["2026-07-31", "2026-09-16"]
