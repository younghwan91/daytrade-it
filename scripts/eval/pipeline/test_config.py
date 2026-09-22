"""The freeze is enforced in code, not by convention."""

import pytest
from config import (
    HoldoutViolation,
    assert_no_holdout_labels,
    consumed_ranges,
    drop_holdout,
    holdout_start,
    is_holdout,
    load,
    stage4_status,
    unlabelled_split,
)


def test_holdout_is_forward_from_2026_09_16_with_a_30_day_target() -> None:
    h = load()["frozen_holdout"]
    assert h["rule"] == "forward"
    assert holdout_start() == "2026-09-16"
    assert h["target_trading_days"] == 30


def test_the_consumed_d55dd90_window_is_recorded_and_is_not_holdout() -> None:
    w = load()["consumed_windows"][0]
    assert len(w["days"]) == 31
    assert w["range"] == ["2026-08-03", "2026-09-15"]
    assert ("2026-08-03", "2026-09-15") in consumed_ranges()
    assert not any(is_holdout(d) for d in w["days"])


def test_stage1_ranges_all_end_before_the_holdout() -> None:
    s1 = load()["stage1"]
    assert s1["train_range"][1] < holdout_start()
    assert s1["post_train_range"][0] > s1["train_range"][1]
    assert s1["post_train_range"][1] < holdout_start()


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("day", ["2026-09-16", "2026-12-01"])
def test_stages_before_4_cannot_load_holdout_outcomes(stage: int, day: str) -> None:
    with pytest.raises(HoldoutViolation):
        assert_no_holdout_labels(["2026-09-15", day], stage=stage)


@pytest.mark.parametrize("stage", [1, 2, 3])
def test_the_consumed_window_may_be_fitted_on(stage: int) -> None:
    assert_no_holdout_labels(["2026-08-03", "2026-09-15"], stage=stage)


def test_stage_4_may() -> None:
    assert_no_holdout_labels(["2026-09-16"], stage=4)


def test_drop_holdout_removes_every_forward_day() -> None:
    rows = [{"day": "2026-09-15"}, {"day": "2026-09-16"}, {"day": "2026-10-01"}]
    assert [r["day"] for r in drop_holdout(rows, stage=1)] == ["2026-09-15"]


def test_unlabelled_rows_split_post_train_before_the_holdout() -> None:
    assert unlabelled_split("2026-08-03") == "post_train"
    assert unlabelled_split("2026-09-15") == "post_train"
    assert unlabelled_split("2026-09-16") == "holdout"


def test_stage4_waits_for_30_eligible_forward_days() -> None:
    few = stage4_status(["2026-09-16", "2026-09-17", "2026-09-15"])
    assert few == {
        "eligible": 2,
        "target": 30,
        "ready": False,
        "first": "2026-09-16",
        "last": "2026-09-17",
    }
    many = stage4_status([f"2026-10-{d:02d}" for d in range(1, 31)])
    assert many["ready"] and many["eligible"] == 30
