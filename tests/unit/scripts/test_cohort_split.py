import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from cohort_split import classify  # noqa: E402


def test_splits_bought_days_from_stratified_ones() -> None:
    cov = {"2026-03-18": (990, 1000), "2026-03-19": (290, 1000), "2026-03-20": (999, 1000)}
    out = classify(cov)
    assert out["census"] == ["2026-03-18", "2026-03-20"]
    assert out["stratified"] == ["2026-03-19"]


def test_a_day_at_99_percent_is_census_not_contaminated() -> None:
    # The stale-CSV error made a 99% day read as 24.8%; at its true coverage it is census.
    assert classify({"d": (983, 1000)})["census"] == ["d"]


def test_a_partially_judged_day_never_counts_as_census() -> None:
    assert classify({"d": (899, 1000)})["stratified"] == ["d"]


def test_empty_day_is_not_census() -> None:
    assert classify({"d": (0, 0)})["stratified"] == ["d"]


def test_threshold_is_configurable() -> None:
    assert classify({"d": (500, 1000)}, census_min=0.5)["census"] == ["d"]
