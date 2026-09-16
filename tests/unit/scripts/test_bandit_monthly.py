import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "eval"))

from bandit_monthly import folds_by_month, lambda_stable, sign_table  # noqa: E402


def test_first_month_is_a_fold_for_nobody() -> None:
    days = [dt.date(2026, 3, 18), dt.date(2026, 3, 27), dt.date(2026, 4, 3)]
    assert [m for m, _ in folds_by_month(days)] == ["2026-04"]


def test_folds_are_in_time_order_and_partition_the_later_months() -> None:
    days = [
        dt.date(2026, 3, 18),
        dt.date(2026, 4, 3),
        dt.date(2026, 4, 10),
        dt.date(2026, 5, 8),
        dt.date(2026, 6, 5),
    ]
    folds = folds_by_month(days)
    assert [m for m, _ in folds] == ["2026-04", "2026-05", "2026-06"]
    assert [d for _, ds in folds for d in ds] == days[1:]


def test_a_fold_never_contains_a_day_from_its_own_training_past() -> None:
    days = [dt.date(2026, 3, 18), dt.date(2026, 4, 3), dt.date(2026, 5, 8)]
    for month, month_days in folds_by_month(days):
        cutoff = min(month_days)
        assert all(d >= cutoff for d in month_days)
        assert all(str(d)[:7] == month for d in month_days)


def test_lambda_stable_accepts_one_grid_step() -> None:
    assert lambda_stable([100.0, 100.0, 30.0])  # adjacent in LAMS
    assert lambda_stable([3.0, 1.0])


def test_lambda_stable_rejects_the_ends_of_the_grid() -> None:
    assert not lambda_stable([100.0, 1.0])
    assert not lambda_stable([100.0, 10.0])  # two steps apart


def test_lambda_stable_is_false_with_no_folds() -> None:
    assert not lambda_stable([])


def _model(w: list[float]) -> dict:
    return {"feature_names": ["score", "premarket", "pre_entry_move"], "w": w}


def test_sign_table_flags_a_feature_that_flips_between_folds() -> None:
    table = sign_table(
        {
            "2026-04": _model([-1.0, 1.0, -1.0]),
            "2026-05": _model([-1.0, -1.0, -1.0]),  # premarket flips
        }
    )
    assert table["score"]["consistent"]
    assert not table["premarket"]["consistent"]
    assert table["premarket"]["modal_share"] == 0.5


def test_sign_table_reports_each_folds_sign() -> None:
    table = sign_table({"2026-04": _model([-1.0, 1.0, 0.0])})
    assert table["score"]["by_fold"] == {"2026-04": "-"}
    assert table["premarket"]["by_fold"] == {"2026-04": "+"}
    assert table["pre_entry_move"]["by_fold"] == {"2026-04": "0"}
