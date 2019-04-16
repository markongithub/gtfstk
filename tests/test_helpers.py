import pytest

import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
import shapely.geometry as sg

from .context import gtfstk, cairns, cairns_dates, cairns_trip_stats
from gtfstk import *


def test_timestr_to_seconds():
    timestr1 = "01:01:01"
    seconds1 = 3600 + 60 + 1
    timestr2 = "25:01:01"
    seconds2 = 25 * 3600 + 60 + 1
    assert timestr_to_seconds(timestr1) == seconds1
    assert timestr_to_seconds(seconds1, inverse=True) == timestr1
    assert timestr_to_seconds(seconds2, inverse=True) == timestr2
    assert timestr_to_seconds(timestr2, mod24=True) == seconds1
    assert timestr_to_seconds(seconds2, mod24=True, inverse=True) == timestr1
    # Test error handling
    assert np.isnan(timestr_to_seconds(seconds1))
    assert np.isnan(timestr_to_seconds(timestr1, inverse=True))


def test_timestr_mod24():
    timestr1 = "01:01:01"
    assert timestr_mod24(timestr1) == timestr1
    timestr2 = "25:01:01"
    assert timestr_mod24(timestr2) == timestr1


def test_datestr_to_date():
    datestr = "20140102"
    date = dt.date(2014, 1, 2)
    assert datestr_to_date(datestr) == date
    assert datestr_to_date(date, inverse=True) == datestr


def test_get_convert_dist():
    di = "mi"
    do = "km"
    f = get_convert_dist(di, do)
    assert f(1) == 1.609_344


def test_get_segment_length():
    s = sg.LineString([(0, 0), (1, 0)])
    p = sg.Point((1 / 2, 0))
    assert get_segment_length(s, p) == 1 / 2
    q = sg.Point((1 / 3, 0))
    assert get_segment_length(s, p, q) == pytest.approx(1 / 6)
    p = sg.Point((0, 1 / 2))
    assert get_segment_length(s, p) == 0


def test_get_max_runs():
    x = [7, 1, 2, 7, 7, 1, 2]
    get = get_max_runs(x)
    expect = np.array([[0, 1], [3, 5]])
    assert_array_equal(get, expect)


def test_get_peak_indices():
    times = [0, 10, 20, 30, 31, 32, 40]
    counts = [7, 1, 2, 7, 7, 1, 2]
    get = get_peak_indices(times, counts)
    expect = [0, 1]
    assert_array_equal(get, expect)


def test_almost_equal():
    f = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    assert almost_equal(f, f)
    g = pd.DataFrame([[4, 3], [2, 1]], columns=["b", "a"])
    assert almost_equal(f, g)
    h = pd.DataFrame([[1, 2], [5, 4]], columns=["a", "b"])
    assert not almost_equal(f, h)
    h = pd.DataFrame()
    assert not almost_equal(f, h)


def test_is_not_null():
    f = None
    c = "foo"
    assert not is_not_null(f, c)

    f = pd.DataFrame(columns=["bar", c])
    assert not is_not_null(f, c)

    f = pd.DataFrame([[1, np.nan]], columns=["bar", c])
    assert not is_not_null(f, c)

    f = pd.DataFrame([[1, np.nan], [2, 2]], columns=["bar", c])
    assert is_not_null(f, c)


def test_get_active_trips_df():
    f = pd.DataFrame(
        {"start_time": [1, 2, 3, 4, 5], "end_time": [6, 7, 8, 9, 10]}
    )
    expect = pd.Series(
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        data=[1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
    )
    get = get_active_trips_df(f)
    assert_series_equal(get, expect)

    f = pd.DataFrame(
        {"start_time": [1, 2, 3, 4, 5], "end_time": [2, 4, 6, 8, 10]}
    )
    expect = pd.Series(
        index=[1, 2, 3, 4, 5, 6, 8, 10], data=[1, 1, 2, 2, 3, 2, 1, 0]
    )
    get = get_active_trips_df(f)
    assert_series_equal(get, expect)


def test_unstack_time_series():
    dates = cairns_dates[:2]
    sts = cairns.compute_stop_time_series(dates, freq="12H")
    rts = cairns.compute_route_time_series(
        cairns_trip_stats, dates, freq="12H"
    )
    fts = cairns.compute_feed_time_series(cairns_trip_stats, dates, freq="12H")
    for id_col, ts in [("stop_id", sts), ("route_id", rts), (None, fts)]:
        f = unstack_time_series(ts)
        expect_cols = {"datetime", "indicator", "value"}
        if id_col:
            expect_cols.add(id_col)
        assert set(f.columns) == expect_cols
        assert f.shape[0] == ts.shape[0] * ts.shape[1]
