from .context import gtfstk, slow, sample, cairns
from gtfstk import *


def test_get_dates():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_dates(feed, as_date_obj)
        d1 = '20140526'
        d2 = '20141228'
        if as_date_obj:
            d1, d2 = map(datestr_to_date, [d1, d2])
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2

def test_get_first_week():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_first_week(feed, as_date_obj)
        d1 = '20140526'
        d2 = '20140601'
        if as_date_obj:
            d1, d2 = map(datestr_to_date, [d1, d2])
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2

def test_restrict_dates():
    feed = cairns.copy()
    dates = feed.get_dates()
    assert restrict_dates(feed, dates[0]) == [dates[0]]
    assert restrict_dates(feed, '9999') == []
    assert restrict_dates(feed, dates + ['999']) == dates
