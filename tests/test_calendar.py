from .context import gtfstk, slow, sample, cairns
from gtfstk import *


def test_get_dates():
    feed = cairns.copy()
    for as_date_obj in [True, False]:
        dates = get_dates(feed, as_date_obj=as_date_obj)
        d1 = '20140526'
        d2 = '20141228'
        if as_date_obj:
            d1 = hp.datestr_to_date(d1)
            d2 = hp.datestr_to_date(d2)
            assert len(dates) == (d2 - d1).days + 1
        assert dates[0] == d1
        assert dates[-1] == d2

def test_get_first_week():
    feed = cairns.copy()
    dates = get_first_week(feed)
    d1 = '20140526'
    d2 = '20140601'
    assert dates[0] == d1
    assert dates[-1] == d2
    assert len(dates) == 7
