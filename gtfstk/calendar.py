"""
Functions about calendar and calendar_dates.
"""
import dateutil.relativedelta as rd

from . import helpers as hp


def get_dates(feed, as_date_obj=False):
    """
    Return a list of dates for which the given Feed is valid, which
    could be the empty list if the Feed has no calendar information.

    Parameters
    ----------
    feed : Feed
    as_date_obj : boolean
        If ``True``, then return the dates as ``datetime.date`` objects;
        otherwise return them as strings

    Returns
    -------
    list
        Dates

    """
    dates = []
    if feed.calendar is not None:
        if 'start_date' in feed.calendar.columns:
            dates.append(feed.calendar['start_date'].min())
        if 'end_date' in feed.calendar.columns:
            dates.append(feed.calendar['end_date'].max())
    if feed.calendar_dates is not None:
        if 'date' in feed.calendar_dates.columns:
            start = feed.calendar_dates['date'].min()
            end = feed.calendar_dates['date'].max()
            dates.extend([start, end])
    if not dates:
        return []

    start_date, end_date = min(dates), max(dates)
    start_date, end_date = map(hp.datestr_to_date, [start_date, end_date])
    num_days = (end_date - start_date).days
    result = [start_date + rd.relativedelta(days=+d)
      for d in range(num_days + 1)]

    # Convert dates back to strings if required
    if not as_date_obj:
        result = [hp.datestr_to_date(x, inverse=True)
          for x in result]

    return result

def get_first_week(feed, as_date_obj=False):
    """
    Return a list of date corresponding to the first Monday--Sunday
    week for which this feed is valid.
    If the given feed does not cover a full Monday--Sunday week,
    then return whatever initial segment of the week it does cover,
    which could be the empty list.

    Parameters
    ----------
    feed : Feed
    as_date_obj : boolean
        If ``True``, then return the dates as ``datetime.date`` objects;
        otherwise return them as strings

    Returns
    -------
    list
        Dates

    """
    dates = feed.get_dates(as_date_obj=True)
    if not dates:
        return []

    # Get first Monday
    monday_index = None
    for (i, date) in enumerate(dates):
        if date.weekday() == 0:
            monday_index = i
            break
    if monday_index is None:
        return []

    result = []
    for j in range(7):
        try:
            result.append(dates[monday_index + j])
        except:
            break

    # Convert to date strings if requested
    if not as_date_obj:
        result = [hp.datestr_to_date(x, inverse=True)
          for x in result]

    return result

def restrict_dates(feed, dates):
    """
    Given a Feed and a date (YYYYMMDD string) or list of dates,
    coerce the date/dates into a list and drop the dates not in
    ``feed.get_dates()``, preserving the original order of ``dates``.
    Intended as a helper function.
    """
    # Coerce string to set
    if isinstance(dates, str):
        dates = [dates]

    # Restrict
    return [d for d in dates if d in feed.get_dates()]
