"""
Functions about calendar and calendar_dates.
"""
import dateutil.relativedelta as rd

from . import helpers as hp


def get_dates(feed, as_date_obj=False):
    """
    Return a chronologically ordered list of dates (strings) for which
    this feed is valid.

    Parameters
    ----------
    feed : Feed
    as_date_obj : boolean
        If ``True``, then return the dates as ``datetime.date`` objects;
        otherwise return them as strings

    Returns
    -------
    list

    Notes
    -----
    If ``feed.calendar`` and ``feed.calendar_dates`` are both ``None``,
    then return the empty list.

    """
    if feed.calendar is not None:
        start_date = feed.calendar['start_date'].min()
        end_date = feed.calendar['end_date'].max()
    elif feed.calendar_dates is not None:
        # Use calendar_dates
        start_date = feed.calendar_dates['date'].min()
        end_date = feed.calendar_dates['date'].max()
    else:
        return []

    start_date = hp.datestr_to_date(start_date)
    end_date = hp.datestr_to_date(end_date)
    num_days = (end_date - start_date).days
    result = [start_date + rd.relativedelta(days=+d)
      for d in range(num_days + 1)]

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
