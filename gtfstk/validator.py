"""
This module contains functions that supplement but do not replace the ``feedvalidator`` module of the `transitfeed package <https://github.com/google/transitfeed>`_.
The latter module checks if GFTS feeds adhere to the `GTFS specification <https://developers.google.com/transit/gtfs/reference?hl=en>`_.
"""  


class GTFSError(Exception):
    """
    Exception raised for Feed objects that do not conform to the GTFS specification.

    Attributes:
    
    - msg: explanation of the error
    """
    def __init__(feed, msg):
        feed.msg = msg
   
    def __str__(feed):
        return repr(feed.msg)


def check_calendar(feed):
    """
    Check that one of ``feed.calendar`` or ``feed.calendar_dates`` is nonempty.
    """
    if (feed.calendar is None and feed.calendar_dates is None) or\
      (feed.calendar.empty and feed.calendar_dates.empty):
        raise GTFSError("calendar or calendar_dates must be nonempty")
