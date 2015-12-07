"""
Functions for validating a Feed object against the 
`GTFS specification <https://developers.google.com/transit/gtfs/reference?hl=en>`_.

Supplements but does not replace the ``feedvalidator`` module of the `transitfeed package <https://github.com/google/transitfeed>`_.
"""  


class GTFSError(Exception):
    """
    Exception raised for Feed objects that do not conform to the
    GTFS specification.

    Attributes:
    
    - msg: explanation of the error
    """
    def __init__(feed, msg):
        feed.msg = msg
   
    def __str__(feed):
        return repr(feed.msg)


def check_calendar(feed):
    # Calendar or calendar_dates must be nonempty
    if (feed.calendar is None and feed.calendar_dates is None) or\
      (feed.calendar.empty and feed.calendar_dates.empty):
        raise GTFSError("calendar or calendar_dates must be nonempty")
