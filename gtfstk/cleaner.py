from . import utilities as utils 

import pandas as pd
"""
Functions for cleaning a Feed object.
"""  

def clean_stop_times(feed):
    """
    In ``feed.stop_times``, prefix a zero to arrival and 
    departure times if necessary.
    This makes sorting by time work as expected.
    Return the resulting stop times data frame.
    """
    st = feed.stop_times.copy()

    def reformat(t):
        if pd.isnull(t):
            return t
        t = t.strip()
        if len(t) == 7:
            t = '0' + t
        return t

    if st is not None:
        st[['arrival_time', 'departure_time']] = st[['arrival_time', 
          'departure_time']].applymap(reformat)

    return st

def clean_routes(feed):
    """
    In ``feed.routes``, disambiguate the ``route_short_name`` column 
    using ``utils.clean_series``.
    Among other things, this will disambiguate duplicate
    route short names.
    Return the resulting routes data frame.
    """
    routes = feed.routes.copy()
    if routes is not None:
        routes['route_short_name'] = utils.clean_series(
          routes['route_short_name'])
    return routes
