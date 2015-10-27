"""
Functions for cleaning a Feed object.
"""  

def reformat_stop_times(feed):
    """
    Clean some feed attributes and return a new feed.
    """
    new_feed = deepcopy(feed)

    # Prefix a 0 to arrival and departure times if necessary.
    # This makes sorting by time work as expected.
    def reformat_times(t):
        if pd.isnull(t):
            return t
        t = t.strip()
        if len(t) == 7:
            t = '0' + t
        return t

    st = new_feed.stop_times
    if st is not None:
        st[['arrival_time', 'departure_time']] = st[['arrival_time', 
          'departure_time']].applymap(reformat_times)
        new_feed.stop_times = st

    return new_feed

def disambiguate_route_short_names(feed):
    """
    Clean the ``route_short_name`` column in ``feed.routes`` 
    using ``utils.clean_series``.
    Among other things, this will disambiguate duplicate
    route short names.
    Return the resulting new feed.
    """
    new_feed = deepcopy(feed)
    new_feed.routes['route_short_name'] = utils.clean_series(
      new_feed.routes['route_short_name'])
