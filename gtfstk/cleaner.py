"""
This module contains functions for cleaning Feed objects.
"""  
from collections import OrderedDict
import math

import pandas as pd

from . import utilities as ut
from . import constants as cs
from .feed import Feed
from .feed import copy as fcopy


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

def clean_route_short_names(feed):
    """
    In ``feed.routes``, assign route IDs to missing route short names.
    Then disambiguate route short names by applying 
    :func:`utilities.clean_series`.
    Return the resulting routes data frame.
    """
    def my_fillna(row):
        rsn, rid = row
        if pd.isnull(rsn):
            rsn = rid
        return rsn

    routes = feed.routes.copy()
    if routes is not None:
        # Fill NaNs
        routes['route_short_name'] = routes[['route_short_name', 
          'route_id']].apply(my_fillna, axis=1)
        # Disambiguate
        routes['route_short_name'] = ut.clean_series(
          routes['route_short_name'])

    return routes

def prune_dead_routes(feed):
    """
    Remove all routes from ``feed.routes`` that do not have
    trips listed in ``feed.trips``.
    Return a new routes data frame.
    """
    live_routes = feed.trips['route_id'].unique()
    r = feed.routes 
    return r[r['route_id'].isin(live_routes)]

def aggregate_routes(feed):
    """
    Group routes by route short name and for each group, 

    1. choose the first route in the group
    2. assign a new route ID to that route
    3. assign all the trips associated with routes in the group to that first route.

    Update ``feed.routes`` and ``feed.trips`` with the new routes, 
    and return the resulting feed.
    """
    feed = fcopy(feed)

    # Create new route IDs
    routes = feed.routes
    n = routes.groupby('route_short_name').ngroups
    pad = int(math.log10(n)) + 1
    nrid_by_orid = dict()
    i = 1
    for rsn, group in routes.groupby('route_short_name'):
        nrid = 'route_{i:0{pad}}'.format(i=i, pad=pad)
        d = {orid: nrid for orid in group['route_id'].values}
        nrid_by_orid.update(d)
        i += 1

    routes['route_id'] = routes['route_id'].map(lambda x: nrid_by_orid[x])
    routes = routes.groupby('route_short_name').first().reset_index()
    feed.routes = routes

    # Update route IDs of trips
    trips = feed.trips
    trips['route_id'] = trips['route_id'].map(lambda x: nrid_by_orid[x])
    feed.trips = trips

    return feed 

def assess(feed):
    """
    Return a Pandas series containing various feed assessments, such as
    the number of trips missing shapes.
    This is not a GTFS validator.
    """
    d = OrderedDict()

    # Count duplicate route short names
    r = feed.routes
    dup = r.duplicated(subset=['route_short_name'])
    n = dup[dup].count()
    d['num_duplicated_route_short_names'] = n
    d['frac_duplicated_route_short_names'] = n/r.shape[0]

    # Has shape_dist_traveled column in stop times?
    st = feed.stop_times
    if 'shape_dist_traveled' in st.columns:
        d['has_shape_dist_traveled'] = True
        # Count missing distances
        n = st[st['shape_dist_traveled'].isnull()].shape[0]
        d['num_missing_dists'] = n
        d['frac_missing_dists'] = n/st.shape[0]
    else:
        d['has_shape_dist_traveled'] = False
        d['num_missing_dists'] = st.shape[0]
        d['frac_missing_dists'] = 1

    # Has direction ID?
    t = feed.trips
    if 'direction_id' in t.columns:
        d['has_direction_id'] = True
        # Count missing directions
        n = t[t['direction_id'].isnull()].shape[0]
        d['num_missing_directions'] = n
        d['frac_missing_directions'] = n/t.shape[0]
    else:
        d['has_direction_id'] = False
        d['num_missing_directions'] = t.shape[0]
        d['frac_missing_directions'] = 1

    # Count trips missing shapes
    if feed.shapes is not None:
        n = t[t['shape_id'].isnull()].shape[0]
    else:
        n = t.shape[0]
    d['num_trips_missing_shapes'] = n
    d['frac_trips_missing_shapes'] = n/t.shape[0]

    # Count missing departure times
    n = st[st['departure_time'].isnull()].shape[0]
    d['num_missing_departure_times'] = n
    d['frac_missing_departure_times'] = n/st.shape[0]

    # Count missing first departure times
    g = st.groupby('trip_id').agg(lambda x: x.iloc[0]).reset_index()
    n = g[g['departure_time'].isnull()].shape[0]
    d['num_missing_first_departure_times'] = n
    d['frac_missing_first_departure_times'] = n/g.shape[0]

    # Express opinion
    if d['frac_missing_departure_times'] >= 0.8 or\
      d['frac_trips_missing_shapes'] >= 0.8:
        d['assessment'] = 'bad feed'
    elif d['frac_missing_directions'] or\
      d['frac_missing_dists'] or\
      d['num_duplicated_route_short_names']:
        d['assessment'] = 'probably a fixable feed'
    else:
        d['assessment'] = 'good feed'

    return pd.Series(d)