"""
This module contains functions for cleaning Feed objects.
"""  
from collections import OrderedDict
import math

import pandas as pd

from . import utilities as ut
from . import constants as cs
from .feed import Feed





def prune_dead_routes(feed):
    """
    Remove all routes from ``feed.routes`` that do not have trips listed in ``feed.trips``.
    Return the result feed.
    """
    feed = feed.copy()
    live_routes = feed.trips['route_id'].unique()
    r = feed.routes 
    feed.routes = r[r['route_id'].isin(live_routes)]
    return feed 

def aggregate_routes(feed, by='route_short_name', route_id_prefix='route_'):
    """
    Given a GTFSTK Feed object, group routes by the ``by`` column of ``feed.routes`` and for each group, 

    1. choose the first route in the group,
    2. assign a new route ID based on the given ``route_id_prefix`` string and a running count, e.g. ``'route_013'``
    3. assign all the trips associated with routes in the group to that first route.

    Update ``feed.routes`` and ``feed.trips`` with the new routes, and return the resulting feed.
    """
    if by not in feed.routes.columns:
        raise ValueError("Column {!s} not in feed.routes".format(
          by))

    feed = feed.copy()

    # Create new route IDs
    routes = feed.routes
    n = routes.groupby(by).ngroups
    k = int(math.log10(n)) + 1 # Number of digits for padding IDs 
    nrid_by_orid = dict()
    i = 1
    for col, group in routes.groupby(by):
        nrid = 'route_{num:0{pad}d}'.format(num=i, pad=k)
        d = {orid: nrid for orid in group['route_id'].values}
        nrid_by_orid.update(d)
        i += 1

    # # Create new route IDs
    # routes = feed.routes
    # n = routes.groupby(by).ngroups
    # nrid_by_orid = dict()
    # for col, group in routes.groupby(by):
    #     nrid = group['route_id'].iat[0]
    #     d = {orid: nrid for orid in group['route_id'].values}
    #     nrid_by_orid.update(d)

    routes['route_id'] = routes['route_id'].map(lambda x: nrid_by_orid[x])
    routes = routes.groupby(by).first().reset_index()
    feed.routes = routes

    # Update route IDs of trips
    trips = feed.trips
    trips['route_id'] = trips['route_id'].map(lambda x: nrid_by_orid[x])
    feed.trips = trips

    # Update route IDs of transfers
    if feed.transfers is not None:
        transfers = feed.transfers
        transfers['route_id'] = transfers['route_id'].map(
          lambda x: nrid_by_orid[x])
        feed.transfers = transfers

    return feed 

def clean(feed):
    """
    Given a GTFSTK Feed instance, apply the following functions to it and return the resulting feed.

    #. :func:`clean_ids`
    #. :func:`clean_stop_times`
    #. :func:`clean_route_short_names`
    #. :func:`prune_dead_routes`
    """
    feed = feed.copy()
    ops = [
      'clean_ids',
      'clean_stop_times',
      'clean_route_short_names',
      'prune_dead_routes',
    ]
    for op in ops:
        feed = globals()[op](feed)

    return feed

def drop_invalid_columns(feed):
    """
    Given a GTFSTK Feed instance, drop all data frame columns not listed in ``constants.VALID_COLS``.
    Return the resulting feed.
    """
    for key, vcols in cs.VALID_COLUMNS_BY_TABLE.items():
        f = getattr(feed, key)
        if f is None:
            continue
        for col in f.columns:
            if col not in vcols:
                print('{!s}: dropping invalid column {!s}'.format(key, col))
                del f[col]

    return feed

def assess(feed):
    """
    Return a Pandas series containing various feed assessments, such as the number of trips missing shapes.
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

    # Count missing last departure times
    g = st.groupby('trip_id').agg(lambda x: x.iloc[-1]).reset_index()
    n = g[g['departure_time'].isnull()].shape[0]
    d['num_missing_last_departure_times'] = n
    d['frac_missing_last_departure_times'] = n/g.shape[0]

    # Express an opinion
    if (d['frac_missing_first_departure_times'] >= 0.1) or\
      (d['frac_missing_last_departure_times'] >= 0.1) or\
      d['frac_trips_missing_shapes'] >= 0.8:
        d['assessment'] = 'bad feed'
    elif d['frac_missing_directions'] or\
      d['frac_missing_dists'] or\
      d['num_duplicated_route_short_names']:
        d['assessment'] = 'probably a fixable feed'
    else:
        d['assessment'] = 'good feed'

    return pd.Series(d)