"""
Functions about cleaning feeds.
"""
import math

import pandas as pd 
import numpy as np
import utm
import shapely.geometry as sg 

from . import constants as cs
from . import helpers as hp


def clean_ids(feed):
    """
    Strip whitespace from all string IDs and then replace every remaining whitespace chunk with an underscore.
    Return the resulting feed.  
    """
    # Alter feed inputs only, and build a new feed from them.
    # The derived feed attributes, such as feed.trips_i, 
    # will be automatically handled when creating the new feed.
    feed = feed.copy()

    for table in cs.GTFS_REF['table']:
        f = getattr(feed, table)
        if f is None:
            continue
        for column in cs.GTFS_REF.loc[cs.GTFS_REF['table'] == table, 'column']:
            if column in f.columns and column.endswith('_id'):
                try:
                    f[column] = f[column].str.strip().str.replace(
                      r'\s+', '_')
                    setattr(feed, table, f)
                except AttributeError:
                    # Column is not of string type
                    continue

    return feed

def clean_stop_times(feed):
    """
    In ``feed.stop_times``, prefix a zero to arrival and departure times if necessary.
    This makes sorting by time work as expected.
    Return the resulting feed.
    """
    feed = feed.copy()
    st = feed.stop_times

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

    feed.stop_times = st 
    return feed

def clean_route_short_names(feed):
    """
    In ``feed.routes``, assign 'n/a' to missing route short names and strip whitespace from route short names.
    Then disambiguate each route short name that is duplicated by appending '-' and its route ID.
    Return the resulting feed.
    """
    feed = feed.copy()
    r = feed.routes
    if r is None:
        return feed

    # Fill NaNs and strip whitespace
    r['route_short_name'] = r['route_short_name'].fillna(
      'n/a').str.strip()
    # Disambiguate
    def disambiguate(row):
        rsn, rid = row
        return rsn + '-' + rid

    r['dup'] = r['route_short_name'].duplicated(keep=False)
    r.loc[r['dup'], 'route_short_name'] = r.loc[
      r['dup'], ['route_short_name', 'route_id']].apply(
      disambiguate, axis=1)
    del r['dup']

    feed.routes = r
    return feed

def drop_dead_routes(feed):
    """
    Remove every route from ``feed.routes`` that does not have trips listed in ``feed.trips``.
    Return the resulting new feed.
    """
    feed = feed.copy()
    live_routes = feed.trips['route_id'].unique()
    r = feed.routes 
    feed.routes = r[r['route_id'].isin(live_routes)]
    return feed 

def aggregate_routes(feed, by='route_short_name', 
  route_id_prefix='route_'):
    """
    Group ``feed.routes`` by the ``by`` column, and for each group 

    1. choose the first route in the group,
    2. assign a new route ID based on the given ``route_id_prefix`` string and a running count, e.g. ``'route_013'``
    3. assign all the trips associated with routes in the group to that first route.

    Return a new feed with the updated routes and trips.
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
    Apply the following functions to this feed and return the resulting new feed.

    #. :func:`clean_ids`
    #. :func:`clean_stop_times`
    #. :func:`clean_route_short_names`
    #. :func:`drop_dead_routes`
    """
    feed = feed.copy()
    ops = [
      'clean_ids',
      'clean_stop_times',
      'clean_route_short_names',
      'drop_dead_routes',
    ]
    for op in ops:
        feed = getattr(feed, op)()

    return feed

def drop_invalid_columns(feed):
    """
    Drop all data frame columns of this feed not listed in the GTFS.
    Return the resulting new feed.
    """
    feed = feed.copy()
    for table, group in cs.GTFS_REF.groupby('table'):
        f = getattr(feed, table)
        if f is None:
            continue
        valid_columns = group['column'].values
        for col in f.columns:
            if col not in valid_columns:
                print('{!s}: dropping invalid column {!s}'.format(table, col))
                del f[col]
        setattr(feed, table, f)

    return feed
