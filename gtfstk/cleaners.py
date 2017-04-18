"""
Functions about cleaning feeds.
"""
import math

import pandas as pd

from . import constants as cs


def clean_column_names(df):
    """
    Strip the whitespace from all column names in the given DataFrame and return the result.
    """
    f = df.copy()
    f.columns = [col.strip() for col in f.columns]
    return f

def drop_zombies(feed):
    """
    Drop stops with no stop times, trips with no stop times, shapes with no trips, routes with no trips, and services with no trips, in that order.
    Return the resulting new feed.
    """
    feed = feed.copy()

    # Drop stops with no stop times
    ids = feed.stop_times['stop_id'].unique()
    f = feed.stops
    feed.stops = f[f['stop_id'].isin(ids)]

    # Drop trips with no stop times
    ids = feed.stop_times['trip_id'].unique()
    f = feed.trips
    feed.trips = f[f['trip_id'].isin(ids)]

    # Drop shapes with no trips
    ids = feed.trips['shape_id'].unique()
    f = feed.shapes
    if f is not None:
        feed.shapes = f[f['shape_id'].isin(ids)]

    # Drop routes with no trips
    ids = feed.trips['route_id'].unique()
    f = feed.routes
    feed.routes = f[f['route_id'].isin(ids)]

    # Drop services with no trips
    ids = feed.trips['service_id'].unique()
    if feed.calendar is not None:
        f = feed.calendar
        feed.calendar = f[f['service_id'].isin(ids)]
    if feed.calendar_dates is not None:
        f = feed.calendar_dates
        feed.calendar_dates = f[f['service_id'].isin(ids)]

    return feed

def clean_ids(feed):
    """
    Strip whitespace from all string IDs and then replace every remaining whitespace chunk with an underscore.
    Return the resulting feed.
    """
    # Alter feed inputs only, and build a new feed from them.
    # The derived feed attributes, such as feed.trips_i,
    # will be automatically handled when creating the new feed.
    feed = feed.copy()

    for table in cs.GTFS_REF['table'].unique():
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

def clean_times(feed):
    """
    Prefix a zero to each H:MM:SS time to make it an HH:MM:SS.
    This makes sorting by time work as expected.
    Return the resulting feed.
    """
    def reformat(t):
        if pd.isnull(t):
            return t
        t = t.strip()
        if len(t) == 7:
            t = '0' + t
        return t

    feed = feed.copy()
    tables_and_columns = [
      ('stop_times', ['arrival_time', 'departure_time']),
      ('frequencies', ['start_time', 'end_time']),
    ]
    for table, columns in tables_and_columns:
        f = getattr(feed, table)
        if f is not None:
            f[columns] = f[columns].applymap(reformat)
        setattr(feed, table, f)

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

def aggregate_routes(feed, by='route_short_name', route_id_prefix='route_'):
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
    k = int(math.log10(n)) + 1  # Number of digits for padding IDs
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
    Apply

    #. :func:`drop_zombies`
    #. :func:`clean_ids`
    #. :func:`clean_times`
    #. :func:`clean_route_short_names`

    to the given feed in that order.
    Return the resulting feed.
    """
    feed = feed.copy()
    ops = [
      'clean_ids',
      'clean_times',
      'clean_route_short_names',
      'drop_zombies',
    ]
    for op in ops:
        feed = globals()[op](feed)

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
