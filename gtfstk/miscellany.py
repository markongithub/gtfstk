"""
Functions about miscellany.
"""
from collections import OrderedDict
import math
import copy

import pandas as pd
import numpy as np
import shapely.geometry as sg

from . import helpers as hp
from . import constants as cs


def summarize(feed, table=None):
    """
    Return a DataFrame summarizing all GTFS tables in the given feed
    or in the given table if specified.

    Parameters
    ----------
    feed : Feed
    table : string
        A GTFS table name, e.g. ``'stop_times'``

    Returns
    -------
    DataFrame
        Columns are

        - ``'table'``: name of the GTFS table, e.g. ``'stops'``
        - ``'column'``: name of a column in the table,
          e.g. ``'stop_id'``
        - ``'num_values'``: number of values in the column
        - ``'num_nonnull_values'``: number of nonnull values in the
          column
        - ``'num_unique_values'``: number of unique values in the
          column, excluding null values
        - ``'min_value'``: minimum value in the column
        - ``'max_value'``: maximum value in the column

    Notes
    -----
    - If the table is not in the feed, then return an empty DataFrame
      with the columns above.
    - If the table is not valid, raise a ValueError.

    """
    gtfs_tables = cs.GTFS_REF.table.unique()

    if table is not None:
        if table not in gtfs_tables:
            raise ValueError('{!s} is not a GTFS table'.format(table))
        else:
            tables = [table]
    else:
        tables = gtfs_tables

    frames = []
    for table in tables:
        f = getattr(feed, table)
        if f is None:
            continue

        def my_agg(col):
            d = {}
            d['column'] = col.name
            d['num_values'] = col.size
            d['num_nonnull_values'] = col.count()
            d['num_unique_values'] = col.nunique()
            d['min_value'] = col.dropna().min()
            d['max_value'] = col.dropna().max()
            return pd.Series(d)

        g = f.apply(my_agg).T.reset_index(drop=True)
        g['table'] = table
        frames.append(g)

    cols = ['table', 'column', 'num_values', 'num_nonnull_values',
      'num_unique_values', 'min_value', 'max_value']

    if not frames:
        f = pd.DataFrame(columns=cols)
    else:
        f = pd.concat(frames)
        # Rearrange columns
        f = f[cols].copy()

    return f

def describe(feed, sample_date=None):
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of routes.
    Specialize some those indicators to the given sample date,
    e.g. number of routes active on the date.

    Parameters
    ----------
    feed : Feed
    sample_date : string
        YYYYMMDD date string specifying the date to compute sample
        stats; defaults to the first Thursday of the Feed's period

    Returns
    -------
    DataFrame
        The columns are

        - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
        - ``'value'``: value of the indicator, e.g. 27

    """
    from . import calendar as cl

    d = OrderedDict()
    dates = cl.get_dates(feed)
    d['agencies'] = feed.agency['agency_name'].tolist()
    d['timezone'] = feed.agency['agency_timezone'].iat[0]
    d['start_date'] = dates[0]
    d['end_date'] = dates[-1]
    d['num_routes'] = feed.routes.shape[0]
    d['num_trips'] = feed.trips.shape[0]
    d['num_stops'] = feed.stops.shape[0]
    if feed.shapes is not None:
        d['num_shapes'] = feed.shapes['shape_id'].nunique()
    else:
        d['num_shapes'] = 0

    if sample_date is None or sample_date not in feed.get_dates():
        sample_date = cl.get_first_week(feed)[3]
    d['sample_date'] = sample_date
    d['num_routes_active_on_sample_date'] = feed.get_routes(
      sample_date).shape[0]
    trips = feed.get_trips(sample_date)
    d['num_trips_active_on_sample_date'] = trips.shape[0]
    d['num_stops_active_on_sample_date'] = feed.get_stops(
      sample_date).shape[0]
    f = pd.DataFrame(list(d.items()), columns=['indicator', 'value'])

    return f

def assess_quality(feed):
    """
    Return a DataFrame of various feed indicators and values,
    e.g. number of trips missing shapes.

    Parameters
    ----------
    feed : Feed

    Returns
    -------
    DataFrame
        The columns are

        - ``'indicator'``: string; name of an indicator, e.g. 'num_routes'
        - ``'value'``: value of the indicator, e.g. 27

    Notes
    -----
    - An odd function, but useful to see roughly how broken a
      feed is
    - Not a GTFS validator

    """
    d = OrderedDict()

    # Count duplicate route short names
    r = feed.routes
    dup = r.duplicated(subset=['route_short_name'])
    n = dup[dup].count()
    d['num_route_short_names_duplicated'] = n
    d['frac_route_short_names_duplicated'] = n/r.shape[0]

    # Count stop times missing shape_dist_traveled values
    st = feed.stop_times.sort_values(['trip_id', 'stop_sequence'])
    if 'shape_dist_traveled' in st.columns:
        # Count missing distances
        n = st[st['shape_dist_traveled'].isnull()].shape[0]
        d['num_stop_time_dists_missing'] = n
        d['frac_stop_time_dists_missing'] = n/st.shape[0]
    else:
        d['num_stop_time_dists_missing'] = st.shape[0]
        d['frac_stop_time_dists_missing'] = 1

    # Count direction_ids missing
    t = feed.trips
    if 'direction_id' in t.columns:
        n = t[t['direction_id'].isnull()].shape[0]
        d['num_direction_ids_missing'] = n
        d['frac_direction_ids_missing'] = n/t.shape[0]
    else:
        d['num_direction_ids_missing'] = t.shape[0]
        d['frac_direction_ids_missing'] = 1

    # Count trips missing shapes
    if feed.shapes is not None:
        n = t[t['shape_id'].isnull()].shape[0]
    else:
        n = t.shape[0]
    d['num_trips_missing_shapes'] = n
    d['frac_trips_missing_shapes'] = n/t.shape[0]

    # Count missing departure times
    n = st[st['departure_time'].isnull()].shape[0]
    d['num_departure_times_missing'] = n
    d['frac_departure_times_missing'] = n/st.shape[0]

    # Count missing first departure times missing
    g = st.groupby('trip_id').first().reset_index()
    n = g[g['departure_time'].isnull()].shape[0]
    d['num_first_departure_times_missing'] = n
    d['frac_first_departure_times_missing'] = n/st.shape[0]

    # Count missing last departure times
    g = st.groupby('trip_id').last().reset_index()
    n = g[g['departure_time'].isnull()].shape[0]
    d['num_last_departure_times_missing'] = n
    d['frac_last_departure_times_missing'] = n/st.shape[0]

    # Opine
    if (d['frac_first_departure_times_missing'] >= 0.1) or\
      (d['frac_last_departure_times_missing'] >= 0.1) or\
      d['frac_trips_missing_shapes'] >= 0.8:
        d['assessment'] = 'bad feed'
    elif d['frac_direction_ids_missing'] or\
      d['frac_stop_time_dists_missing'] or\
      d['num_route_short_names_duplicated']:
        d['assessment'] = 'probably a fixable feed'
    else:
        d['assessment'] = 'good feed'

    f = pd.DataFrame(list(d.items()), columns=['indicator', 'value'])

    return f

def convert_dist(feed, new_dist_units):
    """
    Convert the distances recorded in the ``shape_dist_traveled``
    columns of the given Feed to the given distance units.
    New distance units must lie in :const:`.constants.DIST_UNITS`.
    """
    feed = feed.copy()

    if feed.dist_units == new_dist_units:
        # Nothing to do
        return feed

    old_dist_units = feed.dist_units
    feed.dist_units = new_dist_units

    converter = hp.get_convert_dist(old_dist_units, new_dist_units)

    if hp.is_not_null(feed.stop_times, 'shape_dist_traveled'):
        feed.stop_times['shape_dist_traveled'] =\
          feed.stop_times['shape_dist_traveled'].map(converter)

    if hp.is_not_null(feed.shapes, 'shape_dist_traveled'):
        feed.shapes['shape_dist_traveled'] =\
          feed.shapes['shape_dist_traveled'].map(converter)

    return feed

def compute_feed_stats(feed, trip_stats, dates):
    """
    Compute some feed stats for the given dates and trip stats.

    Parameters
    ----------
    feed : Feed
    trip_stats : DataFrame
        Trip stats to consider in the format output by
        :func:`.trips.compute_trip_stats`
    dates : string or list
        A YYYYMMDD date string or list thereof indicating the date(s)
        for which to compute stats

    Returns
    -------
    DataFrame
        The columns are

        - ``'date'``
        - ``'num_trips'``: number of trips active on the date
        - ``'num_routes'``: number of routes active on the date
        - ``'num_stops'``: number of stops active on the date
        - ``'peak_num_trips'``: maximum number of simultaneous trips in
          service on the date
        - ``'peak_start_time'``: start time of first longest period
          during which the peak number of trips occurs on the date
        - ``'peak_end_time'``: end time of first longest period during
          which the peak number of trips occurs on the date
        - ``'service_distance'``: sum of the service distances for the
          active routes on the date
        - ``'service_duration'``: sum of the service durations for the
          active routes on the date
        - ``'service_speed'``: service_distance/service_duration on the
          date

        Exclude dates that lie outside of the Feed's date range.
        If all the dates given lie outside of the feed's date range,
        then return an empty DataFrame with the specified columns.

    Notes
    -----
    Assume the following feed attributes are not ``None``:

    - Those used in :func:`.trips.get_trips`
    - Those used in :func:`.routes.get_routes`
    - Those used in :func:`.stops.get_stops`

    """
    dates = feed.restrict_dates(dates)
    cols = [
      'date',
      'num_stops',
      'num_routes',
      'num_trips',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_distance',
      'service_duration',
      'service_speed',
    ]
    if not dates:
        return pd.DataFrame([], columns=cols)

    ts = trip_stats.copy()
    activity = feed.compute_trip_activity(dates)
    stop_times = feed.stop_times.copy()

    # Convert timestrings to seconds for quicker calculations
    ts[['start_time', 'end_time']] =\
      ts[['start_time', 'end_time']].applymap(hp.timestr_to_seconds)

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence ->
    # [stats dictionary, date list that stats apply]
    stats_and_dates_by_ids = {}
    for date in dates:
        stats = {}
        ids = tuple(activity.loc[activity[date] > 0, 'trip_id'])
        if ids in stats_and_dates_by_ids:
            # Append date to date list
            stats_and_dates_by_ids[ids][1].append(date)
        elif not ids:
            # Empty stats
            stats = {col: np.nan for col in cols
              if col != 'date'}
            stats_and_dates_by_ids[ids] = [stats, [date]]
        else:
            # Compute stats
            f = ts[ts['trip_id'].isin(ids)].copy()
            stats['num_stops'] = stop_times.loc[
              stop_times['trip_id'].isin(ids), 'stop_id'].nunique()
            stats['num_routes'] = f['route_id'].nunique()
            stats['num_trips'] = f.shape[0]
            stats['service_distance'] = f['distance'].sum()
            stats['service_duration'] = f['duration'].sum()
            stats['service_speed'] =\
              stats['service_distance']/stats['service_duration']

            # Compute peak stats, which is the slowest part
            times = np.unique(f[['start_time', 'end_time']].values)
            counts = [hp.count_active_trips(f, t) for t in times]
            start, end = hp.get_peak_indices(times, counts)
            stats['peak_num_trips'] = counts[start]
            stats['peak_start_time'] = times[start]
            stats['peak_end_time'] = times[end]

            # Record stats
            stats_and_dates_by_ids[ids] = [stats, [date]]

    # Assemble stats into DataFrame
    rows = []
    for stats, dates in stats_and_dates_by_ids.values():
        for date in dates:
            s = copy.copy(stats)
            s['date'] = date
            rows.append(s)
    f = pd.DataFrame(rows).sort_values('date')

    # Convert seconds back to timestrings
    times = ['peak_start_time', 'peak_end_time']
    f[times] = f[times].applymap(
      lambda t: hp.timestr_to_seconds(t, inverse=True))

    return f[cols].copy()

def compute_feed_time_series(feed, trip_stats, dates, freq='5Min'):
    """
    Compute some feed stats in time series form for the given dates
    and trip stats.

    Parameters
    ----------
    feed : Feed
    trip_stats : DataFrame
        Trip stats to consider in the format output by
        :func:`.trips.compute_trip_stats`
    dates : string or list
        A YYYYMMDD date string or list thereof indicating the date(s)
        for which to compute stats
    freq : string
        Pandas frequency string specifying the frequency of the
        resulting time series, e.g. '5Min'; highest frequency allowable
        is one minute ('Min').

    Returns
    -------
    DataFrame
        A time series with a timestamp index across the given dates
        sampled at the given frequency.
        The maximum allowable frequency is 1 minute.

        The columns are

        - ``'num_trip_starts'``: number of trips starting during the
          time period
        - ``'num_trips'``: number of trips in service during during the
          time period
        - ``'service_distance'``: distance traveled by all active trips
          during the time period
        - ``'service_duration'``: duration traveled by all active trips
          during the time period
        - ``'service_speed'``: service_distance/service_duration

        Exclude dates that lie outside of the Feed's date range.
        If all the dates given lie outside of the Feed's date range,
        then return an empty DataFrame with the specified columns.

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - To manually resample the resulting time series, use the following
      methods:

        * for ``'num_trips'`` series, use ``how=np.mean``
        * for the other series, use ``how=np.sum``
        * ``'service_speed'`` cannot be resampled and must be
          recalculated from ``'service_distance'`` and
          ``'service_duration'``

    - To remove the date and seconds from the time series f, do
      ``f.index = [t.time().strftime('%H:%M') for t in f.index.to_datetime()]``
    - If all dates lie outside the feed's date range, then return an
      empty DataFrame with only the column ``'num_trips'``.
    - Dates with no active stops will not appear in the result
      (in contrast to the output of :func:`compute_feed_stats`)
    - Assume the following feed attributes are not ``None``:

       * Those used in :func:`.routes.compute_route_time_series`

    """
    cols = [
      'num_trip_starts',
      'num_trip_ends',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
    ]
    rts = feed.compute_route_time_series(trip_stats, dates, freq=freq)
    if rts.empty:
        return pd.DataFrame(columns=cols).sort_index(axis=1)

    f = pd.concat([rts[col].sum(axis=1) for col in cols], axis=1, keys=cols)
    f['service_speed'] = f['service_distance']/f['service_duration']

    return f.sort_index(axis=1)

def create_shapes(feed, all_trips=False):
    """
    Given a feed, create a shape for every trip that is missing a
    shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips
    tables.

    If ``all_trips``, then create new shapes for all trips by
    connecting stops, and remove the old shapes.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``

    """
    feed = feed.copy()

    if all_trips:
        trip_ids = feed.trips['trip_id']
    else:
        trip_ids = feed.trips[feed.trips['shape_id'].isnull()]['trip_id']

    # Get stop times for given trips
    f = feed.stop_times[feed.stop_times['trip_id'].isin(trip_ids)][
      ['trip_id', 'stop_sequence', 'stop_id']]
    f = f.sort_values(['trip_id', 'stop_sequence'])

    if f.empty:
        # Nothing to do
        return feed

    # Create new shape IDs for given trips.
    # To do this, collect unique stop sequences,
    # sort them to impose a canonical order, and
    # assign shape IDs to them
    stop_seqs = sorted(set(tuple(group['stop_id'].values)
      for trip, group in f.groupby('trip_id')))
    d = int(math.log10(len(stop_seqs))) + 1  # Digits for padding shape IDs
    shape_by_stop_seq = {seq: 'shape_{num:0{pad}d}'.format(num=i, pad=d)
      for i, seq in enumerate(stop_seqs)}

    # Assign these new shape IDs to given trips
    shape_by_trip = {trip: shape_by_stop_seq[tuple(group['stop_id'].values)]
      for trip, group in f.groupby('trip_id')}
    trip_cond = feed.trips['trip_id'].isin(trip_ids)
    feed.trips.loc[trip_cond, 'shape_id'] = feed.trips.loc[trip_cond,
      'trip_id'].map(lambda x: shape_by_trip[x])

    # Build new shapes for given trips
    G = [[shape, i, stop] for stop_seq, shape in shape_by_stop_seq.items()
      for i, stop in enumerate(stop_seq)]
    g = pd.DataFrame(G, columns=['shape_id', 'shape_pt_sequence',
      'stop_id'])
    g = g.merge(feed.stops[['stop_id', 'stop_lon', 'stop_lat']]).sort_values(
      ['shape_id', 'shape_pt_sequence'])
    g = g.drop(['stop_id'], axis=1)
    g = g.rename(columns={
      'stop_lon': 'shape_pt_lon',
      'stop_lat': 'shape_pt_lat',
    })

    if feed.shapes is not None and not all_trips:
        # Update feed shapes with new shapes
        feed.shapes = pd.concat([feed.shapes, g])
    else:
        # Create all new shapes
        feed.shapes = g

    return feed

def compute_bounds(feed):
    """
    Return the tuple (min longitude, min latitude, max longitude,
    max latitude) where the longitudes and latitude vary across all
    the feed's stop coordinates.
    """
    lons, lats = feed.stops['stop_lon'], feed.stops['stop_lat']
    return lons.min(), lats.min(), lons.max(), lats.max()

def compute_convex_hull(feed):
    """
    Return a Shapely Polygon representing the convex hull formed by
    the stops of the given Feed.
    """
    m = sg.MultiPoint(feed.stops[['stop_lon', 'stop_lat']].values)
    return m.convex_hull

def compute_center(feed, num_busiest_stops=None):
    """
    Return the centroid (WGS84 longitude-latitude pair) of the convex
    hull of the stops of the given Feed.
    If ``num_busiest_stops`` (integer) is given,
    then compute the ``num_busiest_stops`` busiest stops in the feed
    on the first Monday of the feed and return the mean of the
    longitudes and the mean of the latitudes of these stops,
    respectively.
    """
    s = feed.stops.copy()
    if num_busiest_stops is None:
        hull = compute_convex_hull(feed)
        lon, lat = list(hull.centroid.coords)[0]
    else:
        date = feed.get_first_week()[0]
        ss = feed.compute_stop_stats([date]).sort_values(
          'num_trips', ascending=False)
        f = ss.head(num_busiest_stops)
        f = s.merge(f)
        lon = f['stop_lon'].mean()
        lat = f['stop_lat'].mean()
    return lon, lat

def restrict_to_routes(feed, route_ids):
    """
    Build a new feed by restricting this one to only the stops,
    trips, shapes, etc. used by the routes with the given list of
    route IDs.
    Return the resulting feed.
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Slice routes
    feed.routes = feed.routes[feed.routes['route_id'].isin(
      route_ids)].copy()

    # Slice trips
    feed.trips = feed.trips[feed.trips['route_id'].isin(route_ids)].copy()

    # Slice stop times
    trip_ids = feed.trips['trip_id']
    feed.stop_times = feed.stop_times[
      feed.stop_times['trip_id'].isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times['stop_id'].unique()
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()

    # Slice calendar
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()

    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()

    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates[
          feed.calendar_dates['service_id'].isin(service_ids)].copy()

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()

    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers[
          feed.transfers['from_stop_id'].isin(stop_ids) |
          feed.transfers['to_stop_id'].isin(stop_ids)].copy()

    return feed

def restrict_to_polygon(feed, polygon):
    """
    Build a new feed by restricting this one to only the trips
    that have at least one stop intersecting the given polygon,
    then restricting stops, routes, stop times, etc. to those
    associated with that subset of trips.
    Return the resulting feed.

    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``
    - ``feed.routes``
    - Those used in :func:`.stops.get_stops_in_polygon`

    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()

    # Get IDs of stops within the polygon
    stop_ids = feed.get_stops_in_polygon(polygon)['stop_id']

    # Get all trips that stop at at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st[st['stop_id'].isin(stop_ids)]['trip_id']
    feed.trips = feed.trips[feed.trips['trip_id'].isin(trip_ids)].copy()

    # Get stop times for trips
    feed.stop_times = st[st['trip_id'].isin(trip_ids)].copy()

    # Get stops for trips
    stop_ids = feed.stop_times['stop_id']
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()

    # Get routes for trips
    route_ids = feed.trips['route_id']
    feed.routes = feed.routes[feed.routes['route_id'].isin(
      route_ids)].copy()

    # Get calendar for trips
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()

    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()

    # Now for the optional files.
    # Get calendar dates for trips.
    cd = feed.calendar_dates
    if cd is not None:
        feed.calendar_dates = cd[cd['service_id'].isin(service_ids)].copy()

    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()

    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()

    # Get transfers for stops
    if feed.transfers is not None:
        t = feed.transfers
        feed.transfers = t[t['from_stop_id'].isin(stop_ids) |
          t['to_stop_id'].isin(stop_ids)].copy()

    return feed

def compute_screen_line_counts(feed, linestring, dates, geo_shapes=None):
    """
    Find all the Feed trips active on the given dates
    that intersect the given Shapely LineString (with WGS84
    longitude-latitude coordinates).

    Parameters
    ----------
    feed : Feed
    linestring : Shapely LineString
    dates : list
        YYYYMMDD date strings

    Returns
    -------
    DataFrame
        The columns are

        - ``'date'``
        - ``'trip_id'``
        - ``'route_id'``
        - ``'route_short_name'``
        - ``'crossing_time'``: time that the trip's vehicle crosses
          the linestring; one trip could cross multiple times
        - ``'orientation'``: 1 or -1; 1 indicates trip travel from the
          left side to the right side of the screen line;
          -1 indicates trip travel in the  opposite direction

    Notes
    -----
    - Requires GeoPandas
    - The first step is to geometrize ``feed.shapes`` via
      :func:`.shapes.geometrize_shapes`. Alternatively, use the
      ``geo_shapes`` GeoDataFrame, if given.
    - Assume ``feed.stop_times`` has an accurate
      ``shape_dist_traveled`` column.
    - Assume that trips travel in the same direction as their
      shapes. That restriction is part of GTFS, by the way.
      To calculate direction quickly and accurately, assume that
      the screen line is straight and doesn't double back on itself.
    - Probably does not give correct results for trips with
      self-intersecting shapes.
    - The algorithm works as follows

        1. Compute all the shapes that intersect the linestring
        2. For each such shape, compute the intersection points
        3. For each point p, scan through all the trips in the feed
           that have that shape
        4. For each date in ``dates``, restrict to trips active on the
           date and interpolate a stop time for p by assuming that the
           feed has the shape_dist_traveled field in stop times
        5. Use that interpolated time as the crossing time of the trip
           vehicle, and compute the trip orientation to the screen line
           via a cross product of a vector in the direction of the
           screen line and a tiny vector in the direction of trip travel

    - Assume the following feed attributes are not ``None``:
         * ``feed.shapes``, if ``geo_shapes`` is not given

    """
    dates = feed.restrict_dates(dates)
    cols = ['date', 'trip_id', 'route_id', 'route_short_name',
      'crossing_time', 'orientation']
    if not dates:
        return pd.DataFrame([], columns=cols)

    # Get all shapes that intersect the screen line
    shapes = feed.get_shapes_intersecting_geometry(linestring, geo_shapes,
      geometrized=True)

    # Convert shapes to UTM
    lat, lon = feed.shapes.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
    crs = hp.get_utm_crs(lat, lon)
    shapes = shapes.to_crs(crs)

    # Convert linestring to UTM
    linestring = hp.linestring_to_utm(linestring)

    # Get all intersection points of shapes and linestring
    shapes['intersection'] = shapes.intersection(linestring)

    # Make a vector in the direction of the screen line
    # to later calculate trip orientation.
    # Does not work in case of a bent screen line.
    p1 = sg.Point(linestring.coords[0])
    p2 = sg.Point(linestring.coords[-1])
    w = np.array([p2.x - p1.x, p2.y - p1.y])

    # Build a dictionary from the shapes DataFrame of the form
    # shape ID -> list of pairs (d, v), one for each intersection point,
    # where d is the distance of the intersection point along shape,
    # and v is a tiny vectors from the point in direction of shape.
    # Assume here that trips travel in the same direction as their shapes.
    dv_by_shape = {}
    eps = 1
    convert_dist = hp.get_convert_dist('m', feed.dist_units)
    for __, sid, geom, intersection in shapes.itertuples():
        # Get distances along shape of intersection points (in meters)
        distances = [geom.project(p) for p in intersection]
        # Build tiny vectors
        vectors = []
        for i, p in enumerate(intersection):
            q = geom.interpolate(distances[i] + eps)
            vector = np.array([q.x - p.x, q.y - p.y])
            vectors.append(vector)
        # Convert distances to units used in feed
        distances = [convert_dist(d) for d in distances]
        dv_by_shape[sid] = list(zip(distances, vectors))

    # Get trips with those shapes
    t = feed.trips
    t = t[t['shape_id'].isin(dv_by_shape.keys())].copy()

    # Merge in route short names and stop times
    t = t.merge(feed.routes[['route_id', 'route_short_name']])\
      .merge(feed.stop_times)

    # Drop NaN departure times and convert to seconds past midnight
    t = t[t['departure_time'].notnull()].copy()
    t['departure_time'] = t['departure_time'].map(hp.timestr_to_seconds)

    # Compile crossings by date
    a = feed.compute_trip_activity(dates)
    rows = []
    for date in dates:
        # Slice to trips active on date
        ids = a.loc[a[date] == 1, 'trip_id']
        f = t[t['trip_id'].isin(ids)].copy()

        # For each shape find the trips that cross the screen line
        # and get crossing times and orientation
        f = f.sort_values(['trip_id', 'stop_sequence'])
        for tid, group in f.groupby('trip_id'):
            sid = group['shape_id'].iat[0]
            rid = group['route_id'].iat[0]
            rsn = group['route_short_name'].iat[0]
            stop_times = group['departure_time'].values
            stop_distances = group['shape_dist_traveled'].values
            for d, v in dv_by_shape[sid]:
                # Interpolate crossing time
                time = np.interp(d, stop_distances, stop_times)
                # Compute direction of trip travel relative to
                # screen line by looking at the sign of the cross
                # product of tiny shape vector and screen line vector
                det = np.linalg.det(np.array([v, w]))
                if det >= 0:
                    orientation = 1
                else:
                    orientation = -1
                # Update rows
                rows.append([date, tid, rid, rsn, time, orientation])

    # Create DataFrame
    g = pd.DataFrame(rows, columns=cols).sort_values(['date', 'crossing_time'])

    # Convert departure times back to time strings
    g['crossing_time'] = g['crossing_time'].map(
      lambda x: hp.timestr_to_seconds(x, inverse=True))

    return g
