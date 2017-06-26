"""
Functions about routes.
"""
from collections import OrderedDict
import json

import pandas as pd
import numpy as np
import shapely.geometry as sg

from . import helpers as hp


def compute_route_stats_base(trip_stats_subset, split_directions=False,
  headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Compute some route stats for the given subset of trips stats.

    Parameters
    ----------
    trip_stats_subset : DataFrame
        Subset of the output of :func:`.trips.compute_trip_stats`
    split_directions : boolean
        If ``True``, then separate the stats by trip direction (0 or 1);
        otherwise aggregate trips visiting from both directions
    headway_start_time : string
        HH:MM:SS time string indicating the start time for computing
        headway stats
    headway_end_time : string
        HH:MM:SS time string indicating the end time for computing
        headway stats

    Returns
    -------
    DataFrame
        Has the columns:

        -``'route_id'``
        -``'route_short_name'``
        -``'route_type'``
        -``'direction_id'``
        -``'num_trips'``: number of trips
        -``'is_loop'``: 1 if at least one of the trips on the route has
          its ``is_loop`` field equal to 1; 0 otherwise
        -``'is_bidirectional'``: 1 if the route has trips in both
          directions; 0 otherwise
        -``'start_time'``: start time of the earliest trip on the route
        -``'end_time'``: end time of latest trip on the route
        -``'max_headway'``: maximum of the durations (in minutes)
          between trip starts on the route between
          ``headway_start_time`` and ``headway_end_time`` on the given
          dates
        -``'min_headway'``: minimum of the durations (in minutes)
          mentioned above
        -``'mean_headway'``: mean of the durations (in minutes)
          mentioned above
        -``'peak_num_trips'``: maximum number of simultaneous trips in
          service (for the given direction, or for both directions when
          ``split_directions==False``)
        -``'peak_start_time'``: start time of first longest period
          during which the peak number of trips occurs
        -``'peak_end_time'``: end time of first longest period during
          which the peak number of trips occurs
        -``'service_duration'``: total of the duration of each trip on
          the route in the given subset of trips; measured in hours
        -``'service_distance'``: total of the distance traveled by each
          trip on the route in the given subset of trips; measured in
          whatever distance units are present in ``trip_stats_subset``;
          contains all ``np.nan`` entries if ``feed.shapes is None``
        -``'service_speed'``: service_distance/service_duration;
          measured in distance units per hour
        -``'mean_trip_distance'``: service_distance/num_trips
        -``'mean_trip_duration'``: service_duration/num_trips

    If not ``split_directions == False``, then remove the direction_id
    column and compute each route's stats, except for headways, using
    its trips running in both directions.
    In this case, (1) compute max headway by taking the max of the max
    headways in both directions; (2) compute mean headway by taking the
    weighted mean of the mean headways in both directions.

    If ``trip_stats_subset`` is empty, return an empty DataFrame with
    the columns specified above.

    """
    cols = [
      'route_id',
      'route_short_name',
      'route_type',
      'num_trips',
      'is_loop',
      'is_bidirectional',
      'start_time',
      'end_time',
      'max_headway',
      'min_headway',
      'mean_headway',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_duration',
      'service_distance',
      'service_speed',
      'mean_trip_distance',
      'mean_trip_duration',
    ]

    if split_directions:
        cols.append('direction_id')

    if trip_stats_subset.empty:
        return pd.DataFrame(columns=cols)

    # Convert trip start and end times to seconds to ease calculations below
    f = trip_stats_subset.copy()
    f[['start_time', 'end_time']] = f[['start_time', 'end_time']
      ].applymap(hp.timestr_to_seconds)

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    def compute_route_stats_split_directions(group):
        # Take this group of all trips stats for a single route
        # and compute route-level stats.
        d = OrderedDict()
        d['route_short_name'] = group['route_short_name'].iat[0]
        d['route_type'] = group['route_type'].iat[0]
        d['num_trips'] = group.shape[0]
        d['is_loop'] = int(group['is_loop'].any())
        d['start_time'] = group['start_time'].min()
        d['end_time'] = group['end_time'].max()

        # Compute max and mean headway
        stimes = group['start_time'].values
        stimes = sorted([stime for stime in stimes
          if headway_start <= stime <= headway_end])
        headways = np.diff(stimes)
        if headways.size:
            d['max_headway'] = np.max(headways)/60  # minutes
            d['min_headway'] = np.min(headways)/60  # minutes
            d['mean_headway'] = np.mean(headways)/60  # minutes
        else:
            d['max_headway'] = np.nan
            d['min_headway'] = np.nan
            d['mean_headway'] = np.nan

        # Compute peak num trips
        times = np.unique(group[['start_time', 'end_time']].values)
        counts = [hp.count_active_trips(group, t) for t in times]
        start, end = hp.get_peak_indices(times, counts)
        d['peak_num_trips'] = counts[start]
        d['peak_start_time'] = times[start]
        d['peak_end_time'] = times[end]

        d['service_distance'] = group['distance'].sum()
        d['service_duration'] = group['duration'].sum()
        return pd.Series(d)

    def compute_route_stats(group):
        d = OrderedDict()
        d['route_short_name'] = group['route_short_name'].iat[0]
        d['route_type'] = group['route_type'].iat[0]
        d['num_trips'] = group.shape[0]
        d['is_loop'] = int(group['is_loop'].any())
        d['is_bidirectional'] = int(group['direction_id'].unique().size > 1)
        d['start_time'] = group['start_time'].min()
        d['end_time'] = group['end_time'].max()

        # Compute headway stats
        headways = np.array([])
        for direction in [0, 1]:
            stimes = group[group['direction_id'] == direction][
              'start_time'].values
            stimes = sorted([stime for stime in stimes
              if headway_start <= stime <= headway_end])
            headways = np.concatenate([headways, np.diff(stimes)])
        if headways.size:
            d['max_headway'] = np.max(headways)/60  # minutes
            d['min_headway'] = np.min(headways)/60  # minutes
            d['mean_headway'] = np.mean(headways)/60  # minutes
        else:
            d['max_headway'] = np.nan
            d['min_headway'] = np.nan
            d['mean_headway'] = np.nan

        # Compute peak num trips
        times = np.unique(group[['start_time', 'end_time']].values)
        counts = [hp.count_active_trips(group, t) for t in times]
        start, end = hp.get_peak_indices(times, counts)
        d['peak_num_trips'] = counts[start]
        d['peak_start_time'] = times[start]
        d['peak_end_time'] = times[end]

        d['service_distance'] = group['distance'].sum()
        d['service_duration'] = group['duration'].sum()

        return pd.Series(d)

    if split_directions:
        g = f.groupby(['route_id', 'direction_id']).apply(
          compute_route_stats_split_directions).reset_index()

        # Add the is_bidirectional column
        def is_bidirectional(group):
            d = {}
            d['is_bidirectional'] = int(
              group['direction_id'].unique().size > 1)
            return pd.Series(d)

        gg = g.groupby('route_id').apply(is_bidirectional).reset_index()
        g = g.merge(gg)
    else:
        g = f.groupby('route_id').apply(
          compute_route_stats).reset_index()

    # Compute a few more stats
    g['service_speed'] = g['service_distance']/g['service_duration']
    g['mean_trip_distance'] = g['service_distance']/g['num_trips']
    g['mean_trip_duration'] = g['service_duration']/g['num_trips']

    # Convert route times to time strings
    g[['start_time', 'end_time', 'peak_start_time', 'peak_end_time']] =\
      g[['start_time', 'end_time', 'peak_start_time', 'peak_end_time']].\
      applymap(lambda x: hp.timestr_to_seconds(x, inverse=True))

    return g

def compute_route_time_series_base(trip_stats_subset,
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Compute some route stats as a time series for the given trip stats.

    Parameters
    ----------
    trip_stats_subset : DataFrame
        A subset of the output of :func:`.trips.compute_trip_stats`
    split_directions : boolean
        If ``True``, then separate each routes's stats by trip direction;
        otherwise aggregate trips in both directions
    freq : Pandas frequency string
        Specifices the frequency with which to resample the time series;
        max frequency is one minute ('Min')
    date_label : string
        YYYYMMDD date string used as the date in the time series index

    Returns
    -------
    DataFrame
        A time series version of the following route stats.

        - number of trips in service by route ID
        - number of trip starts by route ID
        - service duration in hours by route ID
        - service distance in kilometers by route ID
        - service speed in kilometers per hour

        The columns are hierarchical (multi-indexed) with

        - top level: name = ``'indicator'``, values =
          ``['service_distance', 'service_duration', 'num_trip_starts',
          'num_trips', 'service_speed']``
        - middle level: name = 'route_id', values = the active routes
        - bottom level: name = 'direction_id', values = 0s and 1s

        If not ``split_directions``, then don't include the bottom level.

        The time series has a timestamp index for a 24-hour period
        sampled at the given frequency.
        The maximum allowable frequency is 1 minute.
        If ``trip_stats_subset`` is empty, then return an empty
        DataFrame with the columns ``['service_distance',
          'service_duration', 'num_trip_starts',
          'num_trips', 'service_speed']``

    Notes
    -----
    - The time series is computed at a one-minute frequency, then
      resampled at the end to the given frequency
    - To resample the resulting time series manually, use the following
      methods.

        * for ``'num_trips'`` series, use ``how=np.mean``
        * for the other series, use ``how=np.sum``
        * ``'service_speed'`` can't be resampled and must be recalculated
          from ``'service_distance'`` and ``'service_duration'``

    - To remove the date and seconds from the time series f, do
      ``f.index = [t.time().strftime('%H:%M') for t in f.index.to_datetime()]``

    """
    cols = [
      'service_distance',
      'service_duration',
      'num_trip_starts',
      'num_trips',
      'service_speed',
    ]
    if trip_stats_subset.empty:
        return pd.DataFrame(columns=cols)

    tss = trip_stats_subset.copy()
    if split_directions:
        # Alter route IDs to encode direction:
        # <route ID>-0 and <route ID>-1
        tss['route_id'] = tss['route_id'] + '-' +\
          tss['direction_id'].map(str)

    routes = tss['route_id'].unique()
    # Build a dictionary of time series and then merge them all
    # at the end.
    # Assign a uniform generic date for the index
    date_str = date_label
    day_start = pd.to_datetime(date_str + ' 00:00:00')
    day_end = pd.to_datetime(date_str + ' 23:59:00')
    rng = pd.period_range(day_start, day_end, freq='Min')
    indicators = [
      'num_trip_starts',
      'num_trips',
      'service_duration',
      'service_distance',
    ]

    bins = [i for i in range(24*60)]  # One bin for each minute
    num_bins = len(bins)

    # Bin start and end times
    def F(x):
        return (hp.timestr_to_seconds(x)//60) % (24*60)

    tss[['start_index', 'end_index']] = tss[['start_time', 'end_time']
      ].applymap(F)
    routes = sorted(set(tss['route_id'].values))

    # Bin each trip according to its start and end time and weight
    series_by_route_by_indicator = {indicator:
      {route: [0 for i in range(num_bins)] for route in routes}
      for indicator in indicators}
    for index, row in tss.iterrows():
        route = row['route_id']
        start = row['start_index']
        end = row['end_index']
        distance = row['distance']

        if start is None or np.isnan(start) or start == end:
            continue

        # Get bins to fill
        if start <= end:
            bins_to_fill = bins[start:end]
        else:
            bins_to_fill = bins[start:] + bins[:end]

        # Bin trip
        # Do num trip starts
        series_by_route_by_indicator['num_trip_starts'][route][start] += 1
        # Do rest of indicators
        for indicator in indicators[1:]:
            if indicator == 'num_trips':
                weight = 1
            elif indicator == 'service_duration':
                weight = 1/60
            else:
                weight = distance/len(bins_to_fill)
            for bin in bins_to_fill:
                series_by_route_by_indicator[indicator][route][bin] += weight

    # Create one time series per indicator
    rng = pd.date_range(date_str, periods=24*60, freq='Min')
    series_by_indicator = {indicator:
      pd.DataFrame(series_by_route_by_indicator[indicator],
        index=rng).fillna(0)
      for indicator in indicators}

    # Combine all time series into one time series
    g = hp.combine_time_series(series_by_indicator, kind='route',
      split_directions=split_directions)
    return hp.downsample(g, freq=freq)

def get_routes(feed, date=None, time=None):
    """
    Return the section of ``feed.routes`` that contains only routes active on the given date.
    If no date is given, then return all routes.
    If a date and time are given, then return only those routes with trips active at that date and time.
    Do not take times modulo 24.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - Those used in :func:`get_trips`.

    """
    if date is None:
        return feed.routes.copy()

    trips = feed.get_trips(date, time)
    R = trips['route_id'].unique()
    return feed.routes[feed.routes['route_id'].isin(R)]

def compute_route_stats(feed, trip_stats, dates, split_directions=False,
  headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Compute stats for all stops for the given dates, and return
    the result as DataFrame with the following columns.

    - date
    - route_id
    - route_short_name
    - route_type
    - direction_id
    - num_trips: number of trips
    - is_loop: 1 if at least one of the trips on the route has its
      ``is_loop`` field equal to 1; 0 otherwise
    - is_bidirectional: 1 if the route has trips in both directions;
      0 otherwise
    - start_time: start time of the earliest trip on the route
    - end_time: end time of latest trip on the route
    - max_headway: maximum of the durations (in minutes) between trip
      starts on the route between ``headway_start_time`` and
      ``headway_end_time`` on the given dates
    - min_headway: minimum of the durations (in minutes) mentioned above
    - mean_headway: mean of the durations (in minutes) mentioned above
    - peak_num_trips: maximum number of simultaneous trips in service
      (for the given direction, or for both directions when
      ``split_directions==False``)
    - peak_start_time: start time of first longest period during which
      the peak number of trips occurs
    - peak_end_time: end time of first longest period during which the
      peak number of trips occurs
    - service_duration: total of the duration of each trip on the route
      in the given subset of trips; measured in hours
    - service_distance: total of the distance traveled by each trip on
      the route in the given subset of trips; measured in wunits,
      that is, whatever distance units are present in trip_stats_subset;
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - service_speed: service_distance/service_duration;
      measured in distance units per hour
    - mean_trip_distance: service_distance/num_trips
    - mean_trip_duration: service_duration/num_trips

    If ``split_directions == False``, then compute each stop's stats
    using trips visiting it from both directions.

    If there are no stats for the given dates, then return an empty
    DataFrame with the columns above.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`.helpers.compute_route_stats_base`
    """
    if not isinstance(dates, list):
        raise ValueError('dates must be a list')

    cols = [
      'date',
      'route_id',
      'route_short_name',
      'route_type',
      'num_trips',
      'is_bidirectional',
      'is_loop',
      'start_time',
      'end_time',
      'max_headway',
      'min_headway',
      'mean_headway',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_duration',
      'service_distance',
      'service_speed',
      'mean_trip_distance',
      'mean_trip_duration',
      ]
    if split_directions:
        cols.append('direction_id')

    # Restrict to feed dates
    dates = set(dates) & set(feed.get_dates())
    if not dates:
        return pd.DataFrame([], columns=cols)

    ts = trip_stats.copy()
    activity = feed.compute_trip_activity(dates)

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence ->
    # [stats DataFarme, date list that stats apply]
    stats_and_dates_by_ids = {}
    for date in dates:
        ids = tuple(activity.loc[activity[date] > 0, 'trip_id'])
        if ids in stats_and_dates_by_ids:
            # Append date to date list
            stats_and_dates_by_ids[ids][1].append(date)
        else:
            # Compute stats
            t = ts[ts['trip_id'].isin(ids)].copy()
            stats = compute_route_stats_base(t,
              split_directions=split_directions,
              headway_start_time=headway_start_time,
              headway_end_time=headway_end_time)

            # Remember stats
            stats_and_dates_by_ids[ids] = [stats, [date]]

    # Assemble stats into DataFrame
    frames = []
    for stats, dates in stats_and_dates_by_ids.values():
        for date in dates:
            f = stats.copy()
            f['date'] = date
            frames.append(f)
    f = pd.concat(frames).sort_values(['date', 'route_id'])

    return f

def compute_route_time_series(feed, trip_stats, dates, split_directions=False,
  freq='5Min'):
    """
    Compute time series for the given routes on the given dates at the
    given frequency and return the result as a DataFrame of the same
    form as output by :func:`compute_stop_time_series_base`.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_trips`

    If all dates lie outside the feed's date range, then return an
    empty DataFrame with only the columns ``'service_distance'``
    ``'service_duration'``, ``'num_trip_starts'``, ``'num_trips'``,
    and ``'service_speed'``.

    Dates with no active stops will not appear in the result
    (which contrasts with the output of :func:`compute_stop_stats`).
    """
    if not isinstance(dates, list):
        raise ValueError('dates must be a list')

    cols = [
      'service_distance',
      'service_duration',
      'num_trip_starts',
      'num_trips',
      'service_speed',
    ]

    # Restrict to feed dates
    dates = set(dates) & set(feed.get_dates())
    if not dates:
        return pd.DataFrame([], columns=cols)

    if split_directions:
        cols.append('direction_id')

    activity = feed.compute_trip_activity(dates)
    ts = trip_stats.copy()

    # Collect stats for each date, memoizing stats by trip ID sequence
    # to avoid unnecessary recomputations.
    # Store in dictionary of the form
    # trip ID sequence ->
    # [stats DataFarme, date list that stats apply]
    stats_and_dates_by_ids = {}
    for date in dates:
        ids = tuple(activity.loc[activity[date] > 0, 'trip_id'])
        if ids in stats_and_dates_by_ids:
            # Append date to date list
            stats_and_dates_by_ids[ids][1].append(date)
        else:
            # Compute stats
            t = ts[ts['trip_id'].isin(ids)].copy()
            stats = compute_route_time_series_base(t,
              split_directions=split_directions, freq=freq, date_label=date)

            # Remember stats
            stats_and_dates_by_ids[ids] = [stats, [date]]

    # Assemble stats into DataFrame
    frames = []
    for stats, dates in stats_and_dates_by_ids.values():
        if stats.empty:
            # Skip empty stats
            continue
        for date in dates:
            f = stats.copy()
            # Replace date
            d = hp.datestr_to_date(date)
            f.index = f.index.map(lambda t: t.replace(
              year=d.year, month=d.month, day=d.day))
            frames.append(f)
    f = pd.concat(frames).sort_index()

    return f

def build_route_timetable(feed, route_id, date):
    """
    Return a DataFrame encoding the timetable for the given route ID on
    the given date.
    The columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times``.
    The result is sorted by grouping by trip ID and sorting the groups
    by their first departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`

    """
    f = feed.get_trips(date)
    f = f[f['route_id'] == route_id].copy()
    f = pd.merge(f, feed.stop_times)
    # Groupby trip ID and sort groups by their minimum departure time.
    # For some reason NaN departure times mess up the transform below.
    # So temporarily fill NaN departure times as a workaround.
    f['dt'] = f['departure_time'].fillna(method='ffill')
    f['min_dt'] = f.groupby('trip_id')['dt'].transform(min)
    return f.sort_values(['min_dt', 'stop_sequence']).drop(
      ['min_dt', 'dt'], axis=1)

def route_to_geojson(feed, route_id, include_stops=False):
    """
    Given a feed and a route ID (string), return a (decoded) GeoJSON
    feature collection comprising a MultiLinestring feature of distinct shapes
    of the trips on the route.
    If ``include_stops``, then also include one Point feature for each stop
    visited by any trip on the route.
    The MultiLinestring feature will contain as properties all the columns
    in ``feed.routes`` pertaining to the given route, and each Point feature
    will contain as properties all the columns in ``feed.stops`` pertaining
    to the stop, except the ``stop_lat`` and ``stop_lon`` properties.

    Return the empty dictionary if all the route's trips lack shapes.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - ``feed.shapes``
    - ``feed.trips``
    - ``feed.stops``

    """
    # Get the relevant shapes
    t = feed.trips.copy()
    A = t[t['route_id'] == route_id]['shape_id'].unique()
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False,
      shape_ids=A)

    if not geometry_by_shape:
        return {}

    r = feed.routes.copy()
    features = [{
        'type': 'Feature',
        'properties': json.loads(r[r['route_id'] == route_id].to_json(
          orient='records'))[0],
        'geometry': sg.mapping(sg.MultiLineString(
          [linestring for linestring in geometry_by_shape.values()]))
    }]

    if include_stops:
        # Get relevant stops and geometrys
        s = feed.get_stops(route_id=route_id)
        cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
        s = s[list(cols)].copy()
        stop_ids = s['stop_id'].tolist()
        geometry_by_stop = feed.build_geometry_by_stop(stop_ids=stop_ids)
        features.extend([{
            'type': 'Feature',
            'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
              orient='records'))[0],
            'geometry': sg.mapping(geometry_by_stop[stop_id]),
        } for stop_id in stop_ids])

    return {'type': 'FeatureCollection', 'features': features}
