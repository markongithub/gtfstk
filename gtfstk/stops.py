"""
Functions about stops.
"""
import json
from collections import Counter, OrderedDict

import pandas as pd
import numpy as np
import utm
import shapely.geometry as sg

from . import constants as cs
from . import helpers as hp


def compute_stop_stats_base(stop_times, trip_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a stop times DataFrame and a subset of a trips DataFrame, return a DataFrame that provides summary stats about the stops in the (inner) join of the two DataFrames.

    The columns of the output DataFrame are:

    - stop_id
    - direction_id: present if and only if ``split_directions``
    - num_routes: number of routes visiting stop (in the given direction)
    - num_trips: number of trips visiting stop (in the givin direction)
    - max_headway: maximum of the durations (in minutes) between trip departures at the stop between ``headway_start_time`` and      ``headway_end_time`` on the given date
    - min_headway: minimum of the durations (in minutes) mentioned above
    - mean_headway: mean of the durations (in minutes) mentioned above
    - start_time: earliest departure time of a trip from this stop on the given date
    - end_time: latest departure time of a trip from this stop on the given date

    If ``split_directions == False``, then compute each stop's stats using trips visiting it from both directions.

    If ``trip_subset`` is empty, then return an empty DataFrame with the columns specified above.
    """
    cols = [
      'stop_id',
      'num_routes',
      'num_trips',
      'max_headway',
      'min_headway',
      'mean_headway',
      'start_time',
      'end_time',
      ]

    if split_directions:
        cols.append('direction_id')

    if trip_subset.empty:
        return pd.DataFrame(columns=cols)

    f = pd.merge(stop_times, trip_subset)

    # Convert departure times to seconds to ease headway calculations
    f['departure_time'] = f['departure_time'].map(hp.timestr_to_seconds)

    headway_start = hp.timestr_to_seconds(headway_start_time)
    headway_end = hp.timestr_to_seconds(headway_end_time)

    # Compute stats for each stop
    def compute_stop_stats(group):
        # Operate on the group of all stop times for an individual stop
        d = OrderedDict()
        d['num_routes'] = group['route_id'].unique().size
        d['num_trips'] = group.shape[0]
        d['start_time'] = group['departure_time'].min()
        d['end_time'] = group['departure_time'].max()
        headways = []
        dtimes = sorted([dtime for dtime in group['departure_time'].values
          if headway_start <= dtime <= headway_end])
        headways.extend([dtimes[i + 1] - dtimes[i]
          for i in range(len(dtimes) - 1)])
        if headways:
            d['max_headway'] = np.max(headways)/60  # minutes
            d['min_headway'] = np.min(headways)/60  # minutes
            d['mean_headway'] = np.mean(headways)/60  # minutes
        else:
            d['max_headway'] = np.nan
            d['min_headway'] = np.nan
            d['mean_headway'] = np.nan
        return pd.Series(d)

    if split_directions:
        g = f.groupby(['stop_id', 'direction_id'])
    else:
        g = f.groupby('stop_id')

    result = g.apply(compute_stop_stats).reset_index()

    # Convert start and end times to time strings
    result[['start_time', 'end_time']] =\
      result[['start_time', 'end_time']].applymap(
      lambda x: hp.timestr_to_seconds(x, inverse=True))

    return result

def compute_stop_time_series_base(stop_times, trips_subset,
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Given a stop times DataFrame and a subset of a trips DataFrame, return a DataFrame that provides summary stats about the stops in the (inner) join of the two DataFrames.

    The time series is a DataFrame with a timestamp index for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    The timestamp includes the date given by ``date_label``, a date string of the form '%Y%m%d'.
   
    The columns of the DataFrame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['num_trips']
    - middle level: name = 'stop_id', values = the active stop IDs
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
   
    If ``trips_subset`` is empty, then return an empty DataFrame with the indicator columns.

    NOTES:

    - 'num_trips' should be resampled with ``how=np.sum``
    - To remove the date and seconds from the time series f, do ``f.index = [t.time().strftime('%H:%M') for t in f.index.to_datetime()]``
    """ 
    cols = ['num_trips']
    if trips_subset.empty:
        return pd.DataFrame(columns=cols)

    f = pd.merge(stop_times, trips_subset)

    if split_directions:
        # Alter stop IDs to encode trip direction:
        # <stop ID>-0 and <stop ID>-1
        f['stop_id'] = f['stop_id'] + '-' +\
          f['direction_id'].map(str)           
    stops = f['stop_id'].unique()  

    # Create one time series for each stop. Use a list first.   
    bins = [i for i in range(24*60)] # One bin for each minute
    num_bins = len(bins)

    # Bin each stop departure time
    def F(x):
        return (hp.timestr_to_seconds(x)//60) % (24*60)

    f['departure_index'] = f['departure_time'].map(F)

    # Create one time series for each stop
    series_by_stop = {stop: [0 for i in range(num_bins)]
      for stop in stops}

    for stop, group in f.groupby('stop_id'):
        counts = Counter((bin, 0) for bin in bins) +\
          Counter(group['departure_index'].values)
        series_by_stop[stop] = [counts[bin] for bin in bins]

    # Combine lists into one time series.
    # Actually, a dictionary indicator -> time series.
    # Only one indicator in this case, but could add more
    # in the future as was done with routes time series.
    rng = pd.date_range(date_label, periods=24*60, freq='Min')
    series_by_indicator = {'num_trips':
      pd.DataFrame(series_by_stop, index=rng).fillna(0)}

    # Combine all time series into one time series
    g = hp.combine_time_series(series_by_indicator, kind='stop',
      split_directions=split_directions)
    return hp.downsample(g, freq=freq)

def get_stops(feed, date=None, trip_id=None, route_id=None, in_stations=False):
    """
    Return ``feed.stops``.
    If a date is given, then restrict the output to stops that are visited by trips active on the given date.
    If a trip ID (string) is given, then restrict the output possibly further to stops that are visited by the trip.
    Else if a route ID (string) is given, then restrict the output possibly further to stops that are visited by at least one trip on the route.
    If ``in_stations``, then restrict the output further to only include stops are within stations, if station data is available in ``feed.stops``.
    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
    - Those used in :func:`get_stop_times`
    - ``feed.routes``   
    """
    s = feed.stops.copy()
    if date is not None:
        A = feed.get_stop_times(date)['stop_id']
        s = s[s['stop_id'].isin(A)].copy()
    if trip_id is not None:
        st = feed.stop_times.copy()
        B = st[st['trip_id'] == trip_id]['stop_id']
        s = s[s['stop_id'].isin(B)].copy()
    elif route_id is not None:
        A = feed.trips[feed.trips['route_id'] == route_id]['trip_id']
        st = feed.stop_times.copy()
        B = st[st['trip_id'].isin(A)]['stop_id']
        s = s[s['stop_id'].isin(B)].copy()
    if in_stations and\
      set(['location_type', 'parent_station']) <= set(s.columns):
        s = s[(s['location_type'] != 1) & (s['parent_station'].notnull())]

    return s

def build_geometry_by_stop(feed, use_utm=False, stop_ids=None):
    """
    Return a dictionary with structure stop_id -> Shapely point object.
    If ``use_utm``, then return each point in in UTM coordinates.
    Otherwise, return each point in WGS84 longitude-latitude coordinates.
    If a list of stop IDs ``stop_ids`` is given, then only include the given stop IDs.

    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
       
    """
    d = {}
    stops = feed.stops.copy()
    if stop_ids is not None:
        stops = stops[stops['stop_id'].isin(stop_ids)]

    if use_utm:
        for stop, group in stops.groupby('stop_id'):
            lat, lon = group[['stop_lat', 'stop_lon']].values[0]
            d[stop] = sg.Point(utm.from_latlon(lat, lon)[:2])
    else:
        for stop, group in stops.groupby('stop_id'):
            lat, lon = group[['stop_lat', 'stop_lon']].values[0]
            d[stop] = sg.Point([lon, lat])
    return d

def compute_stop_activity(feed, dates):
    """
    Return a  DataFrame with the columns

    - stop_id
    - ``dates[0]``: 1 if the stop has at least one trip visiting it on ``dates[0]``; 0 otherwise
    - ``dates[1]``: 1 if the stop has at least one trip visiting it on ``dates[1]``; 0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the stop has at least one trip visiting it on ``dates[-1]``; 0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an empty DataFrame with the column 'stop_id'.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`compute_trip_activity`
       

    """
    if not dates:
        return pd.DataFrame(columns=['stop_id'])

    trip_activity = feed.compute_trip_activity(dates)
    g = pd.merge(trip_activity, feed.stop_times).groupby('stop_id')
    # Pandas won't allow me to simply return g[dates].max().reset_index().
    # I get ``TypeError: unorderable types: datetime.date() < str()``.
    # So here's a workaround.
    for (i, date) in enumerate(dates):
        if i == 0:
            f = g[date].max().reset_index()
        else:
            f = f.merge(g[date].max().reset_index())
    return f

def compute_stop_stats(feed, date, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Call ``compute_stop_stats_base()`` with the subset of trips active on the given date and with the keyword arguments ``split_directions``,   ``headway_start_time``, and ``headway_end_time``.

    See ``compute_stop_stats_base()`` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_timtes``
    - Those used in :func:`get_trips`
       
    NOTES:

    This is a more user-friendly version of ``compute_stop_stats_base()``.
    The latter function works without a feed, though.
    """
    # Get stop times active on date and direction IDs
    return compute_stop_stats_base(feed.stop_times, feed.get_trips(date),
      split_directions=split_directions,
      headway_start_time=headway_start_time,
      headway_end_time=headway_end_time)

def compute_stop_time_series(feed, date, split_directions=False, freq='5Min'):
    """
    Call :func:`.helpers.compute_stops_times_series_base` with the subset of trips  active on the given date and with the keyword arguments ``split_directions``and ``freq`` and with ``date_label`` equal to ``date``.
    See :func:`.helpers.compute_stop_time_series_base` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`
       
    NOTES:
      - This is a more user-friendly version of :func:`.helpers.compute_stop_time_series_base`. The latter function works without a feed, though.
    """ 
    return compute_stop_time_series_base(feed.stop_times,
      feed.get_trips(date), split_directions=split_directions,
      freq=freq, date_label=date)

def build_stop_timetable(feed, stop_id, date):
    """
    Return a  DataFrame encoding the timetable for the given stop ID on the given date.
    The columns are all those in ``feed.trips`` plus those in ``feed.stop_times``.
    The result is sorted by departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`get_stop_times`
       
    """
    f = feed.get_stop_times(date)
    f = pd.merge(f, feed.trips)
    f = f[f['stop_id'] == stop_id]
    return f.sort_values('departure_time')

def get_stops_in_polygon(feed, polygon, geo_stops=None):
    """
    Return the slice of ``feed.stops`` that contains all stops that lie within the given Shapely Polygon object.
    Assume the polygon specified in WGS84 longitude-latitude coordinates.
   
    To do this, first geometrize ``feed.stops`` via :func:`geometrize_stops`.
    Alternatively, use the ``geo_stops`` GeoDataFrame, if given.
    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.stops``, if ``geo_stops`` is not given
       
    """
    if geo_stops is not None:
        f = geo_stops.copy()
    else:
        f = geometrize_stops(feed.stops)
   
    cols = f.columns
    f['hit'] = f['geometry'].within(polygon)
    f = f[f['hit']][cols]
    return ungeometrize_stops(f)

def geometrize_stops(stops, use_utm=False):
    """
    Given a stops DataFrame, convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 points instead of 'stop_lon' and 'stop_lat' columns.
    If ``use_utm``, then use UTM coordinates for the geometries.
    Requires GeoPandas.
    """
    import geopandas as gpd


    f = stops.copy()
    s = gpd.GeoSeries([sg.Point(p) for p in
      stops[['stop_lon', 'stop_lat']].values])
    f['geometry'] = s
    g = f.drop(['stop_lon', 'stop_lat'], axis=1)
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['stop_lat', 'stop_lon']].values
        crs = get_utm_crs(lat, lon)
        g = g.to_crs(crs)

    return g

def ungeometrize_stops(geo_stops):
    """
    The inverse of :func:`geometrize_stops`.   
    If ``geo_stops`` is in UTM (has a UTM CRS property), then convert UTM coordinates back to WGS84 coordinates,
    """
    f = geo_stops.copy().to_crs(cs.CRS_WGS84)
    f['stop_lon'] = f['geometry'].map(
      lambda p: p.x)
    f['stop_lat'] = f['geometry'].map(
      lambda p: p.y)
    del f['geometry']
    return f
