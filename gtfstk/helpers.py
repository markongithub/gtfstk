import datetime as dt
from collections import OrderedDict, Counter
from functools import wraps

import pandas as pd
import numpy as np
import shapely.geometry as sg
from shapely.ops import transform
import utm

from . import constants as cs


def time_it(f):
    def wrap(*args, **kwargs):
        t1 = dt.datetime.now()
        print('Timing', f.__name__)
        print(t1, 'Began process')
        result = f(*args, **kwargs)
        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)    
        return result
    return wrap

def datestr_to_date(x, format_str='%Y%m%d', inverse=False):
    """
    Given a string object ``x`` representing a date in the given format,     convert it to a datetime.date object and return the result.
    If ``inverse``, then assume that ``x`` is a date object and return its corresponding string in the given format.
    """
    if x is None:
        return None
    if not inverse:
        result = dt.datetime.strptime(x, format_str).date()
    else:
        result = x.strftime(format_str)
    return result

def timestr_to_seconds(x, inverse=False, mod24=False):
    """
    Given a time string of the form '%H:%M:%S', return the number of seconds  past midnight that it represents.
    In keeping with GTFS standards, the hours entry may be greater than 23.
    If ``mod24``, then return the number of seconds modulo ``24*3600``.
    If ``inverse``, then do the inverse operation.
    In this case, if ``mod24`` also, then first take the number of  seconds modulo ``24*3600``.
    """
    if not inverse:
        try:
            hours, mins, seconds = x.split(':')
            result = int(hours)*3600 + int(mins)*60 + int(seconds)
            if mod24:
                result %= 24*3600
        except:
            result = np.nan
    else:
        try:
            seconds = int(x)
            if mod24:
                seconds %= 24*3600
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            result = '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
        except:
            result = np.nan
    return result

def timestr_mod24(timestr):
    """
    Given a GTFS time string in the format %H:%M:%S, return a timestring in the same format but with the hours taken modulo 24.
    """
    try:
        hours, mins, seconds = [int(x) for x in timestr.split(':')]
        hours %= 24
        result = '{:02d}:{:02d}:{:02d}'.format(hours, mins, seconds)
    except:
        result = None
    return result

def weekday_to_str(weekday, inverse=False):
    """
    Given a weekday, that is, an integer in ``range(7)``, return it's corresponding weekday name as a lowercase string.
    Here 0 -> 'monday', 1 -> 'tuesday', and so on.
    If ``inverse``, then perform the inverse operation.
    """
    s = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
      'saturday', 'sunday']
    if not inverse:
        try:
            return s[weekday]
        except:
            return
    else:
        try:
            return s.index(weekday)
        except:
            return

def get_segment_length(linestring, p, q=None):
    """
    Given a Shapely linestring and two Shapely points, project the points onto the linestring, and return the distance along the linestring between the two points.
    If ``q is None``, then return the distance from the start of the linestring to the projection of ``p``.
    The distance is measured in the native coordinates of the linestring.
    """
    # Get projected distances
    d_p = linestring.project(p)
    if q is not None:
        d_q = linestring.project(q)
        d = abs(d_p - d_q)
    else:
        d = d_p
    return d

def get_max_runs(x):
    """
    Given a list of numbers, return a NumPy array of pairs (start index, end index + 1) of the runs of max value.

    EXAMPLES::
    
        >>> get_max_runs([7, 1, 2, 7, 7, 1, 2])
        array([[0, 1],
               [3, 5]])

    Assume x is not empty.
    Recipe from `here <http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array>`_
    """
    # Get 0-1 array where 1 marks the max values of x
    x = np.array(x)
    m = np.max(x)
    y = (x == m)*1
    # Bound y by zeros to detect runs properly
    bounded = np.hstack(([0], y, [0]))
    # Get 1 at run starts and -1 at run ends
    diffs = np.diff(bounded)
    run_starts = np.where(diffs > 0)[0]
    run_ends = np.where(diffs < 0)[0]
    return np.array([run_starts, run_ends]).T
    # # Get lengths of runs and find index of longest
    # idx = np.argmax(run_ends - run_starts)
    # return run_starts[idx], run_ends[idx] 

def get_peak_indices(times, counts):
    """
    Given an increasing list of times as seconds past midnight and a list of  trip counts at those times, return a pair of indices i, j such that times[i] to times[j] is the first longest time period such that for all i <= x < j, counts[x] is the max of counts.
    Assume times and counts have the same nonzero length.
    """
    max_runs = get_max_runs(counts)  

    def get_duration(a):
        return times[a[1]] - times[a[0]]

    index = np.argmax(np.apply_along_axis(get_duration, 1, max_runs))
    return max_runs[index]

def get_convert_dist(dist_units_in, dist_units_out):
    """
    Return a function of the form
      
      distance in the units ``dist_units_in`` -> 
      distance in the units ``dist_units_out``
    
    Only supports distance units in ``DIST_UNITS``.
    """
    di, do = dist_units_in, dist_units_out
    DU = cs.DIST_UNITS
    if not (di in DU and do in DU):
        raise ValueError(
          'Distance units must lie in {!s}'.format(DU))

    d = {
      'ft': {'ft': 1, 'm': 0.3048, 'mi': 1/5280, 'km': 0.0003048,},
      'm':  {'ft': 1/0.3048, 'm': 1, 'mi': 1/1609.344, 'km': 1/1000,},
      'mi': {'ft': 5280, 'm': 1609.344, 'mi': 1, 'km': 1.609344,},
      'km': {'ft': 1/0.0003048, 'm': 1000, 'mi': 1/1.609344, 'km': 1,},
      }
    return lambda x: d[di][do]*x

def almost_equal(f, g):
    """
    Return ``True`` if and only if the given DataFrames are equal after sorting their columns names, sorting their values, and reseting their indices.
    """
    if f.empty or g.empty:
        return f.equals(g)
    else:
        # Put in canonical order
        F = f.sort_index(axis=1).sort_values(list(f.columns)).reset_index(
          drop=True)
        G = g.sort_index(axis=1).sort_values(list(g.columns)).reset_index(
          drop=True)
        return F.equals(G)

def is_not_null(data_frame, column_name):
    """
    Return ``True`` if the given DataFrame has a column of the given name (string), and there exists at least one non-NaN value in that column;  return ``False`` otherwise.
    """
    f = data_frame
    c = column_name
    if isinstance(f, pd.DataFrame) and c in f.columns and f[c].notnull().any():
        return True
    else:
        return False

def get_utm_crs(lat, lon):
    """
    Return a GeoPandas coordinate reference system (CRS) dictionary   corresponding to the UTM projection appropriate to the given WGS84    latitude and longitude.
    """
    zone = utm.from_latlon(lat, lon)[2]
    south = lat < 0
    return {'proj':'utm', 'zone': zone, 'south': south,
      'ellps':'WGS84', 'datum':'WGS84', 'units':'m', 'no_defs':True} 

def linestring_to_utm(linestring):
    """
    Given a Shapely LineString in WGS84 coordinates, convert it to the appropriate UTM coordinates. 
    If ``inverse``, then do the inverse.
    """
    proj = lambda x, y: utm.from_latlon(y, x)[:2]

    return transform(proj, linestring) 

def count_active_trips(trip_times, time):
    """
    Given a DataFrame ``trip_times`` containing the columns

    - trip_id
    - start_time: start time of the trip in seconds past midnight
    - end_time: end time of the trip in seconds past midnight

    and a time in seconds past midnight, return the number of trips in the DataFrame that are active at the given time.
    A trip is a considered active at time t if start_time <= t < end_time.
    """
    t = trip_times
    return t[(t['start_time'] <= time) & (t['end_time'] > time)].shape[0]
    
def compute_route_stats_base(trip_stats_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a subset of the output of ``Feed.compute_trip_stats()``, calculate stats for the routes in that subset.
    
    Return a DataFrame with the following columns:

    - route_id
    - route_short_name
    - route_type
    - direction_id
    - num_trips: number of trips
    - is_loop: 1 if at least one of the trips on the route has its ``is_loop`` field equal to 1; 0 otherwise
    - is_bidirectional: 1 if the route has trips in both directions; 0 otherwise
    - start_time: start time of the earliest trip on the route
    - end_time: end time of latest trip on the route
    - max_headway: maximum of the durations (in minutes) between trip starts on the route between ``headway_start_time`` and ``headway_end_time`` on the given dates
    - min_headway: minimum of the durations (in minutes) mentioned above
    - mean_headway: mean of the durations (in minutes) mentioned above
    - peak_num_trips: maximum number of simultaneous trips in service (for the given direction, or for both directions when ``split_directions==False``)
    - peak_start_time: start time of first longest period during which the peak number of trips occurs
    - peak_end_time: end time of first longest period during which the peak number of trips occurs
    - service_duration: total of the duration of each trip on the route in the given subset of trips; measured in hours
    - service_distance: total of the distance traveled by each trip on the route in the given subset of trips; measured in wunits, that is, whatever distance units are present in trip_stats_subset; contains all ``np.nan`` entries if ``feed.shapes is None``  
    - service_speed: service_distance/service_duration; measured in wunits per hour
    - mean_trip_distance: service_distance/num_trips
    - mean_trip_duration: service_duration/num_trips

    If ``split_directions == False``, then remove the direction_id column and compute each route's stats, except for headways, using its trips running in both directions. 
    In this case, (1) compute max headway by taking the max of the max headways in both directions; (2) compute mean headway by taking the weighted mean of the mean headways in both directions. 

    If ``trip_stats_subset`` is empty, return an empty DataFrame with the columns specified above.

    Assume the following feed attributes are not ``None``: none.
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
    f[['start_time', 'end_time']] = f[['start_time', 'end_time']].\
      applymap(timestr_to_seconds)

    headway_start = timestr_to_seconds(headway_start_time)
    headway_end = timestr_to_seconds(headway_end_time)

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
        counts = [count_active_trips(group, t) for t in times]
        start, end = get_peak_indices(times, counts)
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
        counts = [count_active_trips(group, t) for t in times]
        start, end = get_peak_indices(times, counts)
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
      applymap(lambda x: timestr_to_seconds(x, inverse=True))

    return g

def compute_route_time_series_base(trip_stats_subset,
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Given a subset of the output of ``Feed.compute_trip_stats()``, calculate time series for the routes in that subset.

    Return a time series version of the following route stats:
    
    - number of trips in service by route ID
    - number of trip starts by route ID
    - service duration in hours by route ID
    - service distance in kilometers by route ID
    - service speed in kilometers per hour

    The time series is a DataFrame with a timestamp index for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute. 
    ``date_label`` is used as the date for the timestamp index.

    The columns of the DataFrame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['service_distance', 'service_duration', 'num_trip_starts', 'num_trips', 'service_speed']
    - middle level: name = 'route_id', values = the active routes
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``trip_stats_subset`` is empty, then return an empty DataFrame
    with the indicator columns.

    NOTES:
        - The route stats are all computed a minute frequency and then downsampled to the desired frequency, e.g. hour frequency, but the kind of downsampling depends on the route stat.  For example, ``num_trip_starts`` for the hour is the sum of the (integer) ``num_trip_starts`` for each minute, but ``num_trips`` for the hour is the mean of (integer) ``num_trips`` for each minute. Downsampling by the mean makes sense in the latter case, because ``num_trips`` represents the number of trips in service during the hour, and not all trips that start in the hour run the entire hour.Taking the mean will weight the trip counts by the fraction of the hour for which the trips are in service.  For example, if two trips start in the hour, one runs for the entire hour, and the other runs for half the hour, then ``num_trip_starts`` will be 2 and ``num_trips`` will be 1.5 for that hour.
        - To resample the resulting time series use the following methods:
            * for 'num_trips' series, use ``how=np.mean``
            * for the other series, use ``how=np.sum`` 
            * 'service_speed' can't be resampled and must be recalculated from 'service_distance' and 'service_duration' 
        - To remove the date and seconds from the time series f, do ``f.index = [t.time().strftime('%H:%M') for t in f.index.to_datetime()]``
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
    # at the end
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
    
    bins = [i for i in range(24*60)] # One bin for each minute
    num_bins = len(bins)

    # Bin start and end times
    def F(x):
        return (timestr_to_seconds(x)//60) % (24*60)

    tss[['start_index', 'end_index']] =\
      tss[['start_time', 'end_time']].applymap(F)
    routes = sorted(set(tss['route_id'].values))

    # Bin each trip according to its start and end time and weight
    series_by_route_by_indicator = {indicator: 
      {route: [0 for i in range(num_bins)] for route in routes} 
      for indicator in indicators}
    for index, row in tss.iterrows():
        trip = row['trip_id']
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
    g = combine_time_series(series_by_indicator, kind='route',
      split_directions=split_directions)
    return downsample(g, freq=freq)

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
    f['departure_time'] = f['departure_time'].map(timestr_to_seconds)

    headway_start = timestr_to_seconds(headway_start_time)
    headway_end = timestr_to_seconds(headway_end_time)

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
      lambda x: timestr_to_seconds(x, inverse=True))

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
        return (timestr_to_seconds(x)//60) % (24*60)

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
    g = combine_time_series(series_by_indicator, kind='stop',
      split_directions=split_directions)
    return downsample(g, freq=freq)

def combine_time_series(time_series_dict, kind, split_directions=False):
    """
    Given a dictionary of time series DataFrames, combine the time series
    into one time series DataFrame with multi-index (hierarchical) columns
    and return the result.
    The top level columns are the keys of the dictionary and
    the second and third level columns are 'route_id' and 'direction_id',
    if ``kind == 'route'``, or 'stop_id' and 'direction_id', 
    if ``kind == 'stop'``.
    If ``split_directions == False``, then there is no third column level,
    no 'direction_id' column.
    """
    if kind not in ['stop', 'route']:
        raise ValueError(
          "kind must be 'stop' or 'route'")

    subcolumns = ['indicator']
    if kind == 'stop':
        subcolumns.append('stop_id')
    else:
        subcolumns.append('route_id')

    if split_directions:
        subcolumns.append('direction_id')

    def process_index(k):
        return tuple(k.rsplit('-', 1))

    frames = list(time_series_dict.values())
    new_frames = []
    if split_directions:
        for f in frames:
            ft = f.T
            ft.index = pd.MultiIndex.from_tuples([process_index(k) 
              for (k, v) in ft.iterrows()])
            new_frames.append(ft.T)
    else:
        new_frames = frames
    return pd.concat(new_frames, axis=1, keys=list(time_series_dict.keys()),
      names=subcolumns)

def downsample(time_series, freq):
    """
    Downsample the given route, stop, or feed time series, 
    (outputs of ``Feed.compute_route_time_series()``, 
    ``Feed.compute_stop_time_series()``, or ``Feed.compute_feed_time_series()``,
    respectively) to the given Pandas frequency.
    Return the given time series unchanged if the given frequency is 
    shorter than the original frequency.
    """    
    f = time_series.copy()

    # Can't downsample to a shorter frequency
    if f.empty or\
      pd.tseries.frequencies.to_offset(freq) < f.index.freq:
        return f

    result = None
    if 'route_id' in f.columns.names:
        # It's a routes time series
        has_multiindex = True
        # Resample indicators differently.
        # For some reason in Pandas 0.18.1 the column multi-index gets
        # messed up when i try to do the resampling all at once via
        # f.resample(freq).agg(how_dict).
        # Workaround is to operate on indicators separately.
        inds_and_hows = [
          ('num_trips', 'mean'),
          ('num_trip_starts', 'sum'),
          ('service_distance', 'sum'),
          ('service_duration', 'sum'),
          ]
        frames = []
        for ind, how in inds_and_hows:
            frames.append(f[ind].resample(freq).agg({ind: how}))
        g = pd.concat(frames, axis=1)
        # Calculate speed and add it to f. Can't resample it.
        speed = g['service_distance']/g['service_duration']
        speed = pd.concat({'service_speed': speed}, axis=1)
        result = pd.concat([g, speed], axis=1)
    elif 'stop_id' in time_series.columns.names:
        # It's a stops time series
        has_multiindex = True
        result = f.resample(freq).sum()
    else:
        # It's a feed time series
        has_multiindex = False
        inds_and_hows = [
          ('num_trips', 'mean'),
          ('num_trip_starts', 'sum'),
          ('service_distance', 'sum'),
          ('service_duration', 'sum'),
          ]
        frames = []
        for ind, how in inds_and_hows:
            frames.append(f[ind].resample(freq).agg({ind: how}))
        g = pd.concat(frames, axis=1)
        # Calculate speed and add it to f. Can't resample it.
        speed = g['service_distance']/g['service_duration']
        speed = pd.concat({'service_speed': speed}, axis=1)
        result = pd.concat([g, speed], axis=1)

    # Reset column names in result, because they disappear after resampling.
    # Pandas 0.14.0 bug?
    result.columns.names = f.columns.names
    # Sort the multiindex column to make slicing possible;
    # see http://pandas.pydata.org/pandas-docs/stable/indexing.html#multiindexing-using-slicers
    if has_multiindex:
        result = result.sortlevel(axis=1)
    return result

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

def geometrize_shapes(shapes, use_utm=False):
    """
    Given a shapes DataFrame, convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 line strings instead of 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat', and 'shape_dist_traveled' columns.
    If ``use_utm``, then use UTM coordinates for the geometries.

    Requires GeoPandas.
    """
    import geopandas as gpd


    f = shapes.copy().sort_values(['shape_id', 'shape_pt_sequence'])
    
    def my_agg(group):
        d = {}
        d['geometry'] =\
          sg.LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
        return pd.Series(d)

    g = f.groupby('shape_id').apply(my_agg).reset_index()
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
        crs = get_utm_crs(lat, lon) 
        g = g.to_crs(crs)

    return g 

def ungeometrize_shapes(geo_shapes):
    """
    The inverse of :func:`geometrize_shapes`.
    Produces the columns:

    - shape_id
    - shape_pt_sequence
    - shape_pt_lon
    - shape_pt_lat

    If ``geo_shapes`` is in UTM (has a UTM CRS property), then convert UTM coordinates back to WGS84 coordinates,
    """
    geo_shapes = geo_shapes.to_crs(cs.CRS_WGS84)

    F = []
    for index, row in geo_shapes.iterrows():
        F.extend([[row['shape_id'], i, x, y] for 
        i, (x, y) in enumerate(row['geometry'].coords)])

    return pd.DataFrame(F, 
      columns=['shape_id', 'shape_pt_sequence', 
      'shape_pt_lon', 'shape_pt_lat'])
