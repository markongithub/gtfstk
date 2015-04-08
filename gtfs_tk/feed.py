"""
Tools for computing some stats from a GTFS feed, assuming the feed
is valid.

CONVENTIONS:

In conformance with GTFS and unless specified otherwise, 
dates are encoded as date strings of 
the form '%Y%m%d' and times are encoded as time strings of the form '%H:%M:%S'
with the possibility that the hour is greater than 24.
"""
import datetime as dt
import dateutil.relativedelta as rd
from collections import OrderedDict, Counter
import os
import zipfile
import tempfile
import shutil
import json

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, mapping
import utm

import gtfs_tk.utils as utils


REQUIRED_GTFS_FILES = [
  'agency',  
  'stops',   
  'routes',
  'trips',
  'stop_times',
  'calendar',
  ]
OPTIONAL_GTFS_FILES = [
  'calendar_dates',  
  'fare_attributes',    
  'fare_rules',  
  'shapes',  
  'frequencies',     
  'transfers',   
  'feed_info',
  ]
# Columns that must be formatted as int/str in GTFS and not float;
# useful in export().
INT_COLS = [
  'location_type',
  'parent_station',
  'wheelchair_boarding',
  'route_type',
  'direction_id',
  'stop_sequence',
  'wheelchair_accessible',
  'bikes_allowed',
  'pickup_type',
  'dropoff_type',
  'timepoint',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
  'exception_type',
  'payment_method',
  'transfers',
  'shape_pt_sequence',
  'exact_times',
  'transfer_type',
]

DISTANCE_UNITS = ['km', 'm', 'mi', 'ft']

def count_active_trips(trips, time):
    """
    Given a Pandas data frame containing the rows

    - trip_id
    - start_time: start time of the trip in seconds past midnight
    - end_time: end time of the trip in seconds past midnight

    and a time in seconds past midnight, return the number of 
    trips in the data frame that are active at the given time.
    A trip is a considered active at time t if 
    start_time <= t < end_time.
    """
    return trips[(trips['start_time'] <= time) &\
      (trips['end_time'] > time)].shape[0]

def get_routes_stats(trips_stats_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a subset of the output of ``Feed.get_trips_stats()``, 
    calculate stats for the routes in that subset.
    
    Return a Pandas data frame with the following columns:

    - route_id
    - direction_id
    - num_trips: number of trips
    - is_loop: 1 if at least one of the trips on the route has its
      ``is_loop`` field equal to 1; 0 otherwise
    - is_bidirectional: 1 if the route has trips in both directions;
      0 otherwise
    - start_time: start time of the earliest trip on 
      the route
    - end_time: end time of latest trip on the route
    - max_headway: maximum of the durations (in minutes) between 
      trip starts on the route between ``headway_start_time`` and 
      ``headway_end_time`` on the given dates
    - mean_headway: mean of the durations (in minutes) between 
      trip starts on the route between ``headway_start_time`` and 
      ``headway_end_time`` on the given dates
    - peak_num_trips: maximum number of simultaneous trips in service
      (for the given direction, or for both directions when 
      ``split_directions==False``)
    - peak_start_time: start time of first longest period during which
      the peak number of trips occurs
    - peak_end_time: end time of first longest period during which
      the peak number of trips occurs
    - service_duration: total of the duration of each trip on 
      the route in the given subset of trips; measured in hours
    - service_distance: total of the distance traveled by each trip on 
      the route in the given subset of trips;
      measured in kilometers; 
      contains all ``np.nan`` entries if ``self.shapes is None``  
    - service_speed: service_distance/service_duration;
      measured in kilometers per hour
    - mean_trip_distance: service_distance/num_trips
    - mean_trip_duration: service_duration/num_trips

    If ``split_directions == False``, then remove the direction_id column
    and compute each route's stats, except for headways, using its trips
    running in both directions. 
    In this case, (1) compute max headway by taking the max of the max 
    headways in both directions; 
    (2) compute mean headway by taking the weighted mean of the mean
    headways in both directions. 

    If ``trips_stats_subset`` is empty, return an empty data frame.
    """        
    if trips_stats_subset.empty:
        return pd.DataFrame()

    # Convert trip start and end times to seconds to ease calculations below
    f = trips_stats_subset.copy()
    f[['start_time', 'end_time']] = f[['start_time', 'end_time']].\
      applymap(utils.timestr_to_seconds)

    headway_start = utils.timestr_to_seconds(headway_start_time)
    headway_end = utils.timestr_to_seconds(headway_end_time)

    def get_route_stats_split_directions(group):
        # Take this group of all trips stats for a single route
        # and compute route-level stats.
        d = OrderedDict()
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
            d['mean_headway'] = np.mean(headways)/60  # minutes 
        else:
            d['max_headway'] = np.nan
            d['mean_headway'] = np.nan

        # Compute peak num trips
        times = np.unique(group[['start_time', 'end_time']].values)
        counts = [count_active_trips(group, t) for t in times]
        start, end = utils.get_peak_indices(times, counts)
        d['peak_num_trips'] = counts[start]
        d['peak_start_time'] = times[start]
        d['peak_end_time'] = times[end]

        d['service_distance'] = group['distance'].sum()
        d['service_duration'] = group['duration'].sum()
        return pd.Series(d)

    def get_route_stats(group):
        d = OrderedDict()
        d['num_trips'] = group.shape[0]
        d['is_loop'] = int(group['is_loop'].any())
        d['is_bidirectional'] = int(group['direction_id'].unique().size > 1)
        d['start_time'] = group['start_time'].min()
        d['end_time'] = group['end_time'].max()

        # Compute max and mean headway
        headways = np.array([])
        for direction in [0, 1]:
            stimes = group[group['direction_id'] == direction][
              'start_time'].values
            stimes = sorted([stime for stime in stimes 
              if headway_start <= stime <= headway_end])
            headways = np.concatenate([headways, np.diff(stimes)])
        if headways.size:
            d['max_headway'] = np.max(headways)/60  # minutes 
            d['mean_headway'] = np.mean(headways)/60  # minutes
        else:
            d['max_headway'] = np.nan
            d['mean_headway'] = np.nan

        # Compute peak num trips
        times = np.unique(group[['start_time', 'end_time']].values)
        counts = [count_active_trips(group, t) for t in times]
        start, end = utils.get_peak_indices(times, counts)
        d['peak_num_trips'] = counts[start]
        d['peak_start_time'] = times[start]
        d['peak_end_time'] = times[end]

        d['service_distance'] = group['distance'].sum()
        d['service_duration'] = group['duration'].sum()

        return pd.Series(d)

    if split_directions:
        g = f.groupby(['route_id', 'direction_id']).apply(
          get_route_stats_split_directions).reset_index()
        
        # Add the is_bidirectional column
        def is_bidirectional(group):
            d = {}
            d['is_bidirectional'] = int(
              group['direction_id'].unique().size > 1)
            return pd.Series(d)   

        gg = g.groupby('route_id').apply(is_bidirectional).reset_index()
        g = pd.merge(gg, g)
    else:
        g = f.groupby('route_id').apply(
          get_route_stats).reset_index()

    # Compute a few more stats
    g['service_speed'] = g['service_distance']/g['service_duration']
    g['mean_trip_distance'] = g['service_distance']/g['num_trips']
    g['mean_trip_duration'] = g['service_duration']/g['num_trips']

    # Convert route times to time strings
    g[['start_time', 'end_time', 'peak_start_time', 'peak_end_time']] =\
      g[['start_time', 'end_time', 'peak_start_time', 'peak_end_time']].\
      applymap(lambda x: utils.timestr_to_seconds(x, inverse=True))

    return g

def get_routes_time_series(trips_stats_subset,
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Given a subset of the output of ``Feed.get_trips_stats()``, 
    calculate time series for the routes in that subset.

    Return a time series version of the following route stats:
    
    - number of trips in service by route ID
    - number of trip starts by route ID
    - service duration in hours by route ID
    - service distance in kilometers by route ID
    - service speed in kilometers per hour

    The time series is a Pandas data frame with a timestamp index 
    for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    ``date_label`` is used as the date for the timestamp index.

    The columns of the data frame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['service_distance',
      'service_duration', 'num_trip_starts', 'num_trips', 'service_speed']
    - middle level: name = 'route_id', values = the active routes
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``trips_stats_subset`` is empty, then return an empty data frame.

    NOTES:

    - To resample the resulting time series use the following methods:
        - for 'num_trips' series, use ``how=np.mean``
        - for the other series, use ``how=np.sum`` 
        - 'service_speed' can't be resampled and must be recalculated
          from 'service_distance' and 'service_duration' 
    - To remove the date and seconds from the 
      time series f, do ``f.index = [t.time().strftime('%H:%M') 
      for t in f.index.to_datetime()]``
    """  
    if trips_stats_subset.empty:
        return pd.DataFrame()

    tss = trips_stats_subset.copy()
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
        return (utils.timestr_to_seconds(x)//60) % (24*60)

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

def get_stops_stats(stop_times_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a subset of ``Feed.stop_times``,
    return a Pandas data frame that provides summary stats about
    the stops in that subset.
    If ``split_directions == True``, then the stop times subset
    must be augmented by a ``direction_id`` column indicating the 
    direction of each trip.
    
    The columns of the returned data frame are:

    - stop_id
    - direction_id: present iff ``split_directions == True``
    - num_trips: number of trips visiting stop 
    - max_headway: durations (in minutes) between 
      trip departures at the stop between ``headway_start_time`` and 
      ``headway_end_time`` on the given date
    - mean_headway: durations (in minutes) between 
      trip departures at the stop between ``headway_start_time`` and 
      ``headway_end_time`` on the given date
    - start_time: earliest departure time of a trip from this stop
      on the given date
    - end_time: latest departure time of a trip from this stop
      on the given date

    If ``split_directions == False``, then compute each stop's stats
    using trips visiting it from both directions.

    If ``stop_times_subset`` is empty, then return an empty data frame.
    """
    if stop_times_subset.empty:
        return pd.DataFrame()

    f = stop_times_subset.copy()

    # Convert departure times to seconds to ease headway calculations
    f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)

    headway_start = utils.timestr_to_seconds(headway_start_time)
    headway_end = utils.timestr_to_seconds(headway_end_time)

    # Compute stats for each stop
    def get_stop_stats(group):
        # Operate on the group of all stop times for an individual stop
        d = OrderedDict()
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
            d['mean_headway'] = np.mean(headways)/60  # minutes
        else:
            d['max_headway'] = np.nan
            d['mean_headway'] = np.nan
        return pd.Series(d)

    if split_directions:
        g = f.groupby(['stop_id', 'direction_id'])
    else:
        g = f.groupby('stop_id')

    result = g.apply(get_stop_stats).reset_index()

    # Convert start and end times to time strings
    result[['start_time', 'end_time']] =\
      result[['start_time', 'end_time']].applymap(
      lambda x: utils.timestr_to_seconds(x, inverse=True))

    return result

def get_stops_time_series(stop_times_subset, split_directions=False,
  freq='5Min', date_label='20010101'):
    """
    Given a subset of ``Feed.stop_times``, 
    return a time series that describes the number of trips by stop ID
    in that subset.
    If ``split_directions == True``, then the stop times subset
    must be augmented by a ``direction_id`` column indicating the 
    direction of each trip.

    The time series is a Pandas data frame with a timestamp index 
    for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    The timestamp includes the date given by ``date_label``,
    a date string of the form '%Y%m%d'.
    
    The columns of the data frame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['num_trips']
    - middle level: name = 'stop_id', values = the active stop IDs
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``stop_times_subset`` is empty, then return an empty data frame.

    NOTES:

    - 'num_trips' should be resampled with ``how=np.sum``
    - To remove the date and seconds from 
      the time series f, do ``f.index = [t.time().strftime('%H:%M') 
      for t in f.index.to_datetime()]``
    """  
    if stop_times_subset.empty:
        return pd.DataFrame()

    f = stop_times_subset.copy()

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
        return (utils.timestr_to_seconds(x)//60) % (24*60)

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

def downsample(time_series, freq):
    """
    Downsample the given route, stop, or feed time series, 
    (outputs of ``Feed.get_routes_time_series()``, 
    ``Feed.get_stops_time_series()``, or ``Feed.get_feed_time_series()``,
    respectively) to the given Pandas frequency.
    Return the given time series unchanged if the given frequency is 
    shorter than the original frequency.
    """    
    # Can't downsample to a shorter frequency
    if pd.tseries.frequencies.to_offset(freq) < time_series.index.freq:
        return time_series

    result = None
    if 'route_id' in time_series.columns.names:
        # It's a routes time series
        has_multiindex = True
        # Sums
        how = OrderedDict((col, 'sum') for col in time_series.columns
          if col[0] in ['num_trip_starts', 'service_distance', 
          'service_duration'])
        # Means
        how.update(OrderedDict((col, 'mean') for col in time_series.columns
          if col[0] in ['num_trips']))
        f = time_series.resample(freq, how=how)
        # Calculate speed and add it to f. Can't resample it.
        speed = f['service_distance']/f['service_duration']
        speed = pd.concat({'service_speed': speed}, axis=1)
        result = pd.concat([f, speed], axis=1)
    elif 'stop_id' in time_series.columns.names:
        # It's a stops time series
        has_multiindex = True
        how = OrderedDict((col, 'sum') for col in time_series.columns)
        result = time_series.resample(freq, how=how)
    else:
        # It's a feed time series
        has_multiindex = False
        # Sums
        how = OrderedDict((col, 'sum') for col in time_series.columns
          if col in ['num_trip_starts', 'service_distance', 
          'service_duration'])
        # Means
        how.update(OrderedDict((col, 'mean') for col in time_series.columns
          if col in ['num_trips']))
        f = time_series.resample(freq, how=how)
        # Calculate speed and add it to f. Can't resample it.
        speed = f['service_distance']/f['service_duration']
        speed = pd.concat({'service_speed': speed}, axis=1)
        result = pd.concat([f, speed], axis=1)

    # Reset column names in result, because they disappear after resampling.
    # Pandas 0.14.0 bug?
    result.columns.names = time_series.columns.names
    # Sort the multiindex column to make slicing possible;
    # see http://pandas.pydata.org/pandas-docs/stable/indexing.html#multiindexing-using-slicers
    if has_multiindex:
        result = result.sortlevel(axis=1)
    return result

def combine_time_series(time_series_dict, kind, split_directions=False):
    """
    Given a dictionary of time series data frames, combine the time series
    into one time series data frame with multi-index (hierarchical) columns
    and return the result.
    The top level columns are the keys of the dictionary and
    the second and third level columns are 'route_id' and 'direction_id',
    if ``kind == 'route'``, or 'stop_id' and 'direction_id', 
    if ``kind == 'stop'``.
    If ``split_directions == False``, then there is no third column level,
    no 'direction_id' column.
    """
    assert kind in ['stop', 'route'],\
      "kind must be 'stop' or 'route'"

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

def plot_headways(stats, max_headway_limit=60):
    """
    Given a stops or routes stats data frame, 
    return bar charts of the max and mean headways as a MatplotLib figure.
    Only include the stops/routes with max headways at most 
    ``max_headway_limit`` minutes.
    If ``max_headway_limit is None``, then include them all in a giant plot. 
    If there are no stops/routes within the max headway limit, then return 
    ``None``.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    import matplotlib.pyplot as plt

    # Set Pandas plot style
    pd.options.display.mpl_style = 'default'

    if 'stop_id' in stats.columns:
        index = 'stop_id'
    elif 'route_id' in stats.columns:
        index = 'route_id'
    split_directions = 'direction_id' in stats.columns
    if split_directions:
        # Move the direction_id column to a hierarchical column,
        # select the headway columns, and convert from seconds to minutes
        f = stats.pivot(index=index, columns='direction_id')[['max_headway', 
          'mean_headway']]
        # Only take the stops/routes within the max headway limit
        if max_headway_limit is not None:
            f = f[(f[('max_headway', 0)] <= max_headway_limit) |
              (f[('max_headway', 1)] <= max_headway_limit)]
        # Sort by max headway
        f = f.sort(columns=[('max_headway', 0)], ascending=False)
    else:
        f = stats.set_index(index)[['max_headway', 'mean_headway']]
        if max_headway_limit is not None:
            f = f[f['max_headway'] <= max_headway_limit]
        f = f.sort(columns=['max_headway'], ascending=False)
    if f.empty:
        return

    # Plot max and mean headway separately
    n = f.shape[0]
    data_frames = [f['max_headway'], f['mean_headway']]
    titles = ['Max Headway','Mean Headway']
    ylabels = [index, index]
    xlabels = ['minutes', 'minutes']
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for (i, f) in enumerate(data_frames):
        f.plot(kind='barh', ax=axes[i], figsize=(10, max(n/9, 10)))
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel(ylabels[i])
    return fig

def plot_routes_time_series(routes_time_series):
    """
    Given a routes time series data frame,
    sum each time series indicator over all routes, 
    plot each series indicator using MatplotLib, 
    and return the resulting figure of subplots.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    import matplotlib.pyplot as plt

    rts = routes_time_series
    if 'route_id' not in rts.columns.names:
        return

    # Aggregate time series
    f = get_feed_time_series(rts)

    # Reformat time periods
    f.index = [t.time().strftime('%H:%M') 
      for t in rts.index.to_datetime()]
    
    #split_directions = 'direction_id' in rts.columns.names

    # Split time series by into its component time series by indicator type
    # stats = rts.columns.levels[0].tolist()
    stats = [
      'num_trip_starts',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    ts_dict = {stat: f[stat] for stat in stats}

    # Create plots  
    pd.options.display.mpl_style = 'default'
    titles = [stat.capitalize().replace('_', ' ') for stat in stats]
    units = ['','','km','h', 'kph']
    alpha = 1
    fig, axes = plt.subplots(nrows=len(stats), ncols=1)
    for (i, stat) in enumerate(stats):
        if stat == 'service_speed':
            stacked = False
        else:
            stacked = True
        ts_dict[stat].plot(ax=axes[i], alpha=alpha, 
          kind='bar', figsize=(8, 10), stacked=stacked, width=1)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(units[i])

    return fig


class Feed(object):
    """
    A class to gather all the GTFS files for a feed and store them in memory 
    as Pandas data frames.  
    Make sure you have enough memory!  
    The stop times object can be big.
    """
    def __init__(self, path, original_units='km'):
        """
        Read in all the relevant GTFS text files within the directory or 
        ZIP file given by ``path`` and assign them to instance attributes.
        Assume the zip file unzips as a collection of GTFS text files
        rather than as a directory of GTFS text files.
        Set the ``original_units`` to the original distance units of this 
        feed; valid options are listed in ``DISTANCE_UNITS``.
        All distance units will then be converted to kilometers.
        """
        zipped = False
        if zipfile.is_zipfile(path):
            # Extract to temporary location
            zipped = True
            archive = zipfile.ZipFile(path)
            path = path.rstrip('.zip') + '/'
            archive.extractall(path)

        # Get distance units
        assert original_units in DISTANCE_UNITS,\
            'Units must be one of {!s}'.format(DISTANCE_UNITS)
        self.original_units = original_units

        # Check that the required GTFS files exist
        for f in REQUIRED_GTFS_FILES:
            ff = f + '.txt'
            if ff == 'calendar.txt':
                assert os.path.exists(path + ff) or\
                  os.path.exists(path + 'calendar_dates.txt'),\
                  "File calendar.txt or calendar_dates.txt"\
                  " is required in GTFS feeds"
            else:
                assert os.path.exists(path + ff),\
                  "File {!s} is required in GTFS feeds".format(ff)

        # Get required GTFS files
        self.agency = pd.read_csv(path + 'agency.txt')
        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str, 
          'stop_code': str})
        self.routes = pd.read_csv(path + 'routes.txt', dtype={'route_id': str,
          'route_short_name': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'route_id': str,
          'trip_id': str, 'service_id': str, 'shape_id': str, 'stop_id': str})
        self.trips_t = self.trips.set_index('trip_id')
        st = pd.read_csv(path + 'stop_times.txt', dtype={'stop_id': str,
          'trip_id': str})
    
        # Prefix a 0 to arrival and departure times if necessary.
        # This makes sorting by time work as expected.
        def reformat_times(timestr):
            result = timestr
            if isinstance(result, str) and len(result) == 7:
                result = '0' + result
            return result

        st[['arrival_time', 'departure_time']] =\
          st[['arrival_time', 'departure_time']].applymap(reformat_times)
        # Convert distances to kilometers
        if 'shape_dist_traveled' in st.columns:
            st['shape_dist_traveled'] = st['shape_dist_traveled'].map(
              lambda x: utils.to_km(x, original_units))
        self.stop_times = st

        # One of calendar.txt and calendar_dates.txt is
        # required by the GTFS.
        if os.path.isfile(path + 'calendar.txt'):
            self.calendar = pd.read_csv(path + 'calendar.txt', 
              dtype={'service_id': str, 'start_date': str, 'end_date': str})
            # Index by service ID to make self.is_active_trip() fast
            self.calendar_s = self.calendar.set_index('service_id')
        else:
            self.calendar = None
            self.calendar_s = None
        if os.path.isfile(path + 'calendar_dates.txt'):
            self.calendar_dates = pd.read_csv(path + 'calendar_dates.txt', 
              dtype={'service_id': str, 'date': str})
            # Group by service ID and date to make self.is_active_trip() fast
            self.calendar_dates_g = self.calendar_dates.groupby(
              ['service_id', 'date'])
        else:
            self.calendar_dates = None
            self.calendar_dates_g = None

        # Get optional GTFS files if they exist
        if os.path.isfile(path + 'shapes.txt'):
            shapes = pd.read_csv(path + 'shapes.txt', 
              dtype={'shape_id': str})
            # Convert distances to kilometers
            if 'shape_dist_traveled' in shapes.columns:
                shapes['shape_dist_traveled'] =\
                  shapes['shape_dist_traveled'].map(
                  lambda x: utils.to_km(x, original_units))
            self.shapes = shapes
        else:
            self.shapes = None

        # Load the rest of the optional GTFS files without special formatting
        for f in [f for f in OPTIONAL_GTFS_FILES 
          if f not in ['shapes', 'calendar_dates']]:
            p = path + f + '.txt'
            if os.path.isfile(p):
                setattr(self, f, pd.read_csv(p))
            else:
                setattr(self, f, None)

        # Clean up
        if zipped:
            # Remove extracted directory
            shutil.rmtree(path)
        
    # Trip methods
    # ----------------------------------
    def is_active_trip(self, trip, date):
        """
        If the given trip (trip ID) is active on the given date 
        (string of the form '%Y%m%d'), then return ``True``.
        Otherwise, return ``False``.
        To avoid error checking in the interest of speed, 
        assume ``trip`` is a valid trip ID in the feed and 
        ``date`` is a valid date object.

        Note: This method is key for getting all trips, routes, 
        etc. that are active on a given date, 
        so the method needs to be fast. 
        """
        service = self.trips_t.at[trip, 'service_id']
        # Check self.calendar_dates_g.
        caldg = self.calendar_dates_g
        if caldg is not None:
            if (service, date) in caldg.groups:
                et = caldg.get_group((service, date))['exception_type'].iat[0]
                if et == 1:
                    return True
                else:
                    # Exception type is 2
                    return False
        # Check self.calendar_s
        cals = self.calendar_s
        if cals is not None:
            if service in cals.index:
                weekday_str = utils.weekday_to_str(
                  utils.datestr_to_date(date).weekday())
                if cals.at[service, 'start_date'] <= date <= cals.at[service,
                  'end_date'] and cals.at[service, weekday_str] == 1:
                    return True
                else:
                    return False
        # If you made it here, then something went wrong
        return False

    def get_trips(self, date=None, time=None):
        """
        Return the section of ``self.trips`` that contains
        only trips active on the given date (string of the form '%Y%m%d').
        If ``date is None``, then return all trips.
        If a date and time are given, 
        then return only those trips active at that date and time.
        Do not take times modulo 24.
        """
        f = self.trips.copy()
        if date is None:
            return f

        f['is_active'] = f['trip_id'].map(
          lambda trip: self.is_active_trip(trip, date))
        f = f[f['is_active']]
        del f['is_active']

        if time is not None:
            # Get trips active during given time
            g = pd.merge(f, self.stop_times[['trip_id', 'departure_time']])
          
            def F(group):
                d = {}
                start = group['departure_time'].min()
                end = group['departure_time'].max()
                try:
                    result = start <= time <= end
                except TypeError:
                    result = False
                d['is_active'] = result
                return pd.Series(d)

            h = g.groupby('trip_id').apply(F).reset_index()
            f = pd.merge(f, h[h['is_active']])
            del f['is_active']

        return f

    def get_trips_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - trip_id
        - dates[0]: 1 if the trip is active on the given date; 0 otherwise
        - dates[1]: ditto
        - etc.
        - dates[-1]: ditto

        If ``dates is None``, then return ``None``.
        """
        if not dates:
            return

        f = self.trips.copy()
        for date in dates:
            f[date] = f['trip_id'].map(lambda trip: 
              int(self.is_active_trip(trip, date)))
        return f[['trip_id'] + dates]

    def get_trips_stats(self, get_dist_from_shapes=False):
        """
        Return a Pandas data frame with the following columns:

        - trip_id
        - route_id
        - direction_id
        - shape_id
        - num_stops: number of stops on trip
        - start_time: first departure time of the trip
        - end_time: last departure time of the trip
        - start_stop_id: stop ID of the first stop of the trip 
        - end_stop_id: stop ID of the last stop of the trip
        - is_loop: 1 if the start and end stop are less than 400m apart and
          0 otherwise
        - distance: distance of the trip in kilometers; contains all ``np.nan``
          entries if ``self.shapes is None``
        - duration: duration of the trip in hours
        - speed: distance/duration

        NOTES:

        If ``self.stop_times`` has a ``shape_dist_traveled`` column
        and ``get_dist_from_shapes == False``,
        then use that column to compute the distance column (in km).
        Else if ``self.shapes is not None``, then compute the distance 
        column using the shapes and Shapely. 
        Otherwise, set the distances to ``np.nan``.

        Calculating trip distances with ``get_dist_from_shapes=True``
        seems pretty accurate.
        For example, calculating trip distances on the Portland feed using
        ``get_dist_from_shapes=False`` and ``get_dist_from_shapes=True``,
        yields a difference of at most 0.83km.
        """        

        # Start with stop times and extra trip info.
        # Convert departure times to seconds past midnight to 
        # compute durations.
        f = self.trips[['route_id', 'trip_id', 'direction_id', 'shape_id']]
        f = pd.merge(f, self.stop_times).sort(['trip_id', 'stop_sequence'])
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)
        
        # Compute all trips stats except distance, 
        # which is possibly more involved
        point_by_stop = self.get_point_by_stop(use_utm=True)
        g = f.groupby('trip_id')

        def my_agg(group):
            d = OrderedDict()
            d['route_id'] = group['route_id'].iat[0]
            d['direction_id'] = group['direction_id'].iat[0]
            d['shape_id'] = group['shape_id'].iat[0]
            d['num_stops'] = group.shape[0]
            d['start_time'] = group['departure_time'].iat[0]
            d['end_time'] = group['departure_time'].iat[-1]
            d['start_stop_id'] = group['stop_id'].iat[0]
            d['end_stop_id'] = group['stop_id'].iat[-1]
            dist = point_by_stop[d['start_stop_id']].distance(
              point_by_stop[d['end_stop_id']])
            d['is_loop'] = int(dist < 400)
            d['duration'] = (d['end_time'] - d['start_time'])/3600
            return pd.Series(d)

        # Apply my_agg, but don't reset index yet.
        # Need trip ID as index to line up the results of the 
        # forthcoming distance calculation
        h = g.apply(my_agg)  

        # Compute distance
        if 'shape_dist_traveled' in f.columns and not get_dist_from_shapes:
            # Compute distances using shape_dist_traveled column
            h['distance'] = g.apply(
              lambda group: group['shape_dist_traveled'].max())
        elif self.shapes is not None:
            # Compute distances using the shapes and Shapely
            linestring_by_shape = self.get_linestring_by_shape()
            point_by_stop = self.get_point_by_stop()

            def get_dist(group):
                """
                Return the distance traveled along the trip between the first
                and last stops.
                If that distance is negative or if the trip's linestring 
                intersects itself, then return the length of the trip's 
                linestring instead.
                """
                shape = group['shape_id'].iat[0]
                try:
                    # Get the linestring for this trip
                    linestring = linestring_by_shape[shape]
                except KeyError:
                    # Shape ID is NaN or doesn't exist in shapes.
                    # No can do.
                    return np.nan 
                
                # If the linestring intersects itself, then that can cause
                # errors in the computation below, so just 
                # return the length of the linestring as a good approximation
                if not linestring.is_simple:
                    return linestring.length/1000

                # Otherwise, return the difference of the distances along
                # the linestring of the first and last stop
                start_stop = group['stop_id'].iat[0]
                end_stop = group['stop_id'].iat[-1]
                try:
                    start_point = point_by_stop[start_stop]
                    end_point = point_by_stop[end_stop]
                except KeyError:
                    # One of the two stop IDs is NaN, so just
                    # return the length of the linestring
                    return linestring.length/1000
                d1 = linestring.project(start_point)
                d2 = linestring.project(end_point)
                d = d2 - d1
                if d > 0:
                    return d/1000
                else:
                    # Something is probably wrong, so just
                    # return the length of the linestring
                    return linestring.length/1000

            h['distance'] = g.apply(get_dist)
        else:
            h['distance'] = np.nan

        # Reset index and compute final stats
        h = h.reset_index()
        h['speed'] = h['distance']/h['duration']
        h[['start_time', 'end_time']] = h[['start_time', 'end_time']].\
          applymap(lambda x: utils.timestr_to_seconds(x, inverse=True))
        
        return h.sort(['route_id', 'direction_id', 'start_time'])

    def get_trips_locations(self, date, times):
        """
        Return a Pandas data frame of the positions of all trips
        active on the given date and times 
        Include the columns:

        - trip_id
        - route_id
        - direction_id
        - time
        - rel_dist: number between 0 (start) and 1 (end) indicating 
          the relative distance of the trip along its path
        - lon: longitude of trip at given time
        - lat: latitude of trip at given time

        Assume ``self.stop_times`` has an accurate ``shape_dist_traveled``
        column.
        """
        assert 'shape_dist_traveled' in self.stop_times.columns,\
          "The shape_dist_traveled column is required in self.stop_times. "\
          "You can create it, possibly with some inaccuracies, "\
          "via self.stop_times = self.add_dist_to_stop_times()."
        
        # Start with stop times active on date
        f = self.get_stop_times(date)
        f['departure_time'] = f['departure_time'].map(
          utils.timestr_to_seconds)

        # Compute relative distance of each trip along its path
        # at the given time times.
        # Use linear interpolation based on stop departure times and
        # shape distance traveled.
        linestring_by_shape = self.get_linestring_by_shape(use_utm=False)
        sample_times = np.array([utils.timestr_to_seconds(s) 
          for s in times])
        
        def get_rel_dist(group):
            dists = sorted(group['shape_dist_traveled'].values)
            times = sorted(group['departure_time'].values)
            ts = sample_times[(sample_times >= times[0]) &\
              (sample_times <= times[-1])]
            ds = np.interp(ts, times, dists)
            return pd.DataFrame({'time': ts, 'rel_dist': ds/dists[-1]})
        
        # return f.groupby('trip_id', group_keys=False).\
        #   apply(get_rel_dist).reset_index()
        g = f.groupby('trip_id').apply(get_rel_dist).reset_index()
        
        # Delete extraneous multi-index column
        del g['level_1']
        
        # Convert times back to time strings
        g['time'] = g['time'].map(
          lambda x: utils.timestr_to_seconds(x, inverse=True))

        # Merge in more trip info and
        # compute longitude and latitude of trip from relative distance
        h = pd.merge(self.trips[['trip_id', 'route_id', 'direction_id', 
          'shape_id']], g)
        if not h.shape[0]:
            # Return a data frame with the promised headers but no data.
            # Without this check, result below could be an empty data frame.
            h['lon'] = pd.Series()
            h['lat'] = pd.Series()
            return h

        def get_lonlat(group):
            shape = group['shape_id'].iat[0]
            linestring = linestring_by_shape[shape]
            lonlats = [linestring.interpolate(d, normalized=True).coords[0]
              for d in group['rel_dist'].values]
            group['lon'], group['lat'] = zip(*lonlats)
            return group
        
        return h.groupby('shape_id').apply(get_lonlat)
        
    # Route methods
    # ----------------------------------
    def get_routes(self, date=None, time=None):
        """
        Return the section of ``self.routes`` that contains
        only routes active on the given date.
        If ``date is None``, then return all routes.
        If a date and time are given, then return only those routes with
        trips active at that date and time.
        Do not take times modulo 24.
        """
        if date is None:
            return self.routes.copy()

        trips = self.get_trips(date, time)
        R = trips['route_id'].unique()
        return self.routes[self.routes['route_id'].isin(R)]

    def get_routes_stats(self, trips_stats, date, split_directions=False,
        headway_start_time='07:00:00', headway_end_time='19:00:00'):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, cut it down to the subset ``S`` of trips
        that are active on the given date, and then call
        ``get_routes_stats()`` with ``S`` and the keyword arguments
        ``split_directions``, ``headway_start_time``, and 
        ``headway_end_time``.

        See ``get_routes_stats()`` for a description of the output.

        NOTES:

        This is a more user-friendly version of ``get_routes_stats()``.
        The latter function works without a feed, though.

        Return ``None`` if the date does not lie in this feed's date range.
        """
        # Get the subset of trips_stats that contains only trips active
        # on the given date
        trips_stats_subset = pd.merge(trips_stats, self.get_trips(date))
        return get_routes_stats(trips_stats_subset, 
          split_directions=split_directions,
          headway_start_time=headway_start_time, 
          headway_end_time=headway_end_time)

    def get_routes_time_series(self, trips_stats, date, 
      split_directions=False, freq='5Min'):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, cut it down to the subset ``S`` of trips
        that are active on the given date, and then call
        ``Feed.get_routes_time_series()`` with ``S`` and the given 
        keyword arguments ``split_directions`` and ``freq``
        and with ``date_label = utils.date_to_str(date)``.

        See ``Feed.get_routes_time_series()`` for a description of the output.

        If there are no active trips on the date, then return ``None``.

        NOTES:

        This is a more user-friendly version of ``get_routes_time_series()``.
        The latter function works without a feed, though.
        """  
        trips_stats_subset = pd.merge(trips_stats, self.get_trips(date))
        return get_routes_time_series(trips_stats_subset, 
          split_directions=split_directions, freq=freq, 
          date_label=date)

    def get_route_timetable(self, route_id, date):
        """
        Return a Pandas data frame encoding the timetable
        for the given route ID on the given date.
        The columns are all those in ``self.trips`` plus those in 
        ``self.stop_times``.
        The result is sorted by grouping by trip ID and
        sorting the groups by their first departure time.
        """
        f = self.get_trips(date)
        f = f[f['route_id'] == route_id]
        f = pd.merge(f, self.stop_times)
        # Groupby trip ID and sort groups by their minimum departure time
        g = f.groupby('trip_id')
        new_index = g[['departure_time']].\
          transform(min).\
          sort('departure_time').index
        f.ix[new_index]
        return f

    # Stop methods
    # ----------------------------------
    def get_stops(self, date=None):
        """
        Return the section of ``self.stops`` that contains
        only stops with active stop times on the given date 
        (string of the form '%Y%m%d').
        If ``date is None``, then return all stops.
        """
        if date is None:
            return self.stops.copy()

        stop_times = self.get_stop_times(date)
        S = stop_times['stop_id'].unique()
        return self.stops[self.stops['stop_id'].isin(S)]

    def get_point_by_stop(self, use_utm=True):
        """
        Return a dictionary with structure
        stop_id -> Shapely point object.
        If ``use_utm == True``, then return each point in
        in UTM coordinates.
        Otherwise, return each point in WGS84 longitude-latitude
        coordinates.
        """
        point_by_stop = {}
        if use_utm:
            for stop, group in self.stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                point_by_stop[stop] = Point(utm.from_latlon(lat, lon)[:2]) 
        else:
            for stop, group in self.stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                point_by_stop[stop] = Point([lon, lat]) 
        return point_by_stop

    def get_stops_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - stop_id
        - dates[0]: a series of ones and zeros indicating if a 
          stop has stop times on this date (1) or not (0)
        - etc.
        - dates[-1]: ditto

        If ``dates is None``, then return ``None``.
        Dates are represented as strings of the form '%Y%m%d'.
        """
        if not dates:
            return

        trips_activity = self.get_trips_activity(dates)
        g = pd.merge(trips_activity, self.stop_times).groupby('stop_id')
        # Pandas won't allow me to simply return g[dates].max().reset_index().
        # I get ``TypeError: unorderable types: datetime.date() < str()``.
        # So here's a workaround.
        for (i, date) in enumerate(dates):
            if i == 0:
                f = g[date].max().reset_index()
            else:
                f = pd.merge(f, g[date].max().reset_index())
        return f

    def get_stops_stats(self, date, split_directions=False,
        headway_start_time='07:00:00', headway_end_time='19:00:00'):
        """
        Get all the stop times ``S`` that have trips active on the given
        date and call ``get_stops_stats()`` with ``S`` and the keyword 
        arguments ``split_directions``, ``headway_start_time``, and 
        ``headway_end_time``.

        See ``get_stops_stats()`` for a description of the output.

        NOTES:

        This is a more user-friendly version of ``get_stops_stats()``.
        The latter function works without a feed, though.
        """
        # Get stop times active on date and direction IDs
        f = pd.merge(self.get_trips(date), self.stop_times)

        return get_stops_stats(f, split_directions=split_directions,
          headway_start_time=headway_start_time, 
          headway_end_time=headway_end_time)

    def get_stops_time_series(self, date, split_directions=False,
      freq='5Min'):
        """
        Get all the stop times ``S`` that have trips active on the given
        date and call ``get_stops_stats()`` with ``S`` and the keyword 
        arguments ``split_directions`` and ``freq`` and with 
        ``date_label`` equal to ``date``.
        See ``Feed.get_stops_time_series()`` for a description of the output.

        If there are no active stop times on the date, then return ``None``.

        NOTES:

        This is a more user-friendly version of ``get_stops_time_series()``.
        The latter function works without a feed, though.
        """  
        # Get active stop times for date and direction IDs
        f = pd.merge(self.get_trips(date), self.stop_times)
        return get_stops_time_series(f, split_directions=split_directions,
            freq=freq, date_label=date)

    def get_stop_timetable(self, stop_id, date):
        """
        Return a Pandas data frame encoding the timetable
        for the given stop ID on the given date.
        The columns are all those in ``self.trips`` plus those in
        ``self.stop_times``.
        The result is sorted by departure time.
        """
        f = self.get_stop_times(date)
        f = pd.merge(self.trips, f)
        f = f[f['stop_id'] == stop_id]
        return f.sort('departure_time')

    def get_stops_in_stations(self):
        """
        If this feed has station data, that is, 'location_type' and
        'parent_station' columns in ``self.stops``, then return a Pandas
        data frame that has the same columns as ``self.stops``
        but only includes stops with parent stations, that is, stops with
        location type 0 or blank and nonblank parent station.
        Otherwise, return ``None``.
        """
        f = self.stops
        result = f[(f['location_type'] != 1) & (f['parent_station'].notnull())]
        if result.empty:
            return
        return result

    def get_stations_stats(self, date, split_directions=False,
        headway_start_time='07:00:00', headway_end_time='19:00:00'):
        """
        If this feed has station data, that is, 'location_type' and
        'parent_station' columns in ``self.stops``, then compute
        the same stats that ``self.get_stops_stats()`` does, but for
        stations.
        Otherwise, return ``None``.
        """
        # Get stop times of active trips that visit stops in stations
        sis = self.get_stops_in_stations()
        if sis is None:
            return

        f = self.get_stop_times(date)
        f = pd.merge(f, sis)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)

        headway_start = utils.timestr_to_seconds(headway_start_time)
        headway_end = utils.timestr_to_seconds(headway_end_time)

        # Compute stats for each station
        def get_station_stats(group):
            # Operate on the group of all stop times for an individual stop
            d = OrderedDict()
            d['num_trips'] = group.shape[0]
            d['start_time'] = group['departure_time'].min()
            d['end_time'] = group['departure_time'].max()
            headways = []
            dtimes = sorted([dtime for dtime in group['departure_time'].values
              if headway_start <= dtime <= headway_end])
            headways.extend([dtimes[i + 1] - dtimes[i] 
              for i in range(len(dtimes) - 1)])
            if headways:
                d['max_headway'] = np.max(headways)/60
                d['mean_headway'] = np.mean(headways)/60
            else:
                d['max_headway'] = np.nan
                d['mean_headway'] = np.nan
            return pd.Series(d)

        if split_directions:
            g = f.groupby(['parent_station', 'direction_id'])
        else:
            g = f.groupby('parent_station')

        result = g.apply(get_station_stats).reset_index()

        # Convert start and end times to time strings
        result[['start_time', 'end_time']] =\
          result[['start_time', 'end_time']].applymap(
          lambda x: utils.timestr_to_seconds(x, inverse=True))

        return result

    # Shape methods
    # ----------------------------------
    def get_linestring_by_shape(self, use_utm=True):
        """
        Return a dictionary with structure
        shape_id -> Shapely linestring of shape.
        If ``self.shapes is None``, then return ``None``.
        If ``use_utm == True``, then return each linestring in
        in UTM coordinates.
        Otherwise, return each linestring in WGS84 longitude-latitude
        coordinates.
        """
        if self.shapes is None:
            return

        # Note the output for conversion to UTM with the utm package:
        # >>> u = utm.from_latlon(47.9941214, 7.8509671)
        # >>> print u
        # (414278, 5316285, 32, 'T')
        linestring_by_shape = {}
        if use_utm:
            for shape, group in self.shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                xys = [utm.from_latlon(lat, lon)[:2] 
                  for lat, lon in zip(lats, lons)]
                linestring_by_shape[shape] = LineString(xys)
        else:
            for shape, group in self.shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                lonlats = zip(lons, lats)
                linestring_by_shape[shape] = LineString(lonlats)
        return linestring_by_shape

    def get_shapes_geojson(self):
        """
        Return a string that is a GeoJSON feature collection of 
        linestring features representing ``self.shapes``.
        Each feature will have a ``shape_id`` property. 
        If ``self.shapes is None``, then return ``None``.
        The coordinates reference system is the default one for GeoJSON,
        namely WGS84.
        """

        linestring_by_shape = self.get_linestring_by_shape(use_utm=False)
        if linestring_by_shape is None:
            return

        d = {
          'type': 'FeatureCollection', 
          'features': [{
            'properties': {'shape_id': shape},
            'type': 'Feature',
            'geometry': mapping(linestring),
            }
            for shape, linestring in linestring_by_shape.items()]
          }
        return json.dumps(d)

    def add_dist_to_shapes(self):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field for
        ``self.shapes``.

        NOTE: 

        All of the calculated ``shape_dist_traveled`` values 
        for the Portland feed differ by at most 0.016 km in absolute values
        from of the original values. 
        """
        assert self.shapes is not None,\
          "This method requires the feed to have a shapes.txt file"

        f = self.shapes

        def get_dist(group):
            # Compute the distances of the stops along this trip
            group = group.sort('shape_pt_sequence')
            shape = group['shape_id'].iat[0]
            if not isinstance(shape, str):
                print(trip, 'no shape_id:', shape)
                group['shape_dist_traveled'] = np.nan 
                return group
            points = [Point(utm.from_latlon(lat, lon)[:2]) 
              for lon, lat in group[['shape_pt_lon', 'shape_pt_lat']].values]
            p_prev = points[0]
            d = 0
            distances = [0]
            for  p in points[1:]:
                d += p.distance(p_prev)/1000
                distances.append(d)
                p_prev = p
            group['shape_dist_traveled'] = distances
            return group

        result = f.groupby('shape_id', group_keys=False).apply(get_dist)
        self.shapes = result

    # Stop time methods
    # ----------------------------------
    def get_stop_times(self, date=None):
        """
        Return the section of ``self.stop_times`` that contains
        only trips active on the given date (string of the form '%Y%m%d').
        If ``date is None``, then return all stop times.
        """
        f = self.stop_times.copy()
        if date is None:
            return f

        g = self.get_trips(date)
        return f[f['trip_id'].isin(g['trip_id'])]

    def add_dist_to_stop_times(self, trips_stats):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field in
        ``self.stop_times``.
        Doesn't always give accurate results, as described below.

        ALGORITHM:

        Compute the ``shape_dist_traveled`` field by using Shapely to measure 
        the distance of a stop along its trip linestring.
        If for a given trip this process produces a non-monotonically 
        increasing, hence incorrect, list of (cumulative) distances, then
        fall back to estimating the distances as follows.
        
        Get the average speed of the trip via ``trips_stats`` and
        use is to linearly interpolate distances for stop times, 
        assuming that the first stop is at shape_dist_traveled = 0
        (the start of the shape) and the last stop is 
        at shape_dist_traveled = the length of the trip 
        (taken from trips_stats and equal to the length of the shape,
        unless trips_stats was called with ``get_dist_from_shapes == False``).
        This fallback method usually kicks in on trips with self-intersecting
        linestrings.
        Unfortunately, this fallback method will produce incorrect results
        when the first stop does not start at the start of its shape
        (so shape_dist_traveled != 0).
        This is the case for several trips in the Portland feed, for example. 
        """
        linestring_by_shape = self.get_linestring_by_shape()
        point_by_stop = self.get_point_by_stop()

        # Initialize data frame
        f = pd.merge(trips_stats[['trip_id', 'shape_id', 'distance', 
          'duration' ]], self.stop_times).sort(['trip_id', 'stop_sequence'])

        # Convert departure times to seconds past midnight to ease calculations
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)
        dist_by_stop_by_shape = {shape: {} for shape in linestring_by_shape}

        def get_dist(group):
            # Compute the distances of the stops along this trip
            trip = group['trip_id'].iat[0]
            shape = group['shape_id'].iat[0]
            if not isinstance(shape, str):
                print(trip, 'has no shape_id')
                group['shape_dist_traveled'] = np.nan 
                return group
            elif np.isnan(group['distance'].iat[0]):
                group['shape_dist_traveled'] = np.nan 
                return group
            linestring = linestring_by_shape[shape]
            distances = []
            for stop in group['stop_id'].values:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = utils.get_segment_length(linestring, 
                      point_by_stop[stop])
                    dist_by_stop_by_shape[shape][stop] = d
                # Convert from meters to kilometers
                d /= 1000
                distances.append(d)
            s = sorted(distances)
            if s == distances:
                # Good
                pass
            elif s == distances[::-1]:
                # Reverse. This happens when the direction of a linestring
                # opposes the direction of the bus trip.
                distances = distances[::-1]
            else:
                # Totally redo using trip's average speed and 
                # linear interpolation.
                dt = group['departure_time']
                times = dt.values # seconds
                t0, t1 = times[0], times[-1]                  
                d0, d1 = 0, group['distance'].iat[0]
                # Get indices of nan departure times and 
                # temporarily forward fill them
                # for the purposes of using np.interp smoothly
                nan_indices = np.where(dt.isnull())[0]
                dt.fillna(method='ffill')
                # Interpolate
                distances = np.interp(times, [t0, t1], [d0, d1])
                # Nullify distances with nan departure times
                for i in nan_indices:
                    distances[i] = np.nan

            group['shape_dist_traveled'] = distances
            return group

        result = f.groupby('trip_id', group_keys=False).apply(get_dist)
        # Convert departure times back to time strings
        result['departure_time'] = result['departure_time'].map(lambda x: 
          utils.timestr_to_seconds(x, inverse=True))
        del result['shape_id']
        del result['distance']
        del result['duration']
        self.stop_times = result

    # Feed methods
    # ----------------------------------
    def get_dates(self, as_date_obj=False):
        """
        Return a chronologically ordered list of dates
        for which this feed is valid.
        If ``as_date_obj == True``, then return the dates as
        as ``datetime.date`` objects.  
        """
        if self.calendar is not None:
            start_date = self.calendar['start_date'].min()
            end_date = self.calendar['end_date'].max()
        else:
            # Use calendar_dates
            start_date = self.calendar_dates['date'].min()
            end_date = self.calendar_dates['date'].max()
        start_date = utils.datestr_to_date(start_date)
        end_date = utils.datestr_to_date(end_date)
        num_days = (end_date - start_date).days
        result = [start_date + rd.relativedelta(days=+d) 
          for d in range(num_days + 1)]
        if not as_date_obj:
            result = [utils.datestr_to_date(x, inverse=True)
              for x in result]
        return result

    def get_first_week(self, as_date_obj=False):
        """
        Return a list of date corresponding
        to the first Monday--Sunday week for which this feed is valid.
        In the unlikely event that this feed does not cover a full 
        Monday--Sunday week, then return whatever initial segment of the 
        week it does cover. 
        If ``as_date_obj == True``, then return the dates as
        as ``datetime.date`` objects.          
        """
        dates = self.get_dates(as_date_obj=True)
        # Get first Monday
        monday_index = None
        for (i, date) in enumerate(dates):
            if date.weekday() == 0:
                monday_index = i
                break
        week = []
        for j in range(7):
            try:
                week.append(dates[monday_index + j])
            except:
                break
        # Convert to date strings if requested
        if not as_date_obj:
            week = [utils.datestr_to_date(x, inverse=True)
              for x in week]
        return week

    def get_busiest_date(self, dates):
        """
        Given a list of dates, return the first date that has the 
        maximum number of active trips.
        """
        f = self.get_trips_activity(dates)
        s = [(f[date].sum(), date) for date in dates]
        return max(s)[1]

    def get_feed_stats(self, trips_stats, date):
        """
        Given ``trips_stats``, which is the output of 
        ``self.get_trips_stats()`` and a date,
        return a Pandas data frame including the following feed
        stats for the date.

        - num_trips: number of trips active on the given date
        - num_routes: number of routes active on the given date
        - num_stops: number of stops active on the given date
        - peak_num_trips: maximum number of simultaneous trips in service
        - peak_start_time: start time of first longest period during which
          the peak number of trips occurs
        - peak_end_time: end time of first longest period during which
          the peak number of trips occurs
        - service_distance: sum of the service distances for the active routes
        - service_duration: sum of the service durations for the active routes
        - service_speed: service_distance/service_duration

        If there are no stats for the given date, return an empty data frame.
        """
        d = OrderedDict()
        trips = self.get_trips(date)
        if trips.empty:
            return pd.DataFrame()

        d['num_trips'] = trips.shape[0]
        d['num_routes'] = self.get_routes(date).shape[0]
        d['num_stops'] = self.get_stops(date).shape[0]

        # Compute peak stats
        f = pd.merge(trips, trips_stats)
        f[['start_time', 'end_time']] =\
          f[['start_time', 'end_time']].applymap(utils.timestr_to_seconds)

        times = np.unique(f[['start_time', 'end_time']].values)
        counts = [count_active_trips(f, t) for t in times]
        start, end = utils.get_peak_indices(times, counts)
        d['peak_num_trips'] = counts[start]
        d['peak_start_time'] =\
          utils.timestr_to_seconds(times[start], inverse=True)
        d['peak_end_time'] =\
          utils.timestr_to_seconds(times[end], inverse=True)

        # Compute remaining stats
        d['service_distance'] = f['distance'].sum()
        d['service_duration'] = f['duration'].sum()
        d['service_speed'] = d['service_distance']/d['service_duration']

        return pd.DataFrame(d, index=[0])

    def get_feed_time_series(self, trips_stats, date, freq='5Min'):
        """
        Given trips stats (output of ``self.get_trips_stats()``),
        a date, and a Pandas frequency string,
        return a time series of stats for this feed on the given date
        at the given frequency with the following columns

        - num_trip_starts: number of trips starting at this time
        - num_trips: number of trips in service during this time period
        - service_distance: distance traveled by all active trips during
          this time period
        - service_duration: duration traveled by all active trips during this
          time period
        - service_speed: service_distance/service_duration

        If there is no time series for the given date, 
        return an empty data frame.
        """
        rts = self.get_routes_time_series(trips_stats, date, freq=freq)
        if rts.empty:
            return pd.DataFrame()

        stats = rts.columns.levels[0].tolist()
        # split_directions = 'direction_id' in rts.columns.names
        # if split_directions:
        #     # For each stat and each direction, sum across routes.
        #     frames = []
        #     for stat in stats:
        #         f0 = rts.xs((stat, '0'), level=('indicator', 'direction_id'), 
        #           axis=1).sum(axis=1)
        #         f1 = rts.xs((stat, '1'), level=('indicator', 'direction_id'), 
        #           axis=1).sum(axis=1)
        #         f = pd.concat([f0, f1], axis=1, keys=['0', '1'])
        #         frames.append(f)
        #     F = pd.concat(frames, axis=1, keys=stats, names=['indicator', 
        #       'direction_id'])
        #     # Fix speed
        #     F['service_speed'] = F['service_distance'].divide(
        #       F['service_duration'])
        #     result = F
        f = pd.concat([rts[stat].sum(axis=1) for stat in stats], axis=1, 
          keys=stats)
        f['service_speed'] = f['service_distance']/f['service_duration']
        return f

    def export(self, path, ndigits=6):
        """
        Export this feed to a zip archive located at ``path``.
        Round all decimals to ``ndigits`` decimal places.
        All distances will be displayed in kilometers.
        """
        # Remove '.zip' extension from path, because it gets added
        # automatically below
        path = path.rstrip('.zip')

        # Write files to a temporary directory 
        tmp_dir = tempfile.mkdtemp()
        names = REQUIRED_GTFS_FILES + OPTIONAL_GTFS_FILES
        int_cols_set = set(INT_COLS)
        for name in names:
            f = getattr(self, name)
            if f is None:
                continue
            f = f.copy()
            # Some columns need to be output as integers.
            # If there are integers and NaNs in any such column, 
            # then Pandas will format the column as float, which we don't want.
            s = list(int_cols_set & set(f.columns))
            if s:
                f[s] = f[s].fillna(-1).astype(int).astype(str).\
                  replace('-1', '')
            tmp_path = os.path.join(tmp_dir, name + '.txt')
            f.to_csv(tmp_path, index=False, 
              float_format='%.{!s}f'.format(ndigits))

        # Zip directory 
        shutil.make_archive(path, format="zip", root_dir=tmp_dir)    

        # Delete temporary directory
        shutil.rmtree(tmp_dir)

