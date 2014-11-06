"""
Some tools for computing stats from a GTFS feed, assuming the feed
is valid.

All time estimates below were produced on a 2013 MacBook Pro with a
2.8 GHz Intel Core i7 processor and 16GB of RAM running OS 10.9.

TODO:

- Possibly scoop out main logic from ``Feed.get_stops_stats()`` and 
  ``Feed.get_stops_time_series()`` and put it into top level functions
  for the sake of greater flexibility.  Similar to what i did for 
  ``Feed.get_routes_stats()`` and ``Feed.get_routes_time_series()``. 
"""
import datetime as dt
import dateutil.relativedelta as rd
from collections import OrderedDict, Counter
import os
import zipfile
import tempfile
import shutil

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import utm

import gtfs_toolkit.utils as utils


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
DISTANCE_UNITS = ['km', 'm', 'mi', 'ft']


def get_routes_stats(trips_stats_subset, split_directions=False,
    headway_start_timestr='07:00:00', headway_end_timestr='19:00:00'):
    """
    Given a subset of the output of ``Feed.get_trips_stats()``, 
    calculate stats for the routes in that subset.
    
    Return a Pandas data frame with the following columns:

    - route_id
    - direction_id
    - num_trips: mean daily number of trips
    - start_time: start time of the earliest active trip on 
      the route
    - end_time: end time of latest active trip on the route
    - max_headway: maximum of the durations (in minutes) between 
      trip starts on the route between ``headway_start_timestr`` and 
      ``headway_end_timestr`` on the given dates
    - mean_headway: mean of the durations (in minutes) between 
      trip starts on the route between ``headway_start_timestr`` and 
      ``headway_end_timestr`` on the given dates
    - service_duration: total of the duration of each trip on 
      the route in the given subset of trips; measured in hours
    - service_distance: total of the distance traveled by each trip on 
      the route in the given subset of trips;
      measured in kilometers; 
      contains all ``np.nan`` entries if ``self.shapes is None``  
    - service_speed: service_distance/service_duration;
      measured in kilometers per hour

    If ``split_directions == False``, then remove the direction_id column
    and compute each route's stats, except for headways, using its trips
    running in both directions. 
    In this case, (1) compute max headway by taking the max of the max 
    headways in both directions; 
    (2) compute mean headway by taking the weighted mean of the mean
    headways in both directions. 

    NOTES:

    Takes about 0.2 minutes on the Portland feed given the first
    five weekdays of the feed.
    """        
    # Convert trip start times to seconds to ease headway calculations
    tss = trips_stats_subset
    tss['start_time'] = tss['start_time'].map(
      utils.timestr_to_seconds)

    headway_start = utils.timestr_to_seconds(headway_start_timestr)
    headway_end = utils.timestr_to_seconds(headway_end_timestr)

    def get_route_stats_split_directions(group):
        # Take this group of all trips stats for a single route
        # and compute route-level stats.
        headways = []
        stimes = group['start_time'].values
        stimes = sorted([stime for stime in stimes 
          if headway_start <= stime <= headway_end])
        headways.extend([stimes[i + 1] - stimes[i] 
          for i in range(len(stimes) - 1)])
        if headways:
            max_headway = np.max(headways)/60  # minutes 
            mean_headway = np.mean(headways)/60  # minutes 
        else:
            max_headway = np.nan
            mean_headway = np.nan
        num_trips = group.shape[0]
        start_time = group['start_time'].min()
        end_time = group['end_time'].max()
        service_duration = group['duration'].sum()
        service_distance = group['distance'].sum()
        df = pd.DataFrame([[
          start_time, 
          end_time, 
          num_trips, 
          max_headway, 
          mean_headway, 
          service_duration, 
          service_distance,
          ]], columns=[
          'start_time', 
          'end_time', 
          'num_trips', 
          'max_headway', 
          'mean_headway', 
          'service_duration', 
          'service_distance',
          ])
        df.index.name = 'foo'
        return df

    def get_route_stats(group):
        # Compute headways. Need to separate directions for these.
        headways = []
        for direction in [0, 1]:
            stimes = group[group['direction_id'] == direction][
              'start_time'].values
            stimes = sorted([stime for stime in stimes 
              if headway_start <= stime <= headway_end])
            headways.extend([stimes[i + 1] - stimes[i] 
              for i in range(len(stimes) - 1)])
        if headways:
            max_headway = np.max(headways)/60  # minutes 
            mean_headway = np.mean(headways)/60  # minutes
        else:
            max_headway = np.nan
            mean_headway = np.nan
        # Compute rest of stats
        num_trips = group.shape[0]
        start_time = group['start_time'].min()
        end_time = group['end_time'].max()
        service_duration = group['duration'].sum()
        service_distance = group['distance'].sum()
        df = pd.DataFrame([[
          start_time, 
          end_time, 
          num_trips, 
          max_headway, 
          mean_headway, 
          service_duration, 
          service_distance,
          ]], columns=[
          'start_time', 
          'end_time', 
          'num_trips', 
          'max_headway', 
          'mean_headway', 
          'service_duration', 
          'service_distance',
          ])
        df.index.name = 'foo'
        return df

    if split_directions:
        result = tss.groupby(['route_id', 'direction_id']).apply(
          get_route_stats_split_directions).reset_index()
    else:
        result = tss.groupby('route_id').apply(
          get_route_stats).reset_index()

    del result['foo']

    # Add speed column
    result['service_speed'] = result['service_distance'].\
      divide(result['service_duration'])

    # Convert route start times to time strings
    result['start_time'] = result['start_time'].map(lambda x: 
      utils.timestr_to_seconds(x, inverse=True))

    return result

def get_routes_time_series(trips_stats_subset,
  split_directions=False, freq='5Min', date_label='2001-01-01'):
    """
    Given a subset of the output of ``Feed.get_trips_stats()``, 
    calculate time series for the routes in that subset.

    Return a time series version of the following route stats:
    
    - number of vehicles in service by route ID
    - number of trip starts by route ID
    - service duration in hours by route ID
    - service distance in kilometers by route ID
    - service speed in kilometers per hour

    The time series is a Pandas data frame with a timestamp index 
    for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    ``date_label`` is used as the date for the timestamp index.

    Using a period index instead of a timestamp index would be more
    apppropriate, but 
    `Pandas 0.14.1 doesn't support period index frequencies at multiples of DateOffsets (e.g. '5Min') <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period>`_.


    The columns of the data frame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['service_distance',
      'service_duration', 'num_trip_starts', 
      'num_vehicles', 'service_speed']
    - middle level: name = 'route_id', values = the active routes
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``trips_stats_subset`` is ``None`` or empty, then return ``None``.

    NOTES:

    - To resample the resulting time series use the following methods:
        - for 'num_vehicles' series, use ``how=np.mean``
        - for the other series, use ``how=np.sum`` 
        - 'service_speed' can't be resampled and must be recalculated
          from 'service_distance' and 'service_duration' 
    - To remove the date and seconds from the 
      time series f, do ``f.index = [t.time().strftime('%H:%M') 
      for t in f.index.to_datetime()]``
    - Takes about 0.05 minutes on the Portland feed 
    """  
    if trips_stats_subset is None or trips_stats_subset.empty:
        return None

    # Merge trips_stats with trips activity, get trip weights,
    # and drop 0-weight trips
    tss = trips_stats_subset.copy()

    if split_directions:
        # Alter route IDs to encode direction: 
        # <route ID>-0 and <route ID>-1
        tss['route_id'] = tss['route_id'] + '-' +\
          tss['direction_id'].map(str)
        
    routes = sorted(tss['route_id'].unique())
    
    # Build a dictionary of time series and then merge them all
    # at the end
    # Assign a uniform generic date for the index
    date_str = date_label
    day_start = pd.to_datetime(date_str + ' 00:00:00')
    day_end = pd.to_datetime(date_str + ' 23:59:00')
    rng = pd.period_range(day_start, day_end, freq='Min')
    indicators = [
      'num_trip_starts', 
      'num_vehicles', 
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

        if start is None or start == end:
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
            if indicator == 'num_vehicles':
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

def downsample(time_series, freq):
    """
    Downsample the given route or stop time series, which is the output of 
    ``Feed.get_routes_time_series()`` or ``Feed.get_stops_time_series()``, 
    to the given Pandas-style frequency.
    Can't downsample to frequencies less one minute ('1Min'), because the
    time series are generated with one-minute frequency.
    """
    result = None
    if 'route_id' in time_series.columns.names:
        # It's a routes time series
        # Sums
        how = OrderedDict((col, 'sum') for col in time_series.columns
          if col[0] in ['num_trip_starts', 'service_distance', 
          'service_duration'])
        # Means
        how.update(OrderedDict((col, 'mean') for col in time_series.columns
          if col[0] in ['num_vehicles']))
        f = time_series.resample(freq, how=how)
        # Calculate speed and add it to f. Can't resample it.
        speed = f['service_distance'].divide(f['service_duration'])
        speed = pd.concat({'service_speed': speed}, axis=1)
        result = pd.concat([f, speed], axis=1)
    elif 'stop_id' in time_series.columns.names:
        # It's a stops time series
        how = OrderedDict((col, 'sum') for col in time_series.columns)
        result = time_series.resample(freq, how=how)
    # Reset column names in result, because they disappear after resampling.
    # Pandas 0.14.0 bug?
    result.columns.names = time_series.columns.names
    # Sort the multiindex column to make slicing possible;
    # see http://pandas.pydata.org/pandas-docs/stable/indexing.html#multiindexing-using-slicers
    return result.sortlevel(axis=1)

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

def agg_routes_stats(routes_stats):
    """
    Given ``route_stats`` which is the output of ``get_routes_stats()``,
    return a Pandas data frame with the following columns:

    - direction_id
    - num_trips: the sum of the corresponding column in the
      input across all routes
    - start_time: the minimum of the corresponding column of the input
      across all routes
    - end_time: the maximum of the corresponding column of the input
      across all routes
    - service_duration: the sum of the corresponding column in the
      input across all routes
    - service_distance: the sum of the corresponding column in the
      input across all routes  
    - service_speed: service_distance/service_distance

    If the input has no direction id, then the output won't.
    """
    f = routes_stats
    if 'direction_id' in routes_stats.columns:
        g = f.groupby('direction_id').agg({
          'start_time': min, 
          'end_time': max, 
          'num_trips': sum,
          'service_duration': sum,
          'service_distance': sum,
          }).reset_index()
    else:
        g = pd.DataFrame([[
          f['start_time'].min(), 
          f['end_time'].max(), 
          f['num_trips'].sum(),
          f['service_duration'].sum(),
          f['service_distance'].sum(),
          ]], 
          columns=['start_time', 'end_time', 'num_trips', 
            'service_duration', 'service_distance']
          )

    g['service_speed'] = g['service_distance'].divide(g['service_duration'])
    return g

def agg_routes_time_series(routes_time_series):
    rts = routes_time_series
    stats = rts.columns.levels[0].tolist()
    split_directions = 'direction_id' in rts.columns.names
    if split_directions:
        # For each stat and each direction, sum across routes.
        frames = []
        for stat in stats:
            f0 = rts.xs((stat, '0'), level=('indicator', 'direction_id'), 
              axis=1).sum(axis=1)
            f1 = rts.xs((stat, '1'), level=('indicator', 'direction_id'), 
              axis=1).sum(axis=1)
            f = pd.concat([f0, f1], axis=1, keys=['0', '1'])
            frames.append(f)
        F = pd.concat(frames, axis=1, keys=stats, names=['indicator', 
          'direction_id'])
        # Fix speed
        F['service_speed'] = F['service_distance'].divide(
          F['service_duration'])
        result = F
    else:
        f = pd.concat([rts[stat].sum(axis=1) for stat in stats], axis=1, 
          keys=stats)
        f['service_speed'] = f['service_distance'].divide(f['service_duration'])
        result = f
    return result

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
    f = agg_routes_time_series(rts)

    # Reformat time periods
    f.index = [t.time().strftime('%H:%M') 
      for t in rts.index.to_datetime()]
    
    #split_directions = 'direction_id' in rts.columns.names

    # Split time series by into its component time series by indicator type
    # stats = rts.columns.levels[0].tolist()
    stats = [
      'num_trip_starts',
      'num_vehicles',
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
        Set the native distance units of this feed to the given distance
        units.
        Valid options are listed in ``DISTANCE_UNITS``.
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
        for f in [f for f in OPTIONAL_GTFS_FILES if f != 'shapes']:
            p = path + f + '.txt'
            if os.path.isfile(p):
                setattr(self, f, pd.read_csv(p))
            else:
                setattr(self, f, None)

        # Clean up
        if zipped:
            # Remove extracted directory
            shutil.rmtree(path)

    def get_dates(self, as_date_obj=True):
        """
        Return a chronologically ordered list of date strings
        in the form '%Y%m%d' for which this feed is valid.
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
        Return a list of date strings in the form '%Y%m%d' corresponding
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

    def is_active_trip(self, trip, date):
        """
        If the given trip (trip ID) is active on the given date 
        (string of the form '%Y%m%d'), then return ``True``.
        Otherwise, return ``False``.
        To avoid error checking in the interest of speed, 
        assume ``trip`` is a valid trip ID in the feed and 
        ``date`` is a valid date object.
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

    def get_active_trips(self, date, time=None):
        """
        Return the section of ``self.trips`` that contains
        only trips active on the given date (string of the form '%Y%m%d').
        If a time is given in the form of a GTFS time string %H:%M:%S,
        then return only those trips active at that date and time.
        Do not take times modulo 24.
        """
        f = self.trips.copy()
        if not date:
            return f

        f['is_active'] = f['trip_id'].map(lambda trip: 
          int(self.is_active_trip(trip, date)))
        g = f[f['is_active'] == 1]
        del g['is_active']

        if time is not None:
            # Get trips active during given time
            h = pd.merge(g, self.stop_times[['trip_id', 'departure_time']])
          
            def F(group):
                start = group['departure_time'].min()
                end = group['departure_time'].max()
                try:
                    return start <= time <= end
                except TypeError:
                    return False

            gg = h.groupby('trip_id').apply(F).reset_index()
            g = pd.merge(g, gg[gg[0]])
            del g[0]

        return g

    def get_trips_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - trip_id
        - route_id
        - direction_id
        - dates[0]: a series of ones and zeros indicating if a 
          trip is active (1) on the given date or inactive (0)
        - etc.
        - dates[-1]: ditto

        If ``dates is None``, then return ``None``.
        Dates are strings of the form '%Y%m%d'.
        """
        if not dates:
            return

        f = self.trips.copy()
        for date in dates:
            f[date] = f['trip_id'].map(lambda trip: 
              int(self.is_active_trip(trip, date)))
        return f[['trip_id', 'direction_id', 'route_id'] + dates]

    def get_busiest_date_of_first_week(self):
        """
        Consider the dates in ``self.get_first_week()`` and 
        return the first date that has the maximum number of active trips.
        """
        dates = self.get_first_week()
        f = self.get_trips_activity(dates)
        s = [(f[date].sum(), date) for date in dates]
        return max(s)[1]

    def get_trips_stats(self, get_dist_from_shapes=False):
        """
        Return a Pandas data frame with the following columns:

        - trip_id
        - direction_id
        - route_id
        - shape_id
        - start_time: first departure time of the trip
        - end_time: last departure time of the trip
        - duration: duration of the trip in hours
        - start_stop_id: stop ID of the first stop of the trip 
        - end_stop_id: stop ID of the last stop of the trip
        - num_stops: number of stops on trip
        - distance: distance of the trip in kilometers; contains all ``np.nan``
          entries if ``self.shapes is None``

        NOTES:

        If ``self.stop_times`` has a ``shape_dist_traveled`` column
        and ``get_dist_from_shapes == False``,
        then use that column to compute the distance column (in km).
        Elif ``self.shapes is not None``, then compute the distance 
        column using the shapes and Shapely. 
        Otherwise, set the distances to ``np.nan``.

        Takes about 0.3 minutes on the Portland feed, which has the
        ``shape_dist_traveled`` column.
        Using ``get_dist_from_shapes=True`` on the Portland feed, yields 
        a maximum absolute difference of 0.75 km from using 
        ``get_dist_from_shapes=True``.
        """
        trips = self.trips
        stop_times = self.stop_times
        num_trips = trips.shape[0]
        
        # Initialize data frame. Base it on trips.txt.
        stats = trips[['route_id', 'trip_id', 'direction_id', 'shape_id']]

        # Compute start time, end time, duration
        f = pd.merge(trips, stop_times)

        # Convert departure times to seconds past midnight, 
        # to compute durations below
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)
        g = f.groupby('trip_id')
        h = g['departure_time'].agg(OrderedDict([('start_time', np.min), 
          ('end_time', np.max)]))
        h['duration'] = (h['end_time'] - h['start_time'])/3600

        # Compute start stop and end stop
        def start_stop(group):
            i = group['departure_time'].argmin()
            return group['stop_id'].at[i]

        def end_stop(group):
            i = group['departure_time'].argmax()
            return group['stop_id'].at[i]

        h['start_stop'] = g.apply(start_stop)
        h['end_stop'] = g.apply(end_stop)
        h['num_stops'] = g.size()

        # Convert times back to time strings
        h[['start_time', 'end_time']] = h[['start_time', 'end_time']].\
          applymap(lambda x: utils.timestr_to_seconds(x, inverse=True))

        # Compute trip distance (in kilometers) from shapes
        def get_dist(group):
            shape = group['shape_id'].iat[0]
            try:
                return linestring_by_shape[shape].length/1000
            except KeyError:
                # Shape ID is nan or doesn't exist in shapes
                return np.nan 

        if 'shape_dist_traveled' in f.columns and not get_dist_from_shapes:
            # Compute distances using shape_dist_traveled column
            h['distance'] = g.apply(lambda group: 
            group['shape_dist_traveled'].max())
        elif self.shapes is not None:
            # Compute distances using the shapes and Shapely
            linestring_by_shape = self.get_linestring_by_shape()
            h['distance'] = g.apply(get_dist)
        else:
            h['distance'] = np.nan

        stats = pd.merge(stats, h.reset_index()).sort(['route_id', 
          'direction_id', 'start_time'])
        return stats

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

    def get_vehicles_locations(self, linestring_by_shape, date, times):
        """
        Return a Pandas data frame of the positions of all trips
        active on the given date and times.
        Include the columns:

        - trip_id
        - direction_id
        - route_id
        - time
        - rel_dist: number between 0 (start) and 1 (end) indicating 
          the relative distance of the vehicle along its path
        - lon: longitude of vehicle at given time
        - lat: latitude of vehicle at given time

        Requires input ``self.get_linestring_from_shape(use_utm=False)``.
        Assume ``self.stop_times`` has a ``shape_dist_traveled``
        column, possibly created by ``add_dist_to_stop_times()``.

        NOTES:

        On the Portland feed, can do 24*60 times (minute frequency)
        in 0.4 min.

        """
        assert 'shape_dist_traveled' in self.stop_times.columns,\
          "The shape_dist_traveled column is required in self.stop_times."\
          "You can add it via self.stop_times = self.add_dist_to_stop_times()."
        
        # Get active trips
        at = self.get_active_trips(date)

        # Merge active trips with stop times and convert
        # times to seconds past midnight
        f = pd.merge(at, self.stop_times)
        f['departure_time'] = f['departure_time'].map(
          utils.timestr_to_seconds)

        # Compute relative distance of each trip along its path
        # at the given time times.
        # Use linear interpolation based on stop departure times and
        # shape distance traveled.
        sample_times = np.array([utils.timestr_to_seconds(s) 
          for s in times])
        def F(group):
            dists = sorted(group['shape_dist_traveled'].values)
            times = sorted(group['departure_time'].values)
            ts = sample_times[(sample_times >= times[0]) &\
              (sample_times <= times[-1])]
            ds = np.interp(ts, times, dists)
            # if len(ts):
            return pd.DataFrame({'time': ts, 'rel_dist': ds/dists[-1]})
        g = f.groupby('trip_id').apply(F).reset_index()

        # Delete extraneous multiindex column
        del g['level_1']
        
        # Convert times back to time strings
        g['time'] = g['time'].map(
          lambda x: utils.timestr_to_seconds(x, inverse=True))

        # Compute longitude and latitude of vehicle from relative distance
        h = pd.merge(at, g)
        if not h.shape[0]:
            # Return a data frame with the promised headers but no data.
            # Without this check, result below could be an empty data frame.
            h['lon'] = pd.Series()
            h['lat'] = pd.Series()
            return h

        def G(group):
            shape = group['shape_id'].iat[0]
            linestring = linestring_by_shape[shape]
            lonlats = [linestring.interpolate(d, normalized=True).coords[0]
              for d in group['rel_dist'].values]
            group['lon'], group['lat'] = zip(*lonlats)
            return group
        result = h.groupby('shape_id').apply(G)
        return result

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

    def add_dist_to_stop_times(self, trips_stats):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field in
        ``self.stop_times``.

        Compute the ``shape_dist_traveled`` by using Shapely to measure 
        the distance of a stop along its trip linestring.
        If for a given trip, this process produces a non-monotonically 
        increasing, hence incorrect, list of (cumulative) distances, then
        fall back to estimating the distances as follows.
        Get the average speed of the trip via ``trips_stats`` and
        use is to linearly interpolate distances from stop times.
        This fallback method usually kicks in on trips with self-intersecting
        linestrings.

        NOTE: 

        Takes about 0.75 minutes on the Portland feed.
        98% of calculated 'shape_dist_traveled' values differ by at most
        0.56 km in absolute value from the original values, 
        and the maximum absolute difference is 6.3 km.
        """
        linestring_by_shape = self.get_linestring_by_shape()
        point_by_stop = self.get_point_by_stop()

        # Initialize data frame
        f = pd.merge(trips_stats[['trip_id', 'shape_id', 'distance', 
          'duration' ]], self.stop_times)

        # Convert departure times to seconds past midnight to ease calculations
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)
        dist_by_stop_by_shape = {shape: {} for shape in linestring_by_shape}

        def get_dist(group):
            # Compute the distances of the stops along this trip
            group = group.sort('stop_sequence')
            trip = group['trip_id'].iat[0]
            shape = group['shape_id'].iat[0]
            if not isinstance(shape, str):
                print(trip, 'has no shape_id')
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
                # Redo, and this time estimate distances 
                # using trip's average speed and linear interpolation
                distances = [0]
                dtimes = group['departure_time'].values
                speed = group['distance'].iat[0]/\
                  (group['duration'].iat[0]*3600) # km/s
                d = 0
                t_prev = dtimes[0]
                # Finish calculating distances
                for t in dtimes[1:]:
                    d += speed*(t - t_prev)
                    distances.append(d)
                    t_prev = t
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

    def add_dist_to_shapes(self):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field for
        ``self.shapes``.

        NOTE: 

        Takes about 0.33 minutes on the Portland feed.
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

    def get_active_stops(self, date, time=None):
        """
        Return the section of ``self.stops`` that contains
        only stops active on the given date (string of the form '%Y%m%d').
        If a time is given in the form of a GTFS time string '%H:%M:%S',
        then return only those stops that have a departure time at
        that date and time.
        Do not take times modulo 24.
        """    
        if not date:
            return    
        
        f = pd.merge(self.get_active_trips(date), self.stop_times)
        if time:
            f[f['departure_time'] == time]
        active_stop_ids = set(f['stop_id'].values)
        return self.stops[self.stops['stop_id'].isin(active_stop_ids)]

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
        headway_start_timestr='07:00:00', headway_end_timestr='19:00:00'):
        """
        Return a Pandas data frame with the following columns:

        - stop_id
        - direction_id
        - num_vehicles: number of vehicles visiting stop 
        - max_headway: durations (in minuts) between 
          vehicle departures at the stop between ``headway_start_timestr`` and 
          ``headway_end_timestr`` on the given date
        - mean_headway: durations (in minutes) between 
          vehicle departures at the stop between ``headway_start_timestr`` and 
          ``headway_end_timestr`` on the given date
        - start_time: earliest departure time of a vehicle from this stop
          on the given date
        - end_time: latest departure time of a vehicle from this stop
          on the given date

        If ``split_directions == False``, then compute each stop's stats
        using vehicles visiting it from both directions.
        The input ``date`` must be a string of the form '%Y%m%d'.
        
        NOTES:

        Takes about 0.7 minutes on the Portland feed.
        """
        if not date:
            return 

        # Get active trips and merge with stop times
        f = pd.merge(self.get_active_trips(date), self.stop_times)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)

        headway_start = utils.timestr_to_seconds(headway_start_timestr)
        headway_end = utils.timestr_to_seconds(headway_end_timestr)

        # Compute stats for each stop
        def get_stop_stats(group):
            # Operate on the group of all stop times for an individual stop
            headways = []
            num_vehicles = 0
            dtimes = sorted(group['departure_time'].values)
            num_vehicles += len(dtimes)
            dtimes = [dtime for dtime in dtimes 
              if headway_start <= dtime <= headway_end]
            headways.extend([dtimes[i + 1] - dtimes[i] 
              for i in range(len(dtimes) - 1)])
            if headways:
                max_headway = np.max(headways)/60  # minutes
                mean_headway = np.mean(headways)/60  # minutes
            else:
                max_headway = np.nan
                mean_headway = np.nan
            start_time = group['departure_time'].min()
            end_time = group['departure_time'].max()
            df = pd.DataFrame([[
              start_time, 
              end_time, 
              num_vehicles, 
              max_headway, 
              mean_headway,
              ]], 
              columns=[
              'start_time', 
              'end_time', 
              'num_vehicles', 
              'max_headway', 
              'mean_headway',
              ])
            df.index.name = 'foo'
            return df

        if split_directions:
            g = f.groupby(['stop_id', 'direction_id'])
        else:
            g = f.groupby('stop_id')

        result = g.apply(get_stop_stats).reset_index()

        # Convert start and end times to time strings
        result[['start_time', 'end_time']] =\
          result[['start_time', 'end_time']].applymap(
          lambda x: utils.timestr_to_seconds(x, inverse=True))
        del result['foo']

        return result

    def get_stops_time_series(self, date, split_directions=False,
      freq='5Min'):
        """
        Return a time series version of the following stops stats
        for the given date (string of the form '%Y%m%d'):
        
        - number of vehicles by stop ID

        The time series is a Pandas data frame with a timestamp index 
        for the 24-hour period on the given date sampled at 
        the given frequency.
        The maximum allowable frequency is 1 minute.
        
        Using a period index instead of a timestamp index would be more
        apppropriate, but 
        `Pandas 0.14.1 doesn't support period index frequencies at multiples of DateOffsets (e.g. '5Min') <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period>`_.

        The columns of the data frame are hierarchical (multi-index) with

        - top level: name = 'indicator', values = ['num_vehicles']
        - middle level: name = 'stop_id', values = the active stop IDs
        - bottom level: name = 'direction_id', values = 0s and 1s

        If ``split_directions == False``, then don't include the bottom level.
        
        If there are no active trips on the date, then return ``None``.

        NOTES:

        - 'num_vehicles' should be resampled with ``how=np.sum``
        - To remove the date and seconds from 
          the time series f, do ``f.index = [t.time().strftime('%H:%M') 
          for t in f.index.to_datetime()]``
        - Takes about 6.15 minutes on the Portland feed given the first
          five weekdays of the feed.
        """  
        if not date:
            return 

        # Get active stop times for date
        ast = pd.merge(self.get_active_trips(date), self.stop_times)

        if ast.empty:
            return None

        if split_directions:
            # Alter stop IDs to encode trip direction: 
            # <stop ID>-0 and <stop ID>-1
            ast['stop_id'] = ast['stop_id'] + '-' +\
              ast['direction_id'].map(str)            
        stops = sorted(ast['stop_id'].unique())    

        # Create one time series for each stop. Use a list first.    
        bins = [i for i in range(24*60)] # One bin for each minute
        num_bins = len(bins)

        # Bin each stop departure time
        def F(x):
            return (utils.timestr_to_seconds(x)//60) % (24*60)

        ast['departure_index'] = ast['departure_time'].map(F)

        # Create one time series for each stop
        series_by_stop = {stop: [0 for i in range(num_bins)] 
          for stop in stops} 

        for stop, group in ast.groupby('stop_id'):
            counts = Counter((bin, 0) for bin in bins) +\
              Counter(group['departure_index'].values)
            series_by_stop[stop] = [counts[bin] for bin in bins]

        # Combine lists into one time series.
        # Actually, a dictionary indicator -> time series.
        # Only one indicator in this case, but could add more
        # in the future as was done with routes time series.
        rng = pd.date_range(date, periods=24*60, freq='Min')
        series_by_indicator = {'num_vehicles':
          pd.DataFrame(series_by_stop, index=rng).fillna(0)}

        # Combine all time series into one time series
        g = combine_time_series(series_by_indicator, kind='stop',
          split_directions=split_directions)
        return downsample(g, freq=freq)

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
        headway_start_timestr='07:00:00', headway_end_timestr='19:00:00'):
        """
        If this feed has station data, that is, 'location_type' and
        'parent_station' columns in ``self.stops``, then compute
        the same stats that ``self.get_stops_stats()`` does, but for
        stations.
        Otherwise, return ``None``.

        NOTES:

        Takes about 0.2 minutes on the Portland feed given the first
        five weekdays of the feed.
        """
        # Get stop times of active trips that visit stops in stations
        stop_times = self.stop_times
        sis = self.get_stops_in_stations()
        if sis is None:
            return

        f = pd.merge(stop_times, self.get_active_trips(date))
        f = pd.merge(f, sis)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(utils.timestr_to_seconds)

        headway_start = utils.timestr_to_seconds(headway_start_timestr)
        headway_end = utils.timestr_to_seconds(headway_end_timestr)

        # Compute stats for each station
        def get_station_stats(group):
            # Operate on the group of all stop times for an individual stop
            headways = []
            num_vehicles = 0
            dtimes = sorted(group['departure_time'].values)
            num_vehicles += len(dtimes)
            dtimes = [dtime for dtime in dtimes 
              if headway_start <= dtime <= headway_end]
            headways.extend([dtimes[i + 1] - dtimes[i] 
              for i in range(len(dtimes) - 1)])
            if headways:
                max_headway = np.max(headways)/60
                mean_headway = np.mean(headways)/60
            else:
                max_headway = np.nan
                mean_headway = np.nan
            start_time = group['departure_time'].min()
            end_time = group['departure_time'].max()
            df = pd.DataFrame([[
              start_time, 
              end_time, 
              num_vehicles, 
              max_headway, 
              mean_headway,
              ]], 
              columns=[
              'start_time', 
              'end_time', 
              'num_vehicles', 
              'max_headway', 
              'mean_headway',
              ])
            df.index.name = 'foo'
            return df

        if split_directions:
            g = f.groupby(['parent_station', 'direction_id'])
        else:
            g = f.groupby('parent_station')

        result = g.apply(get_station_stats).reset_index()

        # Convert start and end times to time strings
        result[['start_time', 'end_time']] =\
          result[['start_time', 'end_time']].applymap(
          lambda x: utils.timestr_to_seconds(x, inverse=True))
        del result['foo']

        return result

    def get_routes_stats(self, trips_stats, date, split_directions=False,
        headway_start_timestr='07:00:00', headway_end_timestr='19:00:00'):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, cut it down to the subset ``S`` of trips
        that are active on the given date, and then call
        ``get_routes_stats()`` with ``S`` and the keyword arguments
        ``split_directions``, ``headway_start_timestr``, and 
        ``headway_end_timestr``.

        See ``get_routes_stats()`` for a description of the output.

        NOTES:

        A more user-friendly version of ``get_routes_stats()``.
        The latter function works without a feed, though.
        Takes about 0.2 minutes on the Portland feed.
        """
        # Get the subset of trips_stats that contains only trips active
        # on the given date
        trips_stats_subset = pd.merge(trips_stats, self.get_active_trips(date))
        return get_routes_stats(trips_stats_subset, 
          split_directions=split_directions,
          headway_start_timestr=headway_start_timestr, 
          headway_end_timestr=headway_end_timestr)

    def get_routes_time_series(self, trips_stats, date, 
      split_directions=False, freq='5Min'):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, cut it down to the subset ``S`` of trips
        that are active on the given date, and then call
        ``Feed.get_routes_time_series_0()`` with ``S`` and the given 
        keyword arguments ``split_directions`` and ``freq``
        and with ``date_label = utils.date_to_str(date)``.

        See ``Feed.get_routes_time_series()`` for a description of the output.

        If there are no active trips on the date, then return ``None``.

        NOTES:

        A more user-friendly version of ``get_routes_time_series()``.
        The latter function works without a feed, though.
        Takes about 0.6 minutes on the Portland feed.
        """  
        trips_stats_subset = pd.merge(trips_stats, self.get_active_trips(date))
        return get_routes_time_series(trips_stats_subset, 
          split_directions=split_directions, freq=freq, 
          date_label=date)

    def dump_all_stats(self, directory, date=None, freq='1H', 
      split_directions=False):
        """
        Into the given directory, dump to separate CSV files the outputs of
        
        - ``self.get_stops_stats(date)``
        - ``self.get_stops_time_series(date)``
        - ``trips_stats = self.get_trips_stats()``
        - ``self.get_routes_stats(trips_stats, date)``
        - ``self.get_routes_time_series(date)``

        where each time series is resampled to the given frequency.
        Also include a ``README.txt`` file that contains a few notes
        on units and include some useful charts.

        If no date (string of the form '%Y%m%d') is given, 
        then use the busiest day of the first week of the feed.
        """
        import os
        import textwrap

        if not os.path.exists(directory):
            os.makedirs(directory)
        if date is None:
            date = self.get_busiest_date_of_first_week()

        # Write README.txt, which contains notes on units and date range
        readme = """
        Notes 
        =====
        - Distances are measured in kilometers and durations are measured in
        hours
        - Stats were calculated for {!s}
        """.format(date)
        
        with open(directory + 'notes.rst', 'w') as f:
            f.write(textwrap.dedent(readme))

        # Stops stats
        stops_stats = self.get_stops_stats(date)
        stops_stats.to_csv(directory + 'stops_stats.csv', index=False)

        # Stops time series
        sts = self.get_stops_time_series(date, 
          split_directions=split_directions)
        sts = downsample(sts, freq=freq)
        sts.to_csv(directory + 'stops_time_series_{!s}.csv'.format(freq))

        # Trips stats
        trips_stats = self.get_trips_stats()
        trips_stats.to_csv(directory + 'trips_stats.csv', index=False)

        # Routes stats
        routes_stats = self.get_routes_stats(trips_stats, date,
          split_directions=split_directions)
        routes_stats.to_csv(directory + 'routes_stats.csv', index=False)

        # Routes time series
        rts = self.get_routes_time_series(trips_stats, date,
          split_directions=split_directions)
        rts = downsample(rts, freq=freq)
        rts.to_csv(directory + 'routes_time_series_{!s}.csv'.format(freq))

        # Plot sum of routes stats 
        fig = plot_routes_time_series(rts)
        fig.tight_layout()
        fig.savefig(directory + 'routes_time_series_agg.pdf', dpi=200)

    def export(self, path, ndigits=5):
        """
        Export this feed to a zip archive located at ``path``.
        Round all decimals to ``ndigits`` decimal places.
        """
        # Remove '.zip' extension from path, because it gets added
        # automatically below
        path = path.rstrip('.zip')

        # Write files to a temporary directory 
        tmp_dir = tempfile.mkdtemp()
        names = REQUIRED_GTFS_FILES + OPTIONAL_GTFS_FILES
        for name in names:
            if getattr(self, name) is None:
                continue
            tmp_path = os.path.join(tmp_dir, name + '.txt')
            getattr(self, name).to_csv(tmp_path, index=False, 
              float_format='%.{!s}f'.format(ndigits))

        # Zip directory 
        shutil.make_archive(path, format="zip", root_dir=tmp_dir)    

        # Delete temporary directory
        shutil.rmtree(tmp_dir)

