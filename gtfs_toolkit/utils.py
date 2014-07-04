import datetime as dt
from collections import OrderedDict

import pandas as pd
import numpy as np
from shapely.geometry import Point

def date_to_str(date, format_str='%Y%m%d', inverse=False):
    """
    Given a datetime.date object, convert it to a string in the given format
    and return the result.
    If ``inverse == True``, then assume the given date is in the given
    string format and return its corresponding date object.
    """
    if date is None:
        return None
    if not inverse:
        result = date.strftime(format_str)
    else:
        result = dt.datetime.strptime(date, format_str).date()
    return result

def seconds_to_timestr(seconds, inverse=False):
    """
    Return the given number of integer seconds as the time string '%H:%M:%S'.
    If ``inverse == True``, then do the inverse operation.
    In keeping with GTFS standards, the hours entry may be greater than 23.
    """
    if not inverse:
        try:
            seconds = int(seconds)
            hours, remainder = divmod(seconds, 3600)
            mins, secs = divmod(remainder, 60)
            result = '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
        except:
            result = None
    else:
        try:
            hours, mins, seconds = seconds.split(':')
            result = int(hours)*3600 + int(mins)*60 + int(seconds)
        except:
            result = None
    return result

def timestr_mod_24(timestr):
    """
    Given a GTFS time string in the format %H:%M:%S, return a timestring
    in the same format but with the hours taken modulo 24.
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
    Given a weekday, that is, an integer in ``range(7)``, return
    it's corresponding weekday name as a lowercase string.
    Here 0 -> 'monday', 1 -> 'tuesday', and so on.
    If ``inverse == True``, then perform the inverse operation.
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
    Given a Shapely linestring and two Shapely points or coordinate pairs,
    project the points onto the linestring, and return the distance along
    the linestring between the two points.
    If ``q is None``, then return the distance from the start of the linestring
    to the projection of ``p``.
    The distance is measured in the native coordinates of the linestring.
    """
    # Get projected distances
    d_p = linestring.project(Point(p))
    if q is not None:
        d_q = linestring.project(Point(q))
        d = abs(d_p - d_q)
    else:
        d = d_p
    return d

# def downsample_routes_time_series(routes_time_series, freq):
#     """
#     Resample the given routes time series, which is the output of 
#     ``Feed.get_routes_time_series()``, to the given (Pandas style) frequency.
#     Additionally, add a new 'mean_daily_speed' time series to the output 
#     dictionary.
#     """
#     result = {name: None for name in routes_time_series}
#     result.update({'mean_daily_speed': None})
#     for name in result:
#         if name == 'mean_daily_speed':
#             numer = routes_time_series['mean_daily_distance'].resample(freq,
#               how=np.sum)
#             denom = routes_time_series['mean_daily_duration'].resample(freq,
#               how=np.sum)
#             g = numer.divide(denom)
#         elif name == 'mean_daily_num_trip_starts':
#             f = routes_time_series[name]
#             g = f.resample(freq, how=np.sum)
#         elif name == 'mean_daily_num_vehicles':
#             f = routes_time_series[name]
#             g = f.resample(freq, how=np.mean)
#         elif name == 'mean_daily_duration':
#             f = routes_time_series[name]
#             g = f.resample(freq, how=np.sum)
#         else:
#             # name == 'distance'
#             f = routes_time_series[name]
#             g = f.resample(freq, how=np.sum)
#         result[name] = g
#     return result

def combine_time_series(time_series_dict, kind, split_directions=True):
    """
    Given a dictionary of time series data frames, combine the time series
    into one time series data frame with multi-index (hierarchical columns)
    and return the result.
    The top level columns are the keys of the dictionary and
    the second and third level columns are 'route_id' and 'direction_id',
    if ``kind == 'route'``, or 'stop_id' and 'direction_id', 
    if ``kind == 'stop'``.
    If ``split_directions == False``, then there is no third column level,
    no 'direction_id' column.
    """
    if kind == 'route':
        subcolumns = ['route_id']
    else:
        # Assume kind == 'stop':
        subcolumns = ['stop_id']
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
              for k,v in ft.iterrows()], names=subcolumns)
            new_frames.append(ft.T)
    else:
        new_frames = frames
    return pd.concat(new_frames, axis=1, keys=list(time_series_dict.keys()),
      names=['statistic'])

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
          if col[0] in ['mean_daily_num_trip_starts', 'mean_daily_distance', 
          'mean_daily_duration'])
        # Means
        how.update(OrderedDict((col, 'mean') for col in time_series.columns
          if col[0] in ['mean_daily_num_vehicles']))
        f = time_series.resample(freq, how=how)
        # Calculate speed and add it to f. Can't resample it.
        speed = f['mean_daily_distance'].divide(f['mean_daily_duration'])
        speed = pd.concat({'mean_daily_speed': speed}, axis=1)
        result = pd.concat([f, speed], axis=1)
    elif 'stop_id' in time_series.columns.names:
        # It's a stops time series
        how = OrderedDict((col, 'sum') for col in time_series.columns)
        result = time_series.resample(freq, how=how)
    # Reset column names in result, because they disappear after resampling.
    # Pandas 0.14.0 bug?
    result.columns.names = time_series.columns.names
    return result

def plot_routes_headways(routes_stats):
    pass
    
def plot_routes_time_series(time_series):
    """
    Given a stops or routes time series,
    sum each time series statistic over all stops/routes, 
    plot each series statistic using MatplotLib, 
    and return the resulting figure of subplots.
    """
    import matplotlib.pyplot as plt

    # Reformat time periods
    time_series.index = [t.time().strftime('%H:%M') 
      for t in time_series.index.to_datetime()]
    split_directions = 'direction_id' in time_series.columns.names
    # Split time series into its component time series
    ts_dict = {stat: time_series[stat] 
      for stat in time_series.columns.levels[0]}
    # For each time series sum it across its routes/stops but keep
    # the directions separate if they already are
    if split_directions:
        ts_dict = {stat: f.groupby(level='direction_id', axis=1).sum()
          for (stat, f) in ts_dict.items()}
    else:
        ts_dict = {stat: f.sum(axis=1)
          for (stat, f) in ts_dict.items()}
    # Fix speed
    ts_dict['mean_daily_speed'] = ts_dict['mean_daily_distance'].divide(
      ts_dict['mean_daily_duration'])
    # Create plots  
    names = [
      'mean_daily_num_trip_starts', 
      'mean_daily_num_vehicles', 
      'mean_daily_distance',
      'mean_daily_duration', 
      'mean_daily_speed',
      ]
    titles = [name.capitalize().replace('_', ' ') for name in names]
    units = ['','','km','h', 'kph']
    alpha = 1
    fig, axes = plt.subplots(nrows=len(names), ncols=1)
    for (i, name) in enumerate(names):
        if name == 'mean_daily_speed':
            stacked = False
        else:
            stacked = True
        ts_dict[name].plot(ax=axes[i], alpha=alpha, 
          kind='bar', figsize=(8, 10), stacked=stacked)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(units[i])

    return fig.tight_layout()
    # return fig