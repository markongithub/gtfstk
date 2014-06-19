import datetime as dt

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
    """
    # Get projected distances
    d_p = linestring.project(Point(p))
    if q is not None:
        d_q = linestring.project(Point(q))
        d = abs(d_p - d_q)
    else:
        d = d_p
    return d

def downsample_routes_time_series(routes_time_series, freq):
    """
    Resample the given routes time series, which is the output of 
    ``Feed.get_routes_time_series()``, to the given (Pandas style) frequency.
    Additionally, add a new 'mean_daily_speed' time series to the output 
    dictionary.
    """
    result = {name: None for name in routes_time_series}
    result.update({'mean_daily_speed': None})
    for name in result:
        if name == 'mean_daily_speed':
            numer = routes_time_series['mean_daily_distance'].resample(freq,
              how=np.sum)
            denom = routes_time_series['mean_daily_duration'].resample(freq,
              how=np.sum)
            g = numer.divide(denom)
        elif name == 'mean_daily_num_trip_starts':
            f = routes_time_series[name]
            g = f.resample(freq, how=np.sum)
        elif name == 'mean_daily_num_vehicles':
            f = routes_time_series[name]
            g = f.resample(freq, how=np.mean)
        elif name == 'mean_daily_duration':
            f = routes_time_series[name]
            g = f.resample(freq, how=np.sum)
        else:
            # name == 'distance'
            f = routes_time_series[name]
            g = f.resample(freq, how=np.sum)
        result[name] = g
    return result

def downsample_stops_time_series(stops_time_series, freq):
    """
    Resample the given stops time series, which is the output of 
    ``Feed.get_stops_time_series()``, to the given (Pandas style) frequency.
    """
    result = {name: None for name in stops_time_series}
    for name in result:
        if name == 'mean_daily_num_vehicles':
            f = stops_time_series[name]
            g = f.resample(freq, how=np.sum)
        result[name] = g
    return result

def plot_routes_time_series(routes_ts_dict, big_units=True):
    """
    Given a routes time series dictionary (possibly downsampled),
    sum each time series over all routes, plot each series using
    MatplotLib, and return the resulting figure of four subplots.
    """
    import matplotlib.pyplot as plt

    # Plot sum of routes stats at 30-minute frequency
    F = None
    for name, f in routes_ts_dict.iteritems():
        g = f.T.sum().T
        if name == 'mean_daily_distance':
            # Convert to km
            g /= 1000
        elif name == 'mean_daily_duration':
            # Convert to h
            g /= 3600
        if F is None:
            # Initialize F
            F = pd.DataFrame(g, columns=[name])
        else:
            F[name] = g
    # Fix speed
    F['mean_daily_speed'] = F['mean_daily_distance'].divide(
      F['mean_daily_duration'])
    # Reformat periods
    F.index = [t.time().strftime('%H:%M') for t in F.index.to_datetime()]
    colors = ['red', 'blue', 'yellow', 'purple', 'green'] 
    alpha = 0.7
    names = [
      'mean_daily_num_trip_starts', 
      'mean_daily_num_vehicles', 
      'mean_daily_distance',
      'mean_daily_duration', 
      'mean_daily_speed',
      ]
    titles = [name.capitalize().replace('_', ' ') for name in names]
    units = ['','','km','h', 'kph']
    fig, axes = plt.subplots(nrows=len(names), ncols=1)
    for (i, name) in enumerate(names):
        F[name].plot(ax=axes[i], color=colors[i], alpha=alpha, 
          kind='bar', figsize=(8, 10))
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(units[i])

    #fig.tight_layout()
    return fig