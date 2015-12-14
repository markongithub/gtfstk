import datetime as dt
from collections import OrderedDict
from functools import wraps

import pandas as pd
import numpy as np
from shapely.geometry import Point

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
    Given a string object ``x`` representing a date in the given format,
    convert it to a datetime.date object and return the result
    If ``inverse == True``, then assume that ``x`` is a date object
    and return its corresponding string in the given format.
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
    Given a time string of the form '%H:%M:%S', return the number of seconds
    past midnight that it represents.
    In keeping with GTFS standards, the hours entry may be greater than 23.
    If ``mod24 == True``, then return the number of seconds modulo ``24*3600``.
    If ``inverse == True``, then do the inverse operation.
    In this case, if ``mod24 == True`` also, then first take the number of 
    seconds modulo ``24*3600``.
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

def clean_series(series, nan_prefix='n/a', mark='-'):
    """
    Given a series of items, replace NaN entries
    with ``nan_prefix + '-0'``, ``nan_prefix + '-1'``, 
    ``nan_prefix + '-2'``, etc.
    Replace duplicate items x1, x2, x3, etc. with 
    x1 + '-0', x2 + '-1', x3 + '-2', etc.
    Return the resulting series.
    You can change the dashes to another string by setting
    the ``mark`` parameter.

    I use this for cleaning route short names.
    """
    # Replace NaNs
    s = series.copy()
    nans = s[s.isnull()]
    fill_nans = ['{!s}{!s}{!s}'.format(nan_prefix, mark, i)
      for i in range(nans.shape[0])]
    s.iloc[nans.index] = fill_nans

    # Replace duplicates
    dups = s[s.duplicated()]
    fill_dups = [x + '{!s}{!s}'.format(mark, i) 
      for i, x in enumerate(dups.values)]
    s.iloc[dups.index] = fill_dups
    return s

def get_segment_length(linestring, p, q=None):
    """
    Given a Shapely linestring and two Shapely points,
    project the points onto the linestring, and return the distance along
    the linestring between the two points.
    If ``q is None``, then return the distance from the start of the linestring
    to the projection of ``p``.
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
    Given a list of numbers, return a NumPy array of pairs
    (start index, end index + 1) of the runs of max value.

    EXAMPLES::
    
        >>> get_max_runs([7, 1, 2, 7, 7, 1, 2])
        >>> [[1, 2], [3, 5]]
    
    Assume x is not empty.
    Recipe from 
    `here <http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array>`_
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
    Given an increasing list of times as seconds past midnight and a list of
    trip counts at those times, return a pair of indices i, j
    such that times[i] to times[j] is the first longest time period 
    such that for all i <= x < j, counts[x] is the max of counts.
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
    
    Only supports distance units in ``DISTANCE_UNITS``.
    """
    di, do = dist_units_in, dist_units_out
    DU = cs.DISTANCE_UNITS
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

def equal(x, y):
    """
    Equality testing that works on both standard Python objects 
    and data frames.
    """
    pass