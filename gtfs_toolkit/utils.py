import datetime as dt
from collections import OrderedDict
from functools import wraps

import pandas as pd
import numpy as np
from shapely.geometry import Point

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

def to_km(x, units='km'):
    """
    Given a distance ``x`` in units ``units``,
    convert it to kilometers and return the result.
    """
    if units == 'km':
        return x
    if units == 'm':
        return x/1000
    if units == 'mi':
        return x*1.6093
    if units == 'ft':
        return x*0.00030480

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
