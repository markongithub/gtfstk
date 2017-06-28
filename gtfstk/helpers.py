"""
Functions useful across modules.
"""
import datetime as dt

import pandas as pd
import numpy as np
from shapely.ops import transform
import utm

from . import constants as cs


def datestr_to_date(x, format_str='%Y%m%d', inverse=False):
    """
    Given a string object ``x`` representing a date in the given format,
    convert it to a Datetime Date object and return the result.
    If ``inverse``, then assume that ``x`` is a date object and return
    its corresponding string in the given format.
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
    Given an HH:MM:SS time string ``x``, return the number of seconds
    past midnight that it represents.
    In keeping with GTFS standards, the hours entry may be greater than
    23.
    If ``mod24``, then return the number of seconds modulo ``24*3600``.
    If ``inverse``, then do the inverse operation.
    In this case, if ``mod24`` also, then first take the number of
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
    Given a GTFS HH:MM:SS time string, return a timestring in the same
    format but with the hours taken modulo 24.
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
    Given a weekday number (integer in the range 0, 1, ..., 6),
    return its corresponding weekday name as a lowercase string.
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
    Given a Shapely linestring and two Shapely points,
    project the points onto the linestring, and return the distance
    along the linestring between the two points.
    If ``q is None``, then return the distance from the start of the
    linestring to the projection of ``p``.
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

    Example::

        >>> get_max_runs([7, 1, 2, 7, 7, 1, 2])
        array([[0, 1],
               [3, 5]])

    Assume x is not empty.
    Recipe `from Stack Overflow <http://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array>`_.
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
    Given an increasing list of times as seconds past midnight and a
    list of trip counts at those respective times,
    return a pair of indices i, j such that times[i] to times[j] is
    the first longest time period such that for all i <= x < j,
    counts[x] is the max of counts.
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

    Only supports distance units in :const:`constants.DIST_UNITS`.
    """
    di, do = dist_units_in, dist_units_out
    DU = cs.DIST_UNITS
    if not (di in DU and do in DU):
        raise ValueError(
          'Distance units must lie in {!s}'.format(DU))

    d = {
      'ft': {'ft': 1, 'm': 0.3048, 'mi': 1/5280, 'km': 0.0003048},
      'm': {'ft': 1/0.3048, 'm': 1, 'mi': 1/1609.344, 'km': 1/1000},
      'mi': {'ft': 5280, 'm': 1609.344, 'mi': 1, 'km': 1.609344},
      'km': {'ft': 1/0.0003048, 'm': 1000, 'mi': 1/1.609344, 'km': 1},
    }
    return lambda x: d[di][do]*x

def almost_equal(f, g):
    """
    Return ``True`` if and only if the given DataFrames are equal after
    sorting their columns names, sorting their values, and
    reseting their indices.
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
    Return ``True`` if the given DataFrame has a column of the given
    name (string), and there exists at least one non-NaN value in that
    column; return ``False`` otherwise.
    """
    f = data_frame
    c = column_name
    if isinstance(f, pd.DataFrame) and c in f.columns and f[c].notnull().any():
        return True
    else:
        return False

def get_utm_crs(lat, lon):
    """
    Return a GeoPandas coordinate reference system (CRS) dictionary
    corresponding to the UTM projection appropriate to the given WGS84
    latitude and longitude.
    """
    zone = utm.from_latlon(lat, lon)[2]
    south = lat < 0
    return {'proj': 'utm', 'zone': zone, 'south': south,
      'ellps': 'WGS84', 'datum': 'WGS84', 'units': 'm', 'no_defs': True}

def linestring_to_utm(linestring):
    """
    Given a Shapely LineString in WGS84 coordinates,
    convert it to the appropriate UTM coordinates.
    If ``inverse``, then do the inverse.
    """
    proj = lambda x, y: utm.from_latlon(y, x)[:2]
    return transform(proj, linestring)

def count_active_trips(trip_times, time):
    """
    Count the number of trips in ``trip_times`` that are active
    at the given time.

    Parameters
    ----------
    trip_times : DataFrame
        Contains the columns

        - trip_id
        - start_time: start time of the trip in seconds past midnight
        - end_time: end time of the trip in seconds past midnight

    time : integer
        Number of seconds past midnight

    Returns
    -------
    integer
        Number of trips in ``trip_times`` that are active at ``time``.
        A trip is a considered active at time t if and only if
        start_time <= t < end_time.

    """
    t = trip_times
    return t[(t['start_time'] <= time) & (t['end_time'] > time)].shape[0]

def combine_time_series(time_series_dict, kind, split_directions=False):
    """
    Combine the many time series DataFrames in the given dictionary
    into one time series DataFrame with hierarchical columns.

    Parameters
    ----------
    time_series_dict : dictionary
        Has the form string -> time series
    kind : string
        ``'route'`` or ``'stop'``
    split_directions : boolean
        If ``True``, then assume the original time series contains data
        separated by trip direction; otherwise, assume not.
        The separation is indicated by a suffix ``'-0'`` (direction 0)
        or ``'-1'`` (direction 1) in the route ID or stop ID column
        values.

    Returns
    -------
    DataFrame
        Columns are hierarchical (multi-index).
        The top level columns are the keys of the dictionary and
        the second level columns are ``'route_id'`` and
        ``'direction_id'``, if ``kind == 'route'``, or 'stop_id' and
        ``'direction_id'``, if ``kind == 'stop'``.
        If ``split_directions``, then third column is
        ``'direction_id'``; otherwise, there is no ``'direction_id'``
        column.

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
    (outputs of :func:`.routes.compute_route_time_series``,
    :func:`.stops.compute_stop_time_series``, or
    :func:`.miscellany.compute_feed_time_series`,
    respectively) to the given Pandas frequency string (e.g. '15Min').
    Return the given time series unchanged if the given frequency is
    shorter than the original frequency.
    """
    f = time_series.copy()

    # Can't downsample to a shorter frequency
    if f.empty or pd.tseries.frequencies.to_offset(freq) < f.index.freq:
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
