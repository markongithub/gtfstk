"""
This module contains functions for calculating things about Feed objects, 
such as daily service duration per route. 
"""
import datetime as dt
import dateutil.relativedelta as rd
from collections import OrderedDict, Counter
import json
import math

import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint, MultiLineString, LineString, mapping
import utm

from . import constants as cs
from . import utilities as ut
from .feed import Feed


# -------------------------------------
# Functions about calendars
# -------------------------------------
def get_dates(feed, as_date_obj=False):
    """
    Return a chronologically ordered list of dates for which this feed is valid.
    If ``as_date_obj == True``, then return the dates as ``datetime.date`` objects.  

    If ``feed.calendar`` and ``feed.calendar_dates`` are both 
    ``None``, then return the empty list.
    """
    if feed.calendar is not None:
        start_date = feed.calendar['start_date'].min()
        end_date = feed.calendar['end_date'].max()
    elif feed.calendar_dates is not None:
        # Use calendar_dates
        start_date = feed.calendar_dates['date'].min()
        end_date = feed.calendar_dates['date'].max()
    else:
        return []

    start_date = ut.datestr_to_date(start_date)
    end_date = ut.datestr_to_date(end_date)
    num_days = (end_date - start_date).days
    result = [start_date + rd.relativedelta(days=+d) 
      for d in range(num_days + 1)]
    
    if not as_date_obj:
        result = [ut.datestr_to_date(x, inverse=True)
          for x in result]
    
    return result

def get_first_week(feed, as_date_obj=False):
    """
    Return a list of date corresponding to the first Monday--Sunday week for which this feed is valid.
    If the given feed does not cover a full Monday--Sunday week, then return whatever initial segment of the week it does cover, which could be the empty list.
    If ``as_date_obj == True``, then return the dates as as ``datetime.date`` objects.    
    """
    dates = get_dates(feed, as_date_obj=True)
    if not dates:
        return []

    # Get first Monday
    monday_index = None
    for (i, date) in enumerate(dates):
        if date.weekday() == 0:
            monday_index = i
            break
    if monday_index is None:
        return []

    result = []
    for j in range(7):
        try:
            result.append(dates[monday_index + j])
        except:
            break

    # Convert to date strings if requested
    if not as_date_obj:
        result = [ut.datestr_to_date(x, inverse=True)
          for x in result]
    return result

# -------------------------------------
# Functions about trips
# -------------------------------------
def count_active_trips(trip_times, time):
    """
    Given a data frame ``trip_times`` containing the columns

    - trip_id
    - start_time: start time of the trip in seconds past midnight
    - end_time: end time of the trip in seconds past midnight

    and a time in seconds past midnight, return the number of 
    trips in the data frame that are active at the given time.
    A trip is a considered active at time t if 
    start_time <= t < end_time.        
    """
    t = trip_times
    return t[(t['start_time'] <= time) & (t['end_time'] > time)].shape[0]

def is_active_trip(feed, trip, date):
    """
    If the given trip (trip ID) is active on the given date,
    then return ``True``; otherwise return ``False``.
    To avoid error checking in the interest of speed, 
    assume ``trip`` is a valid trip ID in the given feed and 
    ``date`` is a valid date object.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``

    NOTES: 

    This function is key for getting all trips, routes, 
    etc. that are active on a given date, so the function needs to be fast. 
    """
    service = feed._trips_i.at[trip, 'service_id']
    # Check feed._calendar_dates_g.
    caldg = feed._calendar_dates_g
    if caldg is not None:
        if (service, date) in caldg.groups:
            et = caldg.get_group((service, date))['exception_type'].iat[0]
            if et == 1:
                return True
            else:
                # Exception type is 2
                return False
    # Check feed._calendar_i
    cali = feed._calendar_i
    if cali is not None:
        if service in cali.index:
            weekday_str = ut.weekday_to_str(
              ut.datestr_to_date(date).weekday())
            if cali.at[service, 'start_date'] <= date <= cali.at[service,
              'end_date'] and cali.at[service, weekday_str] == 1:
                return True
            else:
                return False
    # If you made it here, then something went wrong
    return False

def get_trips(feed, date=None, time=None):
    """
    Return the section of ``feed.trips`` that contains
    only trips active on the given date.
    If ``feed.trips`` is ``None`` or the date is ``None``, 
    then return all ``feed.trips``.
    If a date and time are given, 
    then return only those trips active at that date and time.
    Do not take times modulo 24.
    """
    if feed.trips is None or date is None:
        return feed.trips 

    f = feed.trips.copy()
    f['is_active'] = f['trip_id'].map(
      lambda trip: is_active_trip(feed, trip, date))
    f = f[f['is_active']].copy()
    del f['is_active']

    if time is not None:
        # Get trips active during given time
        g = pd.merge(f, feed.stop_times[['trip_id', 'departure_time']])

        def F(group):
            d = {}
            start = group['departure_time'].dropna().min()
            end = group['departure_time'].dropna().max()
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

def compute_trip_activity(feed, dates):
    """
    Return a  data frame with the columns

    - trip_id
    - ``dates[0]``: 1 if the trip is active on ``dates[0]``; 
      0 otherwise
    - ``dates[1]``: 1 if the trip is active on ``dates[1]``; 
      0 otherwise
    - etc.
    - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``; 
      0 otherwise

    If ``dates`` is ``None`` or the empty list, then return an 
    empty data frame with the column 'trip_id'.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`is_active_trip`
        
    """
    if not dates:
        return pd.DataFrame(columns=['trip_id'])

    f = feed.trips.copy()
    for date in dates:
        f[date] = f['trip_id'].map(lambda trip: 
          int(is_active_trip(feed, trip, date)))
    return f[['trip_id'] + dates]

def compute_busiest_date(feed, dates):
    """
    Given a list of dates, return the first date that has the 
    maximum number of active trips.
    If the list of dates is empty, then raise a ``ValueError``.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`compute_trip_activity`
        
    """
    f = compute_trip_activity(feed, dates)
    s = [(f[date].sum(), date) for date in dates]
    return max(s)[1]

def compute_trip_stats(feed, compute_dist_from_shapes=False):
    """
    Return a data frame with the following columns:

    - trip_id
    - route_id
    - route_short_name
    - route_type
    - direction_id
    - shape_id
    - num_stops: number of stops on trip
    - start_time: first departure time of the trip
    - end_time: last departure time of the trip
    - start_stop_id: stop ID of the first stop of the trip 
    - end_stop_id: stop ID of the last stop of the trip
    - is_loop: 1 if the start and end stop are less than 400m apart and
      0 otherwise
    - distance: distance of the trip in ``feed.dist_units_out``; 
      contains all ``np.nan`` entries if ``feed.shapes is None``
    - duration: duration of the trip in hours
    - speed: distance/duration

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - ``feed.routes``
    - ``feed.stop_times``
    - ``feed.shapes`` (optionally)
    - Those used in :func:`build_geometry_by_stop`

    NOTES:

    If ``feed.stop_times`` has a ``shape_dist_traveled`` column with at least
    one non-NaN value and ``compute_dist_from_shapes == False``,
    then use that column to compute the distance column.
    Else if ``feed.shapes is not None``, then compute the distance 
    column using the shapes and Shapely. 
    Otherwise, set the distances to ``np.nan``.

    Calculating trip distances with ``compute_dist_from_shapes=True``
    seems pretty accurate.
    For example, calculating trip distances on the Portland feed using
    ``compute_dist_from_shapes=False`` and ``compute_dist_from_shapes=True``,
    yields a difference of at most 0.83km.
    """        
    # Start with stop times and extra trip info.
    # Convert departure times to seconds past midnight to 
    # compute durations.
    f = feed.trips[['route_id', 'trip_id', 'direction_id', 'shape_id']]
    f = pd.merge(f, 
      feed.routes[['route_id', 'route_short_name', 'route_type']])
    f = pd.merge(f, feed.stop_times).sort_values(['trip_id', 'stop_sequence'])
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)
    
    # Compute all trips stats except distance, 
    # which is possibly more involved
    geometry_by_stop = build_geometry_by_stop(feed, use_utm=True)
    g = f.groupby('trip_id')

    def my_agg(group):
        d = OrderedDict()
        d['route_id'] = group['route_id'].iat[0]
        d['route_short_name'] = group['route_short_name'].iat[0]
        d['route_type'] = group['route_type'].iat[0]
        d['direction_id'] = group['direction_id'].iat[0]
        d['shape_id'] = group['shape_id'].iat[0]
        d['num_stops'] = group.shape[0]
        d['start_time'] = group['departure_time'].iat[0]
        d['end_time'] = group['departure_time'].iat[-1]
        d['start_stop_id'] = group['stop_id'].iat[0]
        d['end_stop_id'] = group['stop_id'].iat[-1]
        dist = geometry_by_stop[d['start_stop_id']].distance(
          geometry_by_stop[d['end_stop_id']])
        d['is_loop'] = int(dist < 400)
        d['duration'] = (d['end_time'] - d['start_time'])/3600
        return pd.Series(d)

    # Apply my_agg, but don't reset index yet.
    # Need trip ID as index to line up the results of the 
    # forthcoming distance calculation
    h = g.apply(my_agg)  

    # Compute distance
    if ut.is_not_null(f, 'shape_dist_traveled') and\
      not compute_dist_from_shapes:
        # Compute distances using shape_dist_traveled column
        h['distance'] = g.apply(
          lambda group: group['shape_dist_traveled'].max())
    elif feed.shapes is not None:
        # Compute distances using the shapes and Shapely
        geometry_by_shape = build_geometry_by_shape(feed, use_utm=True)
        geometry_by_stop = build_geometry_by_stop(feed, use_utm=True)
        m_to_dist = ut.get_convert_dist('m', feed.dist_units_out)

        def compute_dist(group):
            """
            Return the distance traveled along the trip between the first
            and last stops.
            If that distance is negative or if the trip's linestring 
            intersects itfeed, then return the length of the trip's 
            linestring instead.
            """
            shape = group['shape_id'].iat[0]
            try:
                # Get the linestring for this trip
                linestring = geometry_by_shape[shape]
            except KeyError:
                # Shape ID is NaN or doesn't exist in shapes.
                # No can do.
                return np.nan 
            
            # If the linestring intersects itself, then that can cause
            # errors in the computation below, so just 
            # return the length of the linestring as a good approximation
            D = linestring.length
            if not linestring.is_simple:
                return D

            # Otherwise, return the difference of the distances along
            # the linestring of the first and last stop
            start_stop = group['stop_id'].iat[0]
            end_stop = group['stop_id'].iat[-1]
            try:
                start_point = geometry_by_stop[start_stop]
                end_point = geometry_by_stop[end_stop]
            except KeyError:
                # One of the two stop IDs is NaN, so just
                # return the length of the linestring
                return D
            d1 = linestring.project(start_point)
            d2 = linestring.project(end_point)
            d = d2 - d1
            if 0 < d < D + 100:
                return d
            else:
                # Something is probably wrong, so just
                # return the length of the linestring
                return D

        h['distance'] = g.apply(compute_dist)
        # Convert from meters
        h['distance'] = h['distance'].map(m_to_dist)
    else:
        h['distance'] = np.nan

    # Reset index and compute final stats
    h = h.reset_index()
    h['speed'] = h['distance']/h['duration']
    h[['start_time', 'end_time']] = h[['start_time', 'end_time']].\
      applymap(lambda x: ut.timestr_to_seconds(x, inverse=True))
    
    return h.sort_values(['route_id', 'direction_id', 'start_time'])

def compute_trip_locations(feed, date, times):
    """
    Return a  data frame of the positions of all trips
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

    Assume ``feed.stop_times`` has an accurate ``shape_dist_traveled``
    column.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`get_stop_times`
    - Those used in :func:`build_geometry_by_shape`
        
    """
    if not ut.is_not_null(feed.stop_times, 'shape_dist_traveled'):
        raise ValueError(
          "feed.stop_times needs to have a non-null shape_dist_traveled "\
          "column. You can create it, possibly with some inaccuracies, "\
          "via feed.stop_times = feed.add_dist_to_stop_times().")
    
    # Start with stop times active on date
    f = get_stop_times(feed, date)
    f['departure_time'] = f['departure_time'].map(
      ut.timestr_to_seconds)

    # Compute relative distance of each trip along its path
    # at the given time times.
    # Use linear interpolation based on stop departure times and
    # shape distance traveled.
    geometry_by_shape = build_geometry_by_shape(feed, use_utm=False)
    sample_times = np.array([ut.timestr_to_seconds(s) 
      for s in times])
    
    def compute_rel_dist(group):
        dists = sorted(group['shape_dist_traveled'].values)
        times = sorted(group['departure_time'].values)
        ts = sample_times[(sample_times >= times[0]) &\
          (sample_times <= times[-1])]
        ds = np.interp(ts, times, dists)
        return pd.DataFrame({'time': ts, 'rel_dist': ds/dists[-1]})
    
    # return f.groupby('trip_id', group_keys=False).\
    #   apply(compute_rel_dist).reset_index()
    g = f.groupby('trip_id').apply(compute_rel_dist).reset_index()
    
    # Delete extraneous multi-index column
    del g['level_1']
    
    # Convert times back to time strings
    g['time'] = g['time'].map(
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    # Merge in more trip info and
    # compute longitude and latitude of trip from relative distance
    h = pd.merge(g, feed.trips[['trip_id', 'route_id', 'direction_id', 
      'shape_id']])
    if not h.shape[0]:
        # Return a data frame with the promised headers but no data.
        # Without this check, result below could be an empty data frame.
        h['lon'] = pd.Series()
        h['lat'] = pd.Series()
        return h

    def get_lonlat(group):
        shape = group['shape_id'].iat[0]
        linestring = geometry_by_shape[shape]
        lonlats = [linestring.interpolate(d, normalized=True).coords[0]
          for d in group['rel_dist'].values]
        group['lon'], group['lat'] = zip(*lonlats)
        return group
    
    return h.groupby('shape_id').apply(get_lonlat)
    
def trip_to_geojson(feed, trip_id, include_stops=False):
    """
    Given a feed and a trip ID (string), return a (decoded) GeoJSON feature collection comprising a Linestring feature of representing the trip's shape.
    If ``include_stops``, then also include one Point feature for each stop  visited by the trip. 
    The Linestring feature will contain as properties all the columns in ``feed.trips`` pertaining to the given trip, and each Point feature will contain as properties all the columns in ``feed.stops`` pertaining    to the stop, except the ``stop_lat`` and ``stop_lon`` properties.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - ``feed.shapes``
    - ``feed.stops``

    """
    # Get the relevant shapes
    t = feed.trips.copy()
    t = t[t['trip_id'] == trip_id].copy()
    shid = t['shape_id'].iat[0]
    geometry_by_shape = build_geometry_by_shape(feed, use_utm=False, 
      shape_ids=[shid])

    if geometry_by_shape is None:
        return

    features = [{
        'type': 'Feature',
        'properties': json.loads(t.to_json(orient='records')),
        'geometry': mapping(LineString(geometry_by_shape[shid])),
        }]

    if include_stops:
        # Get relevant stops and geometrys
        s = get_stops(feed, trip_id=trip_id)
        cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
        s = s[list(cols)].copy()
        stop_ids = s['stop_id'].tolist()
        geometry_by_stop = build_geometry_by_stop(feed, stop_ids=stop_ids)
        features.extend([{
            'type': 'Feature',
            'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
              orient='records')),
            'geometry': mapping(geometry_by_stop[stop_id]),
            } for stop_id in stop_ids])

    return {'type': 'FeatureCollection', 'features': features}

# -------------------------------------
# Functions about routes
# -------------------------------------
def get_routes(feed, date=None, time=None):
    """
    Return the section of ``feed.routes`` that contains
    only routes active on the given date.
    If no date is given, then return all routes.
    If a date and time are given, then return only those routes with
    trips active at that date and time.
    Do not take times modulo 24.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - Those used in :func:`get_trips`
        
    """
    if date is None:
        return feed.routes.copy()

    trips = get_trips(feed, date, time)
    R = trips['route_id'].unique()
    return feed.routes[feed.routes['route_id'].isin(R)]

def compute_route_stats_base(trips_stats_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a subset of the output of ``Feed.compute_trip_stats()``, 
    calculate stats for the routes in that subset.
    
    Return a data frame with the following columns:

    - route_id
    - route_short_name
    - route_type
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
    - min_headway: minimum of the durations (in minutes) mentioned above
    - mean_headway: mean of the durations (in minutes) mentioned above
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
      measured in wunits, that is, 
      whatever distance units are present in trips_stats_subset; 
      contains all ``np.nan`` entries if ``feed.shapes is None``  
    - service_speed: service_distance/service_duration;
      measured in wunits per hour
    - mean_trip_distance: service_distance/num_trips
    - mean_trip_duration: service_duration/num_trips

    If ``split_directions == False``, then remove the direction_id column
    and compute each route's stats, except for headways, using its trips
    running in both directions. 
    In this case, (1) compute max headway by taking the max of the max 
    headways in both directions; 
    (2) compute mean headway by taking the weighted mean of the mean
    headways in both directions. 

    If ``trips_stats_subset`` is empty, return an empty data frame with
    the columns specified above.

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

    if trips_stats_subset.empty:
        return pd.DataFrame(columns=cols)

    # Convert trip start and end times to seconds to ease calculations below
    f = trips_stats_subset.copy()
    f[['start_time', 'end_time']] = f[['start_time', 'end_time']].\
      applymap(ut.timestr_to_seconds)

    headway_start = ut.timestr_to_seconds(headway_start_time)
    headway_end = ut.timestr_to_seconds(headway_end_time)

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
        start, end = ut.get_peak_indices(times, counts)
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
        start, end = ut.get_peak_indices(times, counts)
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
      applymap(lambda x: ut.timestr_to_seconds(x, inverse=True))

    return g

def compute_route_stats(feed, trips_stats, date, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Take ``trips_stats``, which is the output of 
    ``compute_trip_stats()``, cut it down to the subset ``S`` of trips
    that are active on the given date, and then call
    ``compute_route_stats_base()`` with ``S`` and the keyword arguments
    ``split_directions``, ``headway_start_time``, and 
    ``headway_end_time``.

    See ``compute_route_stats_base()`` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`compute_route_stats_base`
        
    NOTES:

    This is a more user-friendly version of ``compute_route_stats_base()``.
    The latter function works without a feed, though.

    Return ``None`` if the date does not lie in this feed's date range.
    """
    # Get the subset of trips_stats that contains only trips active
    # on the given date
    trips_stats_subset = pd.merge(trips_stats, get_trips(feed, date))
    return compute_route_stats_base(trips_stats_subset, 
      split_directions=split_directions,
      headway_start_time=headway_start_time, 
      headway_end_time=headway_end_time)

def compute_route_time_series_base(trips_stats_subset,
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Given a subset of the output of ``Feed.compute_trip_stats()``, 
    calculate time series for the routes in that subset.

    Return a time series version of the following route stats:
    
    - number of trips in service by route ID
    - number of trip starts by route ID
    - service duration in hours by route ID
    - service distance in kilometers by route ID
    - service speed in kilometers per hour

    The time series is a data frame with a timestamp index 
    for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    ``date_label`` is used as the date for the timestamp index.

    The columns of the data frame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['service_distance',
      'service_duration', 'num_trip_starts', 'num_trips', 'service_speed']
    - middle level: name = 'route_id', values = the active routes
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``trips_stats_subset`` is empty, then return an empty data frame
    with the indicator columns.

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
    cols = [
      'service_distance',
      'service_duration', 
      'num_trip_starts', 
      'num_trips', 
      'service_speed',
      ]
    if trips_stats_subset.empty:
        return pd.DataFrame(columns=cols)

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
        return (ut.timestr_to_seconds(x)//60) % (24*60)

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

def compute_route_time_series(feed, trips_stats, date, 
  split_directions=False, freq='5Min'):
    """
    Take ``trips_stats``, which is the output of 
    ``compute_trip_stats()``, cut it down to the subset ``S`` of trips
    that are active on the given date, and then call
    ``compute_route_time_series_base()`` with ``S`` and the given 
    keyword arguments ``split_directions`` and ``freq``
    and with ``date_label = ut.date_to_str(date)``.

    See ``compute_route_time_series_base()`` for a description of the output.

    If there are no active trips on the date, then return ``None``.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_trips`
        

    NOTES:

    This is a more user-friendly version of 
    ``compute_route_time_series_base()``.
    The latter function works without a feed, though.
    """  
    trips_stats_subset = pd.merge(trips_stats, get_trips(feed, date))
    return compute_route_time_series_base(trips_stats_subset, 
      split_directions=split_directions, freq=freq, 
      date_label=date)

def get_route_timetable(feed, route_id, date):
    """
    Return a data frame encoding the timetable
    for the given route ID on the given date.
    The columns are all those in ``feed.trips`` plus those in 
    ``feed.stop_times``.
    The result is sorted by grouping by trip ID and
    sorting the groups by their first departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`
        
    """
    f = get_trips(feed, date)
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

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - ``feed.shapes``
    - ``feed.trips``
    - ``feed.stops``

    """
    # Get the relevant shapes
    t = feed.trips.copy()
    A = t[t['route_id'] == route_id]['shape_id'].unique()
    geometry_by_shape = build_geometry_by_shape(feed, use_utm=False, 
      shape_ids=A)

    if geometry_by_shape is None:
        return

    r = feed.routes.copy()
    features = [{
        'type': 'Feature',
        'properties': json.loads(r[r['route_id'] == route_id].to_json(
          orient='records')),
        'geometry': mapping(MultiLineString(
          [linestring for linestring in geometry_by_shape.values()]))
        }]

    if include_stops:
        # Get relevant stops and geometrys
        s = get_stops(feed, route_id=route_id)
        cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
        s = s[list(cols)].copy()
        stop_ids = s['stop_id'].tolist()
        geometry_by_stop = build_geometry_by_stop(feed, stop_ids=stop_ids)
        features.extend([{
            'type': 'Feature',
            'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
              orient='records')),
            'geometry': mapping(geometry_by_stop[stop_id]),
            } for stop_id in stop_ids])

    return {'type': 'FeatureCollection', 'features': features}

# -------------------------------------
# Functions about stops
# -------------------------------------
def get_stops(feed, date=None, trip_id=None, route_id=None):
    """
    Return ``feed.stops``.
    If a date is given, then restrict the output to stops that are visited by trips active on the given date.
    If a trip ID (string) is given, then restrict the output possibly further to stops that are visited by the trip.
    Eles if a route ID (string) is given, then restrict the output possibly further to stops that are visited by at least one trip on the route.

    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
    - Those used in :func:`get_stop_times`
    - ``feed.routes``    
    """
    s = feed.stops.copy()
    if date is not None:
        A = get_stop_times(feed, date)['stop_id']
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
    return s

def build_geometry_by_stop(feed, use_utm=False, stop_ids=None):
    """
    Return a dictionary with structure
    stop_id -> Shapely point object.
    If ``use_utm == True``, then return each point in
    in UTM coordinates.
    Otherwise, return each point in WGS84 longitude-latitude
    coordinates.
    If a list of stop IDs ``stop_ids`` is given, then only include
    the given stop IDs.

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
            d[stop] = Point(utm.from_latlon(lat, lon)[:2]) 
    else:
        for stop, group in stops.groupby('stop_id'):
            lat, lon = group[['stop_lat', 'stop_lon']].values[0]
            d[stop] = Point([lon, lat]) 
    return d

def geometrize_stops(stops, use_utm=False):
    """
    Given a stops data frame, 
    convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 points 
    instead of 'stop_lon' and 'stop_lat' columns.
    If ``use_utm == True``, then use UTM coordinates for the geometries.

    Requires GeoPandas.
    """
    import geopandas as gpd 


    f = stops.copy()
    s = gpd.GeoSeries([Point(p) for p in 
      stops[['stop_lon', 'stop_lat']].values])
    f['geometry'] = s 
    g = f.drop(['stop_lon', 'stop_lat'], axis=1)
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['stop_lat', 'stop_lon']].values
        crs = ut.get_utm_crs(lat, lon) 
        g = g.to_crs(crs)

    return g

def ungeometrize_stops(geo_stops):
    """
    The inverse of :func:`geometrize_stops`.    
    If ``geo_stops`` is in UTM (has a UTM CRS property),
    then convert UTM coordinates back to WGS84 coordinates,
    """
    f = geo_stops.copy().to_crs(cs.CRS_WGS84)
    f['stop_lon'] = f['geometry'].map(
      lambda p: p.x)
    f['stop_lat'] = f['geometry'].map(
      lambda p: p.y)
    del f['geometry']
    return f

def get_stops_intersecting_polygon(feed, polygon, geo_stops=None):
    """
    Return the slice of ``feed.stops`` that contains all stops
    that intersect the given Shapely Polygon object.
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
    f['hit'] = f['geometry'].intersects(polygon)
    f = f[f['hit']][cols]
    return ungeometrize_stops(f)

def compute_stop_activity(feed, dates):
    """
    Return a  data frame with the columns

    - stop_id
    - ``dates[0]``: 1 if the stop has at least one trip visiting it 
      on ``dates[0]``; 0 otherwise 
    - ``dates[1]``: 1 if the stop has at least one trip visiting it 
      on ``dates[1]``; 0 otherwise 
    - etc.
    - ``dates[-1]``: 1 if the stop has at least one trip visiting it 
      on ``dates[-1]``; 0 otherwise 

    If ``dates`` is ``None`` or the empty list, 
    then return an empty data frame with the column 'stop_id'.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`compute_trip_activity`
        

    """
    if not dates:
        return pd.DataFrame(columns=['stop_id'])

    trips_activity = compute_trip_activity(feed, dates)
    g = pd.merge(trips_activity, feed.stop_times).groupby('stop_id')
    # Pandas won't allow me to simply return g[dates].max().reset_index().
    # I get ``TypeError: unorderable types: datetime.date() < str()``.
    # So here's a workaround.
    for (i, date) in enumerate(dates):
        if i == 0:
            f = g[date].max().reset_index()
        else:
            f = f.merge(g[date].max().reset_index())
    return f

def compute_stop_stats_base(stop_times, trips_subset, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Given a stop times data frame and a subset of a trips data frame,
    return a data frame that provides summary stats about
    the stops in the (inner) join of the two data frames.

    The columns of the output data frame are:

    - stop_id
    - direction_id: present iff ``split_directions == True``
    - num_routes: number of routes visiting stop (in the given direction)
    - num_trips: number of trips visiting stop (in the givin direction)
    - max_headway: maximum of the durations (in minutes) between 
      trip departures at the stop between ``headway_start_time`` and 
      ``headway_end_time`` on the given date
    - min_headway: minimum of the durations (in minutes) mentioned above
    - mean_headway: mean of the durations (in minutes) mentioned above
    - start_time: earliest departure time of a trip from this stop
      on the given date
    - end_time: latest departure time of a trip from this stop
      on the given date

    If ``split_directions == False``, then compute each stop's stats
    using trips visiting it from both directions.

    If ``trips_subset`` is empty, then return an empty data frame
    with the columns specified above.
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

    if trips_subset.empty:
        return pd.DataFrame(columns=cols)

    f = pd.merge(stop_times, trips_subset)

    # Convert departure times to seconds to ease headway calculations
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)

    headway_start = ut.timestr_to_seconds(headway_start_time)
    headway_end = ut.timestr_to_seconds(headway_end_time)

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
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    return result

def compute_stop_stats(feed, date, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    Call ``compute_stop_stats_base()`` with the subset of trips active on 
    the given date and with the keyword arguments ``split_directions``,
    ``headway_start_time``, and ``headway_end_time``.

    See ``compute_stop_stats_base()`` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_timtes``
    - Those used in :func:`get_trips`
        
    NOTES:

    This is a more user-friendly version of ``compute_stop_stats_base()``.
    The latter function works without a feed, though.
    """
    # Get stop times active on date and direction IDs
    return compute_stop_stats_base(feed.stop_times, get_trips(feed, date),
      split_directions=split_directions,
      headway_start_time=headway_start_time, 
      headway_end_time=headway_end_time)

def compute_stop_time_series_base(stop_times, trips_subset, 
  split_directions=False, freq='5Min', date_label='20010101'):
    """
    Given a stop times data frame and a subset of a trips data frame,
    return a data frame that provides summary stats about
    the stops in the (inner) join of the two data frames.

    The time series is a data frame with a timestamp index 
    for a 24-hour period sampled at the given frequency.
    The maximum allowable frequency is 1 minute.
    The timestamp includes the date given by ``date_label``,
    a date string of the form '%Y%m%d'.
    
    The columns of the data frame are hierarchical (multi-index) with

    - top level: name = 'indicator', values = ['num_trips']
    - middle level: name = 'stop_id', values = the active stop IDs
    - bottom level: name = 'direction_id', values = 0s and 1s

    If ``split_directions == False``, then don't include the bottom level.
    
    If ``trips_subset`` is empty, then return an empty data frame
    with the indicator columns.

    NOTES:

    - 'num_trips' should be resampled with ``how=np.sum``
    - To remove the date and seconds from 
      the time series f, do ``f.index = [t.time().strftime('%H:%M') 
      for t in f.index.to_datetime()]``
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
        return (ut.timestr_to_seconds(x)//60) % (24*60)

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

def compute_stop_time_series(feed, date, split_directions=False,
  freq='5Min'):
    """
    Call ``compute_stops_times_series_base()`` with the subset of trips 
    active on the given date and with the keyword arguments
    ``split_directions``and ``freq`` and with ``date_label`` equal to ``date``.
    See ``compute_stop_time_series_base()`` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`
        
    NOTES:

    This is a more user-friendly version of 
    ``compute_stop_time_series_base()``.
    The latter function works without a feed, though.
    """  
    return compute_stop_time_series_base(feed.stop_times, 
      get_trips(feed, date), split_directions=split_directions, 
      freq=freq, date_label=date)

def get_stop_timetable(feed, stop_id, date):
    """
    Return a  data frame encoding the timetable
    for the given stop ID on the given date.
    The columns are all those in ``feed.trips`` plus those in
    ``feed.stop_times``.
    The result is sorted by departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`get_stop_times`
        
    """
    f = get_stop_times(feed, date)
    f = pd.merge(f, feed.trips)
    f = f[f['stop_id'] == stop_id]
    return f.sort_values('departure_time')

def get_stops_in_stations(feed):
    """
    If this feed has station data, that is, ``location_type`` and
    ``parent_station`` columns in ``feed.stops``, then return a 
    data frame that has the same columns as ``feed.stops``
    but only includes stops with parent stations, that is, stops with
    location type 0 or blank and non-blank parent station.
    Otherwise, return an empty data frame with the specified columns.

    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
        
    """
    f = feed.stops
    return f[(f['location_type'] != 1) & (f['parent_station'].notnull())]

def compute_station_stats(feed, date, split_directions=False,
    headway_start_time='07:00:00', headway_end_time='19:00:00'):
    """
    If this feed has station data, that is, ``location_type`` and
    ``parent_station`` columns in ``feed.stops``, then compute
    the same stats that ``feed.compute_stop_stats()`` does, but for
    stations.
    Otherwise, return an empty data frame with the specified columns.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_stops_in_stations`
    - Those used in :func:`get_stop_times`

    """
    # Get stop times of active trips that visit stops in stations
    sis = get_stops_in_stations(feed)
    if sis.empty:
        return sis

    f = get_stop_times(feed, date)
    f = pd.merge(f, sis)

    # Convert departure times to seconds to ease headway calculations
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)

    headway_start = ut.timestr_to_seconds(headway_start_time)
    headway_end = ut.timestr_to_seconds(headway_end_time)

    # Compute stats for each station
    def compute_station_stats(group):
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

    result = g.apply(compute_station_stats).reset_index()

    # Convert start and end times to time strings
    result[['start_time', 'end_time']] =\
      result[['start_time', 'end_time']].applymap(
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    return result

# -------------------------------------
# Functions about shapes
# -------------------------------------
def build_geometry_by_shape(feed, use_utm=False, shape_ids=None):
    """
    Return a dictionary with structure
    shape_id -> Shapely linestring of shape.
    If ``feed.shapes is None``, then return ``None``.
    If ``use_utm == True``, then return each linestring in
    in UTM coordinates.
    Otherwise, return each linestring in WGS84 longitude-latitude
    coordinates.
    If a list of shape IDs ``shape_ids`` is given, then only include
    the given shape IDs.

    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``

    """
    if feed.shapes is None:
        return

    # Note the output for conversion to UTM with the utm package:
    # >>> u = utm.from_latlon(47.9941214, 7.8509671)
    # >>> print u
    # (414278, 5316285, 32, 'T')
    d = {}
    shapes = feed.shapes.copy()
    if shape_ids is not None:
        shapes = shapes[shapes['shape_id'].isin(shape_ids)]

    if use_utm:
        for shape, group in shapes.groupby('shape_id'):
            lons = group['shape_pt_lon'].values
            lats = group['shape_pt_lat'].values
            xys = [utm.from_latlon(lat, lon)[:2] 
              for lat, lon in zip(lats, lons)]
            d[shape] = LineString(xys)
    else:
        for shape, group in shapes.groupby('shape_id'):
            lons = group['shape_pt_lon'].values
            lats = group['shape_pt_lat'].values
            lonlats = zip(lons, lats)
            d[shape] = LineString(lonlats)
    return d

def shapes_to_geojson(feed):
    """
    Return a (decoded) GeoJSON feature collection of 
    linestring features representing ``feed.shapes``.
    Each feature will have a ``shape_id`` property. 
    If ``feed.shapes`` is ``None``, then return ``None``.
    The coordinates reference system is the default one for GeoJSON,
    namely WGS84.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`build_geometry_by_shape`

    """

    geometry_by_shape = build_geometry_by_shape(feed, use_utm=False)
    if geometry_by_shape is None:
        return

    return {
      'type': 'FeatureCollection', 
      'features': [{
        'properties': {'shape_id': shape},
        'type': 'Feature',
        'geometry': mapping(linestring),
        }
        for shape, linestring in geometry_by_shape.items()]
      }

def geometrize_shapes(shapes, use_utm=False):
    """
    Given a shapes data frame, convert it to a GeoPandas 
    GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 line strings
    instead of 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat',  
    and 'shape_dist_traveled' columns.
    If ``use_utm == True``, then use UTM coordinates for the geometries.

    Requires GeoPandas.
    """
    import geopandas as gpd


    f = shapes.copy().sort_values(['shape_id', 'shape_pt_sequence'])
    
    def my_agg(group):
        d = {}
        d['geometry'] =\
          LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
        return pd.Series(d)

    g = f.groupby('shape_id').apply(my_agg).reset_index()
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
        crs = ut.get_utm_crs(lat, lon) 
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

    If ``geo_shapes`` is in UTM (has a UTM CRS property),
    then convert UTM coordinates back to WGS84 coordinates,
    """
    geo_shapes = geo_shapes.to_crs(cs.CRS_WGS84)

    F = []
    for index, row in geo_shapes.iterrows():
        F.extend([[row['shape_id'], i, x, y] for 
        i, (x, y) in enumerate(row['geometry'].coords)])

    return pd.DataFrame(F, 
      columns=['shape_id', 'shape_pt_sequence', 
      'shape_pt_lon', 'shape_pt_lat'])

def get_shapes_intersecting_geometry(feed, geometry, geo_shapes=None,
  geometrized=False):
    """
    Return the slice of ``feed.shapes`` that contains all shapes
    that intersect the given Shapely geometry object 
    (e.g. a Polygon or LineString).
    Assume the geometry is specified in WGS84 longitude-latitude coordinates.
    
    To do this, first geometrize ``feed.shapes`` via :func:`geometrize_shapes`.
    Alternatively, use the ``geo_shapes`` GeoDataFrame, if given.
    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``, if ``geo_shapes`` is not given

    If ``geometrized`` is ``True``, then return the 
    resulting shapes data frame in geometrized form.
    """
    if geo_shapes is not None:
        f = geo_shapes.copy()
    else:
        f = geometrize_shapes(feed.shapes)
    
    cols = f.columns
    f['hit'] = f['geometry'].intersects(geometry)
    f = f[f['hit']][cols]

    if geometrized:
        return f
    else:
        return ungeometrize_shapes(f)

def add_dist_to_shapes(feed):
    """
    Copy ``feed.shapes``, calculate the optional ``shape_dist_traveled`` 
    GTFS field, and return the resulting shapes data frame.

    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``

    NOTE: 

    All of the calculated ``shape_dist_traveled`` values 
    for the Portland feed differ by at most 0.016 km in absolute values
    from of the original values. 
    """
    if feed.shapes is None:
        raise ValueError(
          "This function requires the feed to have a shapes.txt file")

    f = feed.shapes
    m_to_dist = ut.get_convert_dist('m', feed.dist_units_out)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        group = group.sort_values('shape_pt_sequence')
        shape = group['shape_id'].iat[0]
        if not isinstance(shape, str):
            group['shape_dist_traveled'] = np.nan 
            return group
        points = [Point(utm.from_latlon(lat, lon)[:2]) 
          for lon, lat in group[['shape_pt_lon', 'shape_pt_lat']].values]
        p_prev = points[0]
        d = 0
        distances = [0]
        for  p in points[1:]:
            d += p.distance(p_prev)
            distances.append(d)
            p_prev = p
        group['shape_dist_traveled'] = distances
        return group

    g = f.groupby('shape_id', group_keys=False).apply(compute_dist)
    # Convert from meters
    g['shape_dist_traveled'] = g['shape_dist_traveled'].map(m_to_dist)

    return g

def add_route_type_to_shapes(feed):
    """
    Append a ``route_type`` column to a copy of ``feed.shapes`` and return
    the resulting shapes data frame.

    Note that a single shape can be linked to multiple trips on 
    multiple routes of multiple route types.
    In that case the route type of the shape is the route type of the last
    route (sorted by ID) with a trip with that shape.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - ``feed.trips``
    - ``feed.shapes``

    """        
    f = pd.merge(feed.routes, feed.trips).sort_values(['shape_id', 'route_id'])
    rtype_by_shape = dict(f[['shape_id', 'route_type']].values)
    
    g = feed.shapes.copy()
    g['route_type'] = g['shape_id'].map(lambda x: rtype_by_shape[x])
    
    return g

# -------------------------------------
# Functions about stop times
# -------------------------------------
def get_stop_times(feed, date=None):
    """
    Return the section of ``feed.stop_times`` that contains
    only trips active on the given date.
    If no date is given, then return all stop times.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`

    """
    f = feed.stop_times.copy()
    if date is None:
        return f

    g = get_trips(feed, date)
    return f[f['trip_id'].isin(g['trip_id'])]

def get_start_and_end_times(feed, date=None):
    """
    Return the first departure time and last arrival time (time strings)
    listed in ``feed.stop_times``, respectively.
    Restrict to the given date if specified.
    Return 
    """
    st = get_stop_times(feed, date)
    # if st.empty:
    #     a, b = np.nan, np.nan 
    # else:
    a, b = st['departure_time'].dropna().min(),\
       st['arrival_time'].dropna().max()
    return a, b 

def add_dist_to_stop_times(feed, trips_stats):
    """
    Copy ``feed.stop_times``, compute its optional 
    ``shape_dist_traveled`` GTFS field, and return the resulting
    data frame.
    Does not always give accurate results, as described below.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`build_geometry_by_shape`
    - Those used in :func:`build_geometry_by_stop`

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
    This fallback method usually kicks in on trips with feed-intersecting
    linestrings.
    Unfortunately, this fallback method will produce incorrect results
    when the first stop does not start at the start of its shape
    (so shape_dist_traveled != 0).
    This is the case for several trips in the Portland feed, for example. 
    """
    geometry_by_shape = build_geometry_by_shape(feed, use_utm=True)
    geometry_by_stop = build_geometry_by_stop(feed, use_utm=True)

    # Initialize data frame
    f = pd.merge(feed.stop_times,
      trips_stats[['trip_id', 'shape_id', 'distance', 'duration']]).\
      sort_values(['trip_id', 'stop_sequence'])

    # Convert departure times to seconds past midnight to ease calculations
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)
    dist_by_stop_by_shape = {shape: {} for shape in geometry_by_shape}
    m_to_dist = ut.get_convert_dist('m', feed.dist_units_out)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        trip = group['trip_id'].iat[0]
        shape = group['shape_id'].iat[0]
        if not isinstance(shape, str):
            group['shape_dist_traveled'] = np.nan 
            return group
        elif np.isnan(group['distance'].iat[0]):
            group['shape_dist_traveled'] = np.nan 
            return group
        linestring = geometry_by_shape[shape]
        distances = []
        for stop in group['stop_id'].values:
            if stop in dist_by_stop_by_shape[shape]:
                d = dist_by_stop_by_shape[shape][stop]
            else:
                d = m_to_dist(ut.get_segment_length(linestring, 
                  geometry_by_stop[stop]))
                dist_by_stop_by_shape[shape][stop] = d
            distances.append(d)
        s = sorted(distances)
        D = linestring.length
        distances_are_reasonable = all([d < D + 100 for d in distances])
        if distances_are_reasonable and s == distances:
            # Good
            pass
        elif distances_are_reasonable and s == distances[::-1]:
            # Reverse. This happens when the direction of a linestring
            # opposes the direction of the bus trip.
            distances = distances[::-1]
        else:
            # Totally redo using trip length, first and last stop times,
            # and linear interpolation
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

    result = f.groupby('trip_id', group_keys=False).apply(compute_dist)
    # Convert departure times back to time strings
    result['departure_time'] = result['departure_time'].map(lambda x: 
      ut.timestr_to_seconds(x, inverse=True))
    
    return result.drop(['shape_id', 'distance', 'duration'], axis=1)

# -------------------------------------
# Functions about feeds
# -------------------------------------
def convert_dist(feed, new_dist_units):
    """
    Convert the distances recorded in the ``shape_dist_traveled`` columns of the given feed from the feeds native distance units (recorded in ``feed.dist_units``) to the given new distance units.
    New distance units must lie in ``constants.DIST_UNITS``
    """       
    feed = feed.copy()

    if feed.dist is None or feed.dist == new_dist_units:
        # Nothing to do
        return feed

    if new_dist_units not in cs.DIST_UNITS:
        raise ValueError('Given distance units must lie in {!s}'.format(
          cs.DIST_UNITS))

    converter = ut.get_convert_dist(feed.dist_units, new_dist_units)

    if ut.is_not_null(feed.stop_times, 'shape_dist_traveled'):
        feed.stop_times['shape_dist_traveled'] =\
          feed.stop_times['shape_dist_traveled'].map(converter)

    if ut.is_not_null(feed.shapes, 'shape_dist_traveled'):
        feed.shapes['shape_dist_traveled'] =\
          feed.shapes['shape_dist_traveled'].map(converter)

    feed.dist_units = new_dist_units

    return feed

def compute_feed_stats(feed, trips_stats, date):
    """
    Given ``trips_stats``, which is the output of 
    ``feed.compute_trip_stats()`` and a date,
    return a  data frame including the following feed
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

    If there are no stats for the given date, return an empty data frame
    with the specified columns.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_trips`
    - Those used in :func:`get_routes`
    - Those used in :func:`get_stops`

    """
    cols = [
      'num_trips',
      'num_routes',
      'num_stops',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    d = OrderedDict()
    trips = get_trips(feed, date)
    if trips.empty:
        return pd.DataFrame(columns=cols)

    d['num_trips'] = trips.shape[0]
    d['num_routes'] = get_routes(feed, date).shape[0]
    d['num_stops'] = get_stops(feed, date).shape[0]

    # Compute peak stats
    f = trips.merge(trips_stats)
    f[['start_time', 'end_time']] =\
      f[['start_time', 'end_time']].applymap(ut.timestr_to_seconds)

    times = np.unique(f[['start_time', 'end_time']].values)
    counts = [count_active_trips(f, t) for t in times]
    start, end = ut.get_peak_indices(times, counts)
    d['peak_num_trips'] = counts[start]
    d['peak_start_time'] =\
      ut.timestr_to_seconds(times[start], inverse=True)
    d['peak_end_time'] =\
      ut.timestr_to_seconds(times[end], inverse=True)

    # Compute remaining stats
    d['service_distance'] = f['distance'].sum()
    d['service_duration'] = f['duration'].sum()
    d['service_speed'] = d['service_distance']/d['service_duration']

    return pd.DataFrame(d, index=[0])

def compute_feed_time_series(feed, trips_stats, date, freq='5Min'):
    """
    Given trips stats (output of ``feed.compute_trip_stats()``),
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
    return an empty data frame with specified columns.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`compute_route_time_series`

    """
    cols = [
      'num_trip_starts',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    rts = compute_route_time_series(feed, trips_stats, date, freq=freq)
    if rts.empty:
        return pd.DataFrame(columns=cols)

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

def create_shapes(feed, all_trips=False):
    """
    Given a feed, create a shape for every trip that is missing a shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips data frames.

    If ``all_trips == True``, then create new shapes for all trips 
    by connecting stops, and remove the old shapes.
    
    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``
    """
    feed = feed.copy()

    if all_trips:
        trip_ids = feed.trips['trip_id']
    else:
        trip_ids = feed.trips[feed.trips['shape_id'].isnull()]['trip_id']

    # Get stop times for given trips
    f = feed.stop_times[feed.stop_times['trip_id'].isin(trip_ids)][
      ['trip_id', 'stop_sequence', 'stop_id']]
    f = f.sort_values(['trip_id', 'stop_sequence'])

    if f.empty:
        # Nothing to do
        return feed 

    # Create new shape IDs for given trips.
    # To do this, collect unique stop sequences, 
    # sort them to impose a canonical order, and 
    # assign shape IDs to them
    stop_seqs = sorted(set(tuple(group['stop_id'].values) 
      for trip, group in f.groupby('trip_id')))
    d = int(math.log10(len(stop_seqs))) + 1  # Digits for padding shape IDs  
    shape_by_stop_seq = {seq: 'shape_{num:0{pad}d}'.format(num=i, pad=d) 
      for i, seq in enumerate(stop_seqs)}
 
    # Assign these new shape IDs to given trips 
    shape_by_trip = {trip: shape_by_stop_seq[tuple(group['stop_id'].values)] 
      for trip, group in f.groupby('trip_id')}
    trip_cond = feed.trips['trip_id'].isin(trip_ids)
    feed.trips.loc[trip_cond, 'shape_id'] = feed.trips.loc[trip_cond,
      'trip_id'].map(lambda x: shape_by_trip[x])

    # Build new shapes for given trips
    G = [[shape, i, stop] for stop_seq, shape in shape_by_stop_seq.items() 
      for i, stop in enumerate(stop_seq)]
    g = pd.DataFrame(G, columns=['shape_id', 'shape_pt_sequence', 
      'stop_id'])
    g = g.merge(feed.stops[['stop_id', 'stop_lon', 'stop_lat']]).sort_values(
      ['shape_id', 'shape_pt_sequence'])
    g = g.drop(['stop_id'], axis=1)
    g = g.rename(columns={
      'stop_lon': 'shape_pt_lon',
      'stop_lat': 'shape_pt_lat',
      })

    if feed.shapes is not None and not all_trips:
        # Update feed shapes with new shapes
        feed.shapes = pd.concat([feed.shapes, g])
    else:
        # Create all new shapes
        feed.shapes = g

    return feed

def restrict_by_routes(feed, route_ids):
    """
    Build a new feed by taking the given one and chopping it down to
    only the stops, trips, shapes, etc. used by the routes specified
    in the given list of route IDs. 
    Return the resulting feed.
    """
    # Initialize the new feed as the old feed.
    # Restrict its data frames below.
    feed = feed.copy()
    
    # Slice routes
    feed.routes = feed.routes[feed.routes['route_id'].isin(route_ids)].copy()

    # Slice trips
    feed.trips = feed.trips[feed.trips['route_id'].isin(route_ids)].copy()

    # Slice stop times
    trip_ids = feed.trips['trip_id']
    feed.stop_times = feed.stop_times[
      feed.stop_times['trip_id'].isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times['stop_id'].unique()
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()

    # Slice calendar
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()
    
    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()
            
    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates[
          feed.calendar_dates['service_id'].isin(service_ids)].copy()
    
    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()
        
    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()
        
    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers[
          feed.transfers['from_stop_id'].isin(stop_ids) |\
          feed.transfers['to_stop_id'].isin(stop_ids)].copy()
        
    return feed

def restrict_by_polygon(feed, polygon):
    """
    Build a new feed by taking the given one, keeping only the trips 
    that have at least one stop intersecting the given polygon, and then
    restricting stops, routes, stop times, etc. to those associated with 
    that subset of trips. 
    Return the resulting feed.
    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``
    - ``feed.routes``
    - Those used in :func:`get_stops_intersecting_polygon`

    """
    # Initialize the new feed as the old feed.
    # Restrict its data frames below.
    feed = feed.copy()
    
    # Get IDs of stops within the polygon
    stop_ids = get_stops_intersecting_polygon(
      feed, polygon)['stop_id']
        
    # Get all trips that stop at at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st[st['stop_id'].isin(stop_ids)]['trip_id']
    feed.trips = feed.trips[feed.trips['trip_id'].isin(trip_ids)].copy()
    
    # Get stop times for trips
    feed.stop_times = st[st['trip_id'].isin(trip_ids)].copy()
    
    # Get stops for trips
    stop_ids = feed.stop_times['stop_id']
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()
    
    # Get routes for trips
    route_ids = feed.trips['route_id']
    feed.routes = feed.routes[feed.routes['route_id'].isin(route_ids)].copy()
    
    # Get calendar for trips
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()
    
    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()
            
    # Now for the optional files.
    # Get calendar dates for trips.
    cd = feed.calendar_dates
    if cd is not None:
        feed.calendar_dates = cd[cd['service_id'].isin(service_ids)].copy()
    
    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()
        
    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()
        
    # Get transfers for stops
    if feed.transfers is not None:
        t = feed.transfers
        feed.transfers = t[t['from_stop_id'].isin(stop_ids) |\
          t['to_stop_id'].isin(stop_ids)].copy()
        
    return feed

# -------------------------------------
# Miscellaneous functions
# -------------------------------------
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

@ut.time_it
def compute_screen_line_counts(feed, linestring, date, geo_shapes=None):
    """
    Compute all the trips active in the given feed on the given date 
    that intersect the given Shapely LineString 
    (with WGS84 longitude-latitude coordinates),
    and return a data frame with the columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'crossing_time'``: time that the trip's vehicle crosses the linestring;
      one trip could cross multiple times
    - ``'orientation'``: 1 or -1; 1 indicates trip travel from the left side
     to the right side of the screen line; -1 indicates trip travel in the 
     opposite direction

    Requires GeoPandas.

    The first step is to geometrize ``feed.shapes`` via
    :func:`geometrize_shapes`.
    Alternatively, use the ``geo_shapes`` GeoDataFrame, if given.

    Assume ``feed.stop_times`` has an accurate ``shape_dist_traveled``
    column.
    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``, if ``geo_shapes`` is not given

    Assume that trips travel in the same direction as their shapes.
    That restriction is part of GTFS, by the way.
    To calculate direction quickly and accurately, assume that the 
    screen line is straight and doesn't double back on itself.

    WARNING:

    Probably does not give correct results for trips with self-intersecting
    shapes.
    
    ALGORITHM:

    Compute all the shapes that intersect the linestring.
    For each such shape, compute the intersection points.
    For each point p, scan through all the trips in the feed that have 
    that shape and are active on the given date.
    Interpolate a stop time for p by assuming that the feed has
    the shape_dist_traveled field in stop times.
    Use that interpolated time as the crossing time of the trip vehicle,
    and compute the trip orientation to the screen line via a cross product
    of a vector in the direction of the screen line and a tiny vector in the 
    direction of trip travel.
    """  
    # Get all shapes that intersect the screen line
    shapes = get_shapes_intersecting_geometry(feed, linestring, geo_shapes,
      geometrized=True)

    # Convert shapes to UTM
    lat, lon = feed.shapes.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
    crs = ut.get_utm_crs(lat, lon) 
    shapes = shapes.to_crs(crs)

    # Convert linestring to UTM
    linestring = ut.linestring_to_utm(linestring)

    # Get all intersection points of shapes and linestring
    shapes['intersection'] = shapes.intersection(linestring)

    # Make a vector in the direction of the screen line
    # to later calculate trip orientation.
    # Does not work in case of a bent screen line.
    p1 = Point(linestring.coords[0])
    p2 = Point(linestring.coords[-1])
    w = np.array([p2.x - p1.x, p2.y - p1.y])

    # Build a dictionary from the shapes data frame of the form
    # shape ID -> list of pairs (d, v), one for each intersection point, 
    # where d is the distance of the intersection point along shape,
    # and v is a tiny vectors from the point in direction of shape.
    # Assume here that trips travel in the same direction as their shapes.
    dv_by_shape = {}
    eps = 1
    convert_dist = ut.get_convert_dist('m', feed.dist_units_out)
    for __, sid, geom, intersection in shapes.itertuples():
        # Get distances along shape of intersection points (in meters)
        distances = [geom.project(p) for p in intersection]
        # Build tiny vectors
        vectors = []
        for i, p in enumerate(intersection):
            q = geom.interpolate(distances[i] + eps)
            vector = np.array([q.x - p.x, q.y - p.y])
            vectors.append(vector)
        # Convert distances to units used in feed
        distances = [convert_dist(d) for d in distances]
        dv_by_shape[sid] = list(zip(distances, vectors))

    # Get trips with those shapes that are active on the given date
    trips = get_trips(feed, date)
    trips = trips[trips['shape_id'].isin(dv_by_shape.keys())]

    # Merge in route short names
    trips = trips.merge(feed.routes[['route_id', 'route_short_name']])

    # Merge in stop times
    f = trips.merge(feed.stop_times)

    # Drop NaN departure times and convert to seconds past midnight
    f = f[f['departure_time'].notnull()]
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)

    # For each shape find the trips that cross the screen line
    # and get crossing times and orientation
    f = f.sort_values(['trip_id', 'stop_sequence'])
    G = []  # output table
    for tid, group in f.groupby('trip_id'):
        sid = group['shape_id'].iat[0] 
        rid = group['route_id'].iat[0]
        rsn = group['route_short_name'].iat[0]
        stop_times = group['departure_time'].values
        stop_distances = group['shape_dist_traveled'].values
        for d, v in dv_by_shape[sid]:
            # Interpolate crossing time
            t = np.interp(d, stop_distances, stop_times)
            # Compute direction of trip travel relative to
            # screen line by looking at the sign of the cross
            # product of tiny shape vector and screen line vector
            det = np.linalg.det(np.array([v, w]))
            if det >= 0:
                orientation = 1
            else:
                orientation = -1
            # Update G
            G.append([tid, rid, rsn, t, orientation])
    
    # Create data frame
    g = pd.DataFrame(G, columns=['trip_id', 'route_id', 
      'route_short_name', 'crossing_time', 'orientation']
      ).sort_values('crossing_time')

    # Convert departure times to time strings
    g['crossing_time'] = g['crossing_time'].map(
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    return g

def compute_bounds(feed):   
    """
    Return the tuple 
    (min longitude, min latitude, max longitude, max latitude)
    where the longitudes and latitude vary across all the stop (WGS84) 
    coordinates.
    """
    lons, lats = feed.stops['stop_lon'], feed.stops['stop_lat']
    return lons.min(), lats.min(), lons.max(), lats.max()
    
def compute_center(feed, num_busiest_stops=None):
    """
    Compute the convex hull of all the given feed's stop coordinates and
    return the centroid.
    If an integer ``num_busiest_stops`` is given, then compute
    the ``num_busiest_stops`` busiest stops in the feed on the first
    Monday of the feed and return the mean of the longitudes and the 
    mean of the latitudes of these stops, respectively.
    """
    s = feed.stops.copy()
    if num_busiest_stops is not None:
        n = num_busiest_stops
        date = get_first_week(feed)[0]
        ss = compute_stop_stats(feed, date).sort_values(
          'num_trips', ascending=False)
        f = ss.head(num_busiest_stops)
        f = s.merge(f)
        lon = f['stop_lon'].mean()
        lat = f['stop_lat'].mean()
    else:
        m = MultiPoint(s[['stop_lon', 'stop_lat']].values)
        lon, lat = list(m.convex_hull.centroid.coords)[0]
    return lon, lat