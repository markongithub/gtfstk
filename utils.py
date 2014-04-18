from __future__ import print_function, division
import datetime as dt
from itertools import izip
import dateutil.relativedelta as rd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString, mapping
import utm

def date_to_str(date, format_str='%Y%m%d', inverse=False):
    """
    Given a date object, convert it to a string in the given format
    (which defaults to 'dd-mm-yyyy') and return the result.
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

GTFS_PATH = '/Users/araichev/gtfs_toolkit/test/darwin/'
#GTFS_PATH = 'test/seq_20140128/'
shapes = pd.read_csv(GTFS_PATH + 'shapes.txt', dtype={'shape_id': str})
trips = pd.read_csv(GTFS_PATH + 'trips.txt', dtype={'shape_id': str})
stop_times = pd.read_csv(GTFS_PATH + 'stop_times.txt', dtype={'stop_id': str})
stops = pd.read_csv(GTFS_PATH + 'stops.txt', dtype={'stop_id': str})
calendar = pd.read_csv(GTFS_PATH + 'calendar.txt', dtype={'service_id': str}, 
  date_parser=lambda x:date_to_str(x, inverse=True), 
  parse_dates=['start_date', 'end_date'])

def weekday_to_str(weekday, inverse=False):
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

def seconds_to_timestr(seconds, inverse=False):
    """
    Return the given number of integer seconds as the time string 'HH:MM:SS'.
    If ``inverse == True``, then do the inverse operation.
    """
    if seconds is None:
        return None
    if not inverse:
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        result = '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
    else:
        try:
            hours, mins, seconds = seconds.split(':')
            result = int(hours)*3600 + int(mins)*60 + int(seconds)
        except:
            result = None
    return result

def get_trip_activity(dates=None, ndigits=3):
    merged = pd.merge(trips[['trip_id', 'service_id']], calendar)
    if dates is None:
        # Use the full range of dates for which the feed is valid
        feed_start = merged['start_date'].min()
        feed_end = merged['end_date'].max()
        num_days =  (feed_end - feed_start).days
        dates = [feed_start + rd.relativedelta(days=+d) for d in range(num_days + 1)]
        activity_header = 'fraction of dates active during %s--%s' %\
          (date_to_str(feed_start), date_to_str(feed_end))
    else:
        activity_header = 'fraction of dates active among %s' % '-'.join([date_to_str(d) for d in dates])
    merged[activity_header] = pd.Series()

    n = len(dates)
    for index, row in merged.iterrows():    
        trip = row['trip_id']
        start_date = row['start_date']
        end_date = row['end_date']
        weekdays_running = [i for i in range(7) if row[weekday_to_str(i)] == 1]
        count = 0
        for date in dates:
            if not start_date <= date <= end_date:
                continue
            if date.weekday() in weekdays_running:
                count += 1
        merged.ix[index, activity_header] = round(count/n, ndigits)
    return merged
  
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

def get_linestring_by_shape():
    """
    Return a dictionary with structure
    shape_id -> Shapely linestring of shape in UTM coordinates.
    """
    # Note the output for conversion to UTM with the utm package:
    # >>> u = utm.from_latlon(47.9941214, 7.8509671)
    # >>> print u
    # (414278, 5316285, 32, 'T')
    linestring_by_shape = {}
    for shape_id, group in shapes.groupby('shape_id'):
        lons = group['shape_pt_lon'].values
        lats = group['shape_pt_lat'].values
        xys = [utm.from_latlon(lat, lon)[:2] 
          for lat, lon in izip(lats, lons)]
        linestring_by_shape[shape_id] = LineString(xys)
    return linestring_by_shape

def get_xy_by_stop():
    """
    Return a dictionary with structure
    stop_id -> stop location as a UTM coordinate pair
    """
    xy_by_stop = {}
    for stop_id, group in stops.groupby('stop_id'):
        lat, lon = group[['stop_lat', 'stop_lon']].values[0]
        xy_by_stop[stop_id] = utm.from_latlon(lat, lon)[:2] 
    return xy_by_stop

def get_shapes_with_shape_dist_traveled():
    t1 = dt.datetime.now()
    print(t1, 'Adding shape_dist_traveled field to %s shape points...' %\
      shapes['shape_id'].count())

    new_shapes = shapes.copy()
    new_shapes['shape_dist_traveled'] = pd.Series()
    for shape_id, group in shapes.groupby('shape_id'):
        group.sort('shape_pt_sequence')
        lons = group['shape_pt_lon'].values
        lats = group['shape_pt_lat'].values
        xys = [utm.from_latlon(lat, lon)[:2] 
          for lat, lon in izip(lats, lons)]
        xy_prev = xys[0]
        d_prev = 0
        index = group.index[0]
        for (i, xy) in enumerate(xys):
            p = Point(xy_prev)
            q = Point(xy)
            d = p.distance(q)
            new_shapes.ix[index + i, 'shape_dist_traveled'] =\
              round(d_prev + d, 2)
            xy_prev = xy
            d_prev += d

    t2 = dt.datetime.now()
    minutes = (t2 - t1).seconds/60
    print(t2, 'Finished in %.2f min' % minutes)    
    return new_shapes

def get_stop_times_with_shape_dist_traveled():
    t1 = dt.datetime.now()
    print(t1, 'Adding shape_dist_traveled field to %s stop times...' %\
      stop_times['trip_id'].count())

    linestring_by_shape = get_linestring_by_shape()
    xy_by_stop = get_xy_by_stop()

    failures = []

    # Initialize data frame
    merged = pd.merge(trips[['trip_id', 'shape_id']], stop_times)
    merged['shape_dist_traveled'] = pd.Series()

    # Compute shape_dist_traveled column entries
    dist_by_stop_by_shape = {shape: {} for shape in linestring_by_shape}
    for shape, group in merged.groupby('shape_id'):
        linestring = linestring_by_shape[shape]
        for trip, subgroup in group.groupby('trip_id'):
            # Compute the distances of the stops along this trip
            subgroup.sort('stop_sequence')
            stops = subgroup['stop_id'].values
            distances = []
            for stop in stops:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = round(
                      get_segment_length(linestring, xy_by_stop[stop]), 2)
                    dist_by_stop_by_shape[shape][stop] = d
                distances.append(d)
            if distances[0] > distances[1]:
                # This happens when the shape linestring direction is the
                # opposite of the trip direction. Reverse the distances.
                distances = distances[::-1]
            if distances != sorted(distances):
                # Uh oh
                failures.append(trip)
            # Insert stop distances
            index = subgroup.index[0]
            for (i, d) in enumerate(distances):
                merged.ix[index + i, 'shape_dist_traveled'] = d
    
    # Clean up
    del merged['shape_id']
    t2 = dt.datetime.now()
    minutes = (t2 - t1).seconds/60
    print(t2, 'Finished in %.2f min' % minutes)
    print('%s failures on these trips: %s' % (len(failures), failures))    
    return merged

def get_trip_stats():
    t1 = dt.datetime.now()
    print(t1, 'Creating trip stats for %s trips...' % trips['trip_id'].count())

    linestring_by_shape = get_linestring_by_shape()
    xy_by_stop = get_xy_by_stop()

    # Initialize data frame. Base it on trips.txt.
    trip_stats = trips[['route_id', 'trip_id', 'direction_id']]
    trip_stats.set_index('trip_id', inplace=True)
    trip_stats['start_time'] = pd.Series()
    trip_stats['end_time'] = pd.Series()
    trip_stats['start_stop'] = pd.Series()
    trip_stats['end_stop'] = pd.Series()
    trip_stats['duration (seconds)'] = pd.Series()
    trip_stats['distance (meters)'] = pd.Series()

    # Compute data frame values
    dist_by_stop_pair_by_shape = {shape: {} for shape in linestring_by_shape}
    for trip_id, group in pd.merge(trips, stop_times).groupby('trip_id'):
        # Compute trip time stats
        dep = group[['departure_time','stop_id']]
        dep = dep[dep['departure_time'].notnull()]
        argmin = dep['departure_time'].argmin()
        argmax = dep['departure_time'].argmax()
        start_time = dep['departure_time'].iat[argmin]
        end_time = dep['departure_time'].iat[argmax]
        start_stop = dep['stop_id'].iat[argmin] 
        end_stop = dep['stop_id'].iat[argmax] 
        
        # Compute trip distance
        shape = group['shape_id'].values[0]
        stop_pair = frozenset([start_stop, end_stop])
        if stop_pair in dist_by_stop_pair_by_shape[shape]:
            d = dist_by_stop_pair_by_shape[shape][stop_pair]
        else:
            # Compute distance afresh and store
            linestring = linestring_by_shape[shape]
            p = xy_by_stop[start_stop]
            q = xy_by_stop[end_stop]
            d = get_segment_length(linestring, p, q) 
            if d == 0:
                # Trip is a circuit. 
                # This can even happen when start_stop != end_stop and when the two stops are very close together
                d = linestring.length            
            dist_by_stop_pair_by_shape[shape][stop_pair] = round(d, 2)
            
        # Store stats
        trip_stats.ix[trip_id, 'start_time'] = start_time
        trip_stats.ix[trip_id, 'end_time'] = end_time
        trip_stats.ix[trip_id, 'duration (seconds)'] =\
          seconds_to_timestr(end_time, inverse=True) -\
          seconds_to_timestr(start_time, inverse=True)
        trip_stats.ix[trip_id, 'start_stop'] = start_stop
        trip_stats.ix[trip_id, 'end_stop'] = end_stop
        trip_stats.ix[trip_id, 'distance (meters)'] = d
    trip_stats.sort('route_id')
    t2 = dt.datetime.now()
    minutes = (t2 - t1).seconds/60
    print(t2, 'Finished in %.2f min' % minutes)    
    return trip_stats.reset_index()

if __name__ == '__main__':
    dates = [dt.date(2012, 11, 5), dt.date(2012, 11, 6), dt.date(2012, 11, 10)]
    trip_activity = get_trip_activity()
    trip_activity.to_csv(GTFS_PATH + 'trip_activity.csv', index=False)
    # new_shapes = get_shapes_with_shape_dist_traveled()
    # new_shapes.to_csv(GTFS_PATH + 'new_shapes.txt', index=False)
    # new_stop_times = get_stop_times_with_shape_dist_traveled()
    # new_stop_times.to_csv(GTFS_PATH + 'new_stop_times.txt', index=False)
    # trip_stats = get_trip_stats()
    # trip_stats.to_csv(GTFS_PATH + 'trip_stats.csv', index=False)
