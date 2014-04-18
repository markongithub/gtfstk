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


class Feed(object):

    def __init__(self, path):
        self.filepath = path 
        self.shapes = pd.read_csv(path + 'shapes.txt', dtype={'shape_id': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'shape_id': str})
        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str})
        self.calendar = pd.read_csv(path + 'calendar.txt', 
          dtype={'service_id': str}, 
          date_parser=lambda x: date_to_str(x, inverse=True), 
          parse_dates=['start_date', 'end_date'])
        # Indexed by trip_id
        self.calendar_m = pd.merge(self.trips[['trip_id', 'service_id']],
          self.calendar).set_index('trip_id')
        self.dates = None
        # Indexed by trip_id
        self.dateless_trip_stats = None
        
    def get_stop_times(self):
        """
        Big data frame and don't use it enough to warrant storing as an
        attribute.
        """
        return pd.read_csv(self.path + 'stop_times.txt', 
          dtype={'stop_id': str})

    def is_trip_active(self, trip, date):
        cal = self.calendar_m
        row = cal.ix[trip]
        start_date = row['start_date']
        end_date = row['end_date']
        weekdays_running = [i for i in range(7) 
          if row[weekday_to_str(i)] == 1]
        if start_date <= date <= end_date and\
          date.weekday() in weekdays_running:
            return True
        else:
            return False

    def get_linestring_by_shape(self):
        """
        Return a dictionary with structure
        shape_id -> Shapely linestring of shape in UTM coordinates.
        """
        # Note the output for conversion to UTM with the utm package:
        # >>> u = utm.from_latlon(47.9941214, 7.8509671)
        # >>> print u
        # (414278, 5316285, 32, 'T')
        linestring_by_shape = {}
        for shape, group in self.shapes.groupby('shape_id'):
            lons = group['shape_pt_lon'].values
            lats = group['shape_pt_lat'].values
            xys = [utm.from_latlon(lat, lon)[:2] 
              for lat, lon in izip(lats, lons)]
            linestring_by_shape[shape] = LineString(xys)
        return linestring_by_shape

    def get_xy_by_stop(self):
        """
        Return a dictionary with structure
        stop_id -> stop location as a UTM coordinate pair
        """
        xy_by_stop = {}
        for stop, group in self.stops.groupby('stop_id'):
            lat, lon = group[['stop_lat', 'stop_lon']].values[0]
            xy_by_stop[stop] = utm.from_latlon(lat, lon)[:2] 
        return xy_by_stop

    def get_dateless_trip_stats(self):
        t1 = dt.datetime.now()
        print(t1, 'Creating trip stats for %s trips...' % trips['trip_id'].count())

        linestring_by_shape = self.get_linestring_by_shape()
        xy_by_stop = self.get_xy_by_stop()

        # Initialize data frame. Base it on trips.txt.
        trip_stats = self.trips[['route_id', 'trip_id', 'direction_id']]
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
        return trip_stats

    def create_dateless_trip_stats(self):
        self.dateless_trip_stats = self.get_dateless_trip_stats()

# if __name__ == '__main__':
#     darwin = Feed('/Users/araichev/gtfs_toolkit/test/darwin/')