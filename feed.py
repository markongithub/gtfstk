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

def to_int(x):
    """
    Given a NumPy number type, convert it to an integer if it represents
    an integer, e.g. 1.0 -> 1. 
    Otherwise, return the input.
    """
    try:
        return x.astype(int)
    except:
        return x

class Feed(object):
    """
    A class to gather all the GTFS files and store them as Pandas data frames.
    """
    def __init__(self, path):
        self.path = path 
        self.shapes = pd.read_csv(path + 'shapes.txt', dtype={'shape_id': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'shape_id': str})
        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str})
        self.calendar = pd.read_csv(path + 'calendar.txt', 
          dtype={'service_id': str}, 
          date_parser=lambda x: date_to_str(x, inverse=True), 
          parse_dates=['start_date', 'end_date'])
        # Calendar with trip ids:
        self.calendar_m = pd.merge(self.trips[['trip_id', 'service_id']],
          self.calendar)
        
    def get_stop_times(self):
        """
        Return ``stop_times.txt`` as a Pandas data frame.
        This file is big for big feeds, e.g....
        Currently, i don't use this data frame enough to warrant storing it 
        as a feed attribute.
        """
        return pd.read_csv(self.path + 'stop_times.txt', 
          dtype={'stop_id': str})     

    # TODO: Improve by taking into consideration calendar_dates.txt
    def is_active_trip(self, trip, date):
        """
        If the given trip (trip_id) is active on the given date 
        (``datetime.date`` object), then return ``True``.
        Otherwise, return ``False``.
        """
        cal = self.calendar_m.set_index('trip_id')
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

    def get_trip_stats(self):
        """
        Return a data frame with the following stats for each trip:

        - route (route_id)
        - trip (trip_id)
        - start time
        - end time
        - start stop (stop_id)
        - end stop (stop_id)
        - duration of trip (seconds)
        - distance of trip (meters)
        """
        trips = self.trips
        stop_times = self.get_stop_times()

        t1 = dt.datetime.now()
        print(t1, 'Creating trip stats for %s trips...' % trips['trip_id'].count())

        linestring_by_shape = self.get_linestring_by_shape()
        xy_by_stop = self.get_xy_by_stop()

        # Initialize data frame. Base it on trips.txt.
        trip_stats = trips[['route_id', 'trip_id', 'direction_id']]
        trip_stats.set_index('trip_id', inplace=True)
        trip_stats['start_time'] = pd.Series()
        trip_stats['end_time'] = pd.Series()
        trip_stats['start_stop'] = pd.Series()
        trip_stats['end_stop'] = pd.Series()
        trip_stats['duration'] = pd.Series()
        trip_stats['distance'] = pd.Series()

        # Compute data frame values
        dist_by_stop_pair_by_shape = {shape: {} for shape in linestring_by_shape}
        # Convert departure times to seconds past midnight to ease
        # the calculations below
        stop_times['departure_time'] = stop_times['departure_time'].map(
          lambda x: seconds_to_timestr(x, inverse=True))
        for trip_id, group in pd.merge(trips, stop_times).groupby('trip_id'):
            # Compute trip time stats
            dep = group[['departure_time','stop_id']]
            dep = dep[dep['departure_time'].notnull()]
            argmin = dep['departure_time'].argmin()
            argmax = dep['departure_time'].argmax()
            start_time = int(dep['departure_time'].iat[argmin])
            end_time = int(dep['departure_time'].iat[argmax])
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
                d = int(round(d))        
                dist_by_stop_pair_by_shape[shape][stop_pair] = d
                
            # Store stats
            trip_stats.ix[trip_id, 'start_time'] =\
              seconds_to_timestr(start_time)
            trip_stats.ix[trip_id, 'end_time'] =\
              seconds_to_timestr(end_time)
            trip_stats.ix[trip_id, 'duration'] =\
              end_time - start_time
            trip_stats.ix[trip_id, 'start_stop'] = start_stop
            trip_stats.ix[trip_id, 'end_stop'] = end_stop
            trip_stats.ix[trip_id, 'distance'] = d
        trip_stats.sort('route_id')

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)    
    
        return trip_stats.reset_index().apply(to_int)

    def get_network_ts(self, trip_stats, dates, avg_over_dates=False):
        """
        For each of the given dates, use the given trip stats 
        (that is the output of ``self.get_trip_stats()``) to compute
        a time series with minute (period index) frequency with the 
        following network stats:

        - vehicles in service
        - trip starts
        - service duration (hours)
        - service distance (kilometers)

        Concatenate the series, drop the indices with NaN values, and
        return the result.

        If ``avg_over_dates == True``, then return one 24-hour time series
        that is the average of the daily time series.
        In this case, use '2000-1-1' as a placeholder date to index the series.
        """  
        t1 = dt.datetime.now()
        print(t1, 'Creating network time series for %s trips...' % trip_stats['trip_id'].count())

        if not dates:
            return
        n = len(dates)
        if not avg_over_dates:
            # Call recursively on individual dates and sum
            ts_iter = (self.get_network_ts(trip_stats, [date], 
              avg_over_dates=True) for date in dates)
            return pd.concat(ts_iter).dropna()
        
        # Now avg_over_dates is True, so compute one 24-hour time series
        if n > 1:
            # Assign a uniform generic date for the index
            date_str = '2000-1-1'
        else:
            # Use the given date for the index
            date_str = date_to_str(dates[0]) 
        day_start_p = pd.Period(date_str + ' 00:00:00', freq='Min')
        day_end_p = pd.Period(date_str + ' 23:59:00', freq='Min')
        prng = pd.period_range(day_start_p, day_end_p, freq='Min')
        df = pd.DataFrame(0.0, index=prng, 
          columns=['vehicles', 'trip_starts', 'duration', 'distance'])
        
        # Bin each trip and weight each by the fraction of days that
        # it is active within the given dates
        for row_index, row in trip_stats.iterrows():
            trip = row['trip_id']
            start_time = row['start_time']
            end_time = row['end_time']
            start_p = pd.Period(date_str + ' ' + timestr_mod_24(start_time),
              freq='Min')
            end_p = pd.Period(date_str + ' ' + timestr_mod_24(end_time),
              freq='Min')
            # Weight the trip
            trip_weight = 0
            for date in dates:
                if self.is_active_trip(trip, date):
                    trip_weight += 1
            trip_weight /= n
            vehicles_value = 1*trip_weight
            trip_starts_value = 1*trip_weight
            # Trip duration per minute
            duration_value = 60*trip_weight 
            # Trip distance per minute
            distance_value =\
              row['distance']/(row['duration']/60)*trip_weight
            # Bin the trip
            if start_p <= end_p:
                criterion = (df.index >= start_p) & (df.index < end_p)
            else:
                criterion = ((df.index >= start_p) &\
                  (df.index <= day_end_p)) |\
                  ((df.index >= day_start_p) &\
                  (df.index < end_p))
            df.loc[criterion, 'vehicles'] += vehicles_value
            df.loc[df.index == start_p, 'trip_starts'] += trip_starts_value
            df.loc[criterion, 'duration'] += duration_value
            df.loc[criterion, 'distance'] += distance_value

        # Convert from meters and seconds to hours and kilometers
        df.index.name = 'bin'
        df['duration'] = df['duration'].map(lambda x: x/3600)
        df['distance'] = df['distance'].map(lambda x: x/1000)

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)    

        return df

    @staticmethod
    def resample_network_ts(network_ts, freq=None):
        """
        Resample the given output of ``get_network_ts()`` to the given
        frequency.
        Because the columns need to be resampled in different ways, 
        this function is useful.
        """
        if freq is None:
            return
        # Resample by freq
        return network_ts.resample(freq, how=OrderedDict([
          ('vehicles', np.mean),
          ('trip_starts', np.sum),
          ('duration', np.sum), 
          ('distance', np.sum), 
          ])).dropna()

    # Less useful functions. 
    # TODO: Test these.
    def get_trip_activity(self, dates=None):
        cal = self.calendar_m
        if dates is None:
            # Use the full range of dates for which the feed is valid
            feed_start = cal['start_date'].min()
            feed_end = cal['end_date'].max()
            num_days =  (feed_end - feed_start).days
            dates = [feed_start + rd.relativedelta(days=+d) for d in range(num_days + 1)]
            activity_header = 'fraction of dates active during %s--%s' %\
              (date_to_str(feed_start), date_to_str(feed_end))
        else:
            activity_header = 'fraction of dates active among %s' % '-'.join([date_to_str(d) for d in dates])
        cal[activity_header] = pd.Series()

        n = len(dates)
        for index, row in cal.iterrows():    
            trip = row['trip_id']
            start_date = row['start_date']
            end_date = row['end_date']
            weekdays_running = [i for i in range(7) 
              if row[weekday_to_str(i)] == 1]
            count = 0
            for date in dates:
                if not start_date <= date <= end_date:
                    continue
                if date.weekday() in weekdays_running:
                    count += 1
            cal.ix[index, activity_header] = count/n
        return cal
  
    def get_shapes_with_shape_dist_traveled(self):
        shapes = self.shapes

        t1 = dt.datetime.now()
        print(t1, 'Adding shape_dist_traveled field to %s shape points...' %\
          shapes['shape_id'].count())

        new_shapes = shapes.copy()
        new_shapes['shape_dist_traveled'] = pd.Series()
        for shape, group in shapes.groupby('shape_id'):
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

    def get_stop_times_with_shape_dist_traveled(self):
        trips = self.trips
        stop_times = self.get_stop_times()

        t1 = dt.datetime.now()
        print(t1, 'Adding shape_dist_traveled field to %s stop times...' %\
          stop_times['trip_id'].count())

        linestring_by_shape = self.get_linestring_by_shape()
        xy_by_stop = self.get_xy_by_stop()

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

        del merged['shape_id']

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)
        print('%s failures on these trips: %s' % (len(failures), failures))  

        return merged