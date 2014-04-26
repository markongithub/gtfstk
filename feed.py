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
        self.routes = pd.read_csv(path + 'routes.txt')
        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str})
        self.shapes = pd.read_csv(path + 'shapes.txt', dtype={'shape_id': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'shape_id': str})
        self.calendar = pd.read_csv(path + 'calendar.txt', 
          dtype={'service_id': str}, 
          date_parser=lambda x: date_to_str(x, inverse=True), 
          parse_dates=['start_date', 'end_date'])
        try:
            self.calendar_dates = pd.read_csv(path + 'calendar_dates.txt', 
              dtype={'service_id': str}, 
              date_parser=lambda x: date_to_str(x, inverse=True), 
              parse_dates=['date'])
        except IOError:
            # Doesn't exist, so create an empty data frame
            self.calendar_dates = pd.DataFrame()
        # Merge trips and calendar
        self.calendar_m = pd.merge(self.trips[['trip_id', 'service_id']],
          self.calendar).set_index('trip_id')
        
    def get_stop_times(self):
        """
        Return ``stop_times.txt`` as a Pandas data frame.
        This file is big for big feeds, e.g....
        Currently, i don't use this data frame enough to warrant storing it 
        as a feed attribute.
        """
        return pd.read_csv(self.path + 'stop_times.txt', 
          dtype={'stop_id': str})     

    def is_active_trip(self, trip, date):
        """
        If the given trip (trip_id) is active on the given date 
        (date object), then return ``True``.
        Otherwise, return ``False``.
        """
        cal = self.calendar_m
        row = cal.ix[trip]
        # Check exceptional scenarios first
        cald = self.calendar_dates
        service = row['service_id']
        try:
            exception_type = cald[(cald['service_id'] == service) &\
              (cald['date'] == date)].ix[0, 'exception_type']
            if exception_type == 1:
                return True
            else:
                # exception_type == 2
                return False
        except (KeyError, IndexError): 
            pass
        # Check regular scenario
        start_date = row['start_date']
        end_date = row['end_date']
        weekday_str = weekday_to_str(date.weekday())
        if start_date <= date <= end_date and row[weekday_str] == 1:
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

    def get_trips_stats(self):
        """
        Return a Pandas data frame with the following stats for each trip:

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
        trips_stats = trips[['route_id', 'trip_id', 'direction_id']]
        trips_stats.set_index('trip_id', inplace=True)
        trips_stats['start_time'] = pd.Series()
        trips_stats['end_time'] = pd.Series()
        trips_stats['start_stop'] = pd.Series()
        trips_stats['end_stop'] = pd.Series()
        trips_stats['duration'] = pd.Series()
        trips_stats['distance'] = pd.Series()

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
            trips_stats.ix[trip_id, 'start_time'] =\
              seconds_to_timestr(start_time)
            trips_stats.ix[trip_id, 'end_time'] =\
              seconds_to_timestr(end_time)
            trips_stats.ix[trip_id, 'duration'] =\
              end_time - start_time
            trips_stats.ix[trip_id, 'start_stop'] = start_stop
            trips_stats.ix[trip_id, 'end_stop'] = end_stop
            trips_stats.ix[trip_id, 'distance'] = d
        trips_stats.sort('route_id')

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)    
    
        return trips_stats.reset_index().apply(to_int)


    def get_routes_timeseries(self, trips_stats, dates, routes=None):
        """
        Given ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, a list of date objects, and a
        possibly ``None`` list of route IDs,
        compute and return the following four 24-hour time series as 
        Pandas data frames with minute (period index) frequency:
        
        - mean number of vehicles in service by route ID
        - mean number of trip starts by route ID
        - mean service duration (hours) by route ID
        - mean service distance (kilometers) by route ID

        Here the mean is taken over the dates given.

        Return the time series as values of a dictionary with keys
        'vehicles', 'trip_starts', 'duration', 'distance'.

        If no routes are given, then use all the routes in this feed.

        Regarding resampling methods for the output:

        - for the vehicles series, use ``how=np.mean``
        - the other series, use ``how=np.sum`` 
        """  
        if not dates:
            return 

        # Time the function call
        t1 = dt.datetime.now()
        num_trips = trips_stats['trip_id'].count()
        print(t1, 'Creating routes time series for %s trips...' % num_trips)

        # Initialize routes
        if routes is not None:
            routes_given = True
        else:
            routes_given = False
            routes = sorted(self.routes['route_id'].values)

        # Initialize the series
        n = len(dates)
        if n > 1:
            # Assign a uniform generic date for the index
            date_str = '2000-1-1'
        else:
            # Use the given date for the index
            date_str = date_to_str(dates[0]) 
        day_start = pd.to_datetime(date_str + ' 00:00:00')
        day_end = pd.to_datetime(date_str + ' 23:59:00')
        rng = pd.period_range(day_start, day_end, freq='Min')
        names = ['vehicles', 'trip_starts', 'duration', 'distance']
        series_by_name = {}
        for name in names:
            series_by_name[name] = pd.DataFrame(0.0, index=rng, columns=routes)
        
        # Get the section of trips_stats needed for the given routes
        if routes_given:
            criteria = [trips_stats['route_id'] == route for route in routes]
            criterion = False
            for c in criteria:
                criterion |= c
            trips_stats = trips_stats[criterion]

        # Bin each trip according to its start and end time
        # and weight each trip by the fraction of days that
        # it is active within the given dates
        i = 0
        for row_index, row in trips_stats.iterrows():
            i += 1
            print("Progress {:2.1%}".format(i/num_trips), end="\r")
            trip = row['trip_id']
            route = row['route_id']
            start_time = row['start_time']
            end_time = row['end_time']
            start = pd.to_datetime(date_str + ' ' +\
              timestr_mod_24(start_time))
            end = pd.to_datetime(date_str + ' ' +\
              timestr_mod_24(end_time))
            # Weight the trip
            trip_weight = 0
            for date in dates:
                if self.is_active_trip(trip, date):
                    trip_weight += 1
            trip_weight /= n
            if not trip_weight:
                continue
            # Bin the trip
            criterion_by_name = {}
            if start <= end:
                for name, f in series_by_name.iteritems():
                    criterion_by_name[name] =\
                      (f.index >= start) & (f.index < end)
            else:
                for name, f in series_by_name.iteritems():
                    criterion_by_name[name] =\
                      ((f.index >= start) &\
                      (f.index <= day_end)) |\
                      ((f.index >= day_start) &\
                      (f.index < end))
            series_by_name['vehicles'].loc[
              criterion_by_name['vehicles'], route] +=\
              1*trip_weight
            series_by_name['trip_starts'].loc[
              series_by_name['trip_starts'].index == start, route] +=\
              1*trip_weight
            series_by_name['duration'].loc[
              criterion_by_name['duration'], route] +=\
              1/60*trip_weight
            series_by_name['distance'].loc[
              criterion_by_name['distance'], route] +=\
              (row['distance']/1000)/(row['duration']/60)*\
              trip_weight

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished in %.2f min' % minutes)    

        return series_by_name

    def dump_routes_timeseries(self, routes_ts, freq='1H', directory=None):
        """
        Given ``routes_ts``, which is the output of 
        ``self.get_routes_timeseries()``, 
        resample each series to the given frequency and dump each
        to the given directory, which defaults to ``self.path``.
        """
        if directory is None:
            directory = self.path
        for name, f in routes_ts.iteritems():
            if 'vehicles' in name:
                how = np.mean
            else:
                how = np.sum
            fr = f.resample(freq, how=how)
            fr.T.to_csv(directory + '%s_timeseries_by_route_%s.csv' %\
              (name, freq), index_label='route_id')

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