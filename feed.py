"""
Some tools for computing stats from GTFS feeds.

All time estimates below were produced on a 2013 MacBook Pro with a
2.8 GHz Intel Core i7 processor and 16GB of RAM running OS 10.9.2.
"""
from __future__ import print_function, division
import datetime as dt
from itertools import izip
import dateutil.relativedelta as rd
from collections import OrderedDict

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

    def get_dates(self):
        """
        Return a chronologically ordered list of dates 
        (``datetime.date`` objects) for which this feed is valid. 
        """
        start_date = self.calendar['start_date'].min()
        end_date = self.calendar['end_date'].max()
        days =  (end_date - start_date).days
        return [start_date + rd.relativedelta(days=+d) 
          for d in range(days + 1)]

    def get_first_workweek(self):
        """
        Return a list of dates (``datetime.date`` objects) 
        of the first Monday--Friday week for which this feed is valid.
        In the unlikely event that this feed does not cover a full 
        Monday--Friday week, then return whatever initial segment of the 
        week it does cover. 
        """
        dates = self.get_dates()
        # Get first Monday
        monday_index = None
        for (i, date) in enumerate(dates):
            if date.weekday() == 0:
                monday_index = i
                break
        week = []
        for j in range(5):
            try:
                week.append(dates[monday_index + j])
            except:
                break
        return week

    def is_active_trip(self, trip, date):
        """
        If the given trip (trip_id) is active on the given date 
        (date object), then return ``True``.
        Otherwise, return ``False``.
        """
        cal = self.calendar_m
        row = cal.ix[trip]
        # Check exceptional scenario given by calendar_dates
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
        Return a Pandas data frame with the following columns:

        - trip_id
        - route_id
        - start_time: first departure time of the trip
        - end_time: last departure time of the trip
        - start_stop_id: stop ID of the first stop of the trip 
        - end_stop_id: stop ID of the last stop of the trip
        - duration: duration of the trip (seconds)
        - distance: distance of the trip (meters)

        This method can take a long time, because it processes 
        ``stop_times.txt``, which can have several million lines. 

        NOTES:

        Takes about 11.5 minutes on the SEQ feed.
        """
        trips = self.trips
        stop_times = self.get_stop_times()

        t1 = dt.datetime.now()
        num_trips = trips.shape[0]
        print(t1, 'Creating trip stats for %s trips...' % num_trips)

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
                    # This can even happen when start_stop != end_stop 
                    # if the two stops are very close together
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
        print(t2, 'Finished trips stats in %.2f min' % minutes)    
    
        return trips_stats.reset_index().apply(to_int)

    def get_trips_weights(self, dates):
        """
        Return a Pandas data frame with the columns

        - trip_id
        - fraction of given dates for which the trip is active

        Here ``dates`` is a list of ``datetime.date`` objects.
        """
        if not dates:
            return

        f = self.trips
        n = len(dates)
        f['weight'] = f['trip_id'].map(lambda trip: sum(1 for date in dates 
          if self.is_active_trip(trip, date))/n)
        return f[['trip_id', 'weight']]

    def get_routes_stats(self, trips_stats, dates, routes=None):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, and use it to calculate stats for 
        each of the given routes (list of route IDs) 
        averaged over the given dates (list of ``datetime.date`` objects). 
        If ``routes is None``, then use all routes in the feed.

        To average over ``dates``, assign each trip a weight equal to 
        the fraction of days in ``dates`` for which the trip is active.
        Return a Pandas data frame with the following row entries

        - route_id: route ID
        - num_trips: sum of the weights for each trip on the route
        - start_time: start time of the earliest positive-weighted trip on 
          the route
        - end_time: end time of latest positive-weighted trip on the route
        - duration: sum of the weighted service durations for each trip on 
          the route (hours)
        - distance: sum of the weighted distances for each trip on the route 
          (kilometers)

        NOTES:

        Takes about 3.5 minutes on the SEQ feed.
        """
        if not dates:
            return 

        # Time the function call
        t1 = dt.datetime.now()
        
        # Get the section of trips_stats needed for the given routes
        stats = trips_stats
        if routes is not None:
            criteria = [stats['route_id'] == route for route in routes]
            criterion = False
            for c in criteria:
                criterion |= c
            stats = stats[criterion]

        num_trips = stats.shape[0]
        print(t1, 'Creating routes stats for %s trips...' % num_trips)

        # Merge with trip weights and drop 0-weight trips
        stats = pd.merge(stats, self.get_trips_weights(dates))
        stats = stats[stats.weight > 0]
        
        # Add weighted columns
        stats['wduration'] = stats['duration']*stats['weight']
        stats['wdistance'] = stats['distance']*stats['weight']

        # Group and aggregate
        grouped = stats.groupby('route_id')
        f = grouped.aggregate(OrderedDict([
          ('weight', np.sum),
          ('start_time', np.min),
          ('end_time', np.max),
          ('wduration', lambda x: x.sum()/3600), # hours
          ('wdistance', lambda x: x.sum()/1000), # kilometers
          ]))

        # Rename columns
        f.rename(columns={
          'weight': 'num_trips',
          'wdistance': 'distance', 
          'wduration': 'duration'
          }, inplace=True)

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished routes stats in %.2f min' % minutes)    

        return f.reset_index()

    def get_routes_time_series(self, trips_stats, dates, routes=None):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, and use it to calculate four time series
        for each of the given routes (list of route IDs) 
        averaged over the given dates (list of ``datetime.date`` objects). 
        If ``routes is None``, then use all routes in the feed.

        The four time series are

        - mean number of vehicles in service by route ID
        - mean number of trip starts by route ID
        - mean service duration (hours) by route ID
        - mean service distance (kilometers) by route ID

        Here the mean is taken over the dates given.
        Each time series is a Pandas data frame over a 24-hour 
        period with minute (period index) frequency (00:00 to 23:59).
        
        Return the time series as values of a dictionary with keys
        'num_vehicles', 'num_trip_starts', 'duration', 'distance'.

        Regarding resampling methods for the output:

        - for the vehicles series, use ``how=np.mean``
        - the other series, use ``how=np.sum`` 

        NOTES:

        Takes about 3.5 minutes on the SEQ feed.
        """  
        if not dates:
            return 

        # Time the function call
        t1 = dt.datetime.now()

        # Get the section of trips_stats needed for the given routes
        stats = trips_stats
        if routes is not None:
            criteria = [stats['route_id'] == route for route in routes]
            criterion = False
            for c in criteria:
                criterion |= c
            stats = stats[criterion]
            routes = sorted(routes)
        else:
            # Use all trips and all route ids
            routes = sorted(self.routes['route_id'].values)

        num_trips = stats.shape[0]
        print(t1, 'Creating routes time series for %s trips...' % num_trips)

        # Merge trip stats with trip weights and drop 0-weight trips
        stats = pd.merge(stats, self.get_trips_weights(dates))
        stats = stats[stats.weight > 0]
        
        # Initialize time series
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
        names = ['num_vehicles', 'num_trip_starts', 'duration', 'distance']
        series_by_name = {}
        for name in names:
            series_by_name[name] = pd.DataFrame(0.0, index=rng, columns=routes)
        
        # Bin each trip according to its start and end time and weight
        i = 0
        for index, row in stats.iterrows():
            i += 1
            print("Progress {:2.1%}".format(i/num_trips), end="\r")
            trip = row['trip_id']
            route = row['route_id']
            weight = row['weight']
            start_time = row['start_time']
            end_time = row['end_time']
            start = pd.to_datetime(date_str + ' ' +\
              timestr_mod_24(start_time))
            end = pd.to_datetime(date_str + ' ' +\
              timestr_mod_24(end_time))
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
            series_by_name['num_vehicles'].loc[
              criterion_by_name['num_vehicles'], route] +=\
              1*weight
            series_by_name['num_trip_starts'].loc[
              series_by_name['num_trip_starts'].index == start, route] +=\
              1*weight
            series_by_name['duration'].loc[
              criterion_by_name['duration'], route] +=\
              1/60*weight
            series_by_name['distance'].loc[
              criterion_by_name['distance'], route] +=\
              (row['distance']/1000)/(row['duration']/60)*weight

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished routes time series in %.2f min' % minutes)    

        return series_by_name

    def dump_stats_report(self, dates=None, freq='30Min', directory=None):
        """
        Into the given directory, dump to separate CSV the outputs of
        ``self.get_trips_stats()``, ``self.get_routes_stats(dates)``,
        and ``self.get_routes_time_series(dates)``,
        where the latter is resampled to the given frequency.
        Also include a ``README.txt`` file that contains a few notes
        on units.

        If no dates are given, then use ``self.get_first_workweek()``.
        If no directory is given, then use ``self.path + 'stats/'``. 

        NOTES:

        Takes about 17 minutes on the SEQ feed.
        """
        import os
        import matplotlib.pyplot as plt

        # Time function call
        t1 = dt.datetime.now()
        print(t1, 'Beginning process...')

        if dates is None:
            dates = self.get_first_workweek()
        if directory is None:
            directory = self.path + 'stats/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        dates_str = ' to '.join(
          [date_to_str(d) for d in [dates[0], dates[-1]]])

        # Write README.txt, which contains notes on units and date range
        readme = "- For the trips stats, "\
        "duration is measured in seconds and "\
        "distance is measured in meters\n"\
        "- For the routes stats and time series, "\
        "duration is measured in hours and "\
        "distance is measured in kilometers\n"\
        "- Routes stats and time series were calculated for an average day "\
        "during %s" % dates_str
        with open(directory + 'README.txt', 'w') as f:
            f.write(readme)

        # Trips stats
        trips_stats = self.get_trips_stats()
        trips_stats.to_csv(directory + 'trips_stats.csv', index=False)

        # Routes stats
        routes_stats = self.get_routes_stats(trips_stats, dates)
        routes_stats.to_csv(directory + 'routes_stats.csv', index=False)

        # Routes time series
        routes_time_series = self.get_routes_time_series(trips_stats, dates)
        for name, f in routes_time_series.iteritems():
            if 'vehicles' in name:
                how = np.mean
            else:
                how = np.sum
            fr = f.resample(freq, how=how)
            # Remove date from timestamps
            fr.index = [d.time() for d in fr.index.to_datetime()]
            fr.T.to_csv(directory + 'routes_time_series_%s_%s.csv' %\
              (name, freq), index_label='route_id')

        # Graph routes stats
        F = None
        for name, f in routes_time_series.iteritems():
            if 'vehicles' in name:
                how = np.mean
            else:
                how = np.sum
            g = f.resample(freq, how).T.sum().T
            if F is None:
                F = pd.DataFrame(g, columns=[name])
            else:
                F[name] = g
        F.index = [d.time() for d in F.index]
        colors = ['red', 'blue', 'yellow', 'green'] 
        alpha = 0.7
        columns = [
          'num_trip_starts', 
          'num_vehicles', 
          'duration', 
          'distance',
          ]
        titles = [
          'Number of trip starts',
          'Number of in-service vehicles',
          'In-service duration',
          'In-service distance',
          ]
        ylabels = ['','','hours','kilometers']
        fig, axes = plt.subplots(nrows=4, ncols=1)
        for (i, column) in enumerate(columns):
            F[column].plot(ax=axes[i], color=colors[i], alpha=alpha, kind='bar', 
              figsize=(8, 10))
            axes[i].set_title(titles[i])
            axes[i].set_ylabel(ylabels[i])

        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        fig.savefig(directory + 'routes_stats.pdf', dpi=200)
        
        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished process in %.2f min' % minutes)    

    def dump_routes_time_series(self, routes_ts, freq='1H', directory=None):
        """
        Given ``routes_ts``, which is the output of 
        ``self.get_routes_time_series()``, 
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
            # Remove date from timestamps
            fr.index = [d.time() for d in fr.index.to_datetime()]
            fr.T.to_csv(directory + 'routes_time_series_%s_%s.csv' %\
              (name, freq), index_label='route_id')

    def get_shapes_with_shape_dist_traveled(self):
        """
        Compute the optional ``shape_dist_traveled`` GTFS field for
        ``self.shapes`` and return the resulting Pandas data frame.  
        """
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
        """
        Compute the optional ``shape_dist_traveled`` GTFS field for
        ``self.get_stop_times()`` and return the resulting Pandas data frame.  

        NOTES:

        This takes a *long* time, so probably needs improving.
        On the SEQ feed, i stopped it after 2 hours.
        """
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