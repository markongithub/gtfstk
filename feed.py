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
    A class to gather all the GTFS files for a feed and store them as 
    Pandas data frames.
    """
    def __init__(self, path):
        self.path = path 
        self.routes = pd.read_csv(path + 'routes.txt')
        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str})
        self.shapes = pd.read_csv(path + 'shapes.txt', dtype={'shape_id': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'shape_id': str,
          'stop_id': str, 'direction_id': str})
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
        This frame can be big.
        For example the one for the SEQ feed has roughly 2 million rows.
        So i'll only import it when i need it for now, instead of storing
        it as an attribute.
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

    def get_first_week(self):
        """
        Return a list of dates (``datetime.date`` objects) 
        of the first Monday--Sunday week for which this feed is valid.
        In the unlikely event that this feed does not cover a full 
        Monday--Sunday week, then return whatever initial segment of the 
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
        for j in range(7):
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
        if not cald.empty:
            if not cald[(cald['service_id'] == service) &\
              (cald['date'] == date) &\
              (cald['exception_type'] == 1)].empty:
                return True
            if not cald[(cald['service_id'] == service) &\
              (cald['date'] == date) &\
              (cald['exception_type'] == 2)].empty:
                return False
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
        - direction_id
        - route_id
        - start_time: first departure time of the trip
        - end_time: last departure time of the trip
        - start_stop_id: stop ID of the first stop of the trip 
        - end_stop_id: stop ID of the last stop of the trip
        - duration: duration of the trip (seconds)
        - distance: distance of the trip (meters)

        NOTES:

        Takes about 2.8 minutes on the SEQ feed.
        """
        trips = self.trips
        stop_times = self.get_stop_times()

        t1 = dt.datetime.now()
        num_trips = trips.shape[0]
        print(t1, 'Creating trip stats for %s trips...' % num_trips)

        # Initialize data frame. Base it on trips.txt.
        stats = trips[['route_id', 'trip_id', 'direction_id']]

        # Convert departure times to seconds past midnight, 
        # to compute durations below
        stop_times['departure_time'] = stop_times['departure_time'].map(
          lambda x: seconds_to_timestr(x, inverse=True))

        # Compute start time, end time, duration
        f = pd.merge(trips, stop_times).groupby('trip_id')
        g = f['departure_time'].agg({'start_time': np.min, 'end_time': np.max})
        g['duration'] = g['end_time'] - g['start_time']

        # Compute start stop and end stop
        def start_stop(group):
            i = group['departure_time'].argmin()
            return group['stop_id'].at[i]

        def end_stop(group):
            i = group['departure_time'].argmax()
            return group['stop_id'].at[i]

        g['start_stop'] = f.apply(start_stop)
        g['end_stop'] = f.apply(end_stop)

        # Convert times back to time strings
        g[['start_time', 'end_time']] = g[['start_time', 'end_time']].\
          applymap(lambda x: seconds_to_timestr(int(x)))

        # Compute trip distance, which is more involved
        g['shape_id'] = f['shape_id'].first()
        linestring_by_shape = self.get_linestring_by_shape()
        xy_by_stop = self.get_xy_by_stop()
        dist_by_stop_pair_by_shape = {shape: {} 
          for shape in linestring_by_shape}

        for trip, row in g.iterrows():
            start_stop = row.at['start_stop'] 
            end_stop = row.at['end_stop'] 
            shape = row.at['shape_id']
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
            g.ix[trip, 'distance'] = d

        stats = pd.merge(stats, g.reset_index())
        stats.sort('route_id')

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished trips stats in %.2f min' % minutes)    
    
        return stats

    def get_trips_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - trip_id
        - dates[0]: a series of ones and zeros indicating if a 
        trip is active (1) on the given date or inactive (0)
        ...
        - dates[-1]: ditto

        NOTES:

        Takes about 1 minute on the SEQ feed.
        """
        if not dates:
            return

        print(dt.datetime.now())
        f = self.trips
        for date in dates:
            f[date] = f['trip_id'].map(lambda trip: 
              int(self.is_active_trip(trip, date)))
        print(dt.datetime.now())
        return f[['trip_id'] + dates]

    def get_routes_stats(self, trips_stats, dates, routes=None):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, and use it to calculate stats for 
        each of the given routes (list of route IDs) 
        averaged over the given dates (list of ``datetime.date`` objects). 
        If ``routes is None``, then use all routes in the feed.

        Return a Pandas data frame with the following row entries

        - route_id: route ID
        - mean_daily_num_trips
        - min_start_time: start time of the earliest active trip on 
          the route
        - max_end_time: end time of latest active trip on the route
        - max_headway: maximum of the durations (in seconds) between 
          trip starts on the route between 07:00 and 19:00 
          on the given dates
        - mean_headway: mean of the durations (in seconds) between 
          vehicle departures at the stop between 07:00 and 19:00 
          on the given dates
        - mean_daily_duration: in seconds
        - mean_daily_distance: in meters

        NOTES:

        Takes about 1 minute on the SEQ feed.
        """
        if not dates:
            return 

        # Time the function call
        t1 = dt.datetime.now()
        
        # Get the section of trips_stats needed for the given routes
        if routes is not None:
            criteria = [trips_stats['route_id'] == route for route in routes]
            criterion = False
            for c in criteria:
                criterion |= c
            trips_stats = trips_stats[criterion]

        num_trips = trips_stats.shape[0]
        print(t1, 'Creating routes stats for %s trips...' % num_trips)

        # Merge with trips activity, assign a weight to each trip equal
        # to the fraction of days in dates for which it is active, and
        # and drop 0-weight trips
        trips_stats = pd.merge(trips_stats, self.get_trips_activity(dates))
        trips_stats['weight'] = trips_stats[dates].sum(axis=1)/len(dates)
        trips_stats = trips_stats[trips_stats['weight'] > 0]
        
        # Convert trip start times to seconds to ease headway calculations
        trips_stats['start_time'] = trips_stats['start_time'].map(lambda x: 
          seconds_to_timestr(x, inverse=True))

        def get_route_stats(group):
            # Take this group of all trips stats for a single route
            # and compute route-level stats.
            headways = []
            for date in dates:
                stimes = sorted(group[group[date] > 0]['start_time'].\
                  values)
                stimes = [stime for stime in stimes 
                  if 7*3600 <= stime <= 19*3600]
                headways.extend([stimes[i + 1] - stimes[i] 
                  for i in range(len(stimes) - 1)])
            if headways:
                max_headway = np.max(headways)
                mean_headway = round(np.mean(headways))
            else:
                max_headway = np.nan
                mean_headway = np.nan
            mean_daily_num_trips = group['weight'].sum()
            min_start_time = group['start_time'].min()
            max_end_time = group['end_time'].max()
            mean_daily_duration = (group['duration']*group['weight']).sum()
            mean_daily_distance = (group['distance']*group['weight']).sum()
            df = pd.DataFrame([[min_start_time, max_end_time, 
              mean_daily_num_trips, max_headway, mean_headway, 
              mean_daily_duration, mean_daily_distance]], columns=[
              'min_start_time', 'max_end_time', 
              'mean_daily_num_trips', 'max_headway', 'mean_headway', 
              'mean_daily_duration', 'mean_daily_distance'])
            df.index.name = 'foo'
            return df

        result = trips_stats.groupby('route_id').apply(get_route_stats).\
          reset_index()
        del result['foo']

        # Convert route start times to time strings
        result['min_start_time'] = result['min_start_time'].map(lambda x: 
          seconds_to_timestr(x))

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished routes stats in %.2f min' % minutes)    

        return result

    # TODO: Update the output key names to 'mean_daily_...'
    # Change units to seconds and meters
    def get_routes_time_series(self, trips_stats, dates, routes=None):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, and use it to calculate four time series
        for each of the given routes (list of route IDs) 
        averaged over the given dates (list of ``datetime.date`` objects). 
        If ``routes is None``, then use all routes in the feed.

        The four time series are

        - mean daily number of vehicles in service by route ID
        - mean daily number of trip starts by route ID
        - mean daily service duration (hours) by route ID
        - mean daily service distance (kilometers) by route ID

        Each time series is a Pandas data frame over a 24-hour 
        period with minute (period index) frequency (00:00 to 23:59).
        
        Return the time series as values of a dictionary with keys
        'num_vehicles', 'num_trip_starts', 'duration', 'distance'.

        Regarding resampling methods for the output:

        - for the vehicles series, use ``how=np.mean``
        - for the other series, use ``how=np.sum`` 

        NOTES:

        Takes about 2 minutes on the SEQ feed.
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

        # Merge with trips activity, get trip weights,
        # and drop 0-weight trips
        n = len(dates)
        stats = pd.merge(stats, self.get_trips_activity(dates))
        stats['weight'] = stats[dates].sum(axis=1)/n
        stats = stats[stats.weight > 0]
        
        # Initialize time series
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

            # TODO: Simplify by using
            #for name, f in series_by_name.iteritems():

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

    # TODO: Add stops=None keyword as in get_routes_stats
    def get_stops_stats(self, dates):
        """
        Return a Pandas data frame with the following columns:

        - stop_id
        - mean_daily_num_vehicles: mean daily number of vehicles visiting stop 
        - max_headway: maximum of the durations (in seconds) between 
          vehicle departures at the stop between 07:00 and 19:00 
          on the given dates
        - mean_headway: mean of the durations (in seconds) between 
          vehicle departures at the stop between 07:00 and 19:00 
          on the given dates
        - min_start_time: earliest departure time of a vehicle from this stop
          over the given date range
        - max_end_time: latest departure time of a vehicle from this stop
          over the given date range

        NOTES:

        Takes about 1.8 minutes for the SEQ feed.
        """
        t1 = dt.datetime.now()
        print(t1, 'Calculating stops stats...')

        # Get active trips and merge with stop times
        trips_activity = self.get_trips_activity(dates)
        ta = trips_activity[trips_activity[dates].sum(axis=1) > 0]
        stop_times = self.get_stop_times()
        f = pd.merge(ta, stop_times)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(lambda x: 
          seconds_to_timestr(x, inverse=True))

        # Compute stats for each stop
        def get_stop_stats(group):
            # Operate on the group of all stop times for an individual stop
            headways = []
            num_vehicles = 0
            for date in dates:
                dtimes = sorted(group[group[date] > 0]['departure_time'].\
                  values)
                num_vehicles += len(dtimes)
                dtimes = [dtime for dtime in dtimes 
                  if 7*3600 <= dtime <= 19*3600]
                headways.extend([dtimes[i + 1] - dtimes[i] 
                  for i in range(len(dtimes) - 1)])
            if headways:
                max_headway = np.max(headways)
                mean_headway = round(np.mean(headways))
            else:
                max_headway = np.nan
                mean_headway = np.nan
            mean_daily_num_vehicles = num_vehicles/len(dates)
            min_start_time = group['departure_time'].min()
            max_end_time = group['departure_time'].max()
            df = pd.DataFrame([[min_start_time, max_end_time, 
              mean_daily_num_vehicles, max_headway, mean_headway]], 
              columns=['min_start_time', 'max_end_time', 
              'mean_daily_num_vehicles', 'max_headway', 'mean_headway'])
            df.index.name = 'foo'
            return df

        result = f.groupby('stop_id').apply(get_stop_stats).reset_index()
        del result['foo']

        # Convert start and end times to time strings
        result[['min_start_time', 'max_end_time']] =\
          result[['min_start_time', 'max_end_time']].applymap(
          lambda x: seconds_to_timestr(x))

        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished stops stats in %.2f min' % minutes)    

        return result

    # TODO: Finish this
    def get_stops_time_series(self, dates, stops=None):
        pass

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

        # Plot sum of routes stats at 30-minute frequency
        fig = self.plot_sum_of_routes_time_series(routes_time_series)
        fig.savefig(directory + 'sum_of_routes_time_series.pdf', dpi=200)
        
        t2 = dt.datetime.now()
        minutes = (t2 - t1).seconds/60
        print(t2, 'Finished process in %.2f min' % minutes)    

    def plot_sum_of_routes_time_series(self, routes_ts, freq='30Min'):
        """
        Given the output of ``self.get_routes_times_series()``,
        sum each time series over all routes, plot each using
        MatplotLib, and return the resulting figure of four subplots.
        """
        import matplotlib.pyplot as plt

        # Plot sum of routes stats at 30-minute frequency
        F = None
        for name, f in routes_ts.iteritems():
            if 'vehicles' in name:
                how = np.mean
            else:
                how = np.sum
            g = f.resample('30Min', how).T.sum().T
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
            F[column].plot(ax=axes[i], color=colors[i], alpha=alpha, 
              kind='bar', figsize=(8, 10))
            axes[i].set_title(titles[i])
            axes[i].set_ylabel(ylabels[i])

        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        return fig

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