"""
Some tools for computing stats from a GTFS feed, assuming the feed
is valid.

All time estimates below were produced on a 2013 MacBook Pro with a
2.8 GHz Intel Core i7 processor and 16GB of RAM running OS 10.9.

TODO:

- Add get_active_stops()
- Remove get_trips_activity()? 
- Allow dates to be entered as YYYYMMDD strings?
- Speed up time series calculations
- Add start and end time options for headway calculations
"""
import datetime as dt
import dateutil.relativedelta as rd
from collections import OrderedDict
import os
import zipfile
import shutil

import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
import utm

import gtfs_toolkit.utils as utils

VALID_DISTANCE_UNITS = ['km', 'm', 'mi', 'ft']

def downsample(time_series, freq):
    """
    Downsample the given route or stop time series, which is the output of 
    ``Feed.get_routes_time_series()`` or ``Feed.get_stops_time_series()``, 
    to the given Pandas-style frequency.
    Can't downsample to frequencies less one minute ('1Min'), because the
    time series are generated with one-minute frequency.
    """
    result = None
    if 'route_id' in time_series.columns.names:
        # It's a routes time series
        # Sums
        how = OrderedDict((col, 'sum') for col in time_series.columns
          if col[0] in ['mean_daily_num_trip_starts', 'mean_daily_distance', 
          'mean_daily_duration'])
        # Means
        how.update(OrderedDict((col, 'mean') for col in time_series.columns
          if col[0] in ['mean_daily_num_vehicles']))
        f = time_series.resample(freq, how=how)
        # Calculate speed and add it to f. Can't resample it.
        speed = f['mean_daily_distance'].divide(f['mean_daily_duration'])
        speed = pd.concat({'mean_daily_speed': speed}, axis=1)
        result = pd.concat([f, speed], axis=1)
    elif 'stop_id' in time_series.columns.names:
        # It's a stops time series
        how = OrderedDict((col, 'sum') for col in time_series.columns)
        result = time_series.resample(freq, how=how)
    # Reset column names in result, because they disappear after resampling.
    # Pandas 0.14.0 bug?
    result.columns.names = time_series.columns.names
    # Sort the multiindex column to make slicing possible;
    # see http://pandas.pydata.org/pandas-docs/stable/indexing.html#multiindexing-using-slicers
    return result.sortlevel(axis=1)

def plot_headways(stats, max_headway_limit=60):
    """
    Given a stops or routes stats data frame, 
    return bar charts of the max and mean headways as a MatplotLib figure.
    Only include the stops/routes with max headways at most 
    ``max_headway_limit`` minutes.
    If ``max_headway_limit is None``, then include them all in a giant plot. 
    If there are no stops/routes within the max headway limit, then return 
    ``None``.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    import matplotlib.pyplot as plt

    # Set Pandas plot style
    pd.options.display.mpl_style = 'default'

    if 'stop_id' in stats.columns:
        index = 'stop_id'
    elif 'route_id' in stats.columns:
        index = 'route_id'
    split_directions = 'direction_id' in stats.columns
    if split_directions:
        # Move the direction_id column to a hierarchical column,
        # select the headway columns, and convert from seconds to minutes
        f = stats.pivot(index=index, columns='direction_id')[['max_headway', 
          'mean_headway']]/60
        # Only take the stops/routes within the max headway limit
        if max_headway_limit is not None:
            f = f[(f[('max_headway', 0)] <= max_headway_limit) |
              (f[('max_headway', 1)] <= max_headway_limit)]
        # Sort by max headway
        f = f.sort(columns=[('max_headway', 0)], ascending=False)
    else:
        f = stats.set_index(index)[['max_headway', 'mean_headway']]/60
        if max_headway_limit is not None:
            f = f[f['max_headway'] <= max_headway_limit]
        f = f.sort(columns=['max_headway'], ascending=False)
    if f.empty:
        return

    # Plot max and mean headway separately
    n = f.shape[0]
    data_frames = [f['max_headway'], f['mean_headway']]
    titles = ['Max Headway','Mean Headway']
    ylabels = [index, index]
    xlabels = ['minutes', 'minutes']
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for (i, f) in enumerate(data_frames):
        f.plot(kind='barh', ax=axes[i], figsize=(10, max(n/9, 10)))
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel(ylabels[i])
    return fig

def agg_routes_stats(routes_stats):
    """
    Given ``route_stats`` which is the output of ``get_routes_stats()``,
    return a Pandas data frame with the following columns:

    - direction_id
    - mean_daily_num_trips: the sum of the corresponding column in the
      input across all routes
    - min_start_time: the minimum of the corresponding column of the input
      across all routes
    - max_end_time: the maximum of the corresponding column of the input
      across all routes
    - mean_daily_duration: the sum of the corresponding column in the
      input across all routes
    - mean_daily_distance: the sum of the corresponding column in the
      input across all routes  
    - mean_daily_speed: mean_daily_distance/mean_daily_distance

    If the input has no direction id, then the output won't.
    """
    f = routes_stats
    if 'direction_id' in routes_stats.columns:
        g = f.groupby('direction_id').agg({
          'min_start_time': min, 
          'max_end_time': max, 
          'mean_daily_num_trips': sum,
          'mean_daily_duration': sum,
          'mean_daily_distance': sum,
          }).reset_index()
    else:
        g = pd.DataFrame([[
          f['min_start_time'].min(), 
          f['max_end_time'].max(), 
          f['mean_daily_num_trips'].sum(),
          f['mean_daily_duration'].sum(),
          f['mean_daily_distance'].sum(),
          ]], 
          columns=['min_start_time', 'max_end_time', 'mean_daily_num_trips', 
            'mean_daily_duration', 'mean_daily_distance']
          )

    g['mean_daily_speed'] = g['mean_daily_distance'].divide(g['mean_daily_duration'])
    return g

def agg_routes_time_series(routes_time_series):
    rts = routes_time_series
    stats = rts.columns.levels[0].tolist()
    split_directions = 'direction_id' in rts.columns.names
    if split_directions:
        # For each stat and each direction, sum across routes.
        frames = []
        for stat in stats:
            f0 = rts.xs((stat, '0'), level=('statistic', 'direction_id'), 
              axis=1).sum(axis=1)
            f1 = rts.xs((stat, '1'), level=('statistic', 'direction_id'), 
              axis=1).sum(axis=1)
            f = pd.concat([f0, f1], axis=1, keys=['0', '1'])
            frames.append(f)
        F = pd.concat(frames, axis=1, keys=stats, names=['statistic', 
          'direction_id'])
        # Fix speed
        F['mean_daily_speed'] = F['mean_daily_distance'].divide(
          F['mean_daily_duration'])
        result = F
    else:
        f = pd.concat([rts[stat].sum(axis=1) for stat in stats], axis=1, 
          keys=stats)
        f['mean_daily_speed'] = f['mean_daily_distance'].divide(f['mean_daily_duration'])
        result = f
    return result

def plot_routes_time_series(routes_time_series):
    """
    Given a routes time series data frame,
    sum each time series statistic over all routes, 
    plot each series statistic using MatplotLib, 
    and return the resulting figure of subplots.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    import matplotlib.pyplot as plt

    rts = routes_time_series
    if 'route_id' not in rts.columns.names:
        return

    # Aggregate time series
    f = agg_routes_time_series(rts)

    # Reformat time periods
    f.index = [t.time().strftime('%H:%M') 
      for t in rts.index.to_datetime()]
    
    #split_directions = 'direction_id' in rts.columns.names

    # Split time series by into its component time series by statistic type
    # stats = rts.columns.levels[0].tolist()
    stats = [
      'mean_daily_num_trip_starts',
      'mean_daily_num_vehicles',
      'mean_daily_distance',
      'mean_daily_duration',
      'mean_daily_speed',
      ]
    ts_dict = {stat: f[stat] for stat in stats}

    # Create plots  
    pd.options.display.mpl_style = 'default'
    titles = [stat.capitalize().replace('_', ' ') for stat in stats]
    units = ['','','km','h', 'kph']
    alpha = 1
    fig, axes = plt.subplots(nrows=len(stats), ncols=1)
    for (i, stat) in enumerate(stats):
        if stat == 'mean_daily_speed':
            stacked = False
        else:
            stacked = True
        ts_dict[stat].plot(ax=axes[i], alpha=alpha, 
          kind='bar', figsize=(8, 10), stacked=stacked, width=1)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(units[i])

    return fig


class Feed(object):
    """
    A class to gather all the GTFS files for a feed and store them in memory 
    as Pandas data frames.  
    Make sure you have enough memory!  
    The stop times object can be big.
    """
    def __init__(self, path, original_units='km'):
        """
        Read in all the relevant GTFS text files within the directory or 
        ZIP file given by ``path`` and assign them to instance attributes.
        Assume the zip file unzips as a collection of GTFS text files
        rather than as a directory of GTFS text files.
        Set the native distance units of this feed to the given distance
        units.
        Valid options are listed in ``VALID_DISTANCE_UNITS``.
        All distance units will then be converted to kilometers.
        """
        zipped = False
        if zipfile.is_zipfile(path):
            # Extract to temporary location
            zipped = True
            archive = zipfile.ZipFile(path)
            path = path.rstrip('.zip') + '/'
            archive.extractall(path)

        # Get distance units
        assert original_units in VALID_DISTANCE_UNITS,\
            'Units must be one of {!s}'.format(VALID_DISTANCE_UNITS)
        self.original_units = original_units

        self.stops = pd.read_csv(path + 'stops.txt', dtype={'stop_id': str, 
          'stop_code': str})
        self.routes = pd.read_csv(path + 'routes.txt', dtype={'route_id': str,
          'route_short_name': str})
        self.trips = pd.read_csv(path + 'trips.txt', dtype={'route_id': str,
          'trip_id': str, 'service_id': str, 'shape_id': str, 'stop_id': str})
        self.trips_t = self.trips.set_index('trip_id')
        st = pd.read_csv(path + 'stop_times.txt', dtype={'stop_id': str,
          'trip_id': str})
    
        # Prefix a 0 to arrival and departure times if necessary.
        # This makes sorting by time work as expected.
        def reformat_times(timestr):
            result = timestr
            if isinstance(result, str) and len(result) == 7:
                result = '0' + result
            return result

        st[['arrival_time', 'departure_time']] =\
          st[['arrival_time', 'departure_time']].applymap(reformat_times)
        # Convert distances to kilometers
        if 'shape_dist_traveled' in st.columns:
            st['shape_dist_traveled'] = st['shape_dist_traveled'].map(
              lambda x: utils.to_km(x, original_units))
        self.stop_times = st

        # Note that at least one of calendar.txt or calendar_dates.txt is
        # required by the GTFS.
        if os.path.isfile(path + 'calendar.txt'):
            self.calendar = pd.read_csv(path + 'calendar.txt', 
              dtype={'service_id': str}, 
              date_parser=lambda x: utils.date_to_str(x, inverse=True), 
              parse_dates=['start_date', 'end_date'])
            # Index by service ID to make self.is_active_trip() fast
            self.calendar_s = self.calendar.set_index('service_id')
        else:
            self.calendar = None
            self.calendar_s = None
        if os.path.isfile(path + 'calendar_dates.txt'):
            self.calendar_dates = pd.read_csv(path + 'calendar_dates.txt', 
              dtype={'service_id': str}, 
              date_parser=lambda x: utils.date_to_str(x, inverse=True), 
              parse_dates=['date'])
            # Group by service ID and date to make self.is_active_trip() fast
            self.calendar_dates_g = self.calendar_dates.groupby(
              ['service_id', 'date'])
        else:
            self.calendar_dates = None
            self.calendar_dates_g = None

        if os.path.isfile(path + 'shapes.txt'):
            shapes = pd.read_csv(path + 'shapes.txt', 
              dtype={'shape_id': str})
            # Convert distances to kilometers
            if 'shape_dist_traveled' in shapes.columns:
                shapes['shape_dist_traveled'] =\
                  shapes['shape_dist_traveled'].map(
                  lambda x: utils.to_km(x, original_units))
            self.shapes = shapes
        else:
            self.shapes = None
        
        if zipped:
            # Remove extracted directory
            shutil.rmtree(path)

    def get_dates(self):
        """
        Return a chronologically ordered list of dates 
        (``datetime.date`` objects) for which this feed is valid. 
        """
        if self.calendar is not None:
            start_date = self.calendar['start_date'].min()
            end_date = self.calendar['end_date'].max()
        else:
            # Use calendar_dates
            start_date = self.calendar_dates['date'].min()
            end_date = self.calendar_dates['date'].max()
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
        If the given trip (trip ID) is active on the given date 
        (date object), then return ``True``.
        Otherwise, return ``False``.
        To avoid error checking in the interest of speed, 
        assume ``trip`` is a valid trip ID in the feed and 
        ``date`` is a valid date object.
        """
        service = self.trips_t.at[trip, 'service_id']
        # Check self.calendar_dates_g.
        caldg = self.calendar_dates_g
        if caldg is not None:
            if (service, date) in caldg.groups:
                et = caldg.get_group((service, date))['exception_type'].iat[0]
                if et == 1:
                    return True
                else:
                    # Exception type is 2
                    return False
        # Check self.calendar_g
        cals = self.calendar_s
        if cals is not None:
            if service in cals.index:
                weekday_str = utils.weekday_to_str(date.weekday())
                if cals.at[service, 'start_date'] <= date <= cals.at[service,
                  'end_date'] and cals.at[service, weekday_str] == 1:
                    return True
                else:
                    return False
        # If you made it here, then something went wrong
        return False

    def get_active_trips(self, date, timestr=None):
        """
        Return the section of ``self.trips`` that contains
        only trips active on the given date (``datetime.date`` object).
        If a time is given in the form of a GTFS time string %H:%M:%S,
        then return only those trips active at that date and time.
        Do not take times modulo 24.
        """
        f = self.trips.copy()
        if not date:
            return f

        f['is_active'] = f['trip_id'].map(lambda trip: 
          int(self.is_active_trip(trip, date)))
        g = f[f['is_active'] == 1]
        del g['is_active']

        if timestr is not None:
            # Get trips active during given time
            h = pd.merge(g, self.stop_times[['trip_id', 'departure_time']])
          
            def F(group):
                start = group['departure_time'].min()
                end = group['departure_time'].max()
                try:
                    return start <= timestr <= end
                except TypeError:
                    return False

            gg = h.groupby('trip_id').apply(F).reset_index()
            g = pd.merge(g, gg[gg[0]])
            del g[0]

        return g

    def get_trips_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - trip_id
        - route_id
        - direction_id
        - dates[0]: a series of ones and zeros indicating if a 
          trip is active (1) on the given date or inactive (0)
        - etc.
        - dates[-1]: ditto

        If ``dates is None``, then return ``None``.
        """
        if not dates:
            return

        f = self.trips.copy()
        for date in dates:
            f[date] = f['trip_id'].map(lambda trip: 
              int(self.is_active_trip(trip, date)))
        return f[['trip_id', 'direction_id', 'route_id'] + dates]

    def get_trips_stats(self, get_dist_from_shapes=False):
        """
        Return a Pandas data frame with the following columns:

        - trip_id
        - direction_id
        - route_id
        - shape_id
        - start_time: first departure time of the trip
        - end_time: last departure time of the trip
        - duration: duration of the trip in hours
        - start_stop_id: stop ID of the first stop of the trip 
        - end_stop_id: stop ID of the last stop of the trip
        - num_stops: number of stops on trip
        - distance: distance of the trip in kilometers; contains all ``np.nan``
          entries if ``self.shapes is None``

        NOTES:

        If ``self.stop_times`` has a ``shape_dist_traveled`` column
        and ``get_dist_from_shapes == False``,
        then use that column to compute the distance column (in km).
        Elif ``self.shapes is not None``, then compute the distance 
        column using the shapes and Shapely. 
        Otherwise, set the distances to ``np.nan``.

        Takes about 0.3 minutes on the Portland feed, which has the
        ``shape_dist_traveled`` column.
        Using ``get_dist_from_shapes=True`` on the Portland feed, yields 
        a maximum absolute difference of 0.75 km from using 
        ``get_dist_from_shapes=True``.
        """
        trips = self.trips
        stop_times = self.stop_times
        num_trips = trips.shape[0]
        
        # Initialize data frame. Base it on trips.txt.
        stats = trips[['route_id', 'trip_id', 'direction_id', 'shape_id']]

        # Compute start time, end time, duration
        f = pd.merge(trips, stop_times)

        # Convert departure times to seconds past midnight, 
        # to compute durations below
        f['departure_time'] = f['departure_time'].map(
          lambda x: utils.seconds_to_timestr(x, inverse=True))
        g = f.groupby('trip_id')
        h = g['departure_time'].agg(OrderedDict([('start_time', np.min), 
          ('end_time', np.max)]))
        h['duration'] = (h['end_time'] - h['start_time'])/3600

        # Compute start stop and end stop
        def start_stop(group):
            i = group['departure_time'].argmin()
            return group['stop_id'].at[i]

        def end_stop(group):
            i = group['departure_time'].argmax()
            return group['stop_id'].at[i]

        h['start_stop'] = g.apply(start_stop)
        h['end_stop'] = g.apply(end_stop)
        h['num_stops'] = g.size()

        # Convert times back to time strings
        h[['start_time', 'end_time']] = h[['start_time', 'end_time']].\
          applymap(lambda x: utils.seconds_to_timestr(int(x)))

        # Compute trip distance (in kilometers) from shapes
        def get_dist(group):
            shape = group['shape_id'].iat[0]
            return linestring_by_shape[shape].length/1000

        if 'shape_dist_traveled' in f.columns and not get_dist_from_shapes:
            # Compute distances using shape_dist_traveled column
            h['distance'] = g.apply(lambda group: 
            group['shape_dist_traveled'].max())
        elif self.shapes is not None:
            # Compute distances using the shapes and Shapely
            linestring_by_shape = self.get_linestring_by_shape()
            h['distance'] = g.apply(get_dist)
        else:
            h['distance'] = np.nan

        stats = pd.merge(stats, h.reset_index()).sort(['route_id', 
          'direction_id', 'start_time'])
        return stats

    def get_linestring_by_shape(self, use_utm=True):
        """
        Return a dictionary with structure
        shape_id -> Shapely linestring of shape.
        If ``self.shapes is None``, then return ``None``.
        If ``use_utm == True``, then return each linestring in
        in UTM coordinates.
        Otherwise, return each linestring in WGS84 longitude-latitude
        coordinates.
        """
        if self.shapes is None:
            return

        # Note the output for conversion to UTM with the utm package:
        # >>> u = utm.from_latlon(47.9941214, 7.8509671)
        # >>> print u
        # (414278, 5316285, 32, 'T')
        linestring_by_shape = {}
        if use_utm:
            for shape, group in self.shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                xys = [utm.from_latlon(lat, lon)[:2] 
                  for lat, lon in zip(lats, lons)]
                linestring_by_shape[shape] = LineString(xys)
        else:
            for shape, group in self.shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                lonlats = zip(lons, lats)
                linestring_by_shape[shape] = LineString(lonlats)
        return linestring_by_shape

    def get_vehicles_locations(self, linestring_by_shape, date, timestr):
        """
        Return a Pandas data frame of the positions of all trips 
        active on the given date and time.
        Include the columns:

        - trip_id
        - direction_id
        - route_id
        - rel_dist: number between 0 (start) and 1 (end) indicating 
          the relative distance of the vehicle along its path
        - lon: longitude of vehicle at given time
        - lat: latitude of vehicle at given time

        Requires input ``self.get_linestring_from_shape(use_utm=False)``.
        Assume ``self.stop_times`` has a ``shape_dist_traveled``
        column, possibly created by ``add_dist_to_stop_times()``.
        """
        assert 'shape_dist_traveled' in self.stop_times.columns,\
          "The shape_dist_traveled column is required in self.stop_times."\
          "You can add it via self.stop_times = self.add_dist_to_stop_times()."
        
        # Get active trips
        at = self.get_active_trips(date, timestr)

        # Merge active trips with stop times and convert
        # times to seconds past midnight
        f = pd.merge(at, self.stop_times)
        f['departure_time'] = f['departure_time'].map(
          lambda x: utils.seconds_to_timestr(x, inverse=True))

        # Compute relative distance of each vehicle along its path
        # at the given time.
        # Use linear interpolation based on stop departure times and
        # shape distance traveled.
        t = utils.seconds_to_timestr(timestr, inverse=True)
        def F(group):
            dists = sorted(group['shape_dist_traveled'].values)
            times = sorted(group['departure_time'].values)
            d = np.interp(t, times, dists)
            return d/dists[-1]
        g = f.groupby('trip_id').apply(F).reset_index()
        g.columns = ['trip_id', 'rel_dist']
        h = pd.merge(at, g)

        # Compute longitude and latitude of vehicle from relative distance
        def G(group):
            shape = group['shape_id'].iat[0]
            linestring = linestring_by_shape[shape]
            lonlats = [linestring.interpolate(d, normalized=True).coords[0]
              for d in group['rel_dist'].values]
            group['lon'], group['lat'] = zip(*lonlats)
            return group
        return h.groupby('shape_id').apply(G)

    def get_point_by_stop(self, use_utm=True):
        """
        Return a dictionary with structure
        stop_id -> Shapely point object.
        If ``use_utm == True``, then return each point in
        in UTM coordinates.
        Otherwise, return each point in WGS84 longitude-latitude
        coordinates.
        """
        point_by_stop = {}
        if use_utm:
            for stop, group in self.stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                point_by_stop[stop] = Point(utm.from_latlon(lat, lon)[:2]) 
        else:
            for stop, group in self.stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                point_by_stop[stop] = Point([lon, lat]) 
        return point_by_stop

    def add_dist_to_stop_times(self, trips_stats):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field in
        ``self.stop_times``.

        Compute the ``shape_dist_traveled`` by using Shapely to measure 
        the distance of a stop along its trip linestring.
        If for a given trip, this process produces a non-monotonically 
        increasing, hence incorrect, list of (cumulative) distances, then
        fall back to estimating the distances as follows.
        Get the average speed of the trip via ``trips_stats`` and
        use is to linearly interpolate distances from stop times.
        This fallback method usually kicks in on trips with self-intersecting
        linestrings.

        NOTE: 

        Takes about 0.75 minutes on the Portland feed.
        98% of calculated 'shape_dist_traveled' values differ by at most
        0.56 km in absolute value from the original values, 
        and the maximum absolute difference is 6.3 km.
        """
        linestring_by_shape = self.get_linestring_by_shape()
        point_by_stop = self.get_point_by_stop()

        # Initialize data frame
        f = pd.merge(trips_stats[['trip_id', 'shape_id', 'distance', 
          'duration' ]], self.stop_times)

        # Convert departure times to seconds past midnight to ease calculations
        f['departure_time'] = f['departure_time'].map(lambda x: 
          utils.seconds_to_timestr(x, inverse=True))
        dist_by_stop_by_shape = {shape: {} for shape in linestring_by_shape}

        def get_dist(group):
            # Compute the distances of the stops along this trip
            group = group.sort('stop_sequence')
            trip = group['trip_id'].iat[0]
            shape = group['shape_id'].iat[0]
            if not isinstance(shape, str):
                print(trip, 'no shape_id:', shape)
                group['shape_dist_traveled'] = np.nan 
                return group
            linestring = linestring_by_shape[shape]
            distances = []
            for stop in group['stop_id'].values:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = utils.get_segment_length(linestring, 
                      point_by_stop[stop])
                    dist_by_stop_by_shape[shape][stop] = d
                # Convert from meters to kilometers
                d /= 1000
                distances.append(d)
            s = sorted(distances)
            if s == distances:
                # Good
                pass
            elif s == distances[::-1]:
                # Reverse. This happens when the direction of a linestring
                # opposes the direction of the bus trip.
                distances = distances[::-1]
            else:
                # Redo, and this time estimate distances 
                # using trip's average speed and linear interpolation
                distances = [0]
                dtimes = group['departure_time'].values
                speed = group['distance'].iat[0]/\
                  (group['duration'].iat[0]*3600) # km/s
                d = 0
                t_prev = dtimes[0]
                # Finish calculating distances
                for t in dtimes[1:]:
                    d += speed*(t - t_prev)
                    distances.append(d)
                    t_prev = t
            group['shape_dist_traveled'] = distances
            return group

        result = f.groupby('trip_id', group_keys=False).apply(get_dist)
        # Unconvert departure times from seconds past midnight
        result['departure_time'] = result['departure_time'].map(lambda x: 
          utils.seconds_to_timestr(x))
        del result['shape_id']
        del result['distance']
        del result['duration']
        self.stop_times = result

    def add_dist_to_shapes(self):
        """
        Add/overwrite the optional ``shape_dist_traveled`` GTFS field for
        ``self.shapes``.

        NOTE: 

        Takes about 0.33 minutes on the Portland feed.
        All of the calculated ``shape_dist_traveled`` values 
        for the Portland feed differ by at most 0.016 km in absolute values
        from of the original values. 
        """
        assert self.shapes is not None,\
          "This method requires the feed to have a shapes.txt file"

        f = self.shapes

        def get_dist(group):
            # Compute the distances of the stops along this trip
            group = group.sort('shape_pt_sequence')
            shape = group['shape_id'].iat[0]
            if not isinstance(shape, str):
                print(trip, 'no shape_id:', shape)
                group['shape_dist_traveled'] = np.nan 
                return group
            points = [Point(utm.from_latlon(lat, lon)[:2]) 
              for lon, lat in group[['shape_pt_lon', 'shape_pt_lat']].values]
            p_prev = points[0]
            d = 0
            distances = [0]
            for  p in points[1:]:
                d += p.distance(p_prev)/1000
                distances.append(d)
                p_prev = p
            group['shape_dist_traveled'] = distances
            return group

        result = f.groupby('shape_id', group_keys=False).apply(get_dist)
        self.shapes = result

    def get_stops_activity(self, dates):
        """
        Return a Pandas data frame with the columns

        - stop_id
        - dates[0]: a series of ones and zeros indicating if a 
          stop has stop times on this date (1) or not (0)
        - etc.
        - dates[-1]: ditto

        If ``dates is None``, then return ``None``.
        """
        if not dates:
            return

        trips_activity = self.get_trips_activity(dates)
        g = pd.merge(trips_activity, self.stop_times).groupby('stop_id')
        # Pandas won't allow me to simply return g[dates].max().reset_index().
        # I get ``TypeError: unorderable types: datetime.date() < str()``.
        # So here's a workaround.
        for (i, date) in enumerate(dates):
            if i == 0:
                f = g[date].max().reset_index()
            else:
                f = pd.merge(f, g[date].max().reset_index())
        return f

    def get_stops_stats(self, dates, split_directions=False):
        """
        Return a Pandas data frame with the following columns:

        - stop_id
        - direction_id
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

        If ``split_directions == False``, then compute each stop's stats
        using vehicles visiting it from both directions.

        NOTES:

        Takes about 0.7 minutes on the Portland feed given the first
        five weekdays of the feed.
        """
        # Get active trips and merge with stop times
        trips_activity = self.get_trips_activity(dates)
        ta = trips_activity[trips_activity[dates].sum(axis=1) > 0]
        f = pd.merge(ta, self.stop_times)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(lambda x: 
          utils.seconds_to_timestr(x, inverse=True))

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
            df = pd.DataFrame([[
              min_start_time, 
              max_end_time, 
              mean_daily_num_vehicles, 
              max_headway, 
              mean_headway,
              ]], 
              columns=[
              'min_start_time', 
              'max_end_time', 
              'mean_daily_num_vehicles', 
              'max_headway', 
              'mean_headway',
              ])
            df.index.name = 'foo'
            return df

        if split_directions:
            g = f.groupby(['stop_id', 'direction_id'])
        else:
            g = f.groupby('stop_id')

        result = g.apply(get_stop_stats).reset_index()

        # Convert start and end times to time strings
        result[['min_start_time', 'max_end_time']] =\
          result[['min_start_time', 'max_end_time']].applymap(
          lambda x: utils.seconds_to_timestr(x))
        del result['foo']

        return result

    def get_stops_time_series(self, dates, split_directions=False,
      freq='5Min'):
        """
        Return a time series version of the following stops stats
        for the given dates:
        
        - mean daily number of vehicles by stop ID

        The time series is a Pandas data frame with a timestamp index 
        for a 24-hour period sampled at the given frequency.
        The maximum allowable frequency is 1 minute.
        If multiples dates are given, a generic placeholder date of
        2001-01-01 is used as the date for the timestamp index.
        Otherwise, the given date is used.

        Using a period index instead of a timestamp index would be more
        apppropriate, but 
        `Pandas 0.14.1 doesn't support period index frequencies at multiples of DateOffsets (e.g. '5Min') <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period>`_.

        The columns of the data frame are hierarchical (multi-index) with

        - top level: name = 'statistic', values = ['mean_daily_num_vehicles']
        - middle level: name = 'stop_id', values = the active stop IDs
        - bottom level: name = 'direction_id', values = 0s and 1s

        If ``split_directions == False``, then don't include the bottom level.
        
        NOTES:

        - 'mean_daily_num_vehicles' should be resampled with ``how=np.sum``
        - To remove the placeholder date (2001-1-1) and seconds from 
          the time series f, do ``f.index = [t.time().strftime('%H:%M') 
          for t in f.index.to_datetime()]``
        - Takes about 6.15 minutes on the Portland feed given the first
          five weekdays of the feed.
        """  
        if not dates:
            return 

        # Get active trips and merge with stop times
        trips_activity = self.get_trips_activity(dates)
        ta = trips_activity[trips_activity[dates].sum(axis=1) > 0]
        stats = pd.merge(ta, self.stop_times)
        n = len(dates)
        stats['weight'] = stats[dates].sum(axis=1)/n

        # Create a dictionary of time series and combine them all 
        # at the end
        if n > 1:
            # Assign a uniform generic date for the index
            date_str = '2000-1-1'
        else:
            # Use the given date for the index
            date_str = utils.date_to_str(dates[0]) 
        day_start = pd.to_datetime(date_str + ' 00:00:00')
        day_end = pd.to_datetime(date_str + ' 23:59:00')
        rng = pd.period_range(day_start, day_end, freq='Min')
        names = ['mean_daily_num_vehicles']
        series_by_name = {}

        if split_directions:
            # Alter stop IDs to encode trip direction: 
            # <stop ID>-0 and <stop ID>-1
            stats['stop_id'] = stats['stop_id'] + '-' +\
              stats['direction_id'].map(str)            
        stops = sorted(stats['stop_id'].unique())

        for name in names:
            series_by_name[name] = pd.DataFrame(np.nan, index=rng, 
              columns=stops)
        
        def format(x):
            try:
                return pd.to_datetime(date_str + ' ' + utils.timestr_mod_24(x))
            except TypeError:
                return pd.NaT

        # Convert departure time string to datetime
        stats['departure_time'] = stats['departure_time'].map(format)

        # Bin each trip according to its departure time at the stop
        i = 0
        f = series_by_name['mean_daily_num_vehicles']
        num_rows = stats.shape[0]
        for index, row in stats.iterrows():
            i += 1
            print("Stops time series progress {:2.1%}".format(i/num_rows), 
              end="\r")
            stop = row['stop_id']
            weight = row['weight']
            dtime = row['departure_time']
            # Bin stop time
            criterion = f.index == dtime
            g = f.loc[criterion, stop] 
            # Use fill_value=0 to overwrite NaNs with numbers.
            # Need to use pd.Series() to get fill_value to work.
            f.loc[criterion, stop] = g.add(pd.Series(
              weight, index=g.index), fill_value=0)
      
        # Combine dictionary of time series into one time series
        f = _combine_time_series(series_by_name, kind='stop',
          split_directions=split_directions)
        # Convert to timestamp index, because Pandas 0.14.1 can't handle
        # period index frequencies at multiples of DateOffsets (e.g. '5Min') 
        f = f.to_timestamp()
        return downsample(f, freq=freq)

    def get_stops_in_stations(self):
        """
        If this feed has station data, that is, 'location_type' and
        'parent_station' columns in ``self.stops``, then return a Pandas
        data frame that has the same columns as ``self.stops``
        but only includes stops with parent stations, that is, stops with
        location type 0 or blank and nonblank parent station.
        Otherwise, return ``None``.
        """
        f = self.stops
        result = f[(f['location_type'] != 1) & (f['parent_station'].notnull())]
        if result.empty:
            return
        return result

    def get_stations_stats(self, dates, split_directions=False):
        """
        If this feed has station data, that is, 'location_type' and
        'parent_station' columns in ``self.stops``, then compute
        the same stats that ``self.get_stops_stats()`` does, but for
        stations.
        Otherwise, return ``None``.

        NOTES:

        Takes about 0.2 minutes on the Portland feed given the first
        five weekdays of the feed.
        """
        # Get stop times of active trips that visit stops in stations
        stop_times = self.stop_times
        trips_activity = self.get_trips_activity(dates)
        ta = trips_activity[trips_activity[dates].sum(axis=1) > 0]
        sis = self.get_stops_in_stations()
        if sis is None:
            return

        f = pd.merge(stop_times, ta)
        f = pd.merge(f, sis)

        # Convert departure times to seconds to ease headway calculations
        f['departure_time'] = f['departure_time'].map(lambda x: 
          utils.seconds_to_timestr(x, inverse=True))

        # Compute stats for each station
        def get_station_stats(group):
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
            df = pd.DataFrame([[
              min_start_time, 
              max_end_time, 
              mean_daily_num_vehicles, 
              max_headway, 
              mean_headway,
              ]], 
              columns=[
              'min_start_time', 
              'max_end_time', 
              'mean_daily_num_vehicles', 
              'max_headway', 
              'mean_headway',
              ])
            df.index.name = 'foo'
            return df

        if split_directions:
            g = f.groupby(['parent_station', 'direction_id'])
        else:
            g = f.groupby('parent_station')

        result = g.apply(get_station_stats).reset_index()

        # Convert start and end times to time strings
        result[['min_start_time', 'max_end_time']] =\
          result[['min_start_time', 'max_end_time']].applymap(
          lambda x: utils.seconds_to_timestr(x))
        del result['foo']

        return result

    def get_routes_stats(self, trips_stats, dates, split_directions=False,
        headway_start_timestr='07:00:00', headway_end_timestr='19:00:00'):
        """
        Take ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, and use it to calculate stats for 
        all the routes active on at least one day of the given dates
        (list of ``datetime.date`` objects). 
        
        Return a Pandas data frame with the following columns:

        - route_id
        - direction_id
        - mean_daily_num_trips
        - min_start_time: start time of the earliest active trip on 
          the route
        - max_end_time: end time of latest active trip on the route
        - max_headway: maximum of the durations (in seconds) between 
          trip starts on the route between 07:00 and 19:00 on the given dates
        - mean_headway: mean of the durations (in seconds) between 
          trip starts on the route between 07:00 and 19:00 on the given dates
        - mean_daily_duration: in hours
        - mean_daily_distance: in kilometers; contains all ``np.nan`` entries
          if ``self.shapes is None``  
        - mean_daily_speed: in kilometers per hour

        If ``split_directions == False``, then remove the direction_id column
        and compute each route's stats, except for headways, using its trips
        running in both directions. 
        In this case, (1) compute max headway by taking the max of the max 
        headways in both directions; 
        (2) compute mean headway by taking the weighted mean of the mean
        headways in both directions. 

        NOTES:

        Takes about 0.2 minutes on the Portland feed given the first
        five weekdays of the feed.
        """
        if not dates:
            return 

        # Merge trips stats with trips activity, 
        # assign a weight to each trip equal to the fraction of days in 
        # dates for which it is active, and and drop 0-weight trips
        trips_stats = pd.merge(trips_stats, self.get_trips_activity(dates))
        trips_stats['weight'] = trips_stats[dates].sum(axis=1)/len(dates)
        trips_stats = trips_stats[trips_stats['weight'] > 0]
        
        # Convert trip start times to seconds to ease headway calculations
        trips_stats['start_time'] = trips_stats['start_time'].map(lambda x: 
          utils.seconds_to_timestr(x, inverse=True))

        headway_start = utils.seconds_to_timestr(headway_start_timestr, 
          inverse=True)
        headway_end = utils.seconds_to_timestr(headway_end_timestr, 
          inverse=True)

        def get_route_stats_split_directions(group):
            # Take this group of all trips stats for a single route
            # and compute route-level stats.
            headways = []
            for date in dates:
                stimes = group[(group[date] > 0)]['start_time'].\
                  values
                stimes = sorted([stime for stime in stimes 
                  if headway_start <= stime <= headway_end])
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
            df = pd.DataFrame([[
              min_start_time, 
              max_end_time, 
              mean_daily_num_trips, 
              max_headway, 
              mean_headway, 
              mean_daily_duration, 
              mean_daily_distance,
              ]], columns=[
              'min_start_time', 
              'max_end_time', 
              'mean_daily_num_trips', 
              'max_headway', 
              'mean_headway', 
              'mean_daily_duration', 
              'mean_daily_distance',
              ])
            df.index.name = 'foo'
            return df

        def get_route_stats(group):
            # Compute headways. Need to separate directions for these.
            headways = []
            for direction in [0, 1]:
                for date in dates:
                    stimes = group[(group[date] > 0) &\
                      (group['direction_id'] == direction)]['start_time'].\
                      values
                    stimes = sorted([stime for stime in stimes 
                      if headway_start <= stime <= headway_end])
                    headways.extend([stimes[i + 1] - stimes[i] 
                      for i in range(len(stimes) - 1)])
            if headways:
                max_headway = np.max(headways)
                mean_headway = round(np.mean(headways))
            else:
                max_headway = np.nan
                mean_headway = np.nan
            # Compute rest of stats
            mean_daily_num_trips = group['weight'].sum()
            min_start_time = group['start_time'].min()
            max_end_time = group['end_time'].max()
            mean_daily_duration = (group['duration']*group['weight']).sum()
            mean_daily_distance = (group['distance']*group['weight']).sum()
            df = pd.DataFrame([[
              min_start_time, 
              max_end_time, 
              mean_daily_num_trips, 
              max_headway, 
              mean_headway, 
              mean_daily_duration, 
              mean_daily_distance,
              ]], columns=[
              'min_start_time', 
              'max_end_time', 
              'mean_daily_num_trips', 
              'max_headway', 
              'mean_headway', 
              'mean_daily_duration', 
              'mean_daily_distance',
              ])
            df.index.name = 'foo'
            return df

        if split_directions:
            result = trips_stats.groupby(['route_id', 'direction_id']).apply(
              get_route_stats_split_directions).reset_index()
        else:
            result = trips_stats.groupby('route_id').apply(
              get_route_stats).reset_index()

        del result['foo']

        # Add speed column
        result['mean_daily_speed'] = result['mean_daily_distance'].\
          divide(result['mean_daily_duration'])

        # Convert route start times to time strings
        result['min_start_time'] = result['min_start_time'].map(lambda x: 
          utils.seconds_to_timestr(x))

        return result

    def get_routes_time_series(self, trips_stats, dates, 
      split_directions=False, freq='5Min'):
        """
        Given ``trips_stats``, which is the output of 
        ``self.get_trips_stats()``, return a time series version of the 
        following route stats for the given dates:
        
        - mean daily number of vehicles in service by route ID
        - mean daily number of trip starts by route ID
        - mean daily service duration in hours by route ID
        - mean daily service distance in kilometers by route ID
        - mean daily speed in kilometers per hour

        The time series is a Pandas data frame with a timestamp index 
        for a 24-hour period sampled at the given frequency.
        The maximum allowable frequency is 1 minute.
        If multiples dates are given, a generic placeholder date of
        2001-01-01 is used as the date for the timestamp index.
        Otherwise, the given date is used.

        Using a period index instead of a timestamp index would be more
        apppropriate, but 
        `Pandas 0.14.1 doesn't support period index frequencies at multiples of DateOffsets (e.g. '5Min') <http://pandas.pydata.org/pandas-docs/stable/timeseries.html#period>`_.


        The columns of the data frame are hierarchical (multi-index) with

        - top level: name = 'statistic', values = ['mean_daily_distance',
          'mean_daily_duration', 'mean_daily_num_trip_starts', 
          'mean_daily_num_vehicles', 'mean_daily_speed']
        - middle level: name = 'stop_id', values = the active stop IDs
        - bottom level: name = 'direction_id', values = 0s and 1s

        If ``split_directions == False``, then don't include the bottom level.
        
        NOTES:

        - To resample the resulting time series use the following methods:
            - for 'mean_daily_num_vehicles' series, use ``how=np.mean``
            - for the other series, use ``how=np.sum`` 
            - 'mean_daily_speed' can't be resampled and must be recalculated
              from 'mean_daily_distance' and 'mean_daily_duration' 
        - To remove the placeholder date (2001-1-1) and seconds from the 
          time series f, do ``f.index = [t.time().strftime('%H:%M') 
          for t in f.index.to_datetime()]``
        - Takes about 0.6 minutes on the Portland feed given the first
          five weekdays of the feed.
        """  
        if not dates:
            return 
        # Merge trips_stats with trips activity, get trip weights,
        # and drop 0-weight trips
        n = len(dates)
        stats = pd.merge(trips_stats, self.get_trips_activity(dates))
        stats['weight'] = stats[dates].sum(axis=1)/n
        stats = stats[stats.weight > 0]

        if split_directions:
            # Alter route IDs to encode direction: 
            # <route ID>-0 and <route ID>-1
            stats['route_id'] = stats['route_id'] + '-' +\
              stats['direction_id'].map(str)
            
        routes = sorted(stats['route_id'].unique())
        
        # Build a dictionary of time series and then merge them all
        # at the end
        if n > 1:
            # Assign a uniform generic date for the index
            date_str = '2000-1-1'
        else:
            # Use the given date for the index
            date_str = utils.date_to_str(dates[0]) 
        day_start = pd.to_datetime(date_str + ' 00:00:00')
        day_end = pd.to_datetime(date_str + ' 23:59:00')
        rng = pd.period_range(day_start, day_end, freq='Min')
        names = [
          'mean_daily_num_vehicles', 
          'mean_daily_num_trip_starts', 
          'mean_daily_duration', 
          'mean_daily_distance',
          ]
        series_by_name = {}
        for name in names:
            series_by_name[name] = pd.DataFrame(np.nan, index=rng, 
              columns=routes)
        
        # Bin each trip according to its start and end time and weight
        i = 0
        num_rows = stats.shape[0]
        for index, row in stats.iterrows():
            i += 1
            print("Routes time series progress {:2.1%}".format(i/num_rows), 
              end="\r")
            trip = row['trip_id']
            route = row['route_id']
            weight = row['weight']
            start_time = row['start_time']
            end_time = row['end_time']
            start = pd.to_datetime(date_str + ' ' +\
              utils.timestr_mod_24(start_time))
            end = pd.to_datetime(date_str + ' ' +\
              utils.timestr_mod_24(end_time))

            for name, f in series_by_name.items():
                if start == end:
                    criterion = f.index == start
                    num_bins = 1
                elif start < end:
                    criterion = (f.index >= start) & (f.index < end)
                    num_bins = (end - start).seconds/60
                else:
                    criterion = (f.index >= start) | (f.index < end)
                    # Need to add 1 because day_end is 23:59 not 24:00
                    num_bins = (day_end - start).seconds/60 + 1 +\
                      (end - day_start).seconds/60
                # Bin route
                g = f.loc[criterion, route] 
                # Use fill_value=0 to overwrite NaNs with numbers.
                # Need to use pd.Series() to get fill_value to work.
                if name == 'mean_daily_num_trip_starts':
                    f.loc[f.index == start, route] = g.add(pd.Series(
                      weight, index=g.index), fill_value=0)
                elif name == 'mean_daily_num_vehicles':
                    f.loc[criterion, route] = g.add(pd.Series(
                      weight, index=g.index), fill_value=0)
                elif name == 'mean_daily_duration':
                    f.loc[criterion, route] = g.add(pd.Series(
                      weight/60, index=g.index), fill_value=0)
                else:
                    # name == 'distance'
                    f.loc[criterion, route] = g.add(pd.Series(
                      weight*row['distance']/num_bins, index=g.index),
                      fill_value=0)
                    
        # Combine dictionary of time series into one time series
        g = _combine_time_series(series_by_name, kind='route',
          split_directions=split_directions)
        # Convert to timestamp index, because Pandas 0.14.1 can't handle
        # period index frequencies at multiples of DateOffsets (e.g. '5Min') 
        g = g.to_timestamp()
        return downsample(g, freq=freq)

    def dump_all_stats(self, directory, dates=None, freq='1H', 
      split_directions=False):
        """
        Into the given directory, dump to separate CSV files the outputs of
        
        - ``self.get_stops_stats(dates)``
        - ``self.get_stops_time_series(dates)``
        - ``trips_stats = self.get_trips_stats()``
        - ``self.get_routes_stats(trips_stats, dates)``
        - ``self.get_routes_time_series(dates)``

        where each time series is resampled to the given frequency.
        Also include a ``README.txt`` file that contains a few notes
        on units and include some useful charts.

        If no dates are given, then use ``self.get_first_week()[:5]``.
        """
        import os
        import textwrap

        if not os.path.exists(directory):
            os.makedirs(directory)
        if dates is None:
            dates = self.get_first_week()[:5]
        dates_str = ' to '.join(
          [utils.date_to_str(d) for d in [dates[0], dates[-1]]])

        # Write README.txt, which contains notes on units and date range
        readme = """
        Notes 
        =====
        - Distances are measured in kilometers and durations are measured in
        hours
        - Stats were calculated for the period {!s}
        """.format(dates_str)
        
        with open(directory + 'notes.rst', 'w') as f:
            f.write(textwrap.dedent(readme))

        # Stops stats
        stops_stats = self.get_stops_stats(dates)
        stops_stats.to_csv(directory + 'stops_stats.csv', index=False)

        # Stops time series
        sts = self.get_stops_time_series(dates, 
          split_directions=split_directions)
        sts = downsample(sts, freq=freq)
        sts.to_csv(directory + 'stops_time_series_{!s}.csv'.format(freq))

        # Trips stats
        trips_stats = self.get_trips_stats()
        trips_stats.to_csv(directory + 'trips_stats.csv', index=False)

        # Routes stats
        routes_stats = self.get_routes_stats(trips_stats, dates,
          split_directions=split_directions)
        routes_stats.to_csv(directory + 'routes_stats.csv', index=False)

        # Routes time series
        rts = self.get_routes_time_series(trips_stats, dates,
          split_directions=split_directions)
        rts = downsample(rts, freq=freq)
        rts.to_csv(directory + 'routes_time_series_{!s}.csv'.format(freq))

        # Plot sum of routes stats 
        fig = plot_routes_time_series(rts)
        fig.tight_layout()
        fig.savefig(directory + 'routes_time_series_agg.pdf', dpi=200)

def _combine_time_series(time_series_dict, kind, split_directions=False):
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
    assert kind in ['stop', 'route'],\
      "kind must be 'stop' or 'route'"

    subcolumns = ['statistic']
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
