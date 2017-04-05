"""
This module defines the Feed class, which represents a GTFS feed as a collection of DataFrames, and defines some basic operations on Feed objects.
Every Feed method assumes that every attribute of the feed that represents a GTFS file, such as ``agency`` or ``stops``, is either ``None`` or a DataFrame with the columns required in the `GTFS <https://developers.google.com/transit/gtfs/reference?hl=en>`_.

CONVENTIONS:
    - All dates mentioned below are encoded as YYYYMMDD strings, as in the GTFS

"""
from pathlib import Path
import tempfile
import shutil
from copy import deepcopy
import dateutil.relativedelta as rd
from collections import OrderedDict
import json

import pandas as pd 
import numpy as np
import utm
import shapely.geometry as sg 

from . import constants as cs
from . import helpers as hp


class Feed(object):
    """
    An instance of this class represents a GTFS feed, where GTFS tables are stored as DataFrames.
    Beware, the stop times DataFrame can be big (several gigabytes), so make sure you have enough memory to handle it.

    Public instance attributes:

    - ``dist_units``: a string in ``constants.DIST_UNITS``; specifies the distance units to use when calculating various stats, such as route service distance; should match the implicit distance units of the  ``shape_dist_traveled`` column values, if present
    - ``agency``
    - ``stops``
    - ``routes``
    - ``trips``
    - ``stop_times``
    - ``calendar``
    - ``calendar_dates`` 
    - ``fare_attributes``
    - ``fare_rules``
    - ``shapes``
    - ``frequencies``
    - ``transfers``
    - ``feed_info``

    There are also a few private instance attributes that are derived from public attributes and are automatically updated when those public attributes change.
    However, for this update to work, you must properly update the primary attributes like this::

        feed.trips['route_short_name'] = 'bingo'
        feed.trips = feed.trips

    and **not** like this::

        feed.trips['route_short_name'] = 'bingo'

    The first way ensures that the altered trips DataFrame is saved as the new ``trips`` attribute, but the second way does not.
    """
    def __init__(self, dist_units, agency=None, stops=None, routes=None, 
      trips=None, stop_times=None, calendar=None, calendar_dates=None, 
      fare_attributes=None, fare_rules=None, shapes=None, 
      frequencies=None, transfers=None, feed_info=None):
        """
        Assume that every non-None input is a Pandas DataFrame, except for ``dist_units`` which should be a string in ``constants.DIST_UNITS``.

        No other format checking is performed.
        In particular, a Feed instance need not represent a valid GTFS feed.
        """
        # Set primary attributes; the @property magic below will then
        # validate some and automatically set secondary attributes
        for prop, val in locals().items():
            if prop in cs.FEED_ATTRS_PUBLIC:
                setattr(self, prop, val)        

    @property 
    def dist_units(self):
        """
        A public Feed attribute made into a property for easy validation.
        """
        return self._dist_units

    @dist_units.setter
    def dist_units(self, val):
        if val not in cs.DIST_UNITS:
            raise ValueError('Distance units are required and '\
              'must lie in {!s}'.format(cs.DIST_UNITS))
        else:
            self._dist_units = val

    # If ``self.trips`` changes then update ``self._trips_i``
    @property
    def trips(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the trips DataFrame.
        """
        return self._trips

    @trips.setter
    def trips(self, val):
        self._trips = val 
        if val is not None and not val.empty:
            self._trips_i = self._trips.set_index('trip_id')
        else:
            self._trips_i = None

    # If ``self.calendar`` changes, then update ``self._calendar_i``
    @property
    def calendar(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the calendar DataFrame.
        """
        return self._calendar

    @calendar.setter
    def calendar(self, val):
        self._calendar = val 
        if val is not None and not val.empty:
            self._calendar_i = self._calendar.set_index('service_id')
        else:
            self._calendar_i = None 

    # If ``self.calendar_dates`` changes, 
    # then update ``self._calendar_dates_g``
    @property 
    def calendar_dates(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the calendar dates DataFrame.
        """        
        return self._calendar_dates 

    @calendar_dates.setter
    def calendar_dates(self, val):
        self._calendar_dates = val
        if val is not None and not val.empty:
            self._calendar_dates_g = self._calendar_dates.groupby(
              ['service_id', 'date'])
        else:
            self._calendar_dates_g = None

    def __eq__(self, other):
        """
        Define two feeds be equal if and only if their ``constants.FEED_ATTRS`` attributes are equal, or almost equal in the case of DataFrames (but not groupby DataFrames).
        Almost equality is checked via :func:`utilities.almost_equal`, which   canonically sorts DataFrame rows and columns.
        """
        # Return False if failures
        for key in cs.FEED_ATTRS_PUBLIC:
            x = getattr(self, key)
            y = getattr(other, key)
            # DataFrame case
            if isinstance(x, pd.DataFrame):
                if not isinstance(y, pd.DataFrame) or\
                  not hp.almost_equal(x, y):
                    return False 
            # Other case
            else:
                if x != y:
                    return False
        # No failures
        return True

    def copy(self):
        """
        Return a copy of this feed, that is, a feed with all the same public and private attributes.
        """
        other = Feed(dist_units=self.dist_units)
        for key in set(cs.FEED_ATTRS) - set(['dist_units']):
            value = getattr(self, key)
            if isinstance(value, pd.DataFrame):
                # Pandas copy DataFrame
                value = value.copy()
            elif isinstance(value, pd.core.groupby.DataFrameGroupBy):
                # Pandas does not have a copy method for groupby objects
                # as far as i know
                value = deepcopy(value)
            setattr(other, key, value)
        
        return other

    # -------------------------------------
    # Methods about calendars
    # -------------------------------------
    def get_dates(self, as_date_obj=False):
        """
        Return a chronologically ordered list of dates (strings) for which this feed is valid.
        If ``as_date_obj``, then return the dates as ``datetime.date`` objects.  

        If ``self.calendar`` and ``self.calendar_dates`` are both ``None``, then return the empty list.
        """
        if self.calendar is not None:
            start_date = self.calendar['start_date'].min()
            end_date = self.calendar['end_date'].max()
        elif self.calendar_dates is not None:
            # Use calendar_dates
            start_date = self.calendar_dates['date'].min()
            end_date = self.calendar_dates['date'].max()
        else:
            return []

        start_date = hp.datestr_to_date(start_date)
        end_date = hp.datestr_to_date(end_date)
        num_days = (end_date - start_date).days
        result = [start_date + rd.relativedelta(days=+d) 
          for d in range(num_days + 1)]
        
        if not as_date_obj:
            result = [hp.datestr_to_date(x, inverse=True)
              for x in result]
        
        return result

    def get_first_week(self, as_date_obj=False):
        """
        Return a list of date corresponding to the first Monday--Sunday week for which this feed is valid.
        If the given feed does not cover a full Monday--Sunday week, then return whatever initial segment of the week it does cover, which could be the empty list.
        If ``as_date_obj``, then return the dates as as ``datetime.date`` objects.    
        """
        dates = self.get_dates(as_date_obj=True)
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
            result = [hp.datestr_to_date(x, inverse=True)
              for x in result]
        return result

    # -------------------------------------
    # Methods about trips
    # -------------------------------------
    def is_active_trip(self, trip_id, date):
        """
        If the given trip is active on the given date, then return ``True``; otherwise return ``False``.
        To avoid error checking in the interest of speed, assume ``trip`` is a valid trip ID in the given feed and ``date`` is a valid date object.

        Assume the following feed attributes are not ``None``:

        - ``self.trips``

        NOTES: 
            - This function is key for getting all trips, routes, etc. that are active on a given date, so the function needs to be fast. 
        """
        service = self._trips_i.at[trip_id, 'service_id']
        # Check self._calendar_dates_g.
        caldg = self._calendar_dates_g
        if caldg is not None:
            if (service, date) in caldg.groups:
                et = caldg.get_group((service, date))['exception_type'].iat[0]
                if et == 1:
                    return True
                else:
                    # Exception type is 2
                    return False
        # Check self._calendar_i
        cali = self._calendar_i
        if cali is not None:
            if service in cali.index:
                weekday_str = hp.weekday_to_str(
                  hp.datestr_to_date(date).weekday())
                if cali.at[service, 'start_date'] <= date <= cali.at[service,
                  'end_date'] and cali.at[service, weekday_str] == 1:
                    return True
                else:
                    return False
        # If you made it here, then something went wrong
        return False

    def get_trips(self, date=None, time=None):
        """
        Return the section of ``self.trips`` that contains only trips active on the given date.
        If ``self.trips`` is ``None`` or the date is ``None``, then return all ``self.trips``.
        If a date and time are given, then return only those trips active at that date and time.
        Do not take times modulo 24.
        """
        if self.trips is None or date is None:
            return self.trips 

        f = self.trips.copy()
        f['is_active'] = f['trip_id'].map(
          lambda trip_id: self.is_active_trip(trip_id, date))
        f = f[f['is_active']].copy()
        del f['is_active']

        if time is not None:
            # Get trips active during given time
            g = pd.merge(f, self.stop_times[['trip_id', 'departure_time']])

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

    def compute_trip_activity(self, dates):
        """
        Return a  data frame with the columns

        - trip_id
        - ``dates[0]``: 1 if the trip is active on ``dates[0]``; 0 otherwise
        - ``dates[1]``: 1 if the trip is active on ``dates[1]``; 0 otherwise
        - etc.
        - ``dates[-1]``: 1 if the trip is active on ``dates[-1]``; 0 otherwise

        If ``dates`` is ``None`` or the empty list, then return an empty data frame with the column 'trip_id'.

        Assume the following feed attributes are not ``None``:

        - ``self.trips``
        - Those used in :func:`is_active_trip`
            
        """
        if not dates:
            return pd.DataFrame(columns=['trip_id'])

        f = self.trips.copy()
        for date in dates:
            f[date] = f['trip_id'].map(lambda trip_id: 
              int(self.is_active_trip(trip_id, date)))
        return f[['trip_id'] + dates]

    def compute_busiest_date(self, dates):
        """
        Given a list of dates, return the first date that has the maximum number of active trips.
        If the list of dates is empty, then raise a ``ValueError``.

        Assume the following feed attributes are not ``None``:

        - Those used in :func:`compute_trip_activity`
            
        """
        f = self.compute_trip_activity(dates)
        s = [(f[date].sum(), date) for date in dates]
        return max(s)[1]

    def compute_trip_stats(self, compute_dist_from_shapes=False):
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
        - distance: distance of the trip in ``self.dist_units``; 
          contains all ``np.nan`` entries if ``self.shapes is None``
        - duration: duration of the trip in hours
        - speed: distance/duration

        Assume the following feed attributes are not ``None``:

        - ``self.trips``
        - ``self.routes``
        - ``self.stop_times``
        - ``self.shapes`` (optionally)
        - Those used in :func:`build_geometry_by_stop`

        NOTES:
            If ``self.stop_times`` has a ``shape_dist_traveled`` column with at least one non-NaN value and ``compute_dist_from_shapes == False``, then use that column to compute the distance column.
            Else if ``self.shapes is not None``, then compute the distance column using the shapes and Shapely. 
            Otherwise, set the distances to ``np.nan``.

            Calculating trip distances with ``compute_dist_from_shapes=True`` seems pretty accurate.
            For example, calculating trip distances on the Portland feed at https://transitfeeds.com/p/trimet/43/1400947517 using ``compute_dist_from_shapes=False`` and ``compute_dist_from_shapes=True``, yields a difference of at most 0.83km.
        """        
        # Start with stop times and extra trip info.
        # Convert departure times to seconds past midnight to 
        # compute durations.
        f = self.trips[['route_id', 'trip_id', 'direction_id', 'shape_id']]
        f = pd.merge(f, 
          self.routes[['route_id', 'route_short_name', 'route_type']])
        f = pd.merge(f, self.stop_times).sort_values(['trip_id', 'stop_sequence'])
        f['departure_time'] = f['departure_time'].map(hp.timestr_to_seconds)
        
        # Compute all trips stats except distance, 
        # which is possibly more involved
        geometry_by_stop = self.build_geometry_by_stop(use_utm=True)
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
        if hp.is_not_null(f, 'shape_dist_traveled') and\
          not compute_dist_from_shapes:
            # Compute distances using shape_dist_traveled column
            h['distance'] = g.apply(
              lambda group: group['shape_dist_traveled'].max())
        elif self.shapes is not None:
            # Compute distances using the shapes and Shapely
            geometry_by_shape = self.build_geometry_by_shape(use_utm=True)
            geometry_by_stop = self.build_geometry_by_stop(use_utm=True)
            m_to_dist = hp.get_convert_dist('m', self.dist_units)

            def compute_dist(group):
                """
                Return the distance traveled along the trip between the first and last stops.
                If that distance is negative or if the trip's linestring  intersects itfeed, then return the length of the trip's linestring instead.
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
          applymap(lambda x: hp.timestr_to_seconds(x, inverse=True))
        
        return h.sort_values(['route_id', 'direction_id', 'start_time'])

    def compute_trip_locations(self, date, times):
        """
        Return a  data frame of the positions of all trips active on the given date and times 
        Include the columns:

        - trip_id
        - route_id
        - direction_id
        - time
        - rel_dist: number between 0 (start) and 1 (end) indicating 
          the relative distance of the trip along its path
        - lon: longitude of trip at given time
        - lat: latitude of trip at given time

        Assume ``self.stop_times`` has an accurate ``shape_dist_traveled`` column.

        Assume the following feed attributes are not ``None``:

        - ``self.trips``
        - Those used in :func:`get_stop_times`
        - Those used in :func:`build_geometry_by_shape`
            
        """
        if not hp.is_not_null(self.stop_times, 'shape_dist_traveled'):
            raise ValueError(
              "self.stop_times needs to have a non-null shape_dist_traveled "\
              "column. You can create it, possibly with some inaccuracies, "\
              "via feed2 = self.append_dist_to_stop_times().")
        
        # Start with stop times active on date
        f = self.get_stop_times(date)
        f['departure_time'] = f['departure_time'].map(
          hp.timestr_to_seconds)

        # Compute relative distance of each trip along its path
        # at the given time times.
        # Use linear interpolation based on stop departure times and
        # shape distance traveled.
        geometry_by_shape = self.build_geometry_by_shape(use_utm=False)
        sample_times = np.array([hp.timestr_to_seconds(s) 
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
          lambda x: hp.timestr_to_seconds(x, inverse=True))

        # Merge in more trip info and
        # compute longitude and latitude of trip from relative distance
        h = pd.merge(g, self.trips[['trip_id', 'route_id', 'direction_id', 
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

    def trip_to_geojson(self, trip_id, include_stops=False):
        """
        Given a feed and a trip ID (string), return a (decoded) GeoJSON feature collection comprising a Linestring feature of representing the trip's shape.
        If ``include_stops``, then also include one Point feature for each stop  visited by the trip. 
        The Linestring feature will contain as properties all the columns in ``self.trips`` pertaining to the given trip, and each Point feature will contain as properties all the columns in ``self.stops`` pertaining    to the stop, except the ``stop_lat`` and ``stop_lon`` properties.

        Return the empty dictionary if the trip has no shape.
        """
        # Get the relevant shapes
        t = self.trips.copy()
        t = t[t['trip_id'] == trip_id].copy()
        shid = t['shape_id'].iat[0]
        geometry_by_shape = self.build_geometry_by_shape(use_utm=False, 
          shape_ids=[shid])

        if not geometry_by_shape:
            return {}

        features = [{
            'type': 'Feature',
            'properties': json.loads(t.to_json(orient='records'))[0],
            'geometry': sg.mapping(sg.LineString(geometry_by_shape[shid])),
            }]

        if include_stops:
            # Get relevant stops and geometrys
            s = self.get_stops(trip_id=trip_id)
            cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
            s = s[list(cols)].copy()
            stop_ids = s['stop_id'].tolist()
            geometry_by_stop = self.build_geometry_by_stop(stop_ids=stop_ids)
            features.extend([{
                'type': 'Feature',
                'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
                  orient='records'))[0],
                'geometry': sg.mapping(geometry_by_stop[stop_id]),
                } for stop_id in stop_ids])

        return {'type': 'FeatureCollection', 'features': features}

    # -------------------------------------
    # Methods about routes
    # -------------------------------------
    def get_routes(self, date=None, time=None):
        """
        Return the section of ``self.routes`` that contains only routes active on the given date.
        If no date is given, then return all routes.
        If a date and time are given, then return only those routes with trips active at that date and time.
        Do not take times modulo 24.

        Assume the following feed attributes are not ``None``:

        - ``feed.routes``
        - Those used in :func:`get_trips`.
            
        """
        if date is None:
            return self.routes.copy()

        trips = self.get_trips(date, time)
        R = trips['route_id'].unique()
        return self.routes[self.routes['route_id'].isin(R)]

    def compute_route_stats(self, trip_stats, date, 
      split_directions=False, headway_start_time='07:00:00', 
      headway_end_time='19:00:00'):
        """
        Given a DataFrame of possibly partial trip stats for this Feed in the form output by :func:`compute_trip_stats`, cut the stats down to the subset ``S`` of trips that are active on the given date.
        Then call :func:`compute_route_stats_base` with ``S`` and the keyword arguments ``split_directions``, ``headway_start_time``, and  ``headway_end_time``.
        See :func:`compute_route_stats_base` for a description of the outphp.

        Return an empty data frame if there are no route stats for the given trip stats and date.

        Assume the following feed attributes are not ``None``:

        - Those used in :func:`compute_route_stats_base`
            
        NOTES:
            - This is a more user-friendly version of :func:`compute_route_stats_base`. The latter function works without a feed, though.
        """
        # Get the subset of trips_stats that contains only trips active
        # on the given date
        trip_stats_subset = pd.merge(trip_stats, self.get_trips(date))
        return hp.compute_route_stats_base(trip_stats_subset, 
          split_directions=split_directions,
          headway_start_time=headway_start_time, 
          headway_end_time=headway_end_time)

    def compute_route_time_series(self, trip_stats, date, 
      split_directions=False, freq='5Min'):
        """
        Given a DataFrame of possibly partial trip stats for this Feed in the form output by :func:`compute_trip_stats`, cut the stats down to the subset ``S`` of trips that are active on the given date, then call :func:`compute_route_time_series_base` with ``S`` and the given keyword arguments ``split_directions`` and ``freq`` and with ``date_label = date_to_str(date)``.

        See :func:`compute_route_time_series_base` for a description of the output.

        Return an empty data frame if there are no route stats for the given trip stats and date.

        Assume the following feed attributes are not ``None``:

        - Those used in :func:`get_trips`
            

        NOTES:
            - This is a more user-friendly version of ``compute_route_time_series_base()``. The latter function works without a feed, though.
        """  
        trip_stats_subset = pd.merge(trip_stats, self.get_trips(date))
        return hp.compute_route_time_series_base(trip_stats_subset, 
          split_directions=split_directions, freq=freq, 
          date_label=date)

    def get_route_timetable(self, route_id, date):
        """
        Return a data frame encoding the timetable for the given route ID on the given date.
        The columns are all those in ``self.trips`` plus those in ``self.stop_times``.
        The result is sorted by grouping by trip ID and sorting the groups by their first departure time.

        Assume the following feed attributes are not ``None``:

        - ``self.stop_times``
        - Those used in :func:`get_trips`
            
        """
        f = self.get_trips(date)
        f = f[f['route_id'] == route_id].copy()
        f = pd.merge(f, self.stop_times)
        # Groupby trip ID and sort groups by their minimum departure time.
        # For some reason NaN departure times mess up the transform below.
        # So temporarily fill NaN departure times as a workaround.
        f['dt'] = f['departure_time'].fillna(method='ffill')
        f['min_dt'] = f.groupby('trip_id')['dt'].transform(min)
        return f.sort_values(['min_dt', 'stop_sequence']).drop(
          ['min_dt', 'dt'], axis=1)

    def route_to_geojson(self, route_id, include_stops=False):
        """
        Given a feed and a route ID (string), return a (decoded) GeoJSON 
        feature collection comprising a MultiLinestring feature of distinct shapes
        of the trips on the route.
        If ``include_stops``, then also include one Point feature for each stop 
        visited by any trip on the route. 
        The MultiLinestring feature will contain as properties all the columns
        in ``self.routes`` pertaining to the given route, and each Point feature
        will contain as properties all the columns in ``self.stops`` pertaining 
        to the stop, except the ``stop_lat`` and ``stop_lon`` properties.

        Return the empty dictionary if all the route's trips lack shapes.

        Assume the following feed attributes are not ``None``:

        - ``self.routes``
        - ``self.shapes``
        - ``self.trips``
        - ``self.stops``

        """
        # Get the relevant shapes
        t = self.trips.copy()
        A = t[t['route_id'] == route_id]['shape_id'].unique()
        geometry_by_shape = self.build_geometry_by_shape(use_utm=False, 
          shape_ids=A)

        if not geometry_by_shape:
            return {}

        r = self.routes.copy()
        features = [{
            'type': 'Feature',
            'properties': json.loads(r[r['route_id'] == route_id].to_json(
              orient='records'))[0],
            'geometry': sg.mapping(sg.MultiLineString(
              [linestring for linestring in geometry_by_shape.values()]))
            }]

        if include_stops:
            # Get relevant stops and geometrys
            s = self.get_stops(route_id=route_id)
            cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
            s = s[list(cols)].copy()
            stop_ids = s['stop_id'].tolist()
            geometry_by_stop = self.build_geometry_by_stop(stop_ids=stop_ids)
            features.extend([{
                'type': 'Feature',
                'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
                  orient='records'))[0],
                'geometry': sg.mapping(geometry_by_stop[stop_id]),
                } for stop_id in stop_ids])

        return {'type': 'FeatureCollection', 'features': features}


    # -------------------------------------
    # Methods about stops
    # -------------------------------------
    def get_stops(self, date=None, trip_id=None, route_id=None, in_stations=False):
        """
        Return ``self.stops``.
        If a date is given, then restrict the output to stops that are visited by trips active on the given date.
        If a trip ID (string) is given, then restrict the output possibly further to stops that are visited by the trip.
        Else if a route ID (string) is given, then restrict the output possibly further to stops that are visited by at least one trip on the route.
        If ``in_stations``, then restrict the output further to only include stops are within stations, if station data is available in ``self.stops``.
        Assume the following feed attributes are not ``None``:

        - ``self.stops``
        - Those used in :func:`get_stop_times`
        - ``self.routes``    
        """
        s = self.stops.copy()
        if date is not None:
            A = self.get_stop_times(date)['stop_id']
            s = s[s['stop_id'].isin(A)].copy()
        if trip_id is not None:
            st = self.stop_times.copy()
            B = st[st['trip_id'] == trip_id]['stop_id']
            s = s[s['stop_id'].isin(B)].copy()
        elif route_id is not None:
            A = self.trips[self.trips['route_id'] == route_id]['trip_id']
            st = self.stop_times.copy()
            B = st[st['trip_id'].isin(A)]['stop_id']
            s = s[s['stop_id'].isin(B)].copy()
        if in_stations and\
          set(['location_type', 'parent_station']) <= set(s.columns):
            s = s[(s['location_type'] != 1) & (s['parent_station'].notnull())]

        return s

    def build_geometry_by_stop(self, use_utm=False, stop_ids=None):
        """
        Return a dictionary with structure stop_id -> Shapely point object.
        If ``use_utm``, then return each point in in UTM coordinates.
        Otherwise, return each point in WGS84 longitude-latitude coordinates.
        If a list of stop IDs ``stop_ids`` is given, then only include the given stop IDs.

        Assume the following feed attributes are not ``None``:

        - ``self.stops``
            
        """
        d = {}
        stops = self.stops.copy()
        if stop_ids is not None:
            stops = stops[stops['stop_id'].isin(stop_ids)]

        if use_utm:
            for stop, group in stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                d[stop] = sg.Point(utm.from_latlon(lat, lon)[:2]) 
        else:
            for stop, group in stops.groupby('stop_id'):
                lat, lon = group[['stop_lat', 'stop_lon']].values[0]
                d[stop] = sg.Point([lon, lat]) 
        return d

    def compute_stop_activity(self, dates):
        """
        Return a  data frame with the columns

        - stop_id
        - ``dates[0]``: 1 if the stop has at least one trip visiting it on ``dates[0]``; 0 otherwise 
        - ``dates[1]``: 1 if the stop has at least one trip visiting it on ``dates[1]``; 0 otherwise 
        - etc.
        - ``dates[-1]``: 1 if the stop has at least one trip visiting it on ``dates[-1]``; 0 otherwise 

        If ``dates`` is ``None`` or the empty list, then return an empty data frame with the column 'stop_id'.

        Assume the following feed attributes are not ``None``:

        - ``self.stop_times``
        - Those used in :func:`compute_trip_activity`
            

        """
        if not dates:
            return pd.DataFrame(columns=['stop_id'])

        trip_activity = self.compute_trip_activity(dates)
        g = pd.merge(trip_activity, self.stop_times).groupby('stop_id')
        # Pandas won't allow me to simply return g[dates].max().reset_index().
        # I get ``TypeError: unorderable types: datetime.date() < str()``.
        # So here's a workaround.
        for (i, date) in enumerate(dates):
            if i == 0:
                f = g[date].max().reset_index()
            else:
                f = f.merge(g[date].max().reset_index())
        return f

    def compute_stop_stats(self, date, split_directions=False,
        headway_start_time='07:00:00', headway_end_time='19:00:00'):
        """
        Call ``compute_stop_stats_base()`` with the subset of trips active on the given date and with the keyword arguments ``split_directions``,   ``headway_start_time``, and ``headway_end_time``.

        See ``compute_stop_stats_base()`` for a description of the output.

        Assume the following feed attributes are not ``None``:

        - ``feed.stop_timtes``
        - Those used in :func:`get_trips`
            
        NOTES:

        This is a more user-friendly version of ``compute_stop_stats_base()``.
        The latter function works without a feed, though.
        """
        # Get stop times active on date and direction IDs
        return hp.compute_stop_stats_base(self.stop_times, self.get_trips(date),
          split_directions=split_directions,
          headway_start_time=headway_start_time, 
          headway_end_time=headway_end_time)

    def compute_stop_time_series(self, date, split_directions=False, freq='5Min'):
        """
        Call :func:`compute_stops_times_series_base` with the subset of trips  active on the given date and with the keyword arguments ``split_directions``and ``freq`` and with ``date_label`` equal to ``date``.
        See :func:`compute_stop_time_series_base` for a description of the output.

        Assume the following feed attributes are not ``None``:

        - ``self.stop_times``
        - Those used in :func:`get_trips`
            
        NOTES:
          - This is a more user-friendly version of ``compute_stop_time_series_base()``. The latter function works without a feed, though.
        """  
        return hp.compute_stop_time_series_base(self.stop_times, 
          self.get_trips(date), split_directions=split_directions, 
          freq=freq, date_label=date)

    def get_stop_timetable(self, stop_id, date):
        """
        Return a  data frame encoding the timetable for the given stop ID on the given date.
        The columns are all those in ``self.trips`` plus those in ``self.stop_times``.
        The result is sorted by departure time.

        Assume the following feed attributes are not ``None``:

        - ``self.trips``
        - Those used in :func:`get_stop_times`
            
        """
        f = self.get_stop_times(date)
        f = pd.merge(f, self.trips)
        f = f[f['stop_id'] == stop_id]
        return f.sort_values('departure_time')
 
    # -------------------------------------
    # Methods about stop times
    # -------------------------------------
    def get_stop_times(self, date=None):
        """
        Return the section of ``self.stop_times`` that contains only trips active on the given date.
        If no date is given, then return all stop times.

        Assume the following feed attributes are not ``None``:

        - ``self.stop_times``
        - Those used in :func:`get_trips`

        """
        f = self.stop_times.copy()
        if date is None:
            return f

        g = self.get_trips(date)
        return f[f['trip_id'].isin(g['trip_id'])]

    def append_dist_to_stop_times(self, trips_stats):
        """
        Calculate and append the optional ``shape_dist_traveled`` field in ``self.stop_times`` in terms of the distance units ``self.dist_units``.
        Need trip stats in the form output by :func:`compute_trip_stats` for this.
        Return the resulting Feed.
        Does not always give accurate results, as described below.

        Assume the following feed attributes are not ``None``:

        - ``self.stop_times``
        - Those used in :func:`build_geometry_by_shape`
        - Those used in :func:`build_geometry_by_stop`

        ALGORITHM:
            Compute the ``shape_dist_traveled`` field by using Shapely to measure the distance of a stop along its trip linestring.
            If for a given trip this process produces a non-monotonically    increasing, hence incorrect, list of (cumulative) distances, then   fall back to estimating the distances as follows.
            
            Get the average speed of the trip via ``trips_stats`` and use is to linearly interpolate distances for stop times, assuming that the first stop is at shape_dist_traveled = 0 (the start of the shape) and the last stop is at shape_dist_traveled = the length of the trip (taken from trips_stats and equal to the length of the shape, unless trips_stats was called with ``get_dist_from_shapes == False``).
            This fallback method usually kicks in on trips with feed-intersecting    linestrings.
            Unfortunately, this fallback method will produce incorrect results when the first stop does not start at the start of its shape (so shape_dist_traveled != 0).
            This is the case for several trips in the Portland feed at https://transitfeeds.com/p/trimet/43/1400947517, for example. 
        """
        feed = self.copy()
        geometry_by_shape = feed.build_geometry_by_shape(use_utm=True)
        geometry_by_stop = feed.build_geometry_by_stop(use_utm=True)

        # Initialize data frame
        f = pd.merge(self.stop_times,
          trips_stats[['trip_id', 'shape_id', 'distance', 'duration']]).\
          sort_values(['trip_id', 'stop_sequence'])

        # Convert departure times to seconds past midnight to ease calculations
        f['departure_time'] = f['departure_time'].map(hp.timestr_to_seconds)
        dist_by_stop_by_shape = {shape: {} for shape in geometry_by_shape}
        m_to_dist = hp.get_convert_dist('m', self.dist_units)

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
                    d = m_to_dist(hp.get_segment_length(linestring, 
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

        g = f.groupby('trip_id', group_keys=False).apply(compute_dist)
        # Convert departure times back to time strings
        g['departure_time'] = g['departure_time'].map(lambda x: 
          hp.timestr_to_seconds(x, inverse=True))
        g = g.drop(['shape_id', 'distance', 'duration'], axis=1)
        feed.stop_times = g 

        return feed 

    # -------------------------------------
    # Methods about shapes
    # -------------------------------------
    def build_geometry_by_shape(self, use_utm=False, shape_ids=None):
        """
        Return a dictionary with structure shape_id -> Shapely linestring of shape.
        If ``self.shapes is None``, then return ``None``.
        If ``use_utm``, then return each linestring in in UTM coordinates.
        Otherwise, return each linestring in WGS84 longitude-latitude    coordinates.
        If a list of shape IDs ``shape_ids`` is given, then only include the given shape IDs.

        Return the empty dictionary if ``self.shapes is None``.
        """
        if self.shapes is None:
            return {}

        # Note the output for conversion to UTM with the utm package:
        # >>> u = utm.from_latlon(47.9941214, 7.8509671)
        # >>> print u
        # (414278, 5316285, 32, 'T')
        d = {}
        shapes = self.shapes.copy()
        if shape_ids is not None:
            shapes = shapes[shapes['shape_id'].isin(shape_ids)]

        if use_utm:
            for shape, group in shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                xys = [utm.from_latlon(lat, lon)[:2] 
                  for lat, lon in zip(lats, lons)]
                d[shape] = sg.LineString(xys)
        else:
            for shape, group in shapes.groupby('shape_id'):
                lons = group['shape_pt_lon'].values
                lats = group['shape_pt_lat'].values
                lonlats = zip(lons, lats)
                d[shape] = sg.LineString(lonlats)
        return d

    def shapes_to_geojson(self):
        """
        Return a (decoded) GeoJSON FeatureCollection of linestring features representing ``self.shapes``.
        Each feature will have a ``shape_id`` property. 
        The coordinates reference system is the default one for GeoJSON, namely WGS84.

        Return the empty dictionary of ``self.shapes is None``
        """
        geometry_by_shape = self.build_geometry_by_shape(use_utm=False)
        if geometry_by_shape:
            fc = {
              'type': 'FeatureCollection', 
              'features': [{
                'properties': {'shape_id': shape},
                'type': 'Feature',
                'geometry': sg.mapping(linestring),
                }
                for shape, linestring in geometry_by_shape.items()]
              }
        else:
            fc = {}
        return fc 

# -------------------------------------
# Functions about input and output
# -------------------------------------
def read_gtfs(path, dist_units=None):
    """
    Create a Feed object from the given path and given distance units.
    The path should be a directory containing GTFS text files or a zip file that unzips as a collection of GTFS text files (and not as a directory containing GTFS text files).
    """
    path = Path(path)
    if not path.exists():
        raise ValueError("Path {!s} does not exist".format(path))

    # Unzip path to temporary directory if necessary
    if path.is_file():
        zipped = True
        tmp_dir = tempfile.TemporaryDirectory()
        src_path = Path(tmp_dir.name)
        shutil.unpack_archive(str(path), tmp_dir.name, 'zip')
    else:
        zipped = False
        src_path = path

    # Read files into feed dictionary of DataFrames
    tables = cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL
    feed_dict = {table: None for table in tables}
    for p in src_path.iterdir():
        table = p.stem
        if p.is_file() and table in tables:
            feed_dict[table] = pd.read_csv(p, dtype=cs.DTYPE, 
              encoding='utf-8-sig') 
            # utf-8-sig gets rid of the byte order mark (BOM);
            # see http://stackoverflow.com/questions/17912307/u-ufeff-in-python-string 
        
    feed_dict['dist_units'] = dist_units

    # Delete temporary directory
    if zipped:
        tmp_dir.cleanup()

    # Create feed 
    return Feed(**feed_dict)

def write_gtfs(feed, path, ndigits=6):
    """
    Export the given feed to the given path.
    If the path end in '.zip', then write the feed as a zip archive.
    Otherwise assume the path is a directory, and write the feed as a collection of CSV files to that directory, creating the directory if it does not exist.
    Round all decimals to ``ndigits`` decimal places.
    All distances will be the distance units ``feed.dist_units``.
    """
    path = Path(path)

    if path.suffix == '.zip':
        # Write to temporary directory before zipping
        zipped = True
        tmp_dir = tempfile.TemporaryDirectory()
        new_path = Path(tmp_dir.name)
    else:
        zipped = False
        if not path.exists():
            path.mkdir()
        new_path = path 

    tables = cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL
    for table in tables:
        f = getattr(feed, table)
        if f is None:
            continue

        f = f.copy()
        # Some columns need to be output as integers.
        # If there are NaNs in any such column, 
        # then Pandas will format the column as float, which we don't want.
        f_int_cols = set(cs.INT_COLUMNS) & set(f.columns)
        for s in f_int_cols:
            f[s] = f[s].fillna(-1).astype(int).astype(str).\
              replace('-1', '')
        p = new_path/(table + '.txt')
        f.to_csv(str(p), index=False, float_format='%.{!s}f'.format(ndigits))

    # Zip directory 
    if zipped:
        basename = str(path.parent/path.stem)
        shutil.make_archive(basename, format='zip', root_dir=tmp_dir.name)    
        tmp_dir.cleanup()
