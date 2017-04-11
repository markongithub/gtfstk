import json

import pandas as pd 
import numpy as np
import utm
import shapely.geometry as sg 

from . import constants as cs
from . import helpers as hp


def get_stops(feed, date=None, trip_id=None, route_id=None, in_stations=False):
    """
    Return ``feed.stops``.
    If a date is given, then restrict the output to stops that are visited by trips active on the given date.
    If a trip ID (string) is given, then restrict the output possibly further to stops that are visited by the trip.
    Else if a route ID (string) is given, then restrict the output possibly further to stops that are visited by at least one trip on the route.
    If ``in_stations``, then restrict the output further to only include stops are within stations, if station data is available in ``feed.stops``.
    Assume the following feed attributes are not ``None``:

    - ``feed.stops``
    - Those used in :func:`get_stop_times`
    - ``feed.routes``    
    """
    s = feed.stops.copy()
    if date is not None:
        A = feed.get_stop_times(date)['stop_id']
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
    if in_stations and\
      set(['location_type', 'parent_station']) <= set(s.columns):
        s = s[(s['location_type'] != 1) & (s['parent_station'].notnull())]

    return s

def build_geometry_by_stop(feed, use_utm=False, stop_ids=None):
    """
    Return a dictionary with structure stop_id -> Shapely point object.
    If ``use_utm``, then return each point in in UTM coordinates.
    Otherwise, return each point in WGS84 longitude-latitude coordinates.
    If a list of stop IDs ``stop_ids`` is given, then only include the given stop IDs.

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
            d[stop] = sg.Point(utm.from_latlon(lat, lon)[:2]) 
    else:
        for stop, group in stops.groupby('stop_id'):
            lat, lon = group[['stop_lat', 'stop_lon']].values[0]
            d[stop] = sg.Point([lon, lat]) 
    return d

def compute_stop_activity(feed, dates):
    """
    Return a  DataFrame with the columns

    - stop_id
    - ``dates[0]``: 1 if the stop has at least one trip visiting it on ``dates[0]``; 0 otherwise 
    - ``dates[1]``: 1 if the stop has at least one trip visiting it on ``dates[1]``; 0 otherwise 
    - etc.
    - ``dates[-1]``: 1 if the stop has at least one trip visiting it on ``dates[-1]``; 0 otherwise 

    If ``dates`` is ``None`` or the empty list, then return an empty DataFrame with the column 'stop_id'.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`compute_trip_activity`
        

    """
    if not dates:
        return pd.DataFrame(columns=['stop_id'])

    trip_activity = feed.compute_trip_activity(dates)
    g = pd.merge(trip_activity, feed.stop_times).groupby('stop_id')
    # Pandas won't allow me to simply return g[dates].max().reset_index().
    # I get ``TypeError: unorderable types: datetime.date() < str()``.
    # So here's a workaround.
    for (i, date) in enumerate(dates):
        if i == 0:
            f = g[date].max().reset_index()
        else:
            f = f.merge(g[date].max().reset_index())
    return f

def compute_stop_stats(feed, date, split_directions=False,
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
    return hp.compute_stop_stats_base(feed.stop_times, feed.get_trips(date),
      split_directions=split_directions,
      headway_start_time=headway_start_time, 
      headway_end_time=headway_end_time)

def compute_stop_time_series(feed, date, split_directions=False, freq='5Min'):
    """
    Call :func:`.helpers.compute_stops_times_series_base` with the subset of trips  active on the given date and with the keyword arguments ``split_directions``and ``freq`` and with ``date_label`` equal to ``date``.
    See :func:`.helpers.compute_stop_time_series_base` for a description of the output.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`
        
    NOTES:
      - This is a more user-friendly version of :func:`.helpers.compute_stop_time_series_base`. The latter function works without a feed, though.
    """  
    return hp.compute_stop_time_series_base(feed.stop_times, 
      feed.get_trips(date), split_directions=split_directions, 
      freq=freq, date_label=date)

def get_stop_timetable(feed, stop_id, date):
    """
    Return a  DataFrame encoding the timetable for the given stop ID on the given date.
    The columns are all those in ``feed.trips`` plus those in ``feed.stop_times``.
    The result is sorted by departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.trips``
    - Those used in :func:`get_stop_times`
        
    """
    f = feed.get_stop_times(date)
    f = pd.merge(f, feed.trips)
    f = f[f['stop_id'] == stop_id]
    return f.sort_values('departure_time')

def get_stops_in_polygon(feed, polygon, geo_stops=None):
    """
    Return the slice of ``feed.stops`` that contains all stops that lie within the given Shapely Polygon object.
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
        f = hp.geometrize_stops(feed.stops)
    
    cols = f.columns
    f['hit'] = f['geometry'].within(polygon)
    f = f[f['hit']][cols]
    return hp.ungeometrize_stops(f)
