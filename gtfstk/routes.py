"""
Functions about routes
"""
import json

import pandas as pd 
import numpy as np
import utm
import shapely.geometry as sg 

from . import constants as cs
from . import helpers as hp


def get_routes(feed, date=None, time=None):
    """
    Return the section of ``feed.routes`` that contains only routes active on the given date.
    If no date is given, then return all routes.
    If a date and time are given, then return only those routes with trips active at that date and time.
    Do not take times modulo 24.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - Those used in :func:`get_trips`.
        
    """
    if date is None:
        return feed.routes.copy()

    trips = feed.get_trips(date, time)
    R = trips['route_id'].unique()
    return feed.routes[feed.routes['route_id'].isin(R)]

def compute_route_stats(feed, trip_stats, date, 
  split_directions=False, headway_start_time='07:00:00', 
  headway_end_time='19:00:00'):
    """
    Given a DataFrame of possibly partial trip stats for this Feed in the form output by :func:`compute_trip_stats`, cut the stats down to the subset ``S`` of trips that are active on the given date.
    Then call :func:`.helpers.compute_route_stats_base` with ``S`` and the keyword arguments ``split_directions``, ``headway_start_time``, and  ``headway_end_time``.
    See :func:`.helpers.compute_route_stats_base` for a description of the output.

    Return an empty DataFrame if there are no route stats for the given trip stats and date.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`.helpers.compute_route_stats_base`
        
    NOTES:
        - This is a more user-friendly version of :func:`.helpers.compute_route_stats_base`. The latter function works without a feed, though.
    """
    # Get the subset of trips_stats that contains only trips active
    # on the given date
    trip_stats_subset = pd.merge(trip_stats, feed.get_trips(date))
    return hp.compute_route_stats_base(trip_stats_subset, 
      split_directions=split_directions,
      headway_start_time=headway_start_time, 
      headway_end_time=headway_end_time)

def compute_route_time_series(feed, trip_stats, date, 
  split_directions=False, freq='5Min'):
    """
    Given a DataFrame of possibly partial trip stats for this Feed in the form output by :func:`compute_trip_stats`, cut the stats down to the subset ``S`` of trips that are active on the given date, then call :func:`.helpers.compute_route_time_series_base` with ``S`` and the given keyword arguments ``split_directions`` and ``freq`` and with ``date_label = date_to_str(date)``.

    See :func:`.helpers.compute_route_time_series_base` for a description of the output.

    Return an empty DataFrame if there are no route stats for the given trip stats and date.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_trips`
        

    NOTES:
        - This is a more user-friendly version of :func:`.helpers.compute_route_time_series_base`. The latter function works without a feed, though.
    """  
    trip_stats_subset = pd.merge(trip_stats, feed.get_trips(date))
    return hp.compute_route_time_series_base(trip_stats_subset, 
      split_directions=split_directions, freq=freq, 
      date_label=date)

def get_route_timetable(feed, route_id, date):
    """
    Return a DataFrame encoding the timetable for the given route ID on the given date.
    The columns are all those in ``feed.trips`` plus those in ``feed.stop_times``.
    The result is sorted by grouping by trip ID and sorting the groups by their first departure time.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - Those used in :func:`get_trips`
        
    """
    f = feed.get_trips(date)
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

    Return the empty dictionary if all the route's trips lack shapes.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - ``feed.shapes``
    - ``feed.trips``
    - ``feed.stops``

    """
    # Get the relevant shapes
    t = feed.trips.copy()
    A = t[t['route_id'] == route_id]['shape_id'].unique()
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False, 
      shape_ids=A)

    if not geometry_by_shape:
        return {}

    r = feed.routes.copy()
    features = [{
        'type': 'Feature',
        'properties': json.loads(r[r['route_id'] == route_id].to_json(
          orient='records'))[0],
        'geometry': sg.mapping(sg.MultiLineString(
          [linestring for linestring in geometry_by_shape.values()]))
        }]

    if include_stops:
        # Get relevant stops and geometrys
        s = feed.get_stops(route_id=route_id)
        cols = set(s.columns) - set(['stop_lon', 'stop_lat'])
        s = s[list(cols)].copy()
        stop_ids = s['stop_id'].tolist()
        geometry_by_stop = feed.build_geometry_by_stop(stop_ids=stop_ids)
        features.extend([{
            'type': 'Feature',
            'properties': json.loads(s[s['stop_id'] == stop_id].to_json(
              orient='records'))[0],
            'geometry': sg.mapping(geometry_by_stop[stop_id]),
            } for stop_id in stop_ids])

    return {'type': 'FeatureCollection', 'features': features}
