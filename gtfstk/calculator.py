"""
This module contains functions for calculating properties of Feed objects, such as daily service duration per route. 
"""
import datetime as dt
import dateutil.relativedelta as rd
from collections import OrderedDict, Counter
import json
import math

import pandas as pd
import numpy as np
from shapely.geometry import Point, MultiPoint, MultiLineString, LineString, mapping
import utm

from . import constants as cs
from . import utilities as ut
from .feed import Feed





def restrict_by_routes(feed, route_ids):
    """
    Build a new feed by taking the given one and chopping it down to
    only the stops, trips, shapes, etc. used by the routes specified
    in the given list of route IDs. 
    Return the resulting feed.
    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()
    
    # Slice routes
    feed.routes = feed.routes[feed.routes['route_id'].isin(route_ids)].copy()

    # Slice trips
    feed.trips = feed.trips[feed.trips['route_id'].isin(route_ids)].copy()

    # Slice stop times
    trip_ids = feed.trips['trip_id']
    feed.stop_times = feed.stop_times[
      feed.stop_times['trip_id'].isin(trip_ids)].copy()

    # Slice stops
    stop_ids = feed.stop_times['stop_id'].unique()
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()

    # Slice calendar
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()
    
    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()
            
    # Now for the optional files.
    # Get calendar dates for trips.
    if feed.calendar_dates is not None:
        feed.calendar_dates = feed.calendar_dates[
          feed.calendar_dates['service_id'].isin(service_ids)].copy()
    
    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()
        
    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()
        
    # Get transfers for stops
    if feed.transfers is not None:
        feed.transfers = feed.transfers[
          feed.transfers['from_stop_id'].isin(stop_ids) |\
          feed.transfers['to_stop_id'].isin(stop_ids)].copy()
        
    return feed

def restrict_by_polygon(feed, polygon):
    """
    Build a new feed by taking the given one, keeping only the trips 
    that have at least one stop intersecting the given polygon, and then
    restricting stops, routes, stop times, etc. to those associated with 
    that subset of trips. 
    Return the resulting feed.
    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``
    - ``feed.routes``
    - Those used in :func:`get_stops_intersecting_polygon`

    """
    # Initialize the new feed as the old feed.
    # Restrict its DataFrames below.
    feed = feed.copy()
    
    # Get IDs of stops within the polygon
    stop_ids = get_stops_intersecting_polygon(
      feed, polygon)['stop_id']
        
    # Get all trips that stop at at least one of those stops
    st = feed.stop_times.copy()
    trip_ids = st[st['stop_id'].isin(stop_ids)]['trip_id']
    feed.trips = feed.trips[feed.trips['trip_id'].isin(trip_ids)].copy()
    
    # Get stop times for trips
    feed.stop_times = st[st['trip_id'].isin(trip_ids)].copy()
    
    # Get stops for trips
    stop_ids = feed.stop_times['stop_id']
    feed.stops = feed.stops[feed.stops['stop_id'].isin(stop_ids)].copy()
    
    # Get routes for trips
    route_ids = feed.trips['route_id']
    feed.routes = feed.routes[feed.routes['route_id'].isin(route_ids)].copy()
    
    # Get calendar for trips
    service_ids = feed.trips['service_id']
    if feed.calendar is not None:
        feed.calendar = feed.calendar[
          feed.calendar['service_id'].isin(service_ids)].copy()
    
    # Get agency for trips
    if 'agency_id' in feed.routes.columns:
        agency_ids = feed.routes['agency_id']
        if len(agency_ids):
            feed.agency = feed.agency[
              feed.agency['agency_id'].isin(agency_ids)].copy()
            
    # Now for the optional files.
    # Get calendar dates for trips.
    cd = feed.calendar_dates
    if cd is not None:
        feed.calendar_dates = cd[cd['service_id'].isin(service_ids)].copy()
    
    # Get frequencies for trips
    if feed.frequencies is not None:
        feed.frequencies = feed.frequencies[
          feed.frequencies['trip_id'].isin(trip_ids)].copy()
        
    # Get shapes for trips
    if feed.shapes is not None:
        shape_ids = feed.trips['shape_id']
        feed.shapes = feed.shapes[
          feed.shapes['shape_id'].isin(shape_ids)].copy()
        
    # Get transfers for stops
    if feed.transfers is not None:
        t = feed.transfers
        feed.transfers = t[t['from_stop_id'].isin(stop_ids) |\
          t['to_stop_id'].isin(stop_ids)].copy()
        
    return feed

# -------------------------------------
# Miscellaneous functions
# -------------------------------------


def compute_screen_line_counts(feed, linestring, date, geo_shapes=None):
    """
    Compute all the trips active in the given feed on the given date that intersect the given Shapely LineString (with WGS84 longitude-latitude coordinates), and return a DataFrame with the columns:

    - ``'trip_id'``
    - ``'route_id'``
    - ``'route_short_name'``
    - ``'crossing_time'``: time that the trip's vehicle crosses the linestring; one trip could cross multiple times
    - ``'orientation'``: 1 or -1; 1 indicates trip travel from the left side to the right side of the screen line; -1 indicates trip travel in the  opposite direction

    NOTES:
        - Requires GeoPandas.
        - The first step is to geometrize ``feed.shapes`` via   :func:`geometrize_shapes`. Alternatively, use the ``geo_shapes`` GeoDataFrame, if given.
        - Assume ``feed.stop_times`` has an accurate ``shape_dist_traveled`` column.
        - Assume the following feed attributes are not ``None``:
             * ``feed.shapes``, if ``geo_shapes`` is not given
        - Assume that trips travel in the same direction as their shapes. That restriction is part of GTFS, by the way. To calculate direction quickly and accurately, assume that the screen line is straight and doesn't double back on itself.
        - Probably does not give correct results for trips with self-intersecting shapes.
    
    ALGORITHM:
        #. Compute all the shapes that intersect the linestring.
        #. For each such shape, compute the intersection points.
        #. For each point p, scan through all the trips in the feed that have that shape and are active on the given date.
        #. Interpolate a stop time for p by assuming that the feed has the shape_dist_traveled field in stop times.
        #. Use that interpolated time as the crossing time of the trip vehicle, and compute the trip orientation to the screen line via a cross product of a vector in the direction of the screen line and a tiny vector in the direction of trip travel.
    """  
    # Get all shapes that intersect the screen line
    shapes = get_shapes_intersecting_geometry(feed, linestring, geo_shapes,
      geometrized=True)

    # Convert shapes to UTM
    lat, lon = feed.shapes.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
    crs = ut.get_utm_crs(lat, lon) 
    shapes = shapes.to_crs(crs)

    # Convert linestring to UTM
    linestring = ut.linestring_to_utm(linestring)

    # Get all intersection points of shapes and linestring
    shapes['intersection'] = shapes.intersection(linestring)

    # Make a vector in the direction of the screen line
    # to later calculate trip orientation.
    # Does not work in case of a bent screen line.
    p1 = Point(linestring.coords[0])
    p2 = Point(linestring.coords[-1])
    w = np.array([p2.x - p1.x, p2.y - p1.y])

    # Build a dictionary from the shapes DataFrame of the form
    # shape ID -> list of pairs (d, v), one for each intersection point, 
    # where d is the distance of the intersection point along shape,
    # and v is a tiny vectors from the point in direction of shape.
    # Assume here that trips travel in the same direction as their shapes.
    dv_by_shape = {}
    eps = 1
    convert_dist = ut.get_convert_dist('m', feed.dist_units)
    for __, sid, geom, intersection in shapes.itertuples():
        # Get distances along shape of intersection points (in meters)
        distances = [geom.project(p) for p in intersection]
        # Build tiny vectors
        vectors = []
        for i, p in enumerate(intersection):
            q = geom.interpolate(distances[i] + eps)
            vector = np.array([q.x - p.x, q.y - p.y])
            vectors.append(vector)
        # Convert distances to units used in feed
        distances = [convert_dist(d) for d in distances]
        dv_by_shape[sid] = list(zip(distances, vectors))

    # Get trips with those shapes that are active on the given date
    trips = get_trips(feed, date)
    trips = trips[trips['shape_id'].isin(dv_by_shape.keys())]

    # Merge in route short names
    trips = trips.merge(feed.routes[['route_id', 'route_short_name']])

    # Merge in stop times
    f = trips.merge(feed.stop_times)

    # Drop NaN departure times and convert to seconds past midnight
    f = f[f['departure_time'].notnull()]
    f['departure_time'] = f['departure_time'].map(ut.timestr_to_seconds)

    # For each shape find the trips that cross the screen line
    # and get crossing times and orientation
    f = f.sort_values(['trip_id', 'stop_sequence'])
    G = []  # output table
    for tid, group in f.groupby('trip_id'):
        sid = group['shape_id'].iat[0] 
        rid = group['route_id'].iat[0]
        rsn = group['route_short_name'].iat[0]
        stop_times = group['departure_time'].values
        stop_distances = group['shape_dist_traveled'].values
        for d, v in dv_by_shape[sid]:
            # Interpolate crossing time
            t = np.interp(d, stop_distances, stop_times)
            # Compute direction of trip travel relative to
            # screen line by looking at the sign of the cross
            # product of tiny shape vector and screen line vector
            det = np.linalg.det(np.array([v, w]))
            if det >= 0:
                orientation = 1
            else:
                orientation = -1
            # Update G
            G.append([tid, rid, rsn, t, orientation])
    
    # Create DataFrame
    g = pd.DataFrame(G, columns=['trip_id', 'route_id', 
      'route_short_name', 'crossing_time', 'orientation']
      ).sort_values('crossing_time')

    # Convert departure times to time strings
    g['crossing_time'] = g['crossing_time'].map(
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    return g

    
