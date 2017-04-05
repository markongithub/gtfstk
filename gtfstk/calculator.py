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



# -------------------------------------
# Functions about stops
# -------------------------------------


def geometrize_stops(stops, use_utm=False):
    """
    Given a stops data frame, convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 points instead of 'stop_lon' and 'stop_lat' columns.
    If ``use_utm``, then use UTM coordinates for the geometries.
    Requires GeoPandas.
    """
    import geopandas as gpd 


    f = stops.copy()
    s = gpd.GeoSeries([Point(p) for p in 
      stops[['stop_lon', 'stop_lat']].values])
    f['geometry'] = s 
    g = f.drop(['stop_lon', 'stop_lat'], axis=1)
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['stop_lat', 'stop_lon']].values
        crs = ut.get_utm_crs(lat, lon) 
        g = g.to_crs(crs)

    return g

def ungeometrize_stops(geo_stops):
    """
    The inverse of :func:`geometrize_stops`.    
    If ``geo_stops`` is in UTM (has a UTM CRS property), then convert UTM coordinates back to WGS84 coordinates,
    """
    f = geo_stops.copy().to_crs(cs.CRS_WGS84)
    f['stop_lon'] = f['geometry'].map(
      lambda p: p.x)
    f['stop_lat'] = f['geometry'].map(
      lambda p: p.y)
    del f['geometry']
    return f

def get_stops_intersecting_polygon(feed, polygon, geo_stops=None):
    """
    Return the slice of ``feed.stops`` that contains all stops that intersect the given Shapely Polygon object.
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
        f = geometrize_stops(feed.stops)
    
    cols = f.columns
    f['hit'] = f['geometry'].intersects(polygon)
    f = f[f['hit']][cols]
    return ungeometrize_stops(f)

# -------------------------------------
# Functions about shapes
# -------------------------------------
def geometrize_shapes(shapes, use_utm=False):
    """
    Given a shapes data frame, convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 line strings instead of 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat', and 'shape_dist_traveled' columns.
    If ``use_utm``, then use UTM coordinates for the geometries.

    Requires GeoPandas.
    """
    import geopandas as gpd


    f = shapes.copy().sort_values(['shape_id', 'shape_pt_sequence'])
    
    def my_agg(group):
        d = {}
        d['geometry'] =\
          LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
        return pd.Series(d)

    g = f.groupby('shape_id').apply(my_agg).reset_index()
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
        crs = ut.get_utm_crs(lat, lon) 
        g = g.to_crs(crs)

    return g 

def ungeometrize_shapes(geo_shapes):
    """
    The inverse of :func:`geometrize_shapes`.
    Produces the columns:

    - shape_id
    - shape_pt_sequence
    - shape_pt_lon
    - shape_pt_lat

    If ``geo_shapes`` is in UTM (has a UTM CRS property), then convert UTM coordinates back to WGS84 coordinates,
    """
    geo_shapes = geo_shapes.to_crs(cs.CRS_WGS84)

    F = []
    for index, row in geo_shapes.iterrows():
        F.extend([[row['shape_id'], i, x, y] for 
        i, (x, y) in enumerate(row['geometry'].coords)])

    return pd.DataFrame(F, 
      columns=['shape_id', 'shape_pt_sequence', 
      'shape_pt_lon', 'shape_pt_lat'])

def get_shapes_intersecting_geometry(feed, geometry, geo_shapes=None,
  geometrized=False):
    """
    Return the slice of ``feed.shapes`` that contains all shapes that intersect the given Shapely geometry object (e.g. a Polygon or LineString).
    Assume the geometry is specified in WGS84 longitude-latitude coordinates.
    
    To do this, first geometrize ``feed.shapes`` via :func:`geometrize_shapes`.
    Alternatively, use the ``geo_shapes`` GeoDataFrame, if given.
    Requires GeoPandas.

    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``, if ``geo_shapes`` is not given

    If ``geometrized`` is ``True``, then return the 
    resulting shapes data frame in geometrized form.
    """
    if geo_shapes is not None:
        f = geo_shapes.copy()
    else:
        f = geometrize_shapes(feed.shapes)
    
    cols = f.columns
    f['hit'] = f['geometry'].intersects(geometry)
    f = f[f['hit']][cols]

    if geometrized:
        return f
    else:
        return ungeometrize_shapes(f)

def append_dist_to_shapes(feed):
    """
    Calculate and append the optional ``shape_dist_traveled`` field in ``feed.shapes`` in terms of the distance units ``feed.dist_units``.
    Return the resulting feed.

    Assume the following feed attributes are not ``None``:

    - ``feed.shapes``

    NOTES: 
        - All of the calculated ``shape_dist_traveled`` values for the Portland feed https://transitfeeds.com/p/trimet/43/1400947517 differ by at most 0.016 km in absolute values from of the original values. 
    """
    if feed.shapes is None:
        raise ValueError(
          "This function requires the feed to have a shapes.txt file")

    feed = feed.copy()
    f = feed.shapes
    m_to_dist = ut.get_convert_dist('m', feed.dist_units)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        group = group.sort_values('shape_pt_sequence')
        shape = group['shape_id'].iat[0]
        if not isinstance(shape, str):
            group['shape_dist_traveled'] = np.nan 
            return group
        points = [Point(utm.from_latlon(lat, lon)[:2]) 
          for lon, lat in group[['shape_pt_lon', 'shape_pt_lat']].values]
        p_prev = points[0]
        d = 0
        distances = [0]
        for  p in points[1:]:
            d += p.distance(p_prev)
            distances.append(d)
            p_prev = p
        group['shape_dist_traveled'] = distances
        return group

    g = f.groupby('shape_id', group_keys=False).apply(compute_dist)
    # Convert from meters
    g['shape_dist_traveled'] = g['shape_dist_traveled'].map(m_to_dist)

    feed.shapes = g
    return feed

def append_route_type_to_shapes(feed):
    """
    Append a ``route_type`` column to a copy of ``feed.shapes`` and return    the resulting shapes data frame.
    Note that a single shape can be linked to multiple trips on multiple routes of multiple route types.
    In that case the route type of the shape is the route type of the last    route (sorted by ID) with a trip with that shape.

    Assume the following feed attributes are not ``None``:

    - ``feed.routes``
    - ``feed.trips``
    - ``feed.shapes``

    """        
    f = pd.merge(feed.routes, feed.trips).sort_values(['shape_id', 'route_id'])
    rtype_by_shape = dict(f[['shape_id', 'route_type']].values)
    
    g = feed.shapes.copy()
    g['route_type'] = g['shape_id'].map(lambda x: rtype_by_shape[x])
    
    return g


def get_start_and_end_times(feed, date=None):
    """
    Return the first departure time and last arrival time (time strings) listed in ``feed.stop_times``, respectively.
    Restrict to the given date if specified.
    """
    st = get_stop_times(feed, date)
    # if st.empty:
    #     a, b = np.nan, np.nan 
    # else:
    a, b = st['departure_time'].dropna().min(),\
       st['arrival_time'].dropna().max()
    return a, b 

# -------------------------------------
# Functions about feeds
# -------------------------------------
def convert_dist(feed, new_dist_units):
    """
    Convert the distances recorded in the ``shape_dist_traveled`` columns of the given feed from the feed's native distance units (recorded in ``feed.dist_units``) to the given new distance units.
    New distance units must lie in ``constants.DIST_UNITS``
    """       
    feed = feed.copy()

    if feed.dist_units == new_dist_units:
        # Nothing to do
        return feed

    old_dist_units = feed.dist_units
    feed.dist_units = new_dist_units

    converter = ut.get_convert_dist(old_dist_units, new_dist_units)

    if ut.is_not_null(feed.stop_times, 'shape_dist_traveled'):
        feed.stop_times['shape_dist_traveled'] =\
          feed.stop_times['shape_dist_traveled'].map(converter)

    if ut.is_not_null(feed.shapes, 'shape_dist_traveled'):
        feed.shapes['shape_dist_traveled'] =\
          feed.shapes['shape_dist_traveled'].map(converter)

    return feed

def compute_feed_stats(feed, trips_stats, date):
    """
    Given ``trips_stats``, which is the output of 
    ``feed.compute_trip_stats()`` and a date,
    return a  data frame including the following feed
    stats for the date.

    - num_trips: number of trips active on the given date
    - num_routes: number of routes active on the given date
    - num_stops: number of stops active on the given date
    - peak_num_trips: maximum number of simultaneous trips in service
    - peak_start_time: start time of first longest period during which
      the peak number of trips occurs
    - peak_end_time: end time of first longest period during which
      the peak number of trips occurs
    - service_distance: sum of the service distances for the active routes
    - service_duration: sum of the service durations for the active routes
    - service_speed: service_distance/service_duration

    If there are no stats for the given date, return an empty data frame
    with the specified columns.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`get_trips`
    - Those used in :func:`get_routes`
    - Those used in :func:`get_stops`

    """
    cols = [
      'num_trips',
      'num_routes',
      'num_stops',
      'peak_num_trips',
      'peak_start_time',
      'peak_end_time',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    d = OrderedDict()
    trips = get_trips(feed, date)
    if trips.empty:
        return pd.DataFrame(columns=cols)

    d['num_trips'] = trips.shape[0]
    d['num_routes'] = get_routes(feed, date).shape[0]
    d['num_stops'] = get_stops(feed, date).shape[0]

    # Compute peak stats
    f = trips.merge(trips_stats)
    f[['start_time', 'end_time']] =\
      f[['start_time', 'end_time']].applymap(ut.timestr_to_seconds)

    times = np.unique(f[['start_time', 'end_time']].values)
    counts = [count_active_trips(f, t) for t in times]
    start, end = ut.get_peak_indices(times, counts)
    d['peak_num_trips'] = counts[start]
    d['peak_start_time'] =\
      ut.timestr_to_seconds(times[start], inverse=True)
    d['peak_end_time'] =\
      ut.timestr_to_seconds(times[end], inverse=True)

    # Compute remaining stats
    d['service_distance'] = f['distance'].sum()
    d['service_duration'] = f['duration'].sum()
    d['service_speed'] = d['service_distance']/d['service_duration']

    return pd.DataFrame(d, index=[0])

def compute_feed_time_series(feed, trips_stats, date, freq='5Min'):
    """
    Given trips stats (output of ``feed.compute_trip_stats()``),
    a date, and a Pandas frequency string,
    return a time series of stats for this feed on the given date
    at the given frequency with the following columns

    - num_trip_starts: number of trips starting at this time
    - num_trips: number of trips in service during this time period
    - service_distance: distance traveled by all active trips during
      this time period
    - service_duration: duration traveled by all active trips during this
      time period
    - service_speed: service_distance/service_duration

    If there is no time series for the given date, 
    return an empty data frame with specified columns.

    Assume the following feed attributes are not ``None``:

    - Those used in :func:`compute_route_time_series`

    """
    cols = [
      'num_trip_starts',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    rts = compute_route_time_series(feed, trips_stats, date, freq=freq)
    if rts.empty:
        return pd.DataFrame(columns=cols)

    stats = rts.columns.levels[0].tolist()
    # split_directions = 'direction_id' in rts.columns.names
    # if split_directions:
    #     # For each stat and each direction, sum across routes.
    #     frames = []
    #     for stat in stats:
    #         f0 = rts.xs((stat, '0'), level=('indicator', 'direction_id'), 
    #           axis=1).sum(axis=1)
    #         f1 = rts.xs((stat, '1'), level=('indicator', 'direction_id'), 
    #           axis=1).sum(axis=1)
    #         f = pd.concat([f0, f1], axis=1, keys=['0', '1'])
    #         frames.append(f)
    #     F = pd.concat(frames, axis=1, keys=stats, names=['indicator', 
    #       'direction_id'])
    #     # Fix speed
    #     F['service_speed'] = F['service_distance'].divide(
    #       F['service_duration'])
    #     result = F
    f = pd.concat([rts[stat].sum(axis=1) for stat in stats], axis=1, 
      keys=stats)
    f['service_speed'] = f['service_distance']/f['service_duration']
    return f

def create_shapes(feed, all_trips=False):
    """
    Given a feed, create a shape for every trip that is missing a shape ID.
    Do this by connecting the stops on the trip with straight lines.
    Return the resulting feed which has updated shapes and trips data frames.

    If ``all_trips``, then create new shapes for all trips 
    by connecting stops, and remove the old shapes.
    
    Assume the following feed attributes are not ``None``:

    - ``feed.stop_times``
    - ``feed.trips``
    - ``feed.stops``
    """
    feed = feed.copy()

    if all_trips:
        trip_ids = feed.trips['trip_id']
    else:
        trip_ids = feed.trips[feed.trips['shape_id'].isnull()]['trip_id']

    # Get stop times for given trips
    f = feed.stop_times[feed.stop_times['trip_id'].isin(trip_ids)][
      ['trip_id', 'stop_sequence', 'stop_id']]
    f = f.sort_values(['trip_id', 'stop_sequence'])

    if f.empty:
        # Nothing to do
        return feed 

    # Create new shape IDs for given trips.
    # To do this, collect unique stop sequences, 
    # sort them to impose a canonical order, and 
    # assign shape IDs to them
    stop_seqs = sorted(set(tuple(group['stop_id'].values) 
      for trip, group in f.groupby('trip_id')))
    d = int(math.log10(len(stop_seqs))) + 1  # Digits for padding shape IDs  
    shape_by_stop_seq = {seq: 'shape_{num:0{pad}d}'.format(num=i, pad=d) 
      for i, seq in enumerate(stop_seqs)}
 
    # Assign these new shape IDs to given trips 
    shape_by_trip = {trip: shape_by_stop_seq[tuple(group['stop_id'].values)] 
      for trip, group in f.groupby('trip_id')}
    trip_cond = feed.trips['trip_id'].isin(trip_ids)
    feed.trips.loc[trip_cond, 'shape_id'] = feed.trips.loc[trip_cond,
      'trip_id'].map(lambda x: shape_by_trip[x])

    # Build new shapes for given trips
    G = [[shape, i, stop] for stop_seq, shape in shape_by_stop_seq.items() 
      for i, stop in enumerate(stop_seq)]
    g = pd.DataFrame(G, columns=['shape_id', 'shape_pt_sequence', 
      'stop_id'])
    g = g.merge(feed.stops[['stop_id', 'stop_lon', 'stop_lat']]).sort_values(
      ['shape_id', 'shape_pt_sequence'])
    g = g.drop(['stop_id'], axis=1)
    g = g.rename(columns={
      'stop_lon': 'shape_pt_lon',
      'stop_lat': 'shape_pt_lat',
      })

    if feed.shapes is not None and not all_trips:
        # Update feed shapes with new shapes
        feed.shapes = pd.concat([feed.shapes, g])
    else:
        # Create all new shapes
        feed.shapes = g

    return feed

def restrict_by_routes(feed, route_ids):
    """
    Build a new feed by taking the given one and chopping it down to
    only the stops, trips, shapes, etc. used by the routes specified
    in the given list of route IDs. 
    Return the resulting feed.
    """
    # Initialize the new feed as the old feed.
    # Restrict its data frames below.
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
    # Restrict its data frames below.
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
    Compute all the trips active in the given feed on the given date that intersect the given Shapely LineString (with WGS84 longitude-latitude coordinates), and return a data frame with the columns:

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

    # Build a dictionary from the shapes data frame of the form
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
    
    # Create data frame
    g = pd.DataFrame(G, columns=['trip_id', 'route_id', 
      'route_short_name', 'crossing_time', 'orientation']
      ).sort_values('crossing_time')

    # Convert departure times to time strings
    g['crossing_time'] = g['crossing_time'].map(
      lambda x: ut.timestr_to_seconds(x, inverse=True))

    return g

def compute_bounds(feed):   
    """
    Return the tuple (min longitude, min latitude, max longitude, max latitude) where the longitudes and latitude vary across all the stop (WGS84)coordinates.
    """
    lons, lats = feed.stops['stop_lon'], feed.stops['stop_lat']
    return lons.min(), lats.min(), lons.max(), lats.max()
    
def compute_center(feed, num_busiest_stops=None):
    """
    Compute the convex hull of all the given feed's stop coordinates and return the centroid.
    If an integer ``num_busiest_stops`` is given, then compute the ``num_busiest_stops`` busiest stops in the feed on the first Monday of the feed and return the mean of the longitudes and the mean of the latitudes of these stops, respectively.
    """
    s = feed.stops.copy()
    if num_busiest_stops is not None:
        n = num_busiest_stops
        date = get_first_week(feed)[0]
        ss = compute_stop_stats(feed, date).sort_values(
          'num_trips', ascending=False)
        f = ss.head(num_busiest_stops)
        f = s.merge(f)
        lon = f['stop_lon'].mean()
        lat = f['stop_lat'].mean()
    else:
        m = MultiPoint(s[['stop_lon', 'stop_lat']].values)
        lon, lat = list(m.convex_hull.centroid.coords)[0]
    return lon, lat