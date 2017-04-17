"""
Functions about shapes.
"""
import pandas as pd 
import numpy as np
import utm
import shapely.geometry as sg 

from . import constants as cs
from . import helpers as hp


def build_geometry_by_shape(feed, use_utm=False, shape_ids=None):
    """
    Return a dictionary with structure shape_id -> Shapely linestring of shape.
    If ``feed.shapes is None``, then return ``None``.
    If ``use_utm``, then return each linestring in in UTM coordinates.
    Otherwise, return each linestring in WGS84 longitude-latitude    coordinates.
    If a list of shape IDs ``shape_ids`` is given, then only include the given shape IDs.

    Return the empty dictionary if ``feed.shapes is None``.
    """
    if feed.shapes is None:
        return {}

    # Note the output for conversion to UTM with the utm package:
    # >>> u = utm.from_latlon(47.9941214, 7.8509671)
    # >>> print u
    # (414278, 5316285, 32, 'T')
    d = {}
    shapes = feed.shapes.copy()
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

def shapes_to_geojson(feed):
    """
    Return a (decoded) GeoJSON FeatureCollection of linestring features representing ``feed.shapes``.
    Each feature will have a ``shape_id`` property. 
    The coordinates reference system is the default one for GeoJSON, namely WGS84.

    Return the empty dictionary of ``feed.shapes is None``
    """
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
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
    resulting shapes DataFrame in geometrized form.
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
    Return the resulting Feed.

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
    m_to_dist = hp.get_convert_dist('m', feed.dist_units)

    def compute_dist(group):
        # Compute the distances of the stops along this trip
        group = group.sort_values('shape_pt_sequence')
        shape = group['shape_id'].iat[0]
        if not isinstance(shape, str):
            group['shape_dist_traveled'] = np.nan 
            return group
        points = [sg.Point(utm.from_latlon(lat, lon)[:2]) 
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

def geometrize_shapes(shapes, use_utm=False):
    """
    Given a shapes DataFrame, convert it to a GeoPandas GeoDataFrame and return the result.
    The result has a 'geometry' column of WGS84 line strings instead of 'shape_pt_sequence', 'shape_pt_lon', 'shape_pt_lat', and 'shape_dist_traveled' columns.
    If ``use_utm``, then use UTM coordinates for the geometries.

    Requires GeoPandas.
    """
    import geopandas as gpd


    f = shapes.copy().sort_values(['shape_id', 'shape_pt_sequence'])
    
    def my_agg(group):
        d = {}
        d['geometry'] =\
          sg.LineString(group[['shape_pt_lon', 'shape_pt_lat']].values)
        return pd.Series(d)

    g = f.groupby('shape_id').apply(my_agg).reset_index()
    g = gpd.GeoDataFrame(g, crs=cs.CRS_WGS84)

    if use_utm:
        lat, lon = f.ix[0][['shape_pt_lat', 'shape_pt_lon']].values
        crs = get_utm_crs(lat, lon) 
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
