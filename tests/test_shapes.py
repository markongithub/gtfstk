import pytest
import importlib
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
import utm
import shapely.geometry as sg 

from .context import gtfstk, slow, HAS_GEOPANDAS, DATA_DIR, sample, cairns, cairns_shapeless, cairns_date, cairns_trip_stats
from gtfstk import *
if HAS_GEOPANDAS:
    from geopandas import GeoDataFrame


def test_build_geometry_by_shape():
    feed = cairns.copy()
    shape_ids = feed.shapes['shape_id'].unique()[:2]
    d0 = feed.build_geometry_by_shape()
    d1 = feed.build_geometry_by_shape(shape_ids=shape_ids)
    for d in [d0, d1]:
        # Should be a dictionary
        assert isinstance(d, dict)
        # The first key should be a valid shape ID
        assert list(d.keys())[0] in feed.shapes['shape_id'].values
        # The first value should be a Shapely linestring
        assert isinstance(list(d.values())[0], sg.LineString)
    # Lengths should be right
    assert len(d0) == feed.shapes['shape_id'].nunique()
    assert len(d1) == len(shape_ids)
    # Should be empty if feed.shapes is None
    feed2 = cairns_shapeless.copy()
    assert feed2.build_geometry_by_shape(feed2) == {}

def test_shapes_to_geojson():
    feed = cairns.copy()
    collection = feed.shapes_to_geojson()
    geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
    for f in collection['features']:
        shape = f['properties']['shape_id']
        geom = sg.shape(f['geometry'])
        assert geom.equals(geometry_by_shape[shape])

@pytest.mark.skipif(not HAS_GEOPANDAS, reason="Requires GeoPandas")
def test_get_shapes_intersecting_geometry():
    feed = cairns.copy()
    path = DATA_DIR/'cairns_square_stop_750070.geojson'
    polygon = sg.shape(json.load(path.open())['features'][0]['geometry'])
    pshapes = feed.get_shapes_intersecting_geometry(polygon)
    shape_ids = ['120N0005', '1200010', '1200001']
    assert set(pshapes['shape_id'].unique()) == set(shape_ids)

def test_append_dist_to_shapes():
    feed1 = cairns.copy()
    s1 = feed1.shapes
    feed2 = feed1.append_dist_to_shapes()
    s2 = feed2.shapes
    # Check that colums of st2 equal the columns of st1 plus
    # a shape_dist_traveled column
    cols1 = list(s1.columns.values) + ['shape_dist_traveled']
    cols2 = list(s2.columns.values)
    assert set(cols1) == set(cols2)

    # Check that within each trip the shape_dist_traveled column 
    # is monotonically increasing
    for name, group in s2.groupby('shape_id'):
        sdt = list(group['shape_dist_traveled'].values)
        assert sdt == sorted(sdt)
