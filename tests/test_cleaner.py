import pytest
import importlib
from pathlib import Path 

import pandas as pd 
from pandas.util.testing import assert_frame_equal, assert_series_equal
import numpy as np
import utm
import shapely.geometry as sg 

from .context import gtfstk, slow, sample
from gtfstk import *


def test_clean_ids():
    f1 = sample.copy()
    f1.routes.ix[0, 'route_id'] = '  ho   ho ho '
    f2 = f1.clean_ids()
    expect_rid = 'ho_ho_ho'
    f2.routes.ix[0, 'route_id'] == expect_rid

    f3 = f2.clean_ids()
    f3 == f2

def test_clean_route_short_names():
    f1  = sample.copy()
    
    # Should have no effect on a fine feed
    f2 = f1.clean_route_short_names()
    assert_series_equal(f2.routes['route_short_name'], 
      f1.routes['route_short_name'])
    
    # Make route short name duplicates
    f1.routes.loc[1:5, 'route_short_name'] = np.nan
    f1.routes.loc[6:, 'route_short_name'] = '  he llo  '
    f2 = f1.clean_route_short_names()
    # Should have unique route short names
    assert f2.routes['route_short_name'].nunique() == f2.routes.shape[0]
    # NaNs should be replaced by n/a and route IDs
    expect_rsns = ('n/a-' + sample.routes.ix[1:5]['route_id']).tolist()
    assert f2.routes.ix[1:5]['route_short_name'].values.tolist() == expect_rsns
    # Should have names without leading or trailing whitespace
    assert not f2.routes['route_short_name'].str.startswith(' ').any()
    assert not f2.routes['route_short_name'].str.endswith(' ').any()

def test_prune_dead_routes():
    # Should not change Cairns routes
    f1 = sample.copy()
    f2 = f1.prune_dead_routes()
    assert_frame_equal(f2.routes, f1.routes)

    # Create a dummy route which should be removed
    g = pd.DataFrame([[0 for c in f1.routes.columns]], 
      columns=f1.routes.columns)
    f3 = f1.copy()
    f3.routes = pd.concat([f3.routes, g])
    f4 = f3.prune_dead_routes()
    assert_frame_equal(f4.routes, f1.routes)

def test_aggregate_routes():
    feed1 = sample.copy()
    # Equalize all route short names
    feed1.routes['route_short_name'] = 'bingo'
    feed2 = feed1.aggregate_routes()

    # feed2 should have only one route ID
    assert feed2.routes.shape[0] == 1
    
    # Feeds should have same trip data frames excluding
    # route IDs
    feed1.trips['route_id'] = feed2.trips['route_id']
    assert almost_equal(feed1.trips, feed2.trips)

    # Feeds should have equal attributes excluding
    # routes and trips data frames
    feed2.routes = feed1.routes 
    feed2.trips = feed1.trips
    assert feed1 == feed2

def test_clean():
    f1 = sample.copy()
    rid = f1.routes.ix[0, 'route_id']
    f1.routes.ix[0, 'route_id'] = ' ' + rid + '   '
    f2 = f1.clean()
    assert f2.routes.ix[0, 'route_id'] == rid
    assert_frame_equal(f2.trips, sample.trips)

def test_drop_invalid_fields():
    f1 = sample.copy()
    f1.routes['bingo'] = 'bongo'
    f1.trips['wingo'] = 'wongo'
    f2 = f1.drop_invalid_fields()
    assert f2 == sample
