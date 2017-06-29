import pandas as pd
import numpy as np

from .context import gtfstk, slow, HAS_GEOPANDAS, DATA_DIR, sample, cairns, cairns_shapeless, cairns_date, cairns_trip_stats
from gtfstk import *


def test_is_active_trip():
    feed = cairns.copy()
    trip_id = 'CNS2014-CNS_MUL-Weekday-00-4165878'
    date1 = '20140526'
    date2 = '20120322'
    assert is_active_trip(feed, trip_id, date1)
    assert not is_active_trip(feed, trip_id, date2)

    trip_id = 'CNS2014-CNS_MUL-Sunday-00-4165971'
    date1 = '20140601'
    date2 = '20120602'
    assert is_active_trip(feed, trip_id, date1)
    assert not is_active_trip(feed, trip_id, date2)

def test_get_trips():
    feed = cairns.copy()
    date = cairns_date
    trips1 = get_trips(feed, date)
    # Should be a data frame
    assert isinstance(trips1, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips1.shape[0] <= feed.trips.shape[0]
    assert trips1.shape[1] == feed.trips.shape[1]
    # Should have correct columns
    assert set(trips1.columns) == set(feed.trips.columns)

    trips2 = get_trips(feed, date, "07:30:00")
    # Should be a data frame
    assert isinstance(trips2, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips2.shape[0] <= trips2.shape[0]
    assert trips2.shape[1] == trips1.shape[1]
    # Should have correct columns
    assert set(trips2.columns) == set(feed.trips.columns)

def test_compute_trip_activity():
    feed = cairns.copy()
    dates = get_first_week(feed)
    trips_activity = compute_trip_activity(feed, dates)
    # Should be a data frame
    assert isinstance(trips_activity, pd.core.frame.DataFrame)
    # Should have the correct shape
    assert trips_activity.shape[0] == feed.trips.shape[0]
    assert trips_activity.shape[1] == 1 + len(dates)
    # Date columns should contain only zeros and ones
    assert set(trips_activity[dates].values.flatten()) == {0, 1}

def test_compute_busiest_date():
    feed = cairns.copy()
    dates = get_first_week(feed) + ['19000101']
    date = compute_busiest_date(feed, dates)
    # Busiest day should lie in first week
    assert date in dates

@slow
def test_compute_trip_stats():
    feed = cairns.copy()
    trip_stats = compute_trip_stats(feed)

    # Should be a data frame with the correct number of rows
    assert isinstance(trip_stats, pd.core.frame.DataFrame)
    assert trip_stats.shape[0] == feed.trips.shape[0]

    # Should contain the correct columns
    expect_cols = set([
      'trip_id',
      'direction_id',
      'route_id',
      'route_short_name',
      'route_type',
      'shape_id',
      'num_stops',
      'start_time',
      'end_time',
      'start_stop_id',
      'end_stop_id',
      'distance',
      'duration',
      'speed',
      'is_loop',
      ])
    assert set(trip_stats.columns) == expect_cols

    # Shapeless feeds should have null entries for distance column
    feed2 = cairns_shapeless.copy()
    trip_stats = compute_trip_stats(feed2)
    assert len(trip_stats['distance'].unique()) == 1
    assert np.isnan(trip_stats['distance'].unique()[0])

    # Should contain the correct trips
    get_trips = set(trip_stats['trip_id'].values)
    expect_trips = set(feed.trips['trip_id'].values)
    assert get_trips == expect_trips

@slow
def test_locate_trips():
    feed = cairns.copy()
    trip_stats = cairns_trip_stats
    feed = append_dist_to_stop_times(feed, trip_stats)
    date = cairns_date
    times = ['08:00:00']
    f = locate_trips(feed, date, times)
    g = get_trips(feed, date, times[0])
    # Should be a data frame
    assert isinstance(f, pd.core.frame.DataFrame)
    # Should have the correct number of rows
    assert f.shape[0] == g.shape[0]
    # Should have the correct columns
    expect_cols = set([
      'route_id',
      'trip_id',
      'direction_id',
      'shape_id',
      'time',
      'rel_dist',
      'lon',
      'lat',
      ])
    assert set(f.columns) == expect_cols

def test_trip_to_geojson():
    feed = cairns.copy()
    trip_id = feed.trips['trip_id'].values[0]
    g0 = trip_to_geojson(feed, trip_id)
    g1 = trip_to_geojson(feed, trip_id, include_stops=True)
    for g in [g0, g1]:
        # Should be a dictionary
        assert isinstance(g, dict)

    # Should have the correct number of features
    assert len(g0['features']) == 1
    stop_ids = get_stops(feed, trip_id=trip_id)['stop_id'].values
    assert len(g1['features']) == 1 + len(stop_ids)
