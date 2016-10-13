import unittest
import shutil
import importlib

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

import gtfstk.utilities as ut
import gtfstk.constants as cs
from gtfstk.feed import *
from gtfstk.calculator import *

# Check if GeoPandas is installed
loader = importlib.find_loader('geopandas')
if loader is None:
    HAS_GEOPANDAS = False
else:
    HAS_GEOPANDAS = True
    from geopandas import GeoDataFrame

# Load/create test feeds
DATA_DIR = Path('data')
cairns = read_gtfs(DATA_DIR/'cairns_gtfs.zip')
cairns_shapeless = read_gtfs(DATA_DIR/'cairns_gtfs.zip')
cairns_shapeless.shapes = None
trips = cairns_shapeless.trips.copy()
trips['shape_id'] = np.nan
cairns_shapeless.trips = trips

class TestCalculator(unittest.TestCase):
    # --------------------------------------------
    # Test functions about calendars
    # --------------------------------------------
    def test_get_dates(self):
        feed = cairns.copy()
        for as_date_obj in [True, False]:
            dates = get_dates(feed, as_date_obj=as_date_obj)
            d1 = '20140526'
            d2 = '20141228'
            if as_date_obj:
                d1 = ut.datestr_to_date(d1)
                d2 = ut.datestr_to_date(d2)
                self.assertEqual(len(dates), (d2 - d1).days + 1)
            self.assertEqual(dates[0], d1)
            self.assertEqual(dates[-1], d2)

    def test_get_first_week(self):
        feed = cairns.copy()
        dates = get_first_week(feed)
        d1 = '20140526'
        d2 = '20140601'
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 7)

    # --------------------------------------------
    # Test functions about trips
    # --------------------------------------------
    def test_is_active(self):
        feed = cairns.copy()
        trip = 'CNS2014-CNS_MUL-Weekday-00-4165878'
        date1 = '20140526'
        date2 = '20120322'
        self.assertTrue(is_active_trip(feed, trip, date1))
        self.assertFalse(is_active_trip(feed, trip, date2))

        trip = 'CNS2014-CNS_MUL-Sunday-00-4165971'
        date1 = '20140601'
        date2 = '20120602'
        self.assertTrue(is_active_trip(feed, trip, date1))
        self.assertFalse(is_active_trip(feed, trip, date2))

        feed = read_gtfs(DATA_DIR/'portland_gtfs.zip', dist_units_in='ft')
        trip = '4526377'
        date1 = '20140518'
        date2 = '20120517'
        self.assertTrue(is_active_trip(feed, trip, date1))
        self.assertFalse(is_active_trip(feed, trip, date2))

    def test_get_trips(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trips1 = get_trips(feed, date)
        # Should be a data frame
        self.assertIsInstance(trips1, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(trips1.shape[0] <= feed.trips.shape[0])
        self.assertEqual(trips1.shape[1], feed.trips.shape[1])
        # Should have correct columns
        self.assertEqual(set(trips1.columns), set(feed.trips.columns))

        trips2 = get_trips(feed, date, "07:30:00")
        # Should be a data frame
        self.assertIsInstance(trips2, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(trips2.shape[0] <= trips2.shape[0])
        self.assertEqual(trips2.shape[1], trips1.shape[1])
        # Should have correct columns
        self.assertEqual(set(trips2.columns), set(feed.trips.columns))

    def test_compute_trip_activity(self):
        feed = cairns.copy()
        dates = get_first_week(feed)
        trips_activity = compute_trip_activity(feed, dates)
        # Should be a data frame
        self.assertIsInstance(trips_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(trips_activity.shape[0], feed.trips.shape[0])
        self.assertEqual(trips_activity.shape[1], 1 + len(dates))
        # Date columns should contain only zeros and ones
        self.assertEqual(set(trips_activity[dates].values.flatten()), {0, 1})

    def test_compute_busiest_date(self):
        feed = cairns.copy()
        dates = get_first_week(feed) + ['19000101']
        date = compute_busiest_date(feed, dates)
        # Busiest day should lie in first week
        self.assertTrue(date in dates)

    def test_compute_trip_stats(self):
        feed = cairns.copy()
        trips_stats = compute_trip_stats(feed)
        
        # Should be a data frame with the correct number of rows
        self.assertIsInstance(trips_stats, pd.core.frame.DataFrame)
        self.assertEqual(trips_stats.shape[0], feed.trips.shape[0])
        
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
        self.assertEqual(set(trips_stats.columns), expect_cols)
        
        # Shapeless feeds should have null entries for distance column
        feed2 = cairns_shapeless.copy()
        trips_stats = compute_trip_stats(feed2)
        self.assertEqual(len(trips_stats['distance'].unique()), 1)
        self.assertTrue(np.isnan(trips_stats['distance'].unique()[0]))   
        
        # Should contain the correct trips
        get_trips = set(trips_stats['trip_id'].values)
        expect_trips = set(feed.trips['trip_id'].values)
        self.assertEqual(get_trips, expect_trips)

    def test_compute_trip_locations(self):
        feed = cairns.copy()
        trips_stats = compute_trip_stats(feed)
        feed.stop_times = add_dist_to_stop_times(feed, trips_stats)
        date = get_dates(feed)[0]
        times = ['08:00:00']
        f = compute_trip_locations(feed, date, times)
        g = get_trips(feed, date, times[0])
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct number of rows
        self.assertEqual(f.shape[0], g.shape[0])
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
        self.assertEqual(set(f.columns), expect_cols)
    
    def test_trip_to_geojson(self):
        feed = cairns.copy()
        trip_id = feed.trips['trip_id'].values[0]
        g0 = trip_to_geojson(feed, trip_id)      
        g1 = trip_to_geojson(feed, trip_id, include_stops=True)
        for g in [g0, g1]:
            # Should be a dictionary
            self.assertIsInstance(g, dict)

        # Should have the correct number of features
        self.assertEqual(len(g0['features']), 1)
        stop_ids = get_stops(feed, trip_id=trip_id)['stop_id'].values
        self.assertEqual(len(g1['features']), 1 + len(stop_ids))

    # ----------------------------------
    # Test functions about routes
    # ----------------------------------
    def test_get_routes(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        f = get_routes(feed, date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(f.shape[0] <= feed.routes.shape[0])
        self.assertEqual(f.shape[1], feed.routes.shape[1])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.routes.columns))

        g = get_routes(feed, date, "07:30:00")
        # Should be a data frame
        self.assertIsInstance(g, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(g.shape[0] <= f.shape[0])
        self.assertEqual(g.shape[1], f.shape[1])
        # Should have correct columns
        self.assertEqual(set(g.columns), set(feed.routes.columns))

    def test_compute_route_stats_base(self):
        feed = cairns.copy()
        f = compute_trip_stats(feed)
        for split_directions in [True, False]:
            rs = compute_route_stats_base(f, 
              split_directions=split_directions)

            # Should be a data frame of the correct shape
            self.assertIsInstance(rs, pd.core.frame.DataFrame)
            if split_directions:
                max_num_routes = 2*feed.routes.shape[0]
            else:
                max_num_routes = feed.routes.shape[0]
            self.assertTrue(rs.shape[0] <= max_num_routes)

            # Should contain the correct columns
            expect_cols = set([
              'route_id',
              'route_short_name',
              'route_type',
              'num_trips',
              'is_bidirectional',
              'is_loop',
              'start_time',
              'end_time',
              'max_headway',
              'min_headway',
              'mean_headway', 
              'peak_num_trips',
              'peak_start_time',
              'peak_end_time',
              'service_duration', 
              'service_distance',
              'service_speed',              
              'mean_trip_distance',
              'mean_trip_duration',
              ])
            if split_directions:
                expect_cols.add('direction_id')
            self.assertEqual(set(rs.columns), expect_cols)

        # Empty check
        rs = compute_route_stats_base(pd.DataFrame(), 
          split_directions=split_directions)    
        self.assertTrue(rs.empty)

    def test_compute_route_stats(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trips_stats = compute_trip_stats(feed)
        f = pd.merge(trips_stats, get_trips(feed, date))
        for split_directions in [True, False]:
            rs = compute_route_stats(feed, trips_stats, date, 
              split_directions=split_directions)

            # Should be a data frame of the correct shape
            self.assertIsInstance(rs, pd.core.frame.DataFrame)
            if split_directions:
                max_num_routes = 2*feed.routes.shape[0]
            else:
                max_num_routes = feed.routes.shape[0]
                
            self.assertTrue(rs.shape[0] <= max_num_routes)

            # Should contain the correct columns
            expect_cols = set([
              'route_id',
              'route_short_name',
              'route_type',
              'num_trips',
              'is_bidirectional',
              'is_loop',
              'start_time',
              'end_time',
              'max_headway',
              'min_headway',
              'mean_headway', 
              'peak_num_trips',
              'peak_start_time',
              'peak_end_time',
              'service_duration', 
              'service_distance',
              'service_speed',              
              'mean_trip_distance',
              'mean_trip_duration',
              ])
            if split_directions:
                expect_cols.add('direction_id')
            self.assertEqual(set(rs.columns), expect_cols)

        # Empty check
        f = compute_route_stats(feed, trips_stats, '20010101')
        self.assertTrue(f.empty)

    def test_compute_route_time_series_base(self):
        feed = cairns.copy()
        f = compute_trip_stats(feed)
        for split_directions in [True, False]:
            rs = compute_route_stats_base(f, 
              split_directions=split_directions)
            rts = compute_route_time_series_base(f, 
              split_directions=split_directions, freq='1H')
            
            # Should be a data frame of the correct shape
            self.assertIsInstance(rts, pd.core.frame.DataFrame)
            self.assertEqual(rts.shape[0], 24)
            self.assertEqual(rts.shape[1], 5*rs.shape[0])
            
            # Should have correct column names
            if split_directions:
                expect = ['indicator', 'route_id', 'direction_id']
            else:
                expect = ['indicator', 'route_id']
            self.assertEqual(rts.columns.names, expect)   
            
            # Each route have a correct service distance total
            if split_directions == False:
                g = f.groupby('route_id')
                for route in f['route_id'].values:
                    get = rts['service_distance'][route].sum() 
                    expect = g.get_group(route)['distance'].sum()
                    self.assertTrue(abs((get - expect)/expect) < 0.001)

        # Empty check
        rts = compute_route_time_series_base(pd.DataFrame(), 
          split_directions=split_directions, 
          freq='1H')    
        self.assertTrue(rts.empty)

    def test_compute_route_time_series(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trips_stats = compute_trip_stats(feed)
        ats = pd.merge(trips_stats, get_trips(feed, date))
        for split_directions in [True, False]:
            f = compute_route_stats(feed ,trips_stats, date, 
              split_directions=split_directions)
            rts = compute_route_time_series(feed, trips_stats, date, 
              split_directions=split_directions, freq='1H')
            
            # Should be a data frame of the correct shape
            self.assertIsInstance(rts, pd.core.frame.DataFrame)
            self.assertEqual(rts.shape[0], 24)
            self.assertEqual(rts.shape[1], 5*f.shape[0])
            
            # Should have correct column names
            if split_directions:
                expect = ['indicator', 'route_id', 'direction_id']
            else:
                expect = ['indicator', 'route_id']
            self.assertEqual(rts.columns.names, expect)   
            
            # Each route have a correct service distance total
            if split_directions == False:
                atsg = ats.groupby('route_id')
                for route in ats['route_id'].values:
                    get = rts['service_distance'][route].sum() 
                    expect = atsg.get_group(route)['distance'].sum()
                    self.assertTrue(abs((get - expect)/expect) < 0.001)

        # Empty check
        date = '19000101'
        rts = compute_route_time_series(feed, trips_stats, date, 
          split_directions=split_directions, freq='1H')
        self.assertTrue(rts.empty)

    def test_get_route_timetable(self):
        feed = cairns.copy()
        route = feed.routes['route_id'].values[0]
        date = get_dates(feed)[0]
        f = get_route_timetable(feed, route, date)
        # Should be a data frame 
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct columns
        expect_cols = set(feed.trips.columns) |\
          set(feed.stop_times.columns)
        self.assertEqual(set(f.columns), expect_cols)

    def test_route_to_geojson(self):
        feed = cairns.copy()
        route_id = feed.routes['route_id'].values[0]
        g0 = route_to_geojson(feed, route_id)      
        g1 = route_to_geojson(feed, route_id, include_stops=True)
        for g in [g0, g1]:
            # Should be a dictionary
            self.assertIsInstance(g, dict)

        # Should have the correct number of features
        self.assertEqual(len(g0['features']), 1)
        stop_ids = get_stops(feed, route_id=route_id)['stop_id'].values
        self.assertEqual(len(g1['features']), 1 + len(stop_ids))

    # ----------------------------------
    # Test functions about stops
    # ----------------------------------
    def test_get_stops(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trip_id = feed.trips['trip_id'].iat[0]
        route_id = feed.routes['route_id'].iat[0]
        frames = [
          get_stops(feed), 
          get_stops(feed, date=date),
          get_stops(feed, trip_id=trip_id),
          get_stops(feed, route_id=route_id),
          get_stops(feed, date=date, trip_id=trip_id),
          get_stops(feed, date=date, route_id=route_id),
          get_stops(feed, date=date, trip_id=trip_id, route_id=route_id),
          ]
        for f in frames:
            # Should be a data frame
            self.assertIsInstance(f, pd.core.frame.DataFrame)
            # Should have the correct shape
            self.assertTrue(f.shape[0] <= feed.stops.shape[0])
            self.assertEqual(f.shape[1], feed.stops.shape[1])
            # Should have correct columns
            self.assertEqual(set(f.columns), set(feed.stops.columns))
        # Number of rows should be reasonable
        self.assertTrue(frames[0].shape[0] <= frames[1].shape[0])
        self.assertTrue(frames[2].shape[0] <= frames[4].shape[0])
        self.assertSequenceEqual(frames[4].shape, frames[6].shape)

    def test_build_geometry_by_stop(self):
        feed = cairns.copy()
        stop_ids = feed.stops['stop_id'][:2].values
        d0 = build_geometry_by_stop(feed)
        d1 = build_geometry_by_stop(feed, stop_ids=stop_ids)
        for d in [d0, d1]:
            # Should be a dictionary
            self.assertIsInstance(d, dict)
            # The first key should be a valid shape ID
            self.assertTrue(list(d.keys())[0] in 
              feed.stops['stop_id'].values) 
            # The first value should be a Shapely linestring
            self.assertIsInstance(list(d.values())[0], Point)
        # Lengths should be right
        self.assertEqual(len(d0), feed.stops['stop_id'].nunique())
        self.assertEqual(len(d1), len(stop_ids))

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_geometrize_stops(self):
        stops = cairns.stops.copy()
        geo_stops = geometrize_stops(stops)
        # Should be a GeoDataFrame
        self.assertIsInstance(geo_stops, GeoDataFrame)
        # Should have the correct shape
        self.assertEqual(geo_stops.shape[0], stops.shape[0])
        self.assertEqual(geo_stops.shape[1], stops.shape[1] - 1)
        # Should have the correct columns
        expect_cols = set(list(stops.columns) + ['geometry']) -\
          set(['stop_lon', 'stop_lat'])
        self.assertEqual(set(geo_stops.columns), expect_cols)

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_ungeometrize_stops(self):
        stops = cairns.stops.copy()
        geo_stops = geometrize_stops(stops)
        stops2 = ungeometrize_stops(geo_stops)
        # Test columns are correct
        self.assertEqual(set(stops2.columns), set(stops.columns))
        # Data frames should be equal after sorting columns
        cols = sorted(stops.columns)
        assert_frame_equal(stops2[cols], stops[cols])

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_get_stops_intersecting_polygon(self):
        feed = cairns.copy()
        with (DATA_DIR/'cairns_square_stop_750070.geojson').open() as src:
            polygon = sh_shape(json.load(src)['features'][0]['geometry'])
        pstops = get_stops_intersecting_polygon(feed, polygon)
        stop_ids = ['750070']
        self.assertEqual(pstops['stop_id'].values, stop_ids)

    def test_compute_stop_activity(self):
        feed = cairns.copy()
        dates = get_first_week(feed)
        stops_activity = compute_stop_activity(feed, dates)
        # Should be a data frame
        self.assertIsInstance(stops_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(stops_activity.shape[0], feed.stops.shape[0])
        self.assertEqual(stops_activity.shape[1], len(dates) + 1)
        # Date columns should contain only zeros and ones
        self.assertEqual(set(stops_activity[dates].values.flatten()), {0, 1})

    def test_compute_stop_stats_base(self):
        feed = cairns.copy()
        for split_directions in [True, False]:
            stops_stats = compute_stop_stats_base(feed.stop_times,
              feed.trips, split_directions=split_directions)
            # Should be a data frame
            self.assertIsInstance(stops_stats, pd.core.frame.DataFrame)
            # Should contain the correct columns
            expect_cols = set([
              'stop_id',
              'num_routes',
              'num_trips',
              'max_headway',
              'min_headway',
              'mean_headway',
              'start_time',
              'end_time',
              ])
            if split_directions:
                expect_cols.add('direction_id')
            self.assertEqual(set(stops_stats.columns), expect_cols)
            # Should contain the correct stops
            expect_stops = set(feed.stops['stop_id'].values)
            get_stops = set(stops_stats['stop_id'].values)
            self.assertEqual(get_stops, expect_stops)

        # Empty check
        stats = compute_stop_stats_base(feed.stop_times, pd.DataFrame())    
        self.assertTrue(stats.empty)

    def test_compute_stop_stats(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        stops_stats = compute_stop_stats(feed, date)
        # Should be a data frame
        self.assertIsInstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct stops
        get = set(stops_stats['stop_id'].values)
        f = get_stops(feed, date)
        expect = set(f['stop_id'].values)
        self.assertEqual(get, expect)
        
        # Empty check
        f = compute_stop_stats(feed, '20010101')
        self.assertTrue(f.empty)

    def test_compute_stop_time_series_base(self):
        feed = cairns.copy()
        for split_directions in [True, False]:
            ss = compute_stop_stats_base(feed.stop_times, 
              feed.trips, split_directions=split_directions)
            sts = compute_stop_time_series_base(feed.stop_times, 
              feed.trips, freq='1H',
              split_directions=split_directions) 
            
            # Should be a data frame
            self.assertIsInstance(sts, pd.core.frame.DataFrame)
            
            # Should have the correct shape
            self.assertEqual(sts.shape[0], 24)
            self.assertEqual(sts.shape[1], ss.shape[0])
            
            # Should have correct column names
            if split_directions:
                expect = ['indicator', 'stop_id', 'direction_id']
            else:
                expect = ['indicator', 'stop_id']
            self.assertEqual(sts.columns.names, expect)

            # Each stop should have a correct total trip count
            if split_directions == False:
                stg = feed.stop_times.groupby('stop_id')
                for stop in set(feed.stop_times['stop_id'].values):
                    get = sts['num_trips'][stop].sum() 
                    expect = stg.get_group(stop)['departure_time'].count()
                    self.assertEqual(get, expect)
        
        # Empty check
        stops_ts = compute_stop_time_series_base(feed.stop_times,
          pd.DataFrame(), freq='1H',
          split_directions=split_directions) 
        self.assertTrue(stops_ts.empty)

    def test_compute_stop_time_series(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        ast = pd.merge(get_trips(feed, date), feed.stop_times)
        for split_directions in [True, False]:
            f = compute_stop_stats(feed, date, 
              split_directions=split_directions)
            stops_ts = compute_stop_time_series(feed, date, freq='1H',
              split_directions=split_directions) 
            
            # Should be a data frame
            self.assertIsInstance(stops_ts, pd.core.frame.DataFrame)
            
            # Should have the correct shape
            self.assertEqual(stops_ts.shape[0], 24)
            self.assertEqual(stops_ts.shape[1], f.shape[0])
            
            # Should have correct column names
            if split_directions:
                expect = ['indicator', 'stop_id', 'direction_id']
            else:
                expect = ['indicator', 'stop_id']
            self.assertEqual(stops_ts.columns.names, expect)

            # Each stop should have a correct total trip count
            if split_directions == False:
                astg = ast.groupby('stop_id')
                for stop in set(ast['stop_id'].values):
                    get = stops_ts['num_trips'][stop].sum() 
                    expect = astg.get_group(stop)['departure_time'].count()
                    self.assertEqual(get, expect)
        
        # Empty check
        date = '19000101'
        stops_ts = compute_stop_time_series(feed, date, freq='1H',
          split_directions=split_directions) 
        self.assertTrue(stops_ts.empty)

    def test_get_stop_timetable(self):
        feed = cairns.copy()
        stop = feed.stops['stop_id'].values[0]
        date = get_dates(feed)[0]
        f = get_stop_timetable(feed, stop, date)
        # Should be a data frame 
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct columns
        expect_cols = set(feed.trips.columns) |\
          set(feed.stop_times.columns)
        self.assertEqual(set(f.columns), expect_cols)    

    # ----------------------------------
    # Test functions about shapes
    # ----------------------------------
    def test_build_geometry_by_shape(self):
        feed = cairns.copy()
        shape_ids = feed.shapes['shape_id'].unique()[:2]
        d0 = build_geometry_by_shape(feed)
        d1 = build_geometry_by_shape(feed, shape_ids=shape_ids)
        for d in [d0, d1]:
            # Should be a dictionary
            self.assertIsInstance(d, dict)
            # The first key should be a valid shape ID
            self.assertTrue(list(d.keys())[0] in 
              feed.shapes['shape_id'].values) 
            # The first value should be a Shapely linestring
            self.assertIsInstance(list(d.values())[0], LineString)
        # Lengths should be right
        self.assertEqual(len(d0), feed.shapes['shape_id'].nunique())
        self.assertEqual(len(d1), len(shape_ids))
        # Should be None if feed.shapes is None
        feed2 = cairns_shapeless.copy()
        self.assertIsNone(build_geometry_by_shape(feed2))

    def test_shapes_to_geojson(self):
        feed = cairns.copy()
        collection = shapes_to_geojson(feed)
        geometry_by_shape = build_geometry_by_shape(feed, use_utm=False)
        for f in collection['features']:
            shape = f['properties']['shape_id']
            geom = sh_shape(f['geometry'])
            self.assertTrue(geom.equals(geometry_by_shape[shape]))

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_geometrize_shapes(self):
        shapes = cairns.shapes.copy()
        geo_shapes = geometrize_shapes(shapes)
        # Should be a GeoDataFrame
        self.assertIsInstance(geo_shapes, GeoDataFrame)
        # Should have the correct shape
        self.assertEqual(geo_shapes.shape[0], shapes['shape_id'].nunique())
        self.assertEqual(geo_shapes.shape[1], shapes.shape[1] - 2)
        # Should have the correct columns
        expect_cols = set(list(shapes.columns) + ['geometry']) -\
          set(['shape_pt_lon', 'shape_pt_lat', 'shape_pt_sequence',
          'shape_dist_traveled'])
        self.assertEqual(set(geo_shapes.columns), expect_cols)

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_ungeometrize_shapes(self):
        shapes = cairns.shapes.copy()
        geo_shapes = geometrize_shapes(shapes)
        shapes2 = ungeometrize_shapes(geo_shapes)
        # Test columns are correct
        expect_cols = set(list(shapes.columns)) -\
          set(['shape_dist_traveled'])
        self.assertEqual(set(shapes2.columns), expect_cols)
        # Data frames should agree on certain columns
        cols = ['shape_id', 'shape_pt_lon', 'shape_pt_lat']
        assert_frame_equal(shapes2[cols], shapes[cols])

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_get_shapes_intersecting_geometry(self):
        feed = cairns.copy()
        path = DATA_DIR/'cairns_square_stop_750070.geojson'
        polygon = sh_shape(json.load(path.open())['features'][0]['geometry'])
        pshapes = get_shapes_intersecting_geometry(feed, polygon)
        shape_ids = ['120N0005', '1200010', '1200001']
        self.assertEqual(set(pshapes['shape_id'].unique()), set(shape_ids))

    def test_add_dist_to_shapes(self):
        feed = cairns.copy()
        s1 = feed.shapes.copy()
        feed.shapes = add_dist_to_shapes(feed)
        s2 = feed.shapes
        # Check that colums of st2 equal the columns of st1 plus
        # a shape_dist_traveled column
        cols1 = list(s1.columns.values) + ['shape_dist_traveled']
        cols2 = list(s2.columns.values)
        self.assertEqual(set(cols1), set(cols2))

        # Check that within each trip the shape_dist_traveled column 
        # is monotonically increasing
        for name, group in s2.groupby('shape_id'):
            sdt = list(group['shape_dist_traveled'].values)
            self.assertEqual(sdt, sorted(sdt))

    def test_add_route_type_to_shapes(self):
        feed = cairns.copy()
        shapes = add_route_type_to_shapes(feed)
        # Should contain correct columns
        self.assertEqual(set(shapes.columns), 
          set(feed.shapes.columns) | {'route_type'})

    # ----------------------------------
    # Test functions about stop times
    # ----------------------------------
    def test_get_stop_times(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        f = get_stop_times(feed, date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have a reasonable shape
        self.assertTrue(f.shape[0] <= feed.stop_times.shape[0])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.stop_times.columns))

    def test_get_start_and_end_times(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        st = get_stop_times(feed, date)
        times = get_start_and_end_times(feed, date)
        # Should be strings
        for t in times:
            self.assertIsInstance(t, str)
            # Should lie in stop times
            self.assertTrue(t in 
              st[['departure_time', 'arrival_time']].values.flatten())

        # Should get null times in some cases
        times = get_start_and_end_times(feed, '19690711')
        for t in times:
            self.assertTrue(pd.isnull(t))
        feed.stop_times['departure_time'] = np.nan
        times = get_start_and_end_times(feed)
        self.assertTrue(pd.isnull(times[0]))

    def test_add_dist_to_stop_times(self):
        feed = cairns.copy()
        st1 = feed.stop_times
        trips_stats = compute_trip_stats(feed)
        st2 = add_dist_to_stop_times(feed, trips_stats)
        
        # Check that colums of st2 equal the columns of st1 plus
        # a shape_dist_traveled column
        cols1 = list(st1.columns.values) + ['shape_dist_traveled']
        cols2 = list(st2.columns.values)
        self.assertEqual(set(cols1), set(cols2))

        # Check that within each trip the shape_dist_traveled column 
        # is monotonically increasing
        for trip, group in st2.groupby('trip_id'):
            group = group.sort_values('stop_sequence')
            sdt = list(group['shape_dist_traveled'].values)
            self.assertEqual(sdt, sorted(sdt))

    # ----------------------------------
    # Test functions about feeds
    # ----------------------------------
    def test_compute_feed_stats(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trips_stats = compute_trip_stats(feed)
        f = compute_feed_stats(feed, trips_stats, date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct number of rows
        self.assertEqual(f.shape[0], 1)
        # Should contain the correct columns
        expect_cols = set([
          'num_trips',
          'num_routes',
          'num_stops',
          'peak_num_trips',
          'peak_start_time',
          'peak_end_time',
          'service_duration', 
          'service_distance',
          'service_speed',              
          ])
        self.assertEqual(set(f.columns), expect_cols)

        # Empty check
        f = compute_feed_stats(feed, trips_stats, '20010101')
        self.assertTrue(f.empty)

    def test_compute_feed_time_series(self):
        feed = cairns.copy()
        date = get_dates(feed)[0]
        trips_stats = compute_trip_stats(feed)
        f = compute_feed_time_series(feed, trips_stats, date, freq='1H')
        # Should be a data frame 
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct number of rows
        self.assertEqual(f.shape[0], 24)
        # Should have the correct columns
        expect_cols = set([
          'num_trip_starts',
          'num_trips',
          'service_distance',
          'service_duration',
          'service_speed',
          ])
        self.assertEqual(set(f.columns), expect_cols)

        # Empty check
        f = compute_feed_time_series(feed, trips_stats, '20010101')
        self.assertTrue(f.empty)

    def test_create_shapes(self):
        feed1 = cairns.copy()
        # Remove a trip shape
        trip_id = 'CNS2014-CNS_MUL-Weekday-00-4165878'
        feed1.trips.loc[feed1.trips['trip_id'] == trip_id, 'shape_id'] = np.nan
        feed2 = create_shapes(feed1)
        # Should create only 1 new shape
        self.assertEqual(len(set(feed2.shapes['shape_id']) - set(
          feed1.shapes['shape_id'])), 1)

        feed2 = create_shapes(feed1, all_trips=True)
        # Number of shapes should equal number of unique stop sequences
        st = feed1.stop_times.sort_values(['trip_id', 'stop_sequence'])
        stop_seqs = set([tuple(group['stop_id'].values)
          for __, group in st.groupby('trip_id')])
        self.assertEqual(feed2.shapes['shape_id'].nunique(), len(stop_seqs))

    def test_restrict_by_routes(self):
        feed1 = cairns.copy() 
        route_ids = feed1.routes['route_id'][:2].tolist()
        feed2 = restrict_by_routes(feed1, route_ids)
        # Should have correct routes
        self.assertEqual(set(feed2.routes['route_id']), set(route_ids))
        # Should have correct trips
        trip_ids = feed1.trips[feed1.trips['route_id'].isin(
          route_ids)]['trip_id']
        self.assertEqual(set(feed2.trips['trip_id']), set(trip_ids))
        # Should have correct shapes
        shape_ids = feed1.trips[feed1.trips['trip_id'].isin(
          trip_ids)]['shape_id']
        self.assertEqual(set(feed2.shapes['shape_id']), set(shape_ids))
        # Should have correct stops
        stop_ids = feed1.stop_times[feed1.stop_times['trip_id'].isin(
          trip_ids)]['stop_id']
        self.assertEqual(set(feed2.stop_times['stop_id']), set(stop_ids))

    @unittest.skipIf(not HAS_GEOPANDAS, 'geopandas absent; skipping')
    def test_restrict_by_polygon(self):
        feed1 = cairns.copy() 
        with (DATA_DIR/'cairns_square_stop_750070.geojson').open() as src:
            polygon = sh_shape(json.load(src)['features'][0]['geometry'])
        feed2 = restrict_by_polygon(feed1, polygon)
        # Should have correct routes
        rsns = ['120', '120N']
        self.assertEqual(set(feed2.routes['route_short_name']), set(rsns))
        # Should have correct trips
        route_ids = feed1.routes[feed1.routes['route_short_name'].isin(
          rsns)]['route_id']
        trip_ids = feed1.trips[feed1.trips['route_id'].isin(
          route_ids)]['trip_id']
        self.assertEqual(set(feed2.trips['trip_id']), set(trip_ids))
        # Should have correct shapes
        shape_ids = feed1.trips[feed1.trips['trip_id'].isin(
          trip_ids)]['shape_id']
        self.assertEqual(set(feed2.shapes['shape_id']), set(shape_ids))
        # Should have correct stops
        stop_ids = feed1.stop_times[feed1.stop_times['trip_id'].isin(
          trip_ids)]['stop_id']
        self.assertEqual(set(feed2.stop_times['stop_id']), set(stop_ids))

    # ----------------------------------
    # Test miscellanous functions
    # ----------------------------------
    def test_compute_screen_line_counts(self):
        feed = cairns.copy() 
        # Add distances to feed
        trips_stats = compute_trip_stats(feed, compute_dist_from_shapes=True)
        feed.stop_times = add_dist_to_stop_times(feed, trips_stats)
        
        # Pick date
        date = get_first_week(feed)[0]
        
        # Load screen line
        with (DATA_DIR/'cairns_screen_line.geojson').open() as src:
            line = json.load(src)
            line = sh_shape(line['features'][0]['geometry'])
        
        f = compute_screen_line_counts(feed, line, date)

        # Should have correct columns
        expect_cols = set([
          'trip_id',
          'route_id',
          'route_short_name',
          'crossing_time',
          'orientation',
          ])
        self.assertEqual(set(f.columns), expect_cols)

        # Should have correct routes
        rsns = ['120', '120N']
        self.assertEqual(set(f['route_short_name']), set(rsns))

        # Should have correct number of trips
        expect_num_trips = 34
        self.assertEqual(f['trip_id'].nunique(), expect_num_trips)

        # Should have correct orientations
        for ori in [-1, 1]:
            self.assertEqual(f[f['orientation'] == 1].shape[0], 
              expect_num_trips)

    def test_compute_bounds(self):
        feed = cairns.copy() 
        minlon, minlat, maxlon, maxlat = compute_bounds(feed)
        # Bounds should be in the ball park
        self.assertTrue(145 < minlon < 146)
        self.assertTrue(145 < maxlon < 146)
        self.assertTrue(-18 < minlat < -15)
        self.assertTrue(-18 < maxlat < -15)

    def test_compute_center(self):
        feed = cairns.copy() 
        centers = [compute_center(feed), compute_center(feed, 20)]
        bounds = compute_bounds(feed)
        for lon, lat in centers:
            # Center should be in the ball park
            self.assertTrue(bounds[0] < lon < bounds[2])
            self.assertTrue(bounds[1] < lat < bounds[3])



if __name__ == '__main__':
    unittest.main()