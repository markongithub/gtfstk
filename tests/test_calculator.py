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
cairns = read_gtfs(DATA_DIR/'cairns_gtfs.zip', dist_units='km')
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None
t = cairns_shapeless.trips
t['shape_id'] = np.nan
cairns_shapeless.trips = t


class TestCalculator(unittest.TestCase):

    # --------------------------------------------
    # Test functions about trips
    # --------------------------------------------

    def test_compute_trip_stats(self):
        feed = cairns.copy()
        trip_stats = compute_trip_stats(feed)
        
        # Should be a data frame with the correct number of rows
        self.assertIsInstance(trip_stats, pd.core.frame.DataFrame)
        self.assertEqual(trip_stats.shape[0], feed.trips.shape[0])
        
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
        self.assertEqual(set(trip_stats.columns), expect_cols)
        
        # Shapeless feeds should have null entries for distance column
        feed2 = cairns_shapeless.copy()
        trip_stats = compute_trip_stats(feed2)
        self.assertEqual(len(trip_stats['distance'].unique()), 1)
        self.assertTrue(np.isnan(trip_stats['distance'].unique()[0]))   
        
        # Should contain the correct trips
        get_trips = set(trip_stats['trip_id'].values)
        expect_trips = set(feed.trips['trip_id'].values)
        self.assertEqual(get_trips, expect_trips)
    
    # ----------------------------------
    # Test functions about routes
    # ----------------------------------

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


    # ----------------------------------
    # Test functions about stops
    # ----------------------------------
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

   

    # ----------------------------------
    # Test functions about shapes
    # ----------------------------------
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



    def test_append_dist_to_stop_times(self):
        feed1 = cairns.copy()
        st1 = feed1.stop_times
        trip_stats = compute_trip_stats(feed1)
        feed2 = append_dist_to_stop_times(feed1, trip_stats)
        st2 = feed2.stop_times 

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
    # Test miscellaneous functions
    # ----------------------------------
    def test_compute_screen_line_counts(self):
        feed = cairns.copy() 
        # Add distances to feed
        trip_stats = compute_trip_stats(feed, compute_dist_from_shapes=True)
        feed = append_dist_to_stop_times(feed, trip_stats)
        
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




if __name__ == '__main__':
    unittest.main()