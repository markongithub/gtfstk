import unittest
from copy import copy
import shutil
from types import FunctionType

import pandas as pd 
import numpy as np
from pandas.util.testing import assert_frame_equal, assert_series_equal
from shapely.geometry import Point, LineString, mapping
from shapely.geometry import shape as sh_shape

from gtfstk.calculator import *
import gtfstk.utilities as utils

# Load test feeds
cairns = read_gtfs('data/cairns_gtfs.zip')
cairns_shapeless = read_gtfs('data/cairns_gtfs.zip')
cairns_shapeless.shapes = None

class TestCalculator(unittest.TestCase):

    def test_feed_constructor(self):
        feed = Feed(agency=pd.DataFrame())
        for key, value in feed.__dict__.items():
            if key in ['dist_units_in', 'dist_units_out']:
                self.assertEqual(value, 'km')
            elif key == 'convert_dist':
                self.assertIsInstance(value, FunctionType)
            elif key == 'agency':
                self.assertIsInstance(value, pd.DataFrame)
            else:
                self.assertIsNone(value)

    # --------------------------------------------
    # Test functions about inputs and outputs
    # --------------------------------------------
    def test_read_gtfs(self):
        feed = read_gtfs('data/cairns_gtfs.zip')
        for key, value in feed.__dict__.items():
            if key in ['dist_units_in', 'dist_units_out']:
                self.assertEqual(value, 'km')
            elif key == 'convert_dist':
                self.assertIsInstance(value, FunctionType)
            elif key in REQUIRED_GTFS_FILES and key != 'calendar_dates':
                self.assertIsInstance(value, pd.DataFrame)

        # Bad dist_units_in:
        self.assertRaises(ValueError, read_gtfs, 
          path='data/cairns_gtfs.zip',  
          dist_units_in='bingo')

        # Requires dist_units_in:
        self.assertRaises(ValueError, read_gtfs,
          path='data/portland_gtfs.zip')

    def test_write_gtfs(self):
        feed1 = copy(cairns)

        # Export feed1, import it as feed2, and then test that the
        # attributes of the two feeds are equal.
        path = 'data/test_gtfs.zip'
        write_gtfs(feed1, path)
        feed2 = read_gtfs(path)
        names = REQUIRED_GTFS_FILES + OPTIONAL_GTFS_FILES
        for name in names:
            f1 = getattr(feed1, name)
            f2 = getattr(feed2, name)
            if f1 is None:
                self.assertIsNone(f2)
            else:
                assert_frame_equal(f1, f2)

        # Test that integer columns with NaNs get output properly.
        # To this end, put a NaN, 1.0, and 0.0 in the direction_id column 
        # of trips.txt, export it, and import the column as strings.
        # Should only get np.nan, '0', and '1' entries.
        feed3 = copy(cairns)
        f = feed3.trips.copy()
        f['direction_id'] = f['direction_id'].astype(object)
        f.loc[0, 'direction_id'] = np.nan
        f.loc[1, 'direction_id'] = 1.0
        f.loc[2, 'direction_id'] = 0.0
        feed3.trips = f
        write_gtfs(feed3, path)
        archive = zipfile.ZipFile(path)
        dir_name = path.rstrip('.zip') + '/'
        archive.extractall(dir_name)
        t = pd.read_csv(dir_name + 'trips.txt', dtype={'direction_id': str})
        self.assertTrue(t[~t['direction_id'].isin([np.nan, '0', '1'])].empty)
        
        # Remove extracted directory
        shutil.rmtree(dir_name)

    # --------------------------------------------
    # Test functions about calendars
    # --------------------------------------------
    def test_get_dates(self):
        feed = copy(cairns)
        for as_date_obj in [True, False]:
            dates = get_dates(feed, as_date_obj=as_date_obj)
            d1 = '20140526'
            d2 = '20141228'
            if as_date_obj:
                d1 = utils.datestr_to_date(d1)
                d2 = utils.datestr_to_date(d2)
                self.assertEqual(len(dates), (d2 - d1).days + 1)
            self.assertEqual(dates[0], d1)
            self.assertEqual(dates[-1], d2)

    def test_get_first_week(self):
        feed = copy(cairns)
        dates = get_first_week(feed)
        d1 = '20140526'
        d2 = '20140601'
        self.assertEqual(dates[0], d1)
        self.assertEqual(dates[-1], d2)
        self.assertEqual(len(dates), 7)

    # --------------------------------------------
    # Test functions about trips
    # --------------------------------------------
    def test_count_active_trips(self):
        pass

    def test_is_active(self):
        feed = copy(cairns)
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

        feed = read_gtfs('data/portland_gtfs.zip', dist_units_in='ft')
        trip = '4526377'
        date1 = '20140518'
        date2 = '20120517'
        self.assertTrue(is_active_trip(feed, trip, date1))
        self.assertFalse(is_active_trip(feed, trip, date2))

    def test_get_trips(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        f = feed.get_trips(date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(f.shape[0] <= feed.trips.shape[0])
        self.assertEqual(f.shape[1], feed.trips.shape[1])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.trips.columns))

        g = feed.get_trips(date, "07:30:00")
        # Should be a data frame
        self.assertIsInstance(g, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(g.shape[0] <= f.shape[0])
        self.assertEqual(g.shape[1], f.shape[1])
        # Should have correct columns
        self.assertEqual(set(g.columns), set(feed.trips.columns))

    def test_compute_trips_locations(self):
        feed = copy(cairns)
        trips_stats = feed.compute_trips_stats()
        feed.add_dist_to_stop_times(trips_stats)
        date = feed.get_dates()[0]
        timestrs = ['08:00:00']
        f = feed.compute_trips_locations(date, timestrs)
        g = feed.get_trips(date, timestrs[0])
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
    
    def test_compute_trips_activity(self):
        feed = copy(cairns)
        dates = feed.get_first_week()
        trips_activity = feed.compute_trips_activity(dates)
        # Should be a data frame
        self.assertIsInstance(trips_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(trips_activity.shape[0], feed.trips.shape[0])
        self.assertEqual(trips_activity.shape[1], 1 + len(dates))
        # Date columns should contain only zeros and ones
        self.assertEqual(set(trips_activity[dates].values.flatten()), {0, 1})

    def test_compute_trips_stats(self):
        feed = copy(cairns)
        trips_stats = feed.compute_trips_stats()
        
        # Should be a data frame with the correct number of rows
        self.assertIsInstance(trips_stats, pd.core.frame.DataFrame)
        self.assertEqual(trips_stats.shape[0], feed.trips.shape[0])
        
        # Should contain the correct columns
        expect_cols = set([
          'trip_id',
          'direction_id',
          'route_id',
          'route_short_name',
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
        feed2 = cairns_shapeless
        trips_stats = feed2.compute_trips_stats()
        self.assertEqual(len(trips_stats['distance'].unique()), 1)
        self.assertTrue(np.isnan(trips_stats['distance'].unique()[0]))   
        
        # Should contain the correct trips
        get_trips = set(trips_stats['trip_id'].values)
        expect_trips = set(feed.trips['trip_id'].values)
        self.assertEqual(get_trips, expect_trips)

    def test_get_busiest_date(self):
        feed = copy(cairns)
        dates = get_first_week(feed) + ['19000101']
        date = get_busiest_date(feed, dates)
        # Busiest day should lie in first week
        self.assertTrue(date in dates)

    # ---------------------------------
    # Test functions about routes
    # ---------------------------------
    def test_compute_routes_stats_base(self):
        feed = deepcopy(cairns)
        f = feed.compute_trips_stats()
        for split_directions in [True, False]:
            rs = compute_routes_stats(f, split_directions=split_directions)

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
        rs = compute_routes_stats(pd.DataFrame(), 
          split_directions=split_directions)    
        self.assertTrue(rs.empty)

    def test_compute_routes_time_series_base(self):
        feed = copy(cairns)
        f = feed.compute_trips_stats()
        for split_directions in [True, False]:
            rs = compute_routes_stats(f, split_directions=split_directions)
            rts = compute_routes_time_series(f, 
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
        rts = compute_routes_time_series(pd.DataFrame(), 
          split_directions=split_directions, 
          freq='1H')    
        self.assertTrue(rts.empty)

    def test_get_routes(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        f = feed.get_routes(date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(f.shape[0] <= feed.routes.shape[0])
        self.assertEqual(f.shape[1], feed.routes.shape[1])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.routes.columns))

        g = feed.get_routes(date, "07:30:00")
        # Should be a data frame
        self.assertIsInstance(g, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(g.shape[0] <= f.shape[0])
        self.assertEqual(g.shape[1], f.shape[1])
        # Should have correct columns
        self.assertEqual(set(g.columns), set(feed.routes.columns))

    def test_compute_routes_stats(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        trips_stats = feed.compute_trips_stats()
        f = pd.merge(trips_stats, feed.get_trips(date))
        for split_directions in [True, False]:
            rs = feed.compute_routes_stats(trips_stats, date, 
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
        f = feed.compute_routes_stats(trips_stats, '20010101')
        self.assertTrue(f.empty)

    def test_compute_routes_time_series(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        trips_stats = feed.compute_trips_stats()
        ats = pd.merge(trips_stats, feed.get_trips(date))
        for split_directions in [True, False]:
            f = feed.compute_routes_stats(trips_stats, date, 
              split_directions=split_directions)
            rts = feed.compute_routes_time_series(trips_stats, date, 
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
        rts = feed.compute_routes_time_series(trips_stats, date, 
          split_directions=split_directions, freq='1H')
        self.assertTrue(rts.empty)

    def test_get_route_timetable(self):
        feed = copy(cairns)
        route = feed.routes['route_id'].values[0]
        date = feed.get_dates()[0]
        f = feed.get_route_timetable(route, date)
        # Should be a data frame 
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct columns
        expect_cols = set(feed.trips.columns) |\
          set(feed.stop_times.columns)
        self.assertEqual(set(f.columns), expect_cols)

    
    # Test stop methods
    # ----------------------------------
    def test_compute_stops_stats_base(self):
        feed = copy(cairns)
        for split_directions in [True, False]:
            stops_stats = compute_stops_stats(feed.stop_times,
              feed.trips, 
              split_directions=split_directions)
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
        stats = compute_stops_stats(feed.stop_times, pd.DataFrame())    
        self.assertTrue(stats.empty)

    def test_compute_stops_time_series_base(self):
        feed = copy(cairns)
        for split_directions in [True, False]:
            ss = compute_stops_stats(feed.stop_times, 
              feed.trips, split_directions=split_directions)
            sts = compute_stops_time_series(feed.stop_times, 
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
        stops_ts = compute_stops_time_series(feed.stop_times,
          pd.DataFrame(), freq='1H',
          split_directions=split_directions) 
        self.assertTrue(stops_ts.empty)

    def test_get_stops(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        f = feed.get_stops(date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertTrue(f.shape[0] <= feed.stops.shape[0])
        self.assertEqual(f.shape[1], feed.stops.shape[1])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.stops.columns))

    def test_build_geometry_by_stop(self):
        feed = copy(cairns)
        geometry_by_stop = feed.build_geometry_by_stop()
        # Should be a dictionary
        self.assertIsInstance(geometry_by_stop, dict)
        # The first element should be a Shapely point
        self.assertIsInstance(list(geometry_by_stop.values())[0], Point)
        # Should include all stops
        self.assertEqual(len(geometry_by_stop), feed.stops.shape[0])

    def test_compute_stops_activity(self):
        feed = copy(cairns)
        dates = feed.get_first_week()
        stops_activity = feed.compute_stops_activity(dates)
        # Should be a data frame
        self.assertIsInstance(stops_activity, pd.core.frame.DataFrame)
        # Should have the correct shape
        self.assertEqual(stops_activity.shape[0], feed.stops.shape[0])
        self.assertEqual(stops_activity.shape[1], len(dates) + 1)
        # Date columns should contain only zeros and ones
        self.assertEqual(set(stops_activity[dates].values.flatten()), {0, 1})

    def test_compute_stops_stats(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        stops_stats = feed.compute_stops_stats(date)
        # Should be a data frame
        self.assertIsInstance(stops_stats, pd.core.frame.DataFrame)
        # Should contain the correct stops
        get_stops = set(stops_stats['stop_id'].values)
        f = feed.get_stops(date)
        expect_stops = set(f['stop_id'].values)
        self.assertEqual(get_stops, expect_stops)
        
        # Empty check
        f = feed.compute_stops_stats('20010101')
        self.assertTrue(f.empty)

    def test_compute_stops_time_series(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        ast = pd.merge(feed.get_trips(date), feed.stop_times)
        for split_directions in [True, False]:
            f = feed.compute_stops_stats(date, 
              split_directions=split_directions)
            stops_ts = feed.compute_stops_time_series(date, freq='1H',
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
        stops_ts = feed.compute_stops_time_series(date, freq='1H',
          split_directions=split_directions) 
        self.assertTrue(stops_ts.empty)

    def test_get_stop_timetable(self):
        feed = copy(cairns)
        stop = feed.stops['stop_id'].values[0]
        date = feed.get_dates()[0]
        f = feed.get_stop_timetable(stop, date)
        # Should be a data frame 
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have the correct columns
        expect_cols = set(feed.trips.columns) |\
          set(feed.stop_times.columns)
        self.assertEqual(set(f.columns), expect_cols)    

    # Test shape methods
    # ----------------------------------
    def test_build_geometry_by_shape(self):
        feed = copy(cairns)
        geometry_by_shape = feed.build_geometry_by_shape()
        # Should be a dictionary
        self.assertIsInstance(geometry_by_shape, dict)
        # The first element should be a Shapely linestring
        self.assertIsInstance(list(geometry_by_shape.values())[0], 
          LineString)
        # Should contain all shapes
        self.assertEqual(len(geometry_by_shape), 
          feed.shapes.groupby('shape_id').first().shape[0])
        # Should be None if feed.shapes is None
        feed2 = cairns_shapeless
        self.assertIsNone(feed2.build_geometry_by_shape())

    def test_add_dist_to_shapes(self):
        feed = copy(cairns)
        s1 = feed.shapes.copy()
        feed.add_dist_to_shapes()
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

    def test_build_shapes_geojson(self):
        feed = copy(cairns)
        collection = json.loads(feed.build_shapes_geojson())
        geometry_by_shape = feed.build_geometry_by_shape(use_utm=False)
        for f in collection['features']:
            shape = f['properties']['shape_id']
            geom = sh_shape(f['geometry'])
            self.assertTrue(geom.equals(geometry_by_shape[shape]))

    # Test stop time methods
    # ----------------------------------
    def test_get_stop_times(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        f = feed.get_stop_times(date)
        # Should be a data frame
        self.assertIsInstance(f, pd.core.frame.DataFrame)
        # Should have a reasonable shape
        self.assertTrue(f.shape[0] <= feed.stop_times.shape[0])
        # Should have correct columns
        self.assertEqual(set(f.columns), set(feed.stop_times.columns))

    def test_add_dist_to_stop_times(self):
        feed = copy(cairns)
        st1 = feed.stop_times.copy()
        trips_stats = feed.compute_trips_stats()
        feed.add_dist_to_stop_times(trips_stats)
        st2 = feed.stop_times

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
            print('-'*40)
            print('trip', trip, sdt)
            self.assertEqual(sdt, sorted(sdt))

    # Test feed methods
    # ----------------------------------

    def test_compute_feed_stats(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        trips_stats = feed.compute_trips_stats()
        f = feed.compute_feed_stats(trips_stats, date)
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
        f = feed.compute_feed_stats(trips_stats, '20010101')
        self.assertTrue(f.empty)

    def test_compute_feed_time_series(self):
        feed = copy(cairns)
        date = feed.get_dates()[0]
        trips_stats = feed.compute_trips_stats()
        f = feed.compute_feed_time_series(trips_stats, date, freq='1H')
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
        f = feed.compute_feed_time_series(trips_stats, '20010101')
        self.assertTrue(f.empty)

    

    # Test other methods
    # ----------------------------------
    def test_downsample(self):
        pass


if __name__ == '__main__':
    unittest.main()