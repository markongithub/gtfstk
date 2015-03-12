Changes
========

v0.11.4 (2015-03-12)
---------------------
- Changed name to gtfs-tk

v0.11.3 (2015-03-02)
----------------------
- Add ``get_shapes_geojson()``
- Renamed ``get_active_trips()`` and ``get_active_stops()`` to ``get_trips()`` and ``get_stops()``
- Upgraded to Pandas 0.15.2


v0.11.2 (2014-12-10)
----------------------
- Scooped out main logic from ``Feed.get_stops_stats()`` and ``Feed.get_stops_time_series()`` and put it into top level functions
  for the sake of greater flexibility.  Similar to what i did for 
  ``Feed.get_routes_stats()`` and ``Feed.get_routes_time_series()``
- Fixed a bug in computing the last stop of each trip in ``get_trips_stats()``
- Improved the accuracy of trip distances in ``get_trips_stats()``
- Upgraded to Pandas 0.15.1

v0.11.1 (2014-11-12)
----------------------
- Added ``fill_nan_route_short_names()``
- Switched back to version numbering in the style of major.minor.micro

v0.11 (2014-11-10)
----------------------
- Fixed a bug in ``Feed.get_routes_stats()`` that modified the input data frame and therefore affected the same data frame outside of the function (dumb Pandas gotcha). Changed it to operate on a copy of the data frame instead.

v0.10 (2014-11-06)
----------------------
- Speeded up time series computations by at least a factor of 10
- Switched from representing dates as ``datetime.date`` objects to '%Y%m%d' strings (the GTFS way of representing dates), because that's simpler and faster. Added an export method to feed objects
- Minor tweaks to ``add_dist_to_stop_times()``.

v0.9 (2014-10-29)
----------------------
- Scooped out main logic from ``Feed.get_routes_stats()`` and ``Feed.get_routes_time_series()`` and put it into top level functions for the sake of greater flexibility.  I at least need that flexibility to plug into another project. 

v0.8 (2014-10-24)
----------------------
- Simplified methods to accept a single date instead of a list of dates.

v0.7 (2014-10-08)
----------------------
- Whoops, lost track of the changes for this version.

v0.6 (2014-10-08)
----------------------
- Changed ``seconds_to_timestr()`` to ``timestr_to_seconds().``.  Added ``get_busiest_date_of_first_week()``. 

v0.5 (2014-10-02)
----------------------
- Converted headways to minutes
- Added option to change headway start and end time cutoffs in ``get_stops_stats()`` and ``get_stations_stats()``

v0.4 (2014-10-02)
----------------------
- Fixed a bug in get_trips_stats() that caused a failure when a trip was missing a shape ID

v0.3 (2014-09-29)
----------------------
- Switched from major.minor.micro version numbering to major.minor numbering
- Added ``get_vehicle_locations()``.

v0.2.3 (2014-08-22)
----------------------
- Added ``add_dist_to_stop_times()`` and ``add_dist_to_shapes``

v0.2.2 (2014-08-17)
----------------------
- Changed ``get_xy_by_stop()`` name and output type

v0.2.1 (2014-07-22)
----------------------
- Changed from period indices to timestamp indices for time series, because the latter are better supported in Pandas. 
- Upgraded to Pandas 0.14.1.

v0.2.0 (2014-07-22)
----------------------
- Restructured modules 

v0.1.12 (2014-07-21)
----------------------
- Created stats and time series aggregating functions

v0.1.11 (2014-07-17)
----------------------
- Added ``get_dist_from_shapes`` keyword to ``get_trips_stats()`` 

v0.1.10 (2014-07-17)
----------------------
- Fixed some typos and cleaned up the directory

v0.1.9 (2014-07-17)
----------------------
- Changed ``get_routes_stats()`` headway calculation
- Fixed inconsistent outputs in time series functions.

v0.1.8 (2014-07-16)
----------------------
- Minor tweak to ``downsample()``

v0.1.7 (2014-07-16)
----------------------
- Improved ``get_trips_stats()`` and cleaned up code

v0.1.6 (2014-07-04)
----------------------
- Changed time series format

v0.1.5 (2014-06-23)
----------------------
- Added documentation

v0.1.4 (2014-06-20)
----------------------
- Upgraded to Python 3.4

v0.1.3 (2014-06-01)
----------------------
- Created ``utils.py`` and updated Pandas to 0.14.0

v0.1.2 (2014-05-26)
----------------------
-Minor refactoring and tweaks to packaging

v0.1.1 (2014-05-26)
----------------------
- Minor tweaks to packaging

v0.1.0 (2014-05-26 )
----------------------
- Initial version