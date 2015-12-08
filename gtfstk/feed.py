"""
This module defines the Feed class which represents GTFS files as data frames.
Operations on Feed objects live outside of the class in other modules.
"""
from . import constants as cs
from . import utilities as ut


class Feed(object):
    """
    A class that represents GTFS files as data frames.

    Warning: the stop times data frame can be big (several gigabytes), 
    so make sure you have enough memory to handle it.

    Attributes, almost all of which default to ``None``:

    - ``agency``
    - ``stops``
    - ``routes``
    - ``trips``
    - ``trips_i``: ``trips`` reindexed by its ``'trip_id'`` column;
      speeds up ``is_active_trip()``
    - ``stop_times``
    - ``calendar``
    - ``calendar_i``: ``calendar`` reindexed by its ``'service_id'`` column;
      speeds up ``is_active_trip()``
    - ``calendar_dates`` 
    - ``calendar_dates_g``: ``calendar_dates`` grouped by 
      ``['service_id', 'date']``; speeds up ``is_active_trip()``
    - ``fare_attributes``
    - ``fare_rules``
    - ``shapes``
    - ``frequencies``
    - ``transfers``
    - ``feed_info``
    - ``dist_units_in``: a string in ``constants.DISTANCE_UNITS``;
      specifies the native distance units of the feed; default is 'km'
    - ``dist_units_out``: a string in ``constants.DISTANCE_UNITS``;
      specifies the output distance units for functions that operate
      on Feed objects; default is ``dist_units_in``
    - ``convert_dist``: function that converts from ``dist_units_in`` to
      ``dist_units_out``
    """
    def __init__(self, agency=None, stops=None, routes=None, trips=None, 
      stop_times=None, calendar=None, calendar_dates=None, 
      fare_attributes=None, fare_rules=None, shapes=None, 
      frequencies=None, transfers=None, feed_info=None,
      dist_units_in=None, dist_units_out=None):
        """
        Assume that every non-None input is a Pandas data frame,
        except for ``dist_units_in`` and ``dist_units_out`` which 
        should be strings.

        If the ``shapes`` or ``stop_times`` data frame has the optional
        column ``shape_dist_traveled``,
        then the native distance units used in those data frames must be 
        specified with ``dist_units_in``. 
        Supported distance units are listed in ``constants.DISTANCE_UNITS``.
        
        If ``shape_dist_traveled`` column does not exist, then 
        ``dist_units_in`` is not required and will be set to ``'km'``.
        The parameter ``dist_units_out`` specifies the distance units for 
        the outputs of functions that act on feeds, 
        e.g. ``compute_trips_stats()``.
        If ``dist_units_out`` is not specified, then it will be set to
        ``dist_units_in``.

        No other format checking is performed.
        In particular, a Feed instance need not represent a valid GTFS feed.
        """
        # Set attributes
        for kwarg, value in locals().items():
            if kwarg == 'self':
                continue
            setattr(self, kwarg, value)

        # Check for valid distance units
        # Require dist_units_in if feed has distances
        if (self.stop_times is not None and\
          'shape_dist_traveled' in self.stop_times.columns) or\
          (self.shapes is not None and\
          'shape_dist_traveled' in self.shapes.columns):
            if self.dist_units_in is None:
                raise ValueError(
                  'This feed has distances, so you must specify dist_units_in')    
        DU = cs.DISTANCE_UNITS
        for du in [self.dist_units_in, self.dist_units_out]:
            if du is not None and du not in DU:
                raise ValueError('Distance units must lie in {!s}'.format(DU))

        # Set defaults
        if self.dist_units_in is None:
            self.dist_units_in = 'km'
        if self.dist_units_out is None:
            self.dist_units_out = self.dist_units_in
        
        # Set distance conversion function
        self.convert_dist = ut.get_convert_dist(self.dist_units_in,
          self.dist_units_out)

        # Convert distances to dist_units_out if necessary
        if self.stop_times is not None and\
          'shape_dist_traveled' in self.stop_times.columns:
            self.stop_times['shape_dist_traveled'] =\
              self.stop_times['shape_dist_traveled'].map(self.convert_dist)

        if self.shapes is not None and\
          'shape_dist_traveled' in self. shapes.columns:
            self.shapes['shape_dist_traveled'] =\
              self.shapes['shape_dist_traveled'].map(self.convert_dist)

        # Create some extra data frames for fast searching
        if self.trips is not None and not self.trips.empty:
            self.trips_i = self.trips.set_index('trip_id')
        else:
            self.trips_i = None

        if self.calendar is not None and not self.calendar.empty:
            self.calendar_i = self.calendar.set_index('service_id')
        else:
            self.calendar_i = None 

        if self.calendar_dates is not None and not self.calendar_dates.empty:
            self.calendar_dates_g = self.calendar_dates.groupby(
              ['service_id', 'date'])
        else:
            self.calendar_dates_g = None