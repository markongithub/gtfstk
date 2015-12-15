"""
This module defines the Feed class, which represents GTFS files as data frames,
and defines some basic operations on Feed objects.
Almost all operations on Feed objects are functions that live outside of the Feed class and are not methods of the Feed class.
Every function that acts on a Feed object assumes that every attribute of the feed that represents a GTFS file, such as ``agency`` or ``stops``, is either ``None`` or is a data frame with the columns required in the `GTFS <https://developers.google.com/transit/gtfs/reference?hl=en>`_.
"""
from pathlib import Path
import zipfile
import tempfile
import shutil

import pandas as pd 

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

    def __eq__(self, other):
        """
        Define equality between two feeds as follows.
        Two feeds are equal if and only if their ``contsants.FEED_INPUTS``
        attributes are equal, or almost equal in the case of data frames.
        Almost equality is checked via :func:`utilities.almost_equal`, which
        canonically sorts data frame rows and columns.
        """
        # Return False if failures
        for key in cs.FEED_INPUTS:
            x = getattr(self, key)
            y = getattr(other, key)
            # Data frame case
            if isinstance(x, pd.DataFrame):
                if not isinstance(y, pd.DataFrame) or\
                  not ut.almost_equal(x, y):
                    return False 
            # Other case
            else:
                if x != y:
                    return False
        # No failures
        return True

# -------------------------------------
# Functions about basics
# -------------------------------------
def copy(feed):
    """
    Return a copy of the given feed, using Pandas's copy method to 
    properly copy feed attributes that are data frames.
    """
    # Copy feed attributes necessary to create new feed
    new_feed_input = dict()
    for key in cs.FEED_INPUTS:
        value = getattr(feed, key)
        if isinstance(value, pd.DataFrame):
            # Pandas copy data frame
            value = value.copy()
        new_feed_input[key] = value
    
    return Feed(**new_feed_input)

# def concatenate(feeds, prefixes=None):
#     """
#     Given a list of feeds, concatenate or set equal their attributes.
#     To avoid GTFS ID collisions when doing so, prefix the GTFS IDs
#     of the data frames in ``feeds[j]`` with the string ``prefixes[j]``
#     via :func:`prefix_ids`.
#     Return the resulting feed.

#     If ``feeds`` is empty, then return the empty feed.
#     If there is only one feed in ``feeds``, then return ``feeds[0]``.
#     Raise a ``ValueError`` if the given feeds have different 
#     ``dist_units_in`` or ``dist_units_out`` attributes.
#     If ``prefixes is None``, then set it to ``['feed0_', 'feed1_', ...]``.
#     Raise a ``ValueError`` if ``prefixes`` is not ``None`` and 
#     the lengths of ``feeds`` and ``prefixes`` differ.
#     """
#     # Trivial cases
#     n = len(feeds)
#     if not n:
#         return Feed()
#     if n == 1:
#         return feeds[0]

#     # Raise error if conflicting distance units
#     for i in range(n - 1):
#         if feeds[i].dist_units_in != feeds[i + 1].dist_units_in or\
#           feeds[i].dist_units_out != feeds[i + 1].dist_units_out:
#             raise ValueError('The given feeds must have the same '\
#               'dist_units_in and dist_units_out attributes')

#     # Initialize prefixes if necessary
#     if prefixes is None:
#         prefixes = ['feed{!s}_'.format(i) for i in range(n)]
#     elif len(prefixes) != n:
#         raise ValueError('prefixes must be None or '\
#           'have the same length as feeds')

#     # Ready to go now
#     new_feed_input = dict()
#     for key in cs.FEED_INPUTS:
#         value = None
#         for feed, prefix in zip(feeds, prefixes):
#             v = getattr(feed, key)
#             if isinstance(v, pd.DataFrame):
#                 # Prefix IDs of v
#                 v = ut.prefix_ids(v, prefix)
#                 # Overwrite/concatenate value with v
#                 if value is None:
#                     value = v.copy()
#                 else:
#                     value = pd.concat([value, v])
#             else:
#                 # Set/reset value to v
#                 value = v

#         new_feed_input[key] = value

#     return Feed(**new_feed_input)

# -------------------------------------
# Functions about input and output
# -------------------------------------
def read_gtfs(path, dist_units_in=None, dist_units_out=None):
    """
    Create a Feed object from the given path and 
    given distance units.
    The path points to a directory containing GTFS text files or 
    a zip file that unzips as a collection of GTFS text files
    (but not as a directory containing GTFS text files).
    """
    p = Path(path)
    if not p.exists():
        raise ValueError("Path {!s} does not exist".format(p.as_posix()))

    # Unzip path if necessary
    zipped = False
    if zipfile.is_zipfile(p.as_posix()):
        # Extract to temporary location
        zipped = True
        archive = zipfile.ZipFile(p.as_posix())
        # Strip off .zip extension
        p = p.parent / p.stem
        archive.extractall(p.as_posix())


    # Read files into feed dictionary of data frames
    feed_dict = {}
    for f in cs.REQUIRED_GTFS_FILES + cs.OPTIONAL_GTFS_FILES:
        ff = f + '.txt'
        pp = Path(p, ff)
        if pp.exists():
            feed_dict[f] = pd.read_csv(pp.as_posix(), dtype=cs.DTYPE,
              encoding='utf-8-sig') 
            # utf-8-sig gets rid of the byte order mark (BOM);
            # see http://stackoverflow.com/questions/17912307/u-ufeff-in-python-string 
        else:
            feed_dict[f] = None
        
    feed_dict['dist_units_in'] = dist_units_in
    feed_dict['dist_units_out'] = dist_units_out

    # Remove extracted zip directory
    if zipped:
        shutil.rmtree(p.as_posix())

    # Create feed 
    return Feed(**feed_dict)

def write_gtfs(feed, path, ndigits=6):
    """
    Export the given feed to a zip archive located at ``path``.
    Round all decimals to ``ndigits`` decimal places.
    All distances will be displayed in units ``feed.dist_units_out``.

    Assume the following feed attributes are not ``None``: none.
    """
    # Remove '.zip' extension from path, because it gets added
    # automatically below
    p = Path(path)
    p = p.parent / p.stem

    # Write files to a temporary directory 
    tmp_dir = tempfile.mkdtemp()
    names = cs.REQUIRED_GTFS_FILES + cs.OPTIONAL_GTFS_FILES
    INT_COLUMNS_set = set(cs.INT_COLUMNS)
    for name in names:
        f = getattr(feed, name)
        if f is None:
            continue

        f = f.copy()
        # Some columns need to be output as integers.
        # If there are NaNs in any such column, 
        # then Pandas will format the column as float, which we don't want.
        s = list(INT_COLUMNS_set & set(f.columns))
        if s:
            f[s] = f[s].fillna(-1).astype(int).astype(str).\
              replace('-1', '')
        tmp_path = Path(tmp_dir, name + '.txt')
        f.to_csv(tmp_path.as_posix(), index=False, 
          float_format='%.{!s}f'.format(ndigits))

    # Zip directory 
    shutil.make_archive(p.as_posix(), format='zip', root_dir=tmp_dir)    

    # Delete temporary directory
    shutil.rmtree(tmp_dir)
