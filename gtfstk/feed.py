"""
This module defines the Feed class, which represents GTFS files as data frames, and defines some basic operations on Feed objects.
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
    A class that represents a GTFS feed, where GTFS files/tables are stored as as data frames.

    Warning: the stop times data frame can be big (several gigabytes), so make sure you have enough memory to handle it.

    Feed attributes, almost all of which default to ``None``, are

    - ``agency``
    - ``stops``
    - ``routes``
    - ``trips``
    - ``trips_i``: ``trips`` reindexed by its ``'trip_id'`` column; speeds up :func:`calculator.is_active_trip`
    - ``stop_times``
    - ``calendar``
    - ``calendar_i``: ``calendar`` reindexed by its ``'service_id'`` column;  speeds up :func:`calculator.is_active_trip`
    - ``calendar_dates`` 
    - ``calendar_dates_g``: ``calendar_dates`` grouped by ``['service_id', 'date']``; speeds up :func:`calculator.is_active_trip`
    - ``fare_attributes``
    - ``fare_rules``
    - ``shapes``
    - ``frequencies``
    - ``transfers``
    - ``feed_info``
    - ``dist_units``: a string in ``constants.DIST_UNITS``; specifies the distance units of the feed; defaults to 'km' if no ``shape_dist_traveled`` columns are present in the input data frames
    """
    def __init__(self, agency=None, stops=None, routes=None, trips=None, 
      stop_times=None, calendar=None, calendar_dates=None, 
      fare_attributes=None, fare_rules=None, shapes=None, 
      frequencies=None, transfers=None, feed_info=None,
      dist_units=None):
        """
        Assume that every non-None input is a Pandas data frame, except for ``dist_units`` which should be a string in ``constants.DIST_UNITS``.

        If the ``shapes`` or ``stop_times`` data frame has the optional  column ``shape_dist_traveled``, then the native distance units used in those data frames must be specified with ``dist_units``.
        Otherwise, ``dist_units`` will be set to ``'km'`` (kilometers). 

        No other format checking is performed.
        In particular, a Feed instance need not represent a valid GTFS feed.
        """
        # Set primary attributes
        for kwarg, value in locals().items():
            if kwarg in cs.FEED_ATTRS_PRIMARY:
                setattr(self, kwarg, value)

        # Check for valid distance units
        if self.dist_units is not None and\
          self.dist_units not in cs.DIST_UNITS:
            raise ValueError('Distance units must lie in {!s}'.format(
              cs.DIST_UNITS))

        st_has_dist = ut.is_not_null(self.stop_times, 
          'shape_dist_traveled')
        sh_has_dist = ut.is_not_null(self.shapes, 
          'shape_dist_traveled')

        if self.dist_units is None and (st_has_dist or sh_has_dist):
            raise ValueError(
              'This feed has distances, so you must specify dist_units')    
        
        # Set default distance units
        if self.dist_units is None:
            self.dist_units = 'km'

        # Create secondary attributes for fast searching
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
        Two feeds are equal if and only if their ``constants.FEED_ATTRS``    attributes are equal, or almost equal in the case of data frames.
        Almost equality is checked via :func:`utilities.almost_equal`, which   canonically sorts data frame rows and columns.
        """
        # Return False if failures
        for key in cs.FEED_ATTRS:
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

    # TODO: Finish this
    def __setattr__(self, k, v):
        if k not in cs.FEED_ATTRS_PRIMARY:
            raise ValueError('Can only change attributes {!s}'.format(
              cs.FEED_ATTRS_PRIMARY))
            
        setattr(self, k, v)

        # Recompile secondary attributes if necessary
        Create secondary attributes for fast searching
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

    def copy(self):
        """
        Return a copy of this feed.
        """
        other = Feed()
        for key in cs.FEED_ATTRS:
            value = getattr(self, key)
            if isinstance(value, pd.DataFrame):
                # Pandas copy data frame
                value = value.copy()
            setattr(other, key, value)
        
        return other

    # def copy(self):
    #     """
    #     Return a copy of this feed.
    #     """
    #     # Copy feed attributes necessary to create new feed
    #     new_feed_input = dict()

    #     # New distance units in/out should be old distance units out
    #     new_feed_input['dist_units_in'] = self.dist_units_out
    #     new_feed_input['dist_units_out'] = self.dist_units_out

    #     # Set remaining attributes
    #     input_keys = set(cs.FEED_ATTRS_PRIMARY) - set(['dist_units_in', 'dist_units_out'])
    #     for key in input_keys:
    #         value = getattr(self, key)
    #         if isinstance(value, pd.DataFrame):
    #             # Pandas copy data frame
    #             value = value.copy()
    #         new_feed_input[key] = value
        
    #     return Feed(**new_feed_input)

    def recompile(self):
        """
        Return a new feed created from the primary attributes of this feed (those listed in ``constants.FEED_ATTRS_PRIMARY``).
        In particular, this method creates anew secondary feed attributes (those listed in ``constants.FEED_ATTRS_SECONDARY``) from the primary ones, thereby resyncing the secondary attributes to the primary ones.
        Primary and secondary attributes can get out of sync if a user changes one and not the other correspondingly. 
        """
        new_feed_input = dict()
        for key, value in cs.FEED_ATTRS_PRIMARY.items():
            new_feed_input[key] = value
        return Feed(**new_feed_input)

# -------------------------------------
# Functions about input and output
# -------------------------------------
def read_gtfs(path, dist_units_in=None, dist_units_out=None):
    """
    Create a Feed object from the given path and given distance units.
    The path points to a directory containing GTFS text files or a zip file that unzips as a collection of GTFS text files (but not as a directory containing GTFS text files).
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
    p = p.parent/p.stem

    # Write files to a temporary directory 
    tmp_dir = tempfile.mkdtemp()
    names = cs.REQUIRED_GTFS_FILES + cs.OPTIONAL_GTFS_FILES
    for name in names:
        f = getattr(feed, name)
        if f is None:
            continue

        f = f.copy()
        # Some columns need to be output as integers.
        # If there are NaNs in any such column, 
        # then Pandas will format the column as float, which we don't want.
        f_int_cols = set(cs.INT_COLUMNS) & set(f.columns)
        for s in f_int_cols:
            f[s] = f[s].fillna(-1).astype(int).astype(str).\
              replace('-1', '')
        tmp_path = Path(tmp_dir, name + '.txt')
        f.to_csv(tmp_path.as_posix(), index=False, 
          float_format='%.{!s}f'.format(ndigits))

    # Zip directory 
    shutil.make_archive(p.as_posix(), format='zip', root_dir=tmp_dir)    

    # Delete temporary directory
    shutil.rmtree(tmp_dir)
