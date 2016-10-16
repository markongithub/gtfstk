"""
This module defines the Feed class, which represents a GTFS feed as a collection of data frames, and defines some basic operations on Feed objects.
Almost all other operations on Feed objects are defined as functions living outside of the Feed class rather than methods of the Feed class.
Every function that acts on a Feed object assumes that every attribute of the feed that represents a GTFS file, such as ``agency`` or ``stops``, is either ``None`` or a data frame with the columns required in the `GTFS <https://developers.google.com/transit/gtfs/reference?hl=en>`_.
"""
from pathlib import Path
import zipfile
import tempfile
import shutil
from copy import deepcopy

import pandas as pd 

from . import constants as cs
from . import utilities as ut


class Feed(object):
    """
    A class that represents a GTFS feed, where GTFS tables are stored as data frames.
    Beware, the stop times data frame can be big (several gigabytes), so make sure you have enough memory to handle it.
    Feed (public) attributes are

    - ``dist_units``: a string in ``constants.DIST_UNITS``; specifies the distance units to use when calculating various stats, such as route service distance; should match the implicit distance units of the  ``shape_dist_traveled`` column values, if present
    - ``agency``
    - ``stops``
    - ``routes``
    - ``trips``
    - ``stop_times``
    - ``calendar``
    - ``calendar_dates`` 
    - ``fare_attributes``
    - ``fare_rules``
    - ``shapes``
    - ``frequencies``
    - ``transfers``
    - ``feed_info``

    There are also a few private Feed attributes that are derived from some public attributes and are automatically updated when those public attributes change.
    However, for this update to work, you must properly update the primary attributes like this::

        feed.trips['route_short_name'] = 'bingo'
        feed.trips = feed.trips

    and **not** like this::

        feed.trips['route_short_name'] = 'bingo'

    The first way ensures that the altered trips data frame is saved as the new ``trips`` attribute, but the second way does not.
    """
    def __init__(self, dist_units, agency=None, stops=None, routes=None, 
      trips=None, stop_times=None, calendar=None, calendar_dates=None, 
      fare_attributes=None, fare_rules=None, shapes=None, 
      frequencies=None, transfers=None, feed_info=None):
        """
        Assume that every non-None input is a Pandas data frame, except for ``dist_units`` which should be a string in ``constants.DIST_UNITS``.

        No other format checking is performed.
        In particular, a Feed instance need not represent a valid GTFS feed.
        """
        # Set primary attributes; the @property magic below will then
        # validate some and automatically set secondary attributes
        for prop, val in locals().items():
            if prop in cs.FEED_ATTRS_PUBLIC:
                setattr(self, prop, val)        

    @property 
    def dist_units(self):
        """
        A public Feed attribute made into a property for easy validation.
        """
        return self._dist_units

    @dist_units.setter
    def dist_units(self, val):
        if val not in cs.DIST_UNITS:
            raise ValueError('Distance units are required and '\
              'must lie in {!s}'.format(cs.DIST_UNITS))
        else:
            self._dist_units = val

    # If ``self.trips`` changes then update ``self._trips_i``
    @property
    def trips(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the trips data frame.
        """
        return self._trips

    @trips.setter
    def trips(self, val):
        self._trips = val 
        if val is not None and not val.empty:
            self._trips_i = self._trips.set_index('trip_id')
        else:
            self._trips_i = None

    # If ``self.calendar`` changes, then update ``self._calendar_i``
    @property
    def calendar(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the calendar data frame.
        """
        return self._calendar

    @calendar.setter
    def calendar(self, val):
        self._calendar = val 
        if val is not None and not val.empty:
            self._calendar_i = self._calendar.set_index('service_id')
        else:
            self._calendar_i = None 

    # If ``self.calendar_dates`` changes, 
    # then update ``self._calendar_dates_g``
    @property 
    def calendar_dates(self):
        """
        A public Feed attribute made into a property for easy auto-updating of private feed attributes based on the calendar dates data frame.
        """        
        return self._calendar_dates 

    @calendar_dates.setter
    def calendar_dates(self, val):
        self._calendar_dates = val
        if val is not None and not val.empty:
            self._calendar_dates_g = self._calendar_dates.groupby(
              ['service_id', 'date'])
        else:
            self._calendar_dates_g = None

    def __eq__(self, other):
        """
        Define two feeds be equal if and only if their ``constants.FEED_ATTRS`` attributes are equal, or almost equal in the case of data frames (but not groupby data frames).
        Almost equality is checked via :func:`utilities.almost_equal`, which   canonically sorts data frame rows and columns.
        """
        # Return False if failures
        for key in cs.FEED_ATTRS_PUBLIC:
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

    def copy(self):
        """
        Return a copy of this feed, that is, a feed with all the same public and private attributes.
        """
        other = Feed(dist_units=self.dist_units)
        for key in set(cs.FEED_ATTRS) - set(['dist_units']):
            value = getattr(self, key)
            if isinstance(value, pd.DataFrame):
                # Pandas copy data frame
                value = value.copy()
            elif isinstance(value, pd.core.groupby.DataFrameGroupBy):
                # Pandas does not have a copy method for groupby objects
                # as far as i know
                value = deepcopy(value)
            setattr(other, key, value)
        
        return other

# -------------------------------------
# Functions about input and output
# -------------------------------------
def read_gtfs(path, dist_units=None):
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
    for f in cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL:
        ff = f + '.txt'
        pp = Path(p, ff)
        if pp.exists():
            feed_dict[f] = pd.read_csv(pp.as_posix(), dtype=cs.DTYPE,
              encoding='utf-8-sig') 
            # utf-8-sig gets rid of the byte order mark (BOM);
            # see http://stackoverflow.com/questions/17912307/u-ufeff-in-python-string 
        else:
            feed_dict[f] = None
        
    feed_dict['dist_units'] = dist_units

    # Remove extracted zip directory
    if zipped:
        shutil.rmtree(p.as_posix())

    # Create feed 
    return Feed(**feed_dict)

def write_gtfs(feed, path, ndigits=6):
    """
    Export the given feed to a zip archive located at ``path``.
    Round all decimals to ``ndigits`` decimal places.
    All distances will be displayed in units ``feed.dist_units``.
    """
    # Remove '.zip' extension from path, because it gets added
    # automatically below
    p = Path(path)
    p = p.parent/p.stem

    # Write files to a temporary directory 
    tmp_dir = tempfile.mkdtemp()
    names = cs.GTFS_TABLES_REQUIRED + cs.GTFS_TABLES_OPTIONAL
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
