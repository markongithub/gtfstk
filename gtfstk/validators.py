"""
Functions about validation.
A work in progress.
"""
import re 
import pytz
import datetime as dt 

import pycountry
import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp


TIME_PATTERN1 = re.compile(r'^[0,1,2]\d:\d\d:\d\d$')
TIME_PATTERN2 = re.compile(r'^\d:\d\d:\d\d$')
DATE_FORMAT = '%Y%m%d'
TIMEZONES = set(pytz.all_timezones)
# ISO639-1 language codes
LANGS = set([lang.alpha_2 for lang in pycountry.languages 
  if hasattr(lang, 'alpha_2')])
CURRENCIES = set([c.alpha_3 for c in pycountry.currencies if hasattr(c, 'alpha_3')])
URL_PATTERN = re.compile(
  r'^(?:http)s?://' # http:// or https://
  r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'   #domain...
  r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
  r'(?::\d+)?' # optional port
  r'(?:/?|[/?]\S+)$', 
  re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'[^@]+@[^@]+\.[^@]+')
COLOR_PATTERN = re.compile(r'(?:[0-9a-fA-F]{2}){3}$')


def valid_str(x):
    """
    Return ``True`` if ``x`` is a non-blank string; otherwise return False.
    """
    if isinstance(x, str) and x.strip():
        return True 
    else:
        return False 

def valid_time(x):
    if re.match(TIME_PATTERN1, x) or re.match(TIME_PATTERN2, x):
        return True 
    else:
        return False

def valid_date(x):
    """
    Retrun ``True`` if ``x`` is a valid date; otherwise return ``False``.
    """
    try:
        if x != dt.datetime.strptime(x, DATE_FORMAT).strftime(DATE_FORMAT):
            raise ValueError
        return True
    except ValueError:
        return False

def valid_timezone(x):
    """
    Retrun ``True`` if ``x`` is a valid human-readable timezone string, e.g. 'Africa/Abidjan'; otherwise return ``False``.
    """
    return x in TIMEZONES

def valid_lang(x):
    """
    Return ``True`` if ``x`` is a valid two-letter ISO 639 language code, e.g. 'aa'; otherwise return ``False``.
    """
    return x in LANGS

def valid_currency(x):
    """
    Return ``True`` if ``x`` is a valid three-letter ISO 4217 currency code, e.g. 'AED'; otherwise return ``False``.
    """
    return x in CURRENCIES

def valid_url(x):
    """
    Return ``True`` if ``x`` is a valid URL; otherwise return ``False``.
    """
    if re.match(URL_PATTERN, x):
        return True 
    else:
        return False

def valid_email(x):
    if re.match(EMAIL_PATTERN, x):
        return True 
    else:
        return False

def valid_color(x):
    """
    Return ``True`` if ``x`` a valid hexadecimal color string without the leading hash; otherwise return ``False``.
    """
    if re.match(COLOR_PATTERN, x):
        return True 
    else:
        return False

def check_table(errors, table, df, condition, message):
    """
    Given a list of errors, a table name (string), the DataFrame corresponding to the table, a boolean condition on the DataFrame, and an error message, do the following.
    Record the indices of the rows of the DataFrame that statisfy the condition.
    If this is a nonempty list, then append the given error message to the list of errors along with the offending table indices.
    Otherwise, return the original list of errors.
    """
    indices = df.loc[condition].index.tolist()
    if indices:
        errors.append([table, message, indices])

    return errors

def check_column(errors, table, df, column, column_required, checker):
    """
    Given a list of errors, a table name (string), the DataFrame corresponding to the table, a column name (string), a boolean indicating whether the table is required, and a checker (boolean-valued unary function), do the following.
    Apply the checker to the column entries and record the indices of the rows on which errors occur, if any.
    Append these errors to the given error list and return the new error list.
    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    v = lambda x: pd.notnull(x) and checker(x)
    cond = ~f[column].map(checker)  
    errors = check_table(errors, table, f, cond, 
      '{!s} malformed'.format(column))

    return errors

def check_column_id(errors, table, df, column, column_required=True):
    """
    A modified verios of :func:`check_column` that applies to a column that must have unduplicated IDs. 
    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    v = lambda x: pd.notnull(x) and valid_str(x)
    cond = ~f[column].map(v)
    errors = check_table(errors, table, f, cond, 
      '{!s} malformed'.format(column))

    cond = f[column].duplicated()
    errors = check_table(errors, table, f, cond, 
      '{!s} duplicated'.format(column))

    return errors 

def check_column_linked_id(errors, table, df, column, column_required, 
  target_df, target_column=None):
    """
    """
    if target_column is None:
        target_column = column 

    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].isin(target_df[target_column])
    errors = check_table(errors, table, f, cond, 
      '{!s} undefined'.format(column))

    return errors 

def format_errors(errors, as_df):
    """
    Given a possibly empty list of errors (triples), return a DataFrame with the errors as rows and the columns ['table', 'error', 'error_row_indices'], if ``as_df``.
    If not ``as_df``, then return the original list of errors.
    """
    if as_df:
        errors = pd.DataFrame(errors, 
          columns=['table', 'error', 'error_row_indices'])
    return errors

def check_for_required_tables(feed, as_df=False):
    errors = []
    req_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
    for table in req_tables:
        if getattr(feed, table) is None:
            errors.append([table, 'Missing file', []])
    # Calendar check is different
    if feed.calendar is None and feed.calendar_dates is None:
        errors.append(['calendar/calendar_dates', 
          'Both calendar and calendar_dates files are missing', []])
    return format_errors(errors, as_df) 

def check_for_required_columns(feed, as_df=False):
    errors = []
    for table, group in cs.GTFS_REF.groupby('table'):
        f = getattr(feed, table)
        if f is None:
            continue

        for column, column_required in group[['column', 'column_required']].itertuples(
          index=False):
            if column_required and column not in f.columns:
                errors.append([table, 'Missing column {!s}'.format(column), []])
    return format_errors(errors, as_df) 

def check_agency(feed, as_df=False):
    """
    Check thet ``feed.agency`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('agency', error message, row indices where errors occur).
    """
    f = feed.agency.copy()
    errors = []

    # Check service_id
    errors = check_column_id(errors, 'agency', f, 'agency_id', False)

    # Check agency_name
    errors = check_column(errors, 'agency', f, 'agency_name', True, valid_str)

    # Check agency_url
    errors = check_column(errors, 'agency', f, 'agency_url', True, valid_url)

    # Check agency_timezone
    errors = check_column(errors, 'agency', f, 'agency_timezone', True,
      valid_timezone)

    # Check agency_fare_url
    errors = check_column(errors, 'agency', f, 'agency_fare_url', False, valid_url)

    # Check agency_lang
    errors = check_column(errors, 'agency', f, 'agency_lang', False, valid_lang)
    
    # Check agency_phone
    errors = check_column(errors, 'agency', f, 'agency_phone', False, valid_str)

    # Check agency_email
    errors = check_column(errors, 'agency', f, 'agency_email', False, valid_email)

    return format_errors(errors, as_df)

def check_calendar(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar``.
    """    
    f = feed.calendar.copy()
    errors = []

    # Check service_id
    errors = check_column_id(errors, 'calendar', f, 'service_id')

    # Check weekday columns
    v = lambda x: x in range(2)
    for col in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
      'saturday', 'sunday']:
        errors = check_column(errors, 'calendar', f, col, True, v)

    # Check start_date and end_date
    for col in ['start_date', 'end_date']:
        errors = check_column(errors, 'calendar', f, col, True, valid_date)

    return format_errors(errors, as_df) 

def check_calendar_dates(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.calendar_dates.copy()
    table = 'calendar_dates'
    errors = []

    # Check service_id
    if feed.calendar is not None:
        errors = check_column_linked_id(errors, table, f, 'service_id',
          True, feed.calendar)
    else:
        errors = check_column(errors, table, f, 'service_id', True, valid_str)

    # Check date
    errors = check_column(errors, table, f, 'date', True, valid_date)

    # No duplicate (service_id, date) pairs allowed
    cond = f[['service_id', 'date']].duplicated()
    errors = check_table(errors, table, f, cond, 
      '(service_id, date)-pair duplicated')

    # Check exception_type
    v = lambda x: x in [1, 2]
    errors = check_column(errors, table, f, 'exception_type', True, v)

    return format_errors(errors, as_df) 

def check_fare_attributes(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.fare_attributes.copy()
    table = 'fare_attributes'
    errors = []

    # Check fare_id
    errors  = check_column_id(errors, table, f, 'fare_id')

    # Check currency_type
    errors = check_column(errors, table, f, 'currency_type', True, valid_currency)

    # Check payment_method
    v = lambda x: x in range(2)
    errors = check_column(errors, table, f, 'payment_method', True, v)

    # Check transfers
    v = lambda x: x in [np.nan, 0, 1, 2]
    errors = check_column(errors, table, f, 'transfers', True, v)

    # Check transfer_duration
    v = lambda x: x >= 0
    errors = check_column(errors, table, f, 'transfer_duration', False, v)

    return format_errors(errors, as_df)    

def check_fare_rules(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.fare_rules.copy()
    table = 'fare_rules'
    errors = []

    # Check fare_id
    errors  = check_column_linked_id(errors, table, f, 'fare_id', True, 
      feed.fare_attributes)

    # Check route_id
    errors  = check_column_linked_id(errors, table, f, 'route_id', False, 
      feed.routes)

    # Check origin_id, destination_id, contains_id
    for col in ['origin_id', 'destination_id', 'contains_id']:
        errors  = check_column_linked_id(errors, table, f, col, False, 
          feed.stops, 'zone_id')
    
    return format_errors(errors, as_df)    

def check_feed_info(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.feed_info``.
    """    
    f = feed.feed_info.copy()
    errors = []

    # Check feed_publisher_name
    errors = check_column(errors, 'feed_info', f, 'feed_publisher_name', True,
      valid_str)

    # Check feed_publisher_url
    errors = check_column(errors, 'feed_info', f, 'feed_publisher_url', True,
      valid_url)

    # Check feed_lang
    errors = check_column(errors, 'feed_info', f, 'feed_lang', True,
      valid_lang)

    # Check feed_start_date and feed_end_date
    for col in ['feed_start_date', 'feed_end_date']:
        errors = check_column(errors, 'feed_info', f, col, False, valid_date)

    d1, d2 = f[['feed_start_date', 'feed_end_date']].ix[0].values
    if pd.notnull(d1) and pd.notnull(d2) and d1 > d1:
        errors.append(['feed_info', 'feed_start_date later than feed_end_date', 
          [0]])

    # Check feed_version
    errors = check_column(errors, 'feed_info', f, 'feed_version', False, valid_str)

    return format_errors(errors, as_df)

def check_frequencies(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.frequencies``.
    """    
    f = feed.frequencies.copy()
    table = 'frequencies'
    errors = []

    # Check trip_id
    errors = check_column_linked_id(errors, table, f, 'trip_id', True,
      feed.trips)

    # Check start_time and end_time
    time_cols = ['start_time', 'end_time'] 
    for col in time_cols:
        errors = check_column(errors, table, f, col, True, valid_time)

    for col in time_cols:
        f[col] = f[col].map(hp.timestr_to_seconds)

    # start_time should be earlier than end_time
    cond = f['start_time'] >= f['end_time']
    errors = check_table(errors, table, f, cond, 'start_time not earlier than end_time')

    # Headway periods should not overlap
    f = f.sort_values(['trip_id', 'start_time'])
    for __, group in f.groupby('trip_id'):
        a = group['start_time'].values
        b = group['end_time'].values  
        indices = np.flatnonzero(a[1:] < b[:-1]).tolist()
        if indices:
            errors.append([table, 
             'Headway periods for the same trip overlap', 
              indices])

    # Check headway_secs
    v = lambda x: x >= 0
    errors = check_column(errors, table, f, 'headway_secs', True, v)

    # Check exact_times
    v = lambda x: x in [np.nan, 0, 1]
    error = check_column(errors, table, f, 'exact_times', False, v)

    return format_errors(errors, as_df)

def check_routes(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.routes``.
    """    
    f = feed.routes.copy()
    errors = []

    # Check route_id
    errors = check_column_id(errors, 'routes', f, 'route_id')

    # Check agency_id
    if 'agency_id' in f:
        if 'agency_id' not in feed.agency.columns:
            errors.append('Column agency_id present in routes but not in agency')
        else:
            g = f.dropna(subset=['agency_id'])
            cond = ~g['agency_id'].isin(feed.agency['agency_id'])
            errors = check_table(errors, 'routes', g, cond, 'agency_id undefined')

    # Check route_short_name and route_long_name
    for column in ['route_short_name', 'route_long_name']:
        errors = check_column(errors, 'routes', f, column, False, valid_str)

    cond = ~(f['route_short_name'].notnull() | f['route_long_name'].notnull())
    errors = check_table(errors, 'routes', f, cond, 
      'route_short_name and route_long_name both empty')

    # Check route_type
    v = lambda x: x in range(8)
    errors = check_column(errors, 'routes', f, 'route_type', True, v)

    # Check route_url
    errors = check_column(errors, 'routes', f, 'route_url', False, valid_url)

    # Check route_color and route_text_color
    for column in ['route_color', 'route_text_color']:
        v = lambda x: pd.isnull(x) or valid_color(x)
        if column in f.columns:
            cond = ~f[column].map(v)
            errors = check_table(errors, 'routes', f, cond, 
              '{!s} malformed'.format(column))
    
    return format_errors(errors, as_df)

def check_shapes(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.shapes``.
    """    
    f = feed.shapes.copy()
    errors = []

    # Check shape_id
    errors = check_column(errors, 'shapes', f, 'shape_id', True, valid_str)

    # Check shape_pt_lon and shape_pt_lat
    for column, bound in [('shape_pt_lon', 180), ('shape_pt_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        errors = check_table(errors, 'stops', f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check shape_dist_traveled
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled']).sort_values(
          ['shape_id', 'shape_pt_sequence'])
        for __, group in g.groupby('shape_id'):
            a = group['shape_dist_traveled'].values  
            indices = np.flatnonzero(a[1:] <= a[:-1]).tolist()
            if indices:
                errors.append(['shapes', 
                  'shape_dist_traveled not increasing for shape', 
                  indices])

    return format_errors(errors, as_df)

def check_stops(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.stops``.
    """    
    f = feed.stops.copy()
    errors = []

    # Check stop_id
    errors = check_column_id(errors, 'stops', f, 'stop_id')

    # Check stop_code, stop_desc, zone_id, parent_station
    for column in ['stop_code', 'stop_desc', 'zone_id', 'parent_station']:
        errors = check_column(errors, 'stops', f, column, False, valid_str)

    # Check stop_name
    errors = check_column(errors, 'stops', f, 'stop_name', True, valid_str)

    # Check stop_lon and stop_lat
    for column, bound in [('stop_lon', 180), ('stop_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        errors = check_table(errors, 'stops', f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check stop_url
    errors = check_column(errors, 'stops', f, 'stop_url', False, valid_url)

    # Check location_type
    v = lambda x: x in range(2)
    errors = check_column(errors, 'stops', f, 'location_type', False, v)

    # Check stop_timezone
    errors = check_column(errors, 'stops', f, 'stop_timezone', False, valid_timezone)

    # Check wheelchair_boarding
    v = lambda x: x in range(3)
    errors = check_column(errors, 'stops', f, 'wheelchair_boarding', False, v)

    # Check further location_type and parent_station   
    if 'parent_station' in f.columns:
        if 'location_type' not in f.columns:
            errors.append(['stops', 
              'parent_station column present but location_type column missing', 
              []])
        else:
            # Parent stations must have location type 1
            station_ids = f.loc[f['parent_station'].notnull(), 'parent_station']
            cond = f['stop_id'].isin(station_ids) & (f['location_type'] != 1)
            errors = check_table(errors, 'stops', f, cond, 
              'A parent station must have location_type equal to 1')

            # Parent stations should not have parent stations
            cond = f['parent_station'].notnull() & (f['location_type'] == 1) 
            errors = check_table(errors, 'stops', f, cond, 
              'location_type must equal 1')

    return format_errors(errors, as_df) 

def check_stop_times(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.stop_times``.
    """    
    f = feed.stop_times.copy()
    errors = []

    # Check trip_id
    errors = check_column_linked_id(errors, 'stop_times', f, 'trip_id', True, 
      feed.trips)

    # Check arrival_time and departure_time
    for col in ['arrival_time', 'departure_time']:
        errors = check_column(errors, 'stop_times', f, col, False, valid_time)

    # Check stop_id
    errors = check_column_linked_id(errors, 'stop_times', f, 'stop_id', True, 
      feed.stops)

    # Check stop_headsign
    errors = check_column(errors, 'stop_times', f, 'stop_headsign', False, valid_str)

    # Check pickup_type and drop_off_type
    for col in ['pickup_type', 'drop_off_type']:
        v = lambda x: x in range(4)
        errors = check_column(errors, 'stop_times', f, col, False, v)

    # Check shape_dist_traveled
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled']).sort_values(['trip_id', 'stop_sequence'])
        for __, group in g.groupby('trip_id'):
            a = group['shape_dist_traveled'].values  
            indices = np.flatnonzero(a[1:] <= a[:-1]).tolist()
            if indices:
                errors.append(['stop_times', 
                  'shape_dist_traveled not increasing for trip', 
                  indices])

    # Check timepoint
    v = lambda x: x in range(2)
    errors = check_column(errors, 'stop_times', f, 'timepoint', False, v)

    return format_errors(errors, as_df) 

def check_transfers(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.transfers``.
    """    
    f = feed.transfers.copy()
    errors = []
    
    # Check from_stop_id and to_stop_id
    for col in ['from_stop_id', 'to_stop_id']:
        errors = check_column_linked_id(errors, 'transfers', f, col, True,
          feed.stops, 'stop_id')

    # Check transfer_type
    v = lambda x: x in [np.nan, 0, 1, 2, 3, 4]
    errors = check_column(errors, 'transfers', f, 'transfer_type', True, v)

    # Check min_transfer_time
    v = lambda x: x >= 0
    errors = check_column(errors, 'transfers', f, 'min_transfer_time', False, v)

    return format_errors(errors, as_df)

def check_trips(feed, as_df=False):
    """
    Analog of :func:`check_agency` for ``feed.trips``.
    """    
    f = feed.trips.copy()
    errors = []

    # Check trip_id
    errors = check_column_id(errors, 'trips', f, 'trip_id')

    # Check route_id
    errors = check_column_linked_id(errors, 'trips', f, 'route_id', True, 
      feed.routes)
    
    # Check service_id 
    if feed.calendar is not None:
        cond = ~f['service_id'].isin(feed.calendar['service_id'])
    else:
        cond = ~f['service_id'].isin(feed.calendar_dates['service_id'])
    errors = check_table(errors, 'trips', f, cond, 'service_id undefined')

    # Check direction_id
    v = lambda x: x in range(2)
    errors = check_column(errors, 'trips', f, 'direction_id', False, v)

    # Check block_id
    if 'block_id' in f.columns:
        v = lambda x: pd.isnull(x) or valid_str(x)
        cond = ~f['block_id'].map(v)
        errors = check_table(errors, 'trips', f, cond, 'block_id blank')

        g = f.dropna(subset=['block_id'])
        cond = ~g['block_id'].duplicated(keep=False)
        errors = check_table(errors, 'trips', g, cond, 'block_id unduplicated')

    # Check shape_id
    errors = check_column_linked_id(errors, 'trips', f, 'shape_id', False, feed.shapes)

    # Check wheelchair_accessible and bikes_allowed
    v = lambda x: x in range(3)    
    for column in ['wheelchair_accessible', 'bikes_allowed']:
        errors = check_column(errors, 'trips', f, column, False, v)

    return format_errors(errors, as_df)

def validate(feed, as_df=True):
    """
    Check whether the given feed is valid by doing all the checks above.
    Return the errors found as a possibly empty list of pairs (table name, error message).
    If ``as_df``, then format the error list as a DataFrame with the columns

    - ``'table'``: name of table where error occurs
    - ``'error'``: error message.

    Return early if the feed is missing required tables or required columns.
    """
    errors = []

    # Check for required table and for required columns, and halt if errors
    checkers = [
      'check_for_required_tables',
      'check_for_required_columns',
      ]
    for checker in checkers:
        errors.extend(globals()[checker](feed))
        if errors:
            return format_errors(errors, as_df)

    # Check required tables
    checkers = [
      'check_agency',
      'check_routes',
      'check_stops',
      'check_stop_times',
      'check_trips',
      ]
    for checker in checkers:
        errors.extend(globals()[checker](feed))

    # Check optional tables
    checkers = [
      'check_calendar',
      'check_calendar_dates',
      'check_fare_attributes',
      'check_fare_rules',
      'check_feed_info',
      'check_frequencies',
      'check_transfers',
      'check_shapes',
      ]
    for checker in checkers:
        table = checker.replace('check_', '')
        f = getattr(feed, table)
        if f is not None:
            errors.extend(globals()[checker](feed))

    return format_errors(errors, as_df)