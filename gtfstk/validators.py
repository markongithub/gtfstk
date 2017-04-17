"""
Functions about validation.
"""
import re 
import pytz
import datetime as dt 

import pycountry
import numpy as np
import pandas as pd

from . import constants as cs
from . import helpers as hp


TIME_PATTERN1 = re.compile(r'^[0,1,2,3]\d:\d\d:\d\d$')
TIME_PATTERN2 = re.compile(r'^\d:\d\d:\d\d$')
DATE_FORMAT = '%Y%m%d'
TIMEZONES = set(pytz.all_timezones)
# ISO639-1 language codes, both lower and upper case
LANGS = set([lang.alpha_2 for lang in pycountry.languages 
  if hasattr(lang, 'alpha_2')])
LANGS |= set(x.upper() for x in LANGS)
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
    Return ``True`` if ``x`` is a non-blank string; otherwise return ``False``.
    """
    if isinstance(x, str) and x.strip():
        return True 
    else:
        return False 

def valid_time(x):
    """
    Return ``True`` if ``x`` is a valid H:MM:SS or HH:MM:SS time; otherwise return ``False``.
    """
    if isinstance(x, str) and\
      (re.match(TIME_PATTERN1, x) or re.match(TIME_PATTERN2, x)):
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
    if isinstance(x, str) and re.match(URL_PATTERN, x):
        return True 
    else:
        return False

def valid_email(x):
    """
    Return ``True`` if ``x`` is a valid email address; otherwise return ``False``.
    """
    if isinstance(x, str) and re.match(EMAIL_PATTERN, x):
        return True 
    else:
        return False

def valid_color(x):
    """
    Return ``True`` if ``x`` a valid hexadecimal color string without the leading hash; otherwise return ``False``.
    """
    if isinstance(x, str) and re.match(COLOR_PATTERN, x):
        return True 
    else:
        return False

def check_for_required_columns(problems, table, df):
    """
    Given a list of problems (each a list of length 4), a table name (string), and the DataFrame corresponding to the table, do the following.
    Check that the DataFrame contains the colums required by GTFS, append to the problems list one error for each column missing, and return the resulting problems list.
    """
    r = cs.GTFS_REF
    req_columns = r.loc[(r['table'] == table) & r['column_required'],
      'column'].values
    for col in req_columns:
        if col not in df.columns:
            problems.append(['error', 'Missing column {!s}'.format(col), table, []])
        
    return problems 

def check_for_invalid_columns(problems, table, df):
    """
    Given a list of problems (each a list of length 4), a table name (string), and the DataFrame corresponding to the table, do the following.
    Check whether the DataFrame contains extra columns not in the GTFS, append to the problems list one warning for each extra column, and return the resulting problems list.
    """
    r = cs.GTFS_REF
    valid_columns = r.loc[r['table'] == table, 'column'].values
    for col in df.columns:
        if col not in valid_columns:
            problems.append(['warning', 
              'Unrecognized column {!s}'.format(col),
              table, []])
        
    return problems 

def check_table(problems, table, df, condition, message, type_='error'):
    """
    Given a list of problems (each a list of length 4), a table (string), the DataFrame corresponding to the table, a boolean condition on the DataFrame, a message (string), and a problem type ('error' or 'warning'), do the following.
    Record the indices where the DataFrame statisfies the condition, then if the list of indices is nonempty, append to the problems the item ``[type_, message, table, indices]``.
    If the list of indices is empty, then return the original list of problems.
    """
    indices = df.loc[condition].index.tolist()
    if indices:
        problems.append([type_, message, table, indices])

    return problems

def check_column(problems, table, df, column, column_required, checker, 
  type_='error'):
    """
    Given a list of problems (each a list of length 4), a table name (string), the DataFrame corresponding to the table, a column name (string), a boolean indicating whether the table is required, a checker (boolean-valued unary function), and a probelm type ('error' or 'warning'), do the following.
    Apply the checker to the column entries and record the indices of hits.
    If the list of indices is nonempty, append to the problems the item ``[type_, problem, table, indices]``.
    Otherwise, return the original list of problems.

    If the column is not required, then NaN entries will be ignored in the checking.
    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].map(checker)  
    problems = check_table(problems, table, f, cond, 
      'Invalid {!s}; maybe has extra space characters'.format(column), type_)

    return problems

def check_column_id(problems, table, df, column, column_required=True):
    """
    A modified version of :func:`check_column`.
    The given column must have unduplicated IDs that are valid strings. 

    If the column is not required, then NaN entries will be ignored in the checking.
    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].map(valid_str)
    problems = check_table(problems, table, f, cond, 
      'Invalid {!s}; maybe has extra space characters'.format(column))

    cond = f[column].duplicated()
    problems = check_table(problems, table, f, cond, 
      'Repeated {!s}'.format(column))

    return problems 

def check_column_linked_id(problems, table, df, column, column_required, 
  target_df, target_column=None):
    """
    A modified version of :func:`check_column`.
    The given column must contain IDs that are valid strings and are in the target DataFrame under the target column name, the latter of which defaults to the given column name. 

    If the column is not required, then NaN entries will be ignored in the checking.
    """
    if target_column is None:
        target_column = column 

    f = df.copy()

    if target_df is None:
        g = pd.DataFrame()
        g[target_column] = np.nan
    else:
        g = target_df.copy()

    if target_column not in g.columns:
        g[target_column] = np.nan

    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])
        g = g.dropna(subset=[target_column])


    cond = ~f[column].isin(g[target_column])    
    problems = check_table(problems, table, f, cond, 
      'Undefined {!s}'.format(column))

    return problems 

def format_problems(problems, as_df):
    """
    Given a possibly empty list of problems of the form described in :func:`check_table`, return a DataFrame with the problems as rows and the columns ['type', 'message', 'table', 'rows'], if ``as_df``.
    If not ``as_df``, then return the given list of problems.
    """
    if as_df:
        problems = pd.DataFrame(problems, columns=['type', 'message', 
          'table', 'rows']).sort_values(['type', 'table'])
    return problems

def check_agency(feed, as_df=False, include_warnings=False):
    """
    Check that ``feed.agency`` follows the GTFS.
    Return a list of problems of the form described in :func:`check_table`; the list will be empty if no problems are found.
    """
    table = 'agency'
    problems = []

    # Preliminary checks
    if feed.agency is None:
        problems.append(['error', 'Missing table', table, []])
    else:
        f = feed.agency.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column_id(problems, table, f, 'agency_id', False)

    # Check agency_name
    problems = check_column(problems, table, f, 'agency_name', True, valid_str)

    # Check agency_url
    problems = check_column(problems, table, f, 'agency_url', True, valid_url)

    # Check agency_timezone
    problems = check_column(problems, table, f, 'agency_timezone', True,
      valid_timezone)

    # Check agency_fare_url
    problems = check_column(problems, table, f, 'agency_fare_url', False, valid_url)

    # Check agency_lang
    problems = check_column(problems, table, f, 'agency_lang', False, valid_lang)
    
    # Check agency_phone
    problems = check_column(problems, table, f, 'agency_phone', False, valid_str)

    # Check agency_email
    problems = check_column(problems, table, f, 'agency_email', False, valid_email)

    return format_problems(problems, as_df)

def check_calendar(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar``.
    """    
    table = 'calendar'
    problems = []

    # Preliminary checks
    if feed.calendar is None:
        return problems 

    f = feed.calendar.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column_id(problems, table, f, 'service_id')

    # Check weekday columns
    v = lambda x: x in range(2)
    for col in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
      'saturday', 'sunday']:
        problems = check_column(problems, table, f, col, True, v)

    # Check start_date and end_date
    for col in ['start_date', 'end_date']:
        problems = check_column(problems, table, f, col, True, valid_date)

    if include_warnings:
        # Check if feed has expired
        d = f['end_date'].max()
        if feed.calendar_dates is not None and not feed.calendar_dates.empty:
            table += '/calendar_dates'
            d = max(d, feed.calendar_dates['date'].max())
        if d < dt.datetime.today().strftime(DATE_FORMAT):
            problems.append(['warning', 'Feed expired', table, []])

    return format_problems(problems, as_df) 

def check_calendar_dates(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    table = 'calendar_dates'
    problems = []

    # Preliminary checks
    if feed.calendar_dates is None:
        return problems 

    f = feed.calendar_dates.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check service_id
    problems = check_column(problems, table, f, 'service_id', True, valid_str)

    # Check date
    problems = check_column(problems, table, f, 'date', True, valid_date)

    # No duplicate (service_id, date) pairs allowed
    cond = f[['service_id', 'date']].duplicated()
    problems = check_table(problems, table, f, cond, 
      'Repeated pair (service_id, date)')

    # Check exception_type
    v = lambda x: x in [1, 2]
    problems = check_column(problems, table, f, 'exception_type', True, v)

    return format_problems(problems, as_df) 

def check_fare_attributes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    table = 'fare_attributes'
    problems = []

    # Preliminary checks
    if feed.fare_attributes is None:
        return problems 

    f = feed.fare_attributes.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check fare_id
    problems  = check_column_id(problems, table, f, 'fare_id')

    # Check currency_type
    problems = check_column(problems, table, f, 'currency_type', True, valid_currency)

    # Check payment_method
    v = lambda x: x in range(2)
    problems = check_column(problems, table, f, 'payment_method', True, v)

    # Check transfers
    v = lambda x: pd.isnull(x) or x in range(3)
    problems = check_column(problems, table, f, 'transfers', True, v)

    # Check transfer_duration
    v = lambda x: x >= 0
    problems = check_column(problems, table, f, 'transfer_duration', False, v)

    return format_problems(problems, as_df)    

def check_fare_rules(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    table = 'fare_rules'
    problems = []

    # Preliminary checks
    if feed.fare_rules is None:
        return problems 

    f = feed.fare_rules.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check fare_id
    problems  = check_column_linked_id(problems, table, f, 'fare_id', True, 
      feed.fare_attributes)

    # Check route_id
    problems  = check_column_linked_id(problems, table, f, 'route_id', False, 
      feed.routes)

    # Check origin_id, destination_id, contains_id
    for col in ['origin_id', 'destination_id', 'contains_id']:
        problems  = check_column_linked_id(problems, table, f, col, False, 
          feed.stops, 'zone_id')
    
    return format_problems(problems, as_df)    

def check_feed_info(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.feed_info``.
    """    
    table = 'feed_info'
    problems = []

    # Preliminary checks
    if feed.feed_info is None:
        return problems 

    f = feed.feed_info.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check feed_publisher_name
    problems = check_column(problems, table, f, 'feed_publisher_name', True,
      valid_str)

    # Check feed_publisher_url
    problems = check_column(problems, table, f, 'feed_publisher_url', True,
      valid_url)

    # Check feed_lang
    problems = check_column(problems, table, f, 'feed_lang', True,
      valid_lang)

    # Check feed_start_date and feed_end_date
    cols = ['feed_start_date', 'feed_end_date']
    for col in cols:
        problems = check_column(problems, table, f, col, False, valid_date)

    if set(cols) <= set(f.columns):
        d1, d2 = f[['feed_start_date', 'feed_end_date']].ix[0].values
        if pd.notnull(d1) and pd.notnull(d2) and d1 > d1:
            problems.append(['error', 'feed_start_date later than feed_end_date', 
              table, [0]])

    # Check feed_version
    problems = check_column(problems, table, f, 'feed_version', False, valid_str)

    return format_problems(problems, as_df)

def check_frequencies(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.frequencies``.
    """    
    table = 'frequencies'
    problems = []

    # Preliminary checks
    if feed.frequencies is None:
        return problems 

    f = feed.frequencies.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_linked_id(problems, table, f, 'trip_id', True,
      feed.trips)

    # Check start_time and end_time
    time_cols = ['start_time', 'end_time'] 
    for col in time_cols:
        problems = check_column(problems, table, f, col, True, valid_time)

    for col in time_cols:
        f[col] = f[col].map(hp.timestr_to_seconds)

    # start_time should be earlier than end_time
    cond = f['start_time'] >= f['end_time']
    problems = check_table(problems, table, f, cond, 'start_time not earlier than end_time')

    # Headway periods should not overlap
    f = f.sort_values(['trip_id', 'start_time'])
    for __, group in f.groupby('trip_id'):
        a = group['start_time'].values
        b = group['end_time'].values  
        indices = np.flatnonzero(a[1:] < b[:-1]).tolist()
        if indices:
            problems.append(['error', 'Headway periods for the same trip overlap',
              table, indices])

    # Check headway_secs
    v = lambda x: x >= 0
    problems = check_column(problems, table, f, 'headway_secs', True, v)

    # Check exact_times
    v = lambda x: x in range(2)
    error = check_column(problems, table, f, 'exact_times', False, v)

    return format_problems(problems, as_df)

def check_routes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.routes``.
    """    
    table = 'routes'
    problems = []

    # Preliminary checks
    if feed.routes is None:
        problems.append(['error', 'Missing table', table, []])
    else:
        f = feed.routes.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check route_id
    problems = check_column_id(problems, table, f, 'route_id')

    # Check agency_id
    if 'agency_id' in f:
        if 'agency_id' not in feed.agency.columns:
            problems.append(['error', 
              'agency_id column present in routes but not in agency', table, []])
        else:
            g = f.dropna(subset=['agency_id'])
            cond = ~g['agency_id'].isin(feed.agency['agency_id'])
            problems = check_table(problems, table, g, cond, 'Undefined agency_id')

    # Check route_short_name and route_long_name
    for column in ['route_short_name', 'route_long_name']:
        problems = check_column(problems, table, f, column, False, valid_str)

    cond = ~(f['route_short_name'].notnull() | f['route_long_name'].notnull())
    problems = check_table(problems, table, f, cond, 
      'route_short_name and route_long_name both empty')

    # Check route_type
    v = lambda x: x in range(8)
    problems = check_column(problems, table, f, 'route_type', True, v)

    # Check route_url
    problems = check_column(problems, table, f, 'route_url', False, valid_url)

    # Check route_color and route_text_color
    for col in ['route_color', 'route_text_color']:
        problems = check_column(problems, table, f, col, False, valid_color)
    
    if include_warnings:
        # Check for duplicated (route_short_name, route_long_name) pairs
        cond = f[['route_short_name', 'route_long_name']].duplicated()
        problems = check_table(problems, table, f, cond, 
          'Repeated pair (route_short_name, route_long_name)', 'warning')

        # Check for routes without trips
        s = feed.trips['route_id']
        cond = ~f['route_id'].isin(s)
        problems = check_table(problems, table, f, cond, 'Route has no trips', 'warning')

    return format_problems(problems, as_df)

def check_shapes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.shapes``.
    """    
    table = 'shapes'
    problems = []

    # Preliminary checks
    if feed.shapes is None:
        return problems 

    f = feed.shapes.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check shape_id
    problems = check_column(problems, table, f, 'shape_id', True, valid_str)

    # Check shape_pt_lon and shape_pt_lat
    for column, bound in [('shape_pt_lon', 180), ('shape_pt_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        problems = check_table(problems, table, f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check for duplicated (shape_id, shape_pt_sequence) pairs
    cond = f[['shape_id', 'shape_pt_sequence']].duplicated()
    problems = check_table(problems, table, f, cond, 
      'Repeated pair (shape_id, shape_pt_sequence)')

    # Check if shape_dist_traveled does decreases on a trip
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled'])
        indices = []
        prev_sid = None
        prev_dist = -1
        cols = ['shape_id', 'shape_dist_traveled']
        for i, sid, dist in g[cols].itertuples():
            if sid == prev_sid and dist < prev_dist:
                indices.append(i)

            prev_sid = sid 
            prev_dist = dist 
            
        if indices:
            problems.append(['error', 'shape_dist_traveled decreases on a trip',
              table, indices])

    return format_problems(problems, as_df)

def check_stops(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.stops``.
    """    
    table = 'stops'
    problems = []

    # Preliminary checks
    if feed.stops is None:
        problems.append(['error', 'Missing table', table, []])
    else:
        f = feed.stops.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check stop_id
    problems = check_column_id(problems, table, f, 'stop_id')

    # Check stop_code, stop_desc, zone_id, parent_station
    for column in ['stop_code', 'stop_desc', 'zone_id', 'parent_station']:
        problems = check_column(problems, table, f, column, False, valid_str)

    # Check stop_name
    problems = check_column(problems, table, f, 'stop_name', True, valid_str)

    # Check stop_lon and stop_lat
    for column, bound in [('stop_lon', 180), ('stop_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        problems = check_table(problems, table, f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check stop_url
    problems = check_column(problems, table, f, 'stop_url', False, valid_url)

    # Check location_type
    v = lambda x: x in range(2)
    problems = check_column(problems, table, f, 'location_type', False, v)

    # Check stop_timezone
    problems = check_column(problems, table, f, 'stop_timezone', False, valid_timezone)

    # Check wheelchair_boarding
    v = lambda x: x in range(3)
    problems = check_column(problems, table, f, 'wheelchair_boarding', False, v)

    # Check further location_type and parent_station   
    if 'parent_station' in f.columns:
        if 'location_type' not in f.columns:
            problems.append(['error', 
              'parent_station column present but location_type column missing', 
              table, []])
        else:
            # Stations must have location type 1
            station_ids = f.loc[f['parent_station'].notnull(), 'parent_station']
            cond = f['stop_id'].isin(station_ids) & (f['location_type'] != 1)
            problems = check_table(problems, table, f, cond, 
              'A station must have location_type 1')

            # Stations must not lie in stations
            cond = (f['location_type'] == 1) & f['parent_station'].notnull() 
            problems = check_table(problems, table, f, cond, 
              'A station must not lie in another station')

    if include_warnings:
        # Check for stops without trips
        s = feed.stop_times['stop_id']
        cond = ~feed.stops['stop_id'].isin(s)
        problems = check_table(problems, table, f, cond, 'Stop has no stop times', 'warning')

    return format_problems(problems, as_df) 

def check_stop_times(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.stop_times``.
    """    
    table = 'stop_times'
    problems = []

    # Preliminary checks
    if feed.stop_times is None:
        problems.append(['error', 'Missing table', table, []])
    else:
        f = feed.stop_times.copy().sort_values(['trip_id', 'stop_sequence'])
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_linked_id(problems, table, f, 'trip_id', True, 
      feed.trips)

    # Check arrival_time and departure_time
    v = lambda x: pd.isnull(x) or valid_time(x)
    for col in ['arrival_time', 'departure_time']:
        problems = check_column(problems, table, f, col, True, v)

    # Check that arrival and departure times exist for the first and last stop of each trip and for each timepoint.
    # For feeds with many trips, iterating through the stop time rows is faster than uisg groupby.
    if 'timepoint' not in f.columns:
        f['timepoint'] = np.nan  # This will not mess up later timepoint check

    indices = []
    prev_tid = None
    prev_atime = 1
    prev_dtime = 1    
    for i, tid, atime, dtime, tp in f[['trip_id', 'arrival_time', 
      'departure_time', 'timepoint']].itertuples():
        if tid != prev_tid:
            # Check last stop of previous trip
            if pd.isnull(prev_atime) or pd.isnull(prev_dtime):
                indices.append(i - 1)
            # Check first stop of current trip
            if pd.isnull(atime) or pd.isnull(dtime):
                indices.append(i)
        elif tp == 1 and (pd.isnull(atime) or pd.isnull(dtime)):
            # Failure at timepoint
            indices.append(i)
            
        prev_tid = tid
        prev_atime = atime
        prev_dtime = dtime

    if indices:
        problems.append(['error', 'First/last/time point arrival/departure time missing',
          table, indices])

    # Check stop_id
    problems = check_column_linked_id(problems, table, f, 'stop_id', True, 
      feed.stops)

    # Check for duplicated (trip_id, stop_sequence) pairs
    cond = f[['trip_id', 'stop_sequence']].dropna().duplicated()
    problems = check_table(problems, table, f, cond, 
      'Repeated pair (trip_id, stop_sequence)')

    # Check stop_headsign
    problems = check_column(problems, table, f, 'stop_headsign', False, valid_str)

    # Check pickup_type and drop_off_type
    for col in ['pickup_type', 'drop_off_type']:
        v = lambda x: x in range(4)
        problems = check_column(problems, table, f, col, False, v)

    # Check if shape_dist_traveled decreases on a trip
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled'])
        indices = []
        prev_tid = None
        prev_dist = -1
        for i, tid, dist in g[['trip_id', 'shape_dist_traveled']].itertuples():
            if tid == prev_tid and dist < prev_dist:
                indices.append(i)

            prev_tid = tid 
            prev_dist = dist 

        if indices:
            problems.append(['error', 'shape_dist_traveled decreases on a trip',
              table, indices])

    # Check timepoint
    v = lambda x: x in range(2)
    problems = check_column(problems, table, f, 'timepoint', False, v)

    if include_warnings:
        # Check for duplicated (trip_id, departure_time) pairs
        cond = f[['trip_id', 'departure_time']].duplicated()
        problems = check_table(problems, table, f, cond, 
          'Repeated pair (trip_id, departure_time)', 'warning')

    return format_problems(problems, as_df) 

def check_transfers(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.transfers``.
    """    
    table = 'transfers'
    problems = []

    # Preliminary checks
    if feed.transfers is None:
        return problems 

    f = feed.transfers.copy()
    problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)
    
    # Check from_stop_id and to_stop_id
    for col in ['from_stop_id', 'to_stop_id']:
        problems = check_column_linked_id(problems, table, f, col, True,
          feed.stops, 'stop_id')

    # Check transfer_type
    v = lambda x: pd.isnull(x) or x in range(5)
    problems = check_column(problems, table, f, 'transfer_type', False, v)

    # Check min_transfer_time
    v = lambda x: x >= 0
    problems = check_column(problems, table, f, 'min_transfer_time', False, v)

    return format_problems(problems, as_df)

def check_trips(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.trips``.
    """    
    table = 'trips'
    problems = []

    # Preliminary checks
    if feed.trips is None:
        problems.append(['error', 'Missing table', table, []])
    else:
        f = feed.trips.copy()
        problems = check_for_required_columns(problems, table, f)
    if problems:
        return format_problems(problems, as_df)

    if include_warnings:
        problems = check_for_invalid_columns(problems, table, f)

    # Check trip_id
    problems = check_column_id(problems, table, f, 'trip_id')

    # Check route_id
    problems = check_column_linked_id(problems, table, f, 'route_id', True, 
      feed.routes)
    
    # Check service_id 
    g = pd.DataFrame()
    if feed.calendar is not None:
        g = pd.concat([g, feed.calendar])
    if feed.calendar_dates is not None:
        g = pd.concat([g, feed.calendar_dates])
    problems = check_column_linked_id(problems, table, f, 'service_id', True,
      g)

    # Check direction_id
    v = lambda x: x in range(2)
    problems = check_column(problems, table, f, 'direction_id', False, v)

    # Check block_id
    if 'block_id' in f.columns:
        v = lambda x: pd.isnull(x) or valid_str(x)
        cond = ~f['block_id'].map(v)
        problems = check_table(problems, table, f, cond, 'Blank block_id')

        g = f.dropna(subset=['block_id'])
        cond = ~g['block_id'].duplicated(keep=False)
        problems = check_table(problems, table, g, cond, 'Unrepeated block_id')

    # Check shape_id
    problems = check_column_linked_id(problems, table, f, 'shape_id', False, feed.shapes)

    # Check wheelchair_accessible and bikes_allowed
    v = lambda x: x in range(3)    
    for column in ['wheelchair_accessible', 'bikes_allowed']:
        problems = check_column(problems, table, f, column, False, v)

    # Check for trips with no stop times
    if include_warnings:
        s = feed.stop_times['trip_id']
        cond = ~f['trip_id'].isin(s)
        problems = check_table(problems, table, f, cond, 'Trip has no stop times', 'warning')

    return format_problems(problems, as_df)

def validate(feed, as_df=True, include_warnings=True):
    """
    Check whether the given feed satisfies the GTFS by running the functions all the table-checking functions: :func:`check_agency`, :func:`check_calendar`, etc.
    Return the problems found as a possibly empty list of items [problem type, message, table, rows].
    If ``as_df``, then format the error list as a DataFrame with the columns

    - ``'type'``: 'error' or 'warning'; 'error' means the GTFS is violated; 'warning' means there is a problem but it's not a GTFS violation
    - ``'message'``: description of the problem
    - ``'table'``: table in which problem occurs, e.g. 'routes'
    - ``'rows'``: rows of the table's DataFrame where problem occurs

    Return early if the feed is missing required tables or required columns.

    Include warning problems only if ``include_warning``.

    NOTES:
        - This function interprets the GTFS liberally, classifying problems as warnings rather than errors where the GTFS is unclear. For example if a trip_id listed in the trips table is not listed in the stop times table (a trip with no stop times), then that's a warning and not an error. 
        - Timing benchmark: on my 2.80 GHz processor machine with 16 GB of memory, this function can check the 31 MB Southeast Queensland feed at http://transitfeeds.com/p/translink/21/20170310 in 22 seconds (including warnings).
    """
    problems = []

    # Check for invalid columns and check the required tables
    checkers = [
      'check_agency',
      'check_calendar',
      'check_calendar_dates',
      'check_fare_attributes',
      'check_fare_rules',
      'check_feed_info',
      'check_frequencies',
      'check_routes',
      'check_shapes',
      'check_stops',
      'check_stop_times',
      'check_transfers',
      'check_trips',
      ]
    for checker in checkers:
        problems.extend(globals()[checker](feed, include_warnings=include_warnings))

    # Check calendar/calendar_dates combo
    if feed.calendar is None and feed.calendar_dates is None:
        problems.append(['error', 'Missing both tables',
          'calendar & calendar_dates', []])

    return format_problems(problems, as_df)