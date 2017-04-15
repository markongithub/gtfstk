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
    """
    Return ``True`` if ``x`` is a valid email address; otherwise return ``False``.
    """
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

def check_table(msgs, table, df, condition, msg, msg_type='error'):
    """
    Given a list of messages (each a list of length 4), a table (string), the DataFrame corresponding to the table, a boolean condition on the DataFrame, a message (string), and a message type ('error' or 'warning'), do the following.
    Record the indices where the DataFrame statisfies the condition, then if the list of indices is nonempty, append to the messages the item ``[msg_type, msg, table, indices]``.
    If the list of indices is empty, then return the original list of messages.
    """
    indices = df.loc[condition].index.tolist()
    if indices:
        msgs.append([msg_type, msg, table, indices])

    return msgs

def check_column(msgs, table, df, column, column_required, checker, 
  msg_type='error'):
    """
    Given a list of messages (each a list of length 4), a table name (string), the DataFrame corresponding to the table, a column name (string), a boolean indicating whether the table is required, a checker (boolean-valued unary function), and a message type ('error' or 'warning'), do the following.
    Apply the checker to the column entries and record the indices of hits.
    If the list of indices is nonempty, append to the messages the item ``[msg_type, msg, table, indices]``.
    Otherwise, return the original list of messages.

    If the column is not required, then NaN entries will be ignored in the checking.
    """
    f = df.copy()
    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        f = f.dropna(subset=[column])

    cond = ~f[column].map(checker)  
    msgs = check_table(msgs, table, f, cond, 
      '{!s} malformed'.format(column), msg_type)

    return msgs

def check_column_id(msgs, table, df, column, column_required=True):
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
    msgs = check_table(msgs, table, f, cond, 
      '{!s} malformed'.format(column))

    cond = f[column].duplicated()
    msgs = check_table(msgs, table, f, cond, 
      '{!s} duplicated'.format(column))

    return msgs 

def check_column_linked_id(msgs, table, df, column, column_required, 
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
    else:
        g = target_df.copy()

    if not column_required:
        if column not in f.columns:
            f[column] = np.nan
        if target_column not in g.columns:
            g[target_column] = np.nan
        f = f.dropna(subset=[column])
        g = g.dropna(subset=[target_column])

    cond = ~f[column].isin(g[target_column])
    msgs = check_table(msgs, table, f, cond, 
      '{!s} undefined'.format(column))

    return msgs 

def format_msgs(msgs, as_df):
    """
    Given a possibly empty list of messages of the form described in :func:`check_table`, return a DataFrame with the messages as rows and the columns ['message_type', 'message', 'table', 'rows'], if ``as_df``.
    If not ``as_df``, then return the given list of messages.
    """
    if as_df:
        msgs = pd.DataFrame(msgs, columns=['message_type', 'message', 'table', 
          'rows']).sort_values(['message_type', 'table'])
    return msgs

def check_for_required_tables(feed, as_df=False, include_warnings=False):
    """
    Check the given feed for all the required GTFS tables.
    Return a list of messages of the form described in :func:`check_table`; the list will be empty if no problems are found.
    """
    msgs = []
    req_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
    for table in req_tables:
        if getattr(feed, table) is None:
            msgs.append(['error', 'Missing file', table, []])
    # Calendar check is different
    if feed.calendar is None and feed.calendar_dates is None:
        msgs.append(['error', 'Both calendar and calendar_dates tables missing', []])

    return format_msgs(msgs, as_df) 

def check_for_required_columns(feed, as_df=False, include_warnings=False):
    """
    Check the given feed for all the required GTFS columns.
    Return a list of messages of the form described in :func:`check_table`; the list will be empty if no problems are found.
    """
    msgs = []
    for table, group in cs.GTFS_REF.groupby('table'):
        f = getattr(feed, table)
        if f is None:
            continue

        for column, column_required in group[['column', 'column_required']].itertuples(
          index=False):
            if column_required and column not in f.columns:
                msgs.append(['error', 'Column {!s} is missing'.format(column), 
                  table, []])

    return format_msgs(msgs, as_df) 

def check_for_invalid_columns(feed, as_df=False, include_warnings=False):
    """
    Check the given feed for columns not mentioned in the GTFS.
    Return a list of messages of the form described in :func:`check_table`; the list will be empty if no problems are found.
    """
    msgs = []
    for table, group in cs.GTFS_REF.groupby('table'):
        f = getattr(feed, table)
        if f is None:
            continue
        valid_columns = group['column'].values
        for col in f.columns:
            if col not in valid_columns:
                msgs.append(['error', '{!s} is not a valid column name'.format(col),
                  table, []])
        
    return format_msgs(msgs, as_df)

def check_agency(feed, as_df=False, include_warnings=False):
    """
    Check that ``feed.agency`` follows the GTFS.
    Return a list of messages of the form described in :func:`check_table`; the list will be empty if no problems are found.
    """
    f = feed.agency.copy()
    table = 'agency'
    msgs = []

    # Check service_id
    msgs = check_column_id(msgs, table, f, 'agency_id', False)

    # Check agency_name
    msgs = check_column(msgs, table, f, 'agency_name', True, valid_str)

    # Check agency_url
    msgs = check_column(msgs, table, f, 'agency_url', True, valid_url)

    # Check agency_timezone
    msgs = check_column(msgs, table, f, 'agency_timezone', True,
      valid_timezone)

    # Check agency_fare_url
    msgs = check_column(msgs, table, f, 'agency_fare_url', False, valid_url)

    # Check agency_lang
    msgs = check_column(msgs, table, f, 'agency_lang', False, valid_lang)
    
    # Check agency_phone
    msgs = check_column(msgs, table, f, 'agency_phone', False, valid_str)

    # Check agency_email
    msgs = check_column(msgs, table, f, 'agency_email', False, valid_email)

    return format_msgs(msgs, as_df)

def check_calendar(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar``.
    """    
    f = feed.calendar.copy()
    table = 'calendar'
    msgs = []

    # Check service_id
    msgs = check_column_id(msgs, table, f, 'service_id')

    # Check weekday columns
    v = lambda x: x in range(2)
    for col in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
      'saturday', 'sunday']:
        msgs = check_column(msgs, table, f, col, True, v)

    # Check start_date and end_date
    for col in ['start_date', 'end_date']:
        msgs = check_column(msgs, table, f, col, True, valid_date)

    if include_warnings:
        # Check if feed has expired
        d = f['end_date'].max()
        if feed.calendar_dates is not None and not feed.calendar_dates.empty:
            table += '/calendar_dates'
            d = max(d, feed.calendar_dates['date'].max())
        if d < dt.datetime.today().strftime(DATE_FORMAT):
            msgs.append(['warning', 'This feed has expired', table, []])

    return format_msgs(msgs, as_df) 

def check_calendar_dates(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.calendar_dates.copy()
    table = 'calendar_dates'
    msgs = []

    # Check service_id
    msgs = check_column(msgs, table, f, 'service_id', True, valid_str)

    # Check date
    msgs = check_column(msgs, table, f, 'date', True, valid_date)

    # No duplicate (service_id, date) pairs allowed
    cond = f[['service_id', 'date']].duplicated()
    msgs = check_table(msgs, table, f, cond, 
      '(service_id, date)-pair duplicated')

    # Check exception_type
    v = lambda x: x in [1, 2]
    msgs = check_column(msgs, table, f, 'exception_type', True, v)

    return format_msgs(msgs, as_df) 

def check_fare_attributes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.fare_attributes.copy()
    table = 'fare_attributes'
    msgs = []

    # Check fare_id
    msgs  = check_column_id(msgs, table, f, 'fare_id')

    # Check currency_type
    msgs = check_column(msgs, table, f, 'currency_type', True, valid_currency)

    # Check payment_method
    v = lambda x: x in range(2)
    msgs = check_column(msgs, table, f, 'payment_method', True, v)

    # Check transfers
    v = lambda x: pd.isnull(x) or x in range(3)
    msgs = check_column(msgs, table, f, 'transfers', True, v)

    # Check transfer_duration
    v = lambda x: x >= 0
    msgs = check_column(msgs, table, f, 'transfer_duration', False, v)

    return format_msgs(msgs, as_df)    

def check_fare_rules(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.calendar_dates``.
    """    
    f = feed.fare_rules.copy()
    table = 'fare_rules'
    msgs = []

    # Check fare_id
    msgs  = check_column_linked_id(msgs, table, f, 'fare_id', True, 
      feed.fare_attributes)

    # Check route_id
    msgs  = check_column_linked_id(msgs, table, f, 'route_id', False, 
      feed.routes)

    # Check origin_id, destination_id, contains_id
    for col in ['origin_id', 'destination_id', 'contains_id']:
        msgs  = check_column_linked_id(msgs, table, f, col, False, 
          feed.stops, 'zone_id')
    
    return format_msgs(msgs, as_df)    

def check_feed_info(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.feed_info``.
    """    
    f = feed.feed_info.copy()
    table = 'feed_info'
    msgs = []

    # Check feed_publisher_name
    msgs = check_column(msgs, table, f, 'feed_publisher_name', True,
      valid_str)

    # Check feed_publisher_url
    msgs = check_column(msgs, table, f, 'feed_publisher_url', True,
      valid_url)

    # Check feed_lang
    msgs = check_column(msgs, table, f, 'feed_lang', True,
      valid_lang)

    # Check feed_start_date and feed_end_date
    for col in ['feed_start_date', 'feed_end_date']:
        msgs = check_column(msgs, table, f, col, False, valid_date)

    d1, d2 = f[['feed_start_date', 'feed_end_date']].ix[0].values
    if pd.notnull(d1) and pd.notnull(d2) and d1 > d1:
        msgs.append(['error', 'feed_start_date later than feed_end_date', 
          table, [0]])

    # Check feed_version
    msgs = check_column(msgs, table, f, 'feed_version', False, valid_str)

    return format_msgs(msgs, as_df)

def check_frequencies(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.frequencies``.
    """    
    f = feed.frequencies.copy()
    table = 'frequencies'
    msgs = []

    # Check trip_id
    msgs = check_column_linked_id(msgs, table, f, 'trip_id', True,
      feed.trips)

    # Check start_time and end_time
    time_cols = ['start_time', 'end_time'] 
    for col in time_cols:
        msgs = check_column(msgs, table, f, col, True, valid_time)

    for col in time_cols:
        f[col] = f[col].map(hp.timestr_to_seconds)

    # start_time should be earlier than end_time
    cond = f['start_time'] >= f['end_time']
    msgs = check_table(msgs, table, f, cond, 'start_time not earlier than end_time')

    # Headway periods should not overlap
    f = f.sort_values(['trip_id', 'start_time'])
    for __, group in f.groupby('trip_id'):
        a = group['start_time'].values
        b = group['end_time'].values  
        indices = np.flatnonzero(a[1:] < b[:-1]).tolist()
        if indices:
            msgs.append(['error', 'Headway periods for the same trip overlap',
              table, indices])

    # Check headway_secs
    v = lambda x: x >= 0
    msgs = check_column(msgs, table, f, 'headway_secs', True, v)

    # Check exact_times
    v = lambda x: x in range(2)
    error = check_column(msgs, table, f, 'exact_times', False, v)

    return format_msgs(msgs, as_df)

def check_routes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.routes``.
    """    
    f = feed.routes.copy()
    table = 'routes'
    msgs = []

    # Check route_id
    msgs = check_column_id(msgs, table, f, 'route_id')

    # Check agency_id
    if 'agency_id' in f:
        if 'agency_id' not in feed.agency.columns:
            msgs.append(['error', 
              'agency_id column present in routes but not in agency', table, []])
        else:
            g = f.dropna(subset=['agency_id'])
            cond = ~g['agency_id'].isin(feed.agency['agency_id'])
            msgs = check_table(msgs, table, g, cond, 'agency_id undefined')

    # Check route_short_name and route_long_name
    for column in ['route_short_name', 'route_long_name']:
        msgs = check_column(msgs, table, f, column, False, valid_str)

    cond = ~(f['route_short_name'].notnull() | f['route_long_name'].notnull())
    msgs = check_table(msgs, table, f, cond, 
      'route_short_name and route_long_name both empty')

    # Check route_type
    v = lambda x: x in range(8)
    msgs = check_column(msgs, table, f, 'route_type', True, v)

    # Check route_url
    msgs = check_column(msgs, table, f, 'route_url', False, valid_url)

    # Check route_color and route_text_color
    for col in ['route_color', 'route_text_color']:
        msgs = check_column(msgs, table, f, col, False, valid_color)
    
    if include_warnings:
        # Check for duplicated (route_short_name, route_long_name) pairs
        cond = f[['route_short_name', 'route_long_name']].duplicated()
        msgs = check_table(msgs, table, f, cond, 
          'Duplicate (route_short_name, route_long_name) pair', 'warning')

        # Check for routes without trips
        s = feed.trips['route_id']
        cond = ~f['route_id'].isin(s)
        msgs = check_table(msgs, table, f, cond, 'Route has no trips', 'warning')

    return format_msgs(msgs, as_df)

def check_shapes(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.shapes``.
    """    
    f = feed.shapes.copy()
    table = 'shapes'
    msgs = []

    # Check shape_id
    msgs = check_column(msgs, table, f, 'shape_id', True, valid_str)

    # Check shape_pt_lon and shape_pt_lat
    for column, bound in [('shape_pt_lon', 180), ('shape_pt_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        msgs = check_table(msgs, table, f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check for duplicated (shape_id, shape_pt_sequence) pairs
    cond = f[['shape_id', 'shape_pt_sequence']].duplicated()
    msgs = check_table(msgs, table, f, cond, 
      'Duplicate (shape_id, shape_pt_sequence) pair')

    # Check shape_dist_traveled
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled']).sort_values(
          ['shape_id', 'shape_pt_sequence'])
        for shape_id, group in g.groupby('shape_id', sort=False):
            a = group['shape_dist_traveled'].values  
            indices = np.flatnonzero(a[1:] < a[:-1]) # relative indices
            indices = group.index[indices].tolist() # absolute indices
            if indices:
                msgs.append(['error', 
                  'shape_dist_traveled decreases for shape_id {!s}'.format(
                  shape_id), table, indices])

    return format_msgs(msgs, as_df)

def check_stops(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.stops``.
    """    
    f = feed.stops.copy()
    table = 'stops'
    msgs = []

    # Check stop_id
    msgs = check_column_id(msgs, table, f, 'stop_id')

    # Check stop_code, stop_desc, zone_id, parent_station
    for column in ['stop_code', 'stop_desc', 'zone_id', 'parent_station']:
        msgs = check_column(msgs, table, f, column, False, valid_str)

    # Check stop_name
    msgs = check_column(msgs, table, f, 'stop_name', True, valid_str)

    # Check stop_lon and stop_lat
    for column, bound in [('stop_lon', 180), ('stop_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[column].map(v)
        msgs = check_table(msgs, table, f, cond, 
          '{!s} out of bounds {!s}'.format(column, [-bound, bound]))

    # Check stop_url
    msgs = check_column(msgs, table, f, 'stop_url', False, valid_url)

    # Check location_type
    v = lambda x: x in range(2)
    msgs = check_column(msgs, table, f, 'location_type', False, v)

    # Check stop_timezone
    msgs = check_column(msgs, table, f, 'stop_timezone', False, valid_timezone)

    # Check wheelchair_boarding
    v = lambda x: x in range(3)
    msgs = check_column(msgs, table, f, 'wheelchair_boarding', False, v)

    # Check further location_type and parent_station   
    if 'parent_station' in f.columns:
        if 'location_type' not in f.columns:
            msgs.append(['error', 
              'parent_station column present but location_type column missing', 
              table, []])
        else:
            # Parent stations must have location type 1
            station_ids = f.loc[f['parent_station'].notnull(), 'parent_station']
            cond = f['stop_id'].isin(station_ids) & (f['location_type'] != 1)
            msgs = check_table(msgs, table, f, cond, 
              'A parent station must have location_type equal to 1')

            # Parent stations should not have parent stations
            cond = f['parent_station'].notnull() & (f['location_type'] == 1) 
            msgs = check_table(msgs, table, f, cond, 
              'location_type must equal 1')

    if include_warnings:
        # Check for stops without trips
        s = feed.stop_times['stop_id']
        cond = ~feed.stops['stop_id'].isin(s)
        msgs = check_table(msgs, table, f, cond, 'Stop has no stop times', 'warning')

    return format_msgs(msgs, as_df) 

def check_stop_times(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.stop_times``.
    """    
    f = feed.stop_times.copy().sort_values(['trip_id', 'stop_sequence'])
    table = 'stop_times'
    msgs = []

    # Check trip_id
    msgs = check_column_linked_id(msgs, table, f, 'trip_id', True, 
      feed.trips)

    # Check arrival_time and departure_time
    v = lambda x: pd.isnull(x) or valid_time(x)
    for col in ['arrival_time', 'departure_time']:
        msgs = check_column(msgs, table, f, col, True, v)

    # Check stop_id
    msgs = check_column_linked_id(msgs, table, f, 'stop_id', True, 
      feed.stops)

    # Check for duplicated (trip_id, stop_sequence) pairs
    cond = f[['trip_id', 'stop_sequence']].dropna().duplicated()
    msgs = check_table(msgs, table, f, cond, 
      'Duplicate (trip_id, stop_sequence) pair')

    # Check stop_headsign
    msgs = check_column(msgs, table, f, 'stop_headsign', False, valid_str)

    # Check pickup_type and drop_off_type
    for col in ['pickup_type', 'drop_off_type']:
        v = lambda x: x in range(4)
        msgs = check_column(msgs, table, f, col, False, v)

    # Check shape_dist_traveled
    if 'shape_dist_traveled' in f.columns:
        g = f.dropna(subset=['shape_dist_traveled'])
        for trip_id, group in g.groupby('trip_id', sort=False):
            a = group['shape_dist_traveled'].values  
            indices = np.flatnonzero(a[1:] < a[:-1]) # relative indices
            indices = group.index[indices].tolist() # absolute indices
            if indices:
                msgs.append(['error', 
                  'shape_dist_traveled decreases for trip_id {!s}'.format(
                  trip_id), table, indices])

    # Check timepoint
    v = lambda x: x in range(2)
    msgs = check_column(msgs, table, f, 'timepoint', False, v)

    # Check that arrival and departure times exists for the first and last stop of each trip and for each timepoint



    if include_warnings:
        # Check for duplicated (trip_id, departure_time) pairs
        cond = f[['trip_id', 'departure_time']].duplicated()
        msgs = check_table(msgs, table, f, cond, 
          'Duplicate (trip_id, departure_time) pair', 'warning')

    return format_msgs(msgs, as_df) 

def check_transfers(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.transfers``.
    """    
    f = feed.transfers.copy()
    table = 'transfers'
    msgs = []
    
    # Check from_stop_id and to_stop_id
    for col in ['from_stop_id', 'to_stop_id']:
        msgs = check_column_linked_id(msgs, table, f, col, True,
          feed.stops, 'stop_id')

    # Check transfer_type
    v = lambda x: pd.isnull(x) or x in range(5)
    msgs = check_column(msgs, table, f, 'transfer_type', False, v)

    # Check min_transfer_time
    v = lambda x: x >= 0
    msgs = check_column(msgs, table, f, 'min_transfer_time', False, v)

    return format_msgs(msgs, as_df)

def check_trips(feed, as_df=False, include_warnings=False):
    """
    Analog of :func:`check_agency` for ``feed.trips``.
    """    
    f = feed.trips.copy()
    table = 'trips'
    msgs = []

    # Check trip_id
    msgs = check_column_id(msgs, table, f, 'trip_id')

    # Check route_id
    msgs = check_column_linked_id(msgs, table, f, 'route_id', True, 
      feed.routes)
    
    # Check service_id 
    sids = []
    if feed.calendar is not None:
        sids.extend(feed.calendar['service_id'].unique().tolist())
    if feed.calendar_dates is not None:
        sids.extend(feed.calendar_dates['service_id'].unique().tolist())
    # Assume sids is nonempty now
    cond = ~f['service_id'].isin(sids)
    msgs = check_table(msgs, table, f, cond, 'service_id undefined')

    # Check direction_id
    v = lambda x: x in range(2)
    msgs = check_column(msgs, table, f, 'direction_id', False, v)

    # Check block_id
    if 'block_id' in f.columns:
        v = lambda x: pd.isnull(x) or valid_str(x)
        cond = ~f['block_id'].map(v)
        msgs = check_table(msgs, table, f, cond, 'block_id blank')

        g = f.dropna(subset=['block_id'])
        cond = ~g['block_id'].duplicated(keep=False)
        msgs = check_table(msgs, table, g, cond, 'block_id unduplicated')

    # Check shape_id
    msgs = check_column_linked_id(msgs, table, f, 'shape_id', False, feed.shapes)

    # Check wheelchair_accessible and bikes_allowed
    v = lambda x: x in range(3)    
    for column in ['wheelchair_accessible', 'bikes_allowed']:
        msgs = check_column(msgs, table, f, column, False, v)

    # Check for trips with no stop times
    if include_warnings:
        s = feed.stop_times['trip_id']
        cond = ~f['trip_id'].isin(s)
        msgs = check_table(msgs, table, f, cond, 'Trip has no stop times', 'warning')

    return format_msgs(msgs, as_df)

def validate(feed, as_df=True, include_warnings=True):
    """
    Check whether the given feed follows the GTFS by running the functions

    #. :func:`check_for_required_tables`
    #. :func:`check_for_required_columns`
    #: :func:`check_for_invalid_columns`
    #: :func:`check_agency`
    #: :func:`check_routes`
    #: :func:`check_stops`
    #: :func:`check_stop_times`
    #: :func:`check_trips`
    #: :func:`check_calendar`
    #: :func:`check_calendar_dates`
    #: :func:`check_fare_attributes`
    #: :func:`check_fare_rules`
    #: :func:`check_feed_info`
    #: :func:`check_frequencies`
    #: :func:`check_transfers`
    #: :func:`check_shapes`

    in that order.
    Return the problems found as a possibly empty list of tuples (message type, message, table, rows).
    If ``as_df``, then format the error list as a DataFrame with the columns

    - ``'message_type'``: 'error' or 'warning'; 'error' means the GTFS is violated; 'warning' means there is a problem but it's not a GTFS violation
    - ``'message'``: description of the problem
    - ``'table'``: table in which problem occurs, e.g. 'routes'
    - ``'rows'``: rows of the table's DataFrame where problem occurs

    Return early if the feed is missing required tables or required columns.

    Include warning messages only if ``include_warning``.

    NOTES:
        - This function interprets the GTFS liberally, classifying problems as warnings rather than errors where the GTFS is unclear. For example if a trip_id listed in the trips table is not listed in the stop times table (a trip with no stop times), then that's a warning and not an error. 
        - Timing benchmark: on my 2.80 GHz processor machine with 16 GB of memory, this function can check the 31 MB Southeast Queensland feed at http://transitfeeds.com/p/translink/21/20170310 in 17 seconds (including warnings).
    """
    msgs = []

    # Check for required tables and for required columns, and halt if msgs
    checkers = [
      'check_for_required_tables',
      'check_for_required_columns',
      ]
    for checker in checkers:
        msgs.extend(globals()[checker](feed, include_warnings=include_warnings))
        if msgs:
            return format_msgs(msgs, as_df)

    # Check for invalid columns and check the required tables
    checkers = [
      'check_for_invalid_columns',
      'check_agency',
      'check_routes',
      'check_stops',
      'check_stop_times',
      'check_trips',
      ]
    for checker in checkers:
        msgs.extend(globals()[checker](feed, include_warnings=include_warnings))

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
            msgs.extend(globals()[checker](feed, include_warnings=include_warnings))

    return format_msgs(msgs, as_df)