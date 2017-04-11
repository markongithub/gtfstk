"""
A GTFS validator.
A work in progress.
"""
import re 

import pandas as pd

from . import constants as cs


COLOR_PATTERN = re.compile(r'(?:[0-9a-fA-F]{2}){3}$')
URL_PATTERN = re.compile(
        r'^(?:http)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

def valid_int(x, valid_range):
    """
    Return ``True`` if ``x in valid_range``; otherwise return ``False``.
    """
    if x in valid_range:
        return True 
    else:
        return False

def valid_string(x):
    """
    Return ``True`` if ``x`` is a non-blank string; otherwise return False.
    """
    if isinstance(x, str) and x.strip():
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

def valid_url(x):
    """
    Return ``True`` if ``x`` is a valid URL; otherwise return ``False``.
    """
    if re.match(URL_PATTERN, x):
        return True 
    else:
        return False

def check_table(messages, table, condition, id_column, message):
    """
    Given a list of messages, a table, a boolean condition on the table, an ID column of the table, and a message, do the following.
    If some rows of the table statisfy the condition, then get the values of the ID column for those rows, make and error message from the given message and the IDs, and append that messages to the list of messages.
    Otherwise, return the original list of messages.
    """
    bad_ids = table.loc[condition, id_column].tolist()
    if bad_ids:
        messages.append('{!s}; see {!s}s {!s}'.format(message, id_column, bad_ids))
    return messages 

def check_table_id(messages, table, id_column):
    """
    Given a list of messages, a table, and an ID column of the table, do the following.
    If any of the ID column values are emtpy, blank, or duplicated, then add those errors to the list of messages.
    Otherwise, return the original list of messages.
    """
    cond = table[id_column].isnull() | ~table[id_column].map(valid_string)
    messages = check_table(messages, table, cond, id_column, 'Empty or blank route_id')

    cond = table[id_column].duplicated()
    messages = check_table(messages, table, cond, id_column, 'Duplicated route_id')

    return messages

def build_errors(table_name, messages):
    """
    Given the name of a GTFS table and a list of error messages regarding that table, return the list ``[(table_name, m) for m in messages]``.
    """
    return [(table_name, m) for m in messages]

def check_for_required_tables(feed):
    errors = []
    required_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
    for table in required_tables:
        if getattr(feed, table) is None:
            errors.append([table, 'Missing file'])
    # Calendar check is different
    if feed.calendar is None and feed.calendar_dates is None:
        errors.append(['calendar/calendar_dates', 
          'Both calendar and calendar_dates files are missing'])

    return errors 

def check_for_required_fields(feed):
    errors = []
    for table, group in cs.GTFS_REF.groupby('table'):
        f = getattr(feed, table)
        if f is None:
            continue

        for field, field_required in group[['field', 'field_required']].itertuples(
          index=False):
            if field_required and field not in f.columns:
                errors.append([table, 'Missing field {!s}'.format(field)])
    return errors 

def check_routes(feed):
    """
    Check that ``feed.routes`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('routes', error message).
    """
    f = feed.routes.copy()
    msgs = []

    # Check route_id
    msgs = check_table_id(msgs, f, 'route_id')

    # Check agency_id
    if 'agency_id' in f:
        if 'agency_id' not in feed.agency.columns:
            msgs.append('Column agency_id present in routes but not in agency')
        else:
            g = f.dropna(subset=['agency_id'])
            cond = ~g['agency_id'].isin(feed.agency['agency_id'])
            msgs = check_table(msgs, g, cond, 'route_id', 'Undefined agency_id')

    # Check route_short_name and route_long_name
    for field in ['route_short_name', 'route_long_name']:
        g = f.dropna(subset=[field])
        cond = ~g[field].map(valid_string)
        msgs = check_table(msgs, g, cond, 'route_id', 'Blank {!s}'.format(field))

    cond = ~(f['route_short_name'].notnull() | f['route_long_name'].notnull())
    msgs = check_table(msgs, f, cond, 'route_id', 'Both route_short_name and route_long_name empty')

    # Check route_type
    r = list(range(8))
    cond = ~f['route_type'].isin(r)
    msgs = check_table(msgs, f, cond, 'route_id', 
      'route_type out of range {!s}'.format(r))

    # Check route_url
    if 'route_url' in f.columns:
        v = lambda x: pd.isnull(x) or valid_url(x)
        cond = ~f['route_url'].map(v)
        msgs = check_table(msgs, f, cond, 'route_id', 'Malformed route_url')

    # Check route_color and route_text_color
    for field in ['route_color', 'route_text_color']:
        v = lambda x: pd.isnull(x) or valid_color(x)
        if field in f.columns:
            cond = ~f[field].map(v)
            msgs = check_table(msgs, f, cond, 'route_id', 
              'Malformed {!s}'.format(field))
    
    return build_errors('routes', msgs)

def check_stops(feed):
    """
    Check ``feed.stops`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('stops', error message).
    """
    f = feed.stops.copy()
    msgs = []

    # Check stop_id
    msgs = check_table_id(msgs, f, 'stop_id')

def check_trips(feed):
    """
    Check ``feed.trips`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('trips', error message).
    """
    f = feed.trips.copy()
    msgs = []

    # Check trip_id
    msgs = check_table_id(msgs, f, 'trips_id')

    # Check route_id present in routes
    cond = ~f['route_id'].isin(feed.routes['route_id'])
    msgs = check_table(msgs, f, cond, 'trip_id', 'Undefined route_id')

    # Check service_id present in calendar or calendar dates
    if feed.calendar is not None:
        cond = ~f['service_id'].isin(feed.calendar['service_id'])
    else:
        cond = ~f['service_id'].isin(feed.calendar_dates['service_id'])
    msgs = check_table(msgs, f, cond, 'trip_id', 'Undefined service_id')

    # Check direction_id
    if 'direction_id' in f.columns:
        g = f.dropna(subset=['direction_id'])
        r = list(range(2))
        cond = ~g['direction_id'].isin(r)
        msgs = check_table(msgs, g, cond, 'trip_id', 
          'direction_id out of range {!s}'.format(r))

    # Check block_id
    if 'block_id' in f.columns:
        v = lambda x: pd.isnull(x) or valid_string(x)
        cond = ~f['block_id'].map(v)
        msgs = check_table(msgs, f, cond, 'trip_id', 'Blank block_id')

        g = f.dropna(subset=['block_id'])
        cond = ~g['block_id'].duplicated(keep=False)
        msgs = check_table(msgs, g, cond, 'trip_id', 'Unduplicated block_id')

    # Check shape_id present in shapes
    if 'shape_id' in f.columns:
        g = f.dropna(subset=['shape_id'])
        cond = ~g['shape_id'].isin(feed.shapes['shape_id'])
        msgs = check_table(msgs, g, cond, 'trip_id', 'Undefined shape_id')

    # Check wheelchair_accessible and bikes_allowed
    for field in ['wheelchair_accessible', 'bikes_allowed']:
        if field in f.columns:
            g = f.dropna(subset=[field])
            r = list(range(3))
            cond = ~g[field].isin(r)
            msgs = check_table(msgs, g, cond, 'trip_id', 
              '{!s} out of range {!s}'.format(field, r))

    return build_errors('trips', msgs)

def validate(feed, as_df=False):
    """
    Validate the given feed by doing all the checks above.
    Return the errors found as a possibly empty list of pairs (table name, error message).
    If ``as_df``, then format the error list as a DataFrame with the columns

    - ``'table'``: name of table where error occurs
    - ``'error'``: error message.

    Return early if the feed is missing required tables or required fields.
    """
    errors = []
    if as_df:
        format = lambda errors: pd.DataFrame(errors, columns=['table', 'error'])
    else:
        format = lambda x: x

    # Halt if the following critical checks reveal errors
    ops = [
      'check_for_required_tables',
      'check_for_required_fields',
      ]
    for op in ops:
        errors.extend(globals()[op](feed))
        if errors:
            return format(errors)

    # Carry on assuming that all the required tables and fields are present
    ops = [
      'check_routes',
      'check_trips',
      ]
    for op in ops:
        errors.extend(globals()[op](feed))

    return format(errors)