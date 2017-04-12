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

def valid_str(x):
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

def valid_timezone(x):
    """
    Retrun ``True`` if ``x`` is a valid human-readable timezone string, e.g. 'Africa/Abidjan'; otherwise return ``False``.
    """
    return x in cs.TIMEZONES

def check_table(errors, table_name, table, condition, message):
    """
    Given a list of errors, a table name, the corresponding table (DataFrame), a boolean condition on the table, and an error message, do the following.
    Record the indices of the rows of the table that statisfy the condition.
    If this is a nonempty list, then append the given error message to the list of errors along with the offending table indices.
    Otherwise, return the original list of errors.
    """
    indices = table.loc[condition].index.tolist()
    if indices:
        errors.append([table_name, message, indices])
    return errors 

def check_field_id(errors, table_name, table, field):
    """
    """
    v = lambda x: pd.notnull(x) and valid_str(x)
    cond = ~table[field].map(v)
    errors = check_table(errors, table_name, table, cond, 
      '{!s} empty or blank'.format(field))

    cond = table[field].duplicated()
    errors = check_table(errors, table_name, table, cond, 
      '{!s} duplicated'.format(field))

    return errors 

def check_field_url(errors, table_name, table, field, field_required):
    """
    """
    v = valid_url
    if field_required:
        checker = lambda x: pd.notnull(x) and v(x)
        cond = ~table[field].map(checker)  
        errors = check_table(errors, table_name, table, cond, 
          '{!s} empty or blank'.format(field))
    else:
        if field in table.columns:
            g = table.dropna(subset=[field])
            cond = ~g[field].map(v)
            errors = check_table(errors, table_name, g, cond, 
              '{!s} blank'.format(field))
    return errors

def check_field_str(errors, table_name, table, field, field_required):
    """
    """
    v = valid_str
    if field_required:
        checker = lambda x: pd.notnull(x) and v(x)
        cond = ~table[field].map(checker)  
        errors = check_table(errors, table_name, table, cond, 
          '{!s} empty or blank'.format(field))
    else:
        if field in table.columns:
            g = table.dropna(subset=[field])
            cond = ~g[field].map(v)
            errors = check_table(errors, table_name, g, cond, 
              '{!s} blank'.format(field))
    return errors

def check_field_timezone(errors, table_name, table, field, field_required):
    """
    """
    v = valid_timezone
    if field_required:
        checker = lambda x: pd.notnull(x) and v(x)
        cond = ~table[field].map(checker)  
        errors = check_table(errors, table_name, table, cond, 
          '{!s} empty or blank'.format(field))
    else:
        if field in table.columns:
            g = table.dropna(subset=[field])
            cond = ~g[field].map(v)
            errors = check_table(errors, table_name, g, cond, 
              '{!s} blank'.format(field))
    return errors

def check_field_range(errors, table_name, table, field, field_required, range_):
    """
    """
    v = lambda x: x in range_
    if field_required:
        checker = lambda x: pd.notnull(x) and v(x)
        cond = ~table[field].map(checker)  
        errors = check_table(errors, table_name, table, cond, 
          '{!s} empty or out of range'.format(field))
    else:
        if field in table.columns:
            g = table.dropna(subset=[field])
            cond = ~g[field].map(v)
            errors = check_table(errors, table_name, g, cond, 
              '{!s} out of range'.format(field))
    return errors 

def check_field_linked_id(errors, table_name, table, field, field_required, 
  target_table):
    """
    """
    if field_required:
        cond = ~(table[field].notnull() & table[field].isin(target_table[field]))  
        errors = check_table(errors, table_name, table, cond, 
          '{!s} undefiend'.format(field))
    else:
        if field in table.columns:
            g = table.dropna(subset=[field])
            cond = ~g[field].isin(target_table[field])
            errors = check_table(errors, table_name, g, cond, 
              '{!s} undefined'.format(field))
    return errors 

def check_for_required_tables(feed):
    errors = []
    required_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
    for table in required_tables:
        if getattr(feed, table) is None:
            errors.append([table, 'Missing file', []])
    # Calendar check is different
    if feed.calendar is None and feed.calendar_dates is None:
        errors.append(['calendar/calendar_dates', 
          'Both calendar and calendar_dates files are missing', []])
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
                errors.append([table, 'Missing field {!s}'.format(field), []])
    return errors 

def check_routes(feed):
    """
    Check that ``feed.routes`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('routes', error message).
    """
    f = feed.routes.copy()
    errors = []

    # Check route_id
    errors = check_field_id(errors, 'routes', f, 'route_id')

    # Check agency_id
    if 'agency_id' in f:
        if 'agency_id' not in feed.agency.columns:
            errors.append('Column agency_id present in routes but not in agency')
        else:
            g = f.dropna(subset=['agency_id'])
            cond = ~g['agency_id'].isin(feed.agency['agency_id'])
            errors = check_table(errors, 'routes', g, cond, 'agency_id undefined')

    # Check route_short_name and route_long_name
    for field in ['route_short_name', 'route_long_name']:
        errors = check_field_str(errors, 'routes', f, field, False)

    cond = ~(f['route_short_name'].notnull() | f['route_long_name'].notnull())
    errors = check_table(errors, 'routes', f, cond, 
      'route_short_name and route_long_name both empty')

    # Check route_type
    errors = check_field_range(errors, 'routes', f, 'route_type', True, range(8))

    # Check route_url
    errors = check_field_url(errors, 'routes', f, 'route_url', False)

    # Check route_color and route_text_color
    for field in ['route_color', 'route_text_color']:
        v = lambda x: pd.isnull(x) or valid_color(x)
        if field in f.columns:
            cond = ~f[field].map(v)
            errors = check_table(errors, 'routes', f, cond, 
              '{!s} malformed'.format(field))
    
    return errors

def check_stops(feed):
    """
    Check ``feed.stops`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('stops', error message).
    """
    f = feed.stops.copy()
    errors = []

    # Check stop_id
    errors = check_field_id(errors, 'stops', f, 'stop_id')

    # Check stop_code, stop_desc, zone_id, parent_station
    for field in ['stop_code', 'stop_desc', 'zone_id', 'parent_station']:
        errors = check_field_str(errors, 'stops', f, field, False)

    # Check stop_name
    errors = check_field_str(errors, 'stops', f, 'stop_name', True)

    # Check stop_lon and stop_lat
    for field, bound in [('stop_lon', 180), ('stop_lat', 90)]:
        v = lambda x: pd.notnull(x) and -bound <= x <= bound
        cond = ~f[field].map(v)
        errors = check_table(errors, 'stops', f, cond, 
          '{!s} out of bounds {!s}'.format(field, [-bound, bound]))

    # Check stop_url
    errors = check_field_url(errors, 'stops', f, 'stop_url', False)

    # Check location_type
    errors = check_field_range(errors, 'stops', f, 'location_type', False, range(2))

    # Check stop_timezone
    errors = check_field_timezone(errors, 'stops', f, 'stop_timezone', False)

    # Check wheelchair_boarding
    errors = check_field_range(errors, 'stops', f, 'wheelchair_boarding', False, 
      range(3))

    # Check further location_type and parent_station   
    if 'parent_station' in f.columns:
        if 'location_type' not in f.columns:
            errors.append(['stops', 'location_type field missing', []])
        else:
            # A nonnull parent station entry must refer to a stop with location_type 1
            sids = f.loc[f['parent_station'].notnull(), 'stop_id']
            cond = ~(f['stop_id'].isin(sids) & f['location_type'] == 1)
            errors = check_table(errors, 'stops', f, cond, 
              'location_type must equal 1')

            # A location_type of 1 must have a null parent_station 
            cond = ~(f['location_type'] == 1 & f['parent_station'].isnull())
            errors = check_table(errors, 'stops', f, cond, 
              'location_type must equal 1')

    return errors 

def check_trips(feed):
    """
    Check ``feed.trips`` follows the GTFS and output a possibly empty list of errors found as pairs of the form ('trips', error message).
    """
    f = feed.trips.copy()
    errors = []

    # Check trip_id
    errors = check_field_id(errors, 'trips', f, 'trip_id')

    # Check route_id
    errors = check_field_linked_id(errors, 'trips', f, 'route_id', True, feed.routes)
    
    # Check service_id 
    if feed.calendar is not None:
        cond = ~f['service_id'].isin(feed.calendar['service_id'])
    else:
        cond = ~f['service_id'].isin(feed.calendar_dates['service_id'])
    errors = check_table(errors, 'trips', f, cond, 'service_id undefined')

    # Check direction_id
    errors = check_field_range(errors, 'trips', f, 'direction_id', False, range(2))

    # Check block_id
    if 'block_id' in f.columns:
        v = lambda x: pd.isnull(x) or valid_str(x)
        cond = ~f['block_id'].map(v)
        errors = check_table(errors, 'trips', f, cond, 'block_id blank')

        g = f.dropna(subset=['block_id'])
        cond = ~g['block_id'].duplicated(keep=False)
        errors = check_table(errors, 'trips', g, cond, 'block_id unduplicated')

    # Check shape_id
    errors = check_field_linked_id(errors, 'trips', f, 'shape_id', False, feed.shapes)

    # Check wheelchair_accessible and bikes_allowed
    for field in ['wheelchair_accessible', 'bikes_allowed']:
        errors = check_field_range(errors, 'trips', f, field, False, range(3))

    return errors

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
        format = lambda errors: pd.DataFrame(errors, 
          columns=['table', 'error', 'row numbers - 1'])
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
      'check_stops',
      'check_trips',
      ]
    for op in ops:
        errors.extend(globals()[op](feed))

    return format(errors)