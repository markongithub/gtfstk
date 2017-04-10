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
	"""
	f = feed.routes.copy()
	msgs = []

	# Check route_id
	cond = f['route_id'].isnull() | ~f['route_id'].map(valid_string)
	msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_id')

	cond = f['route_id'].duplicated()
	msgs = check_table(msgs, f, cond, 'route_id', 'Duplicated route_id')
	
	# Check agency_id
	if 'agency_id' in f:
		if 'agency_id' not in feed.agency.columns:
			msgs.append('Column agency_id present in routes but not in agency')
		else:
			cond = ~f['agency_id'].isin(feed.agency['agency_id'])
			msgs = check_table(msgs, f, cond, 'route_id', 'Undefined agency_id')

	# Check route_short_name
	v = lambda x: pd.notnull(x) and valid_string(x)
	cond = ~f['route_short_name'].map(v)
	msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_short_name')

	# Check route_long_name
	cond = ~(f['route_short_name'].map(v) | f['route_long_name'].map(v))
	msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_long_name')

	# Check route_type
	cond = ~f['route_type'].isin(range(8))
	msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_type')

	# Check route_url
	if 'route_url' in f.columns:
		v = lambda x: pd.isnull(x) or valid_url(x)
		cond = ~f['route_url'].map(v)
		msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_url')

	# Check route_color
	v = lambda x: pd.isnull(x) or valid_color(x)
	if 'route_color' in f.columns:
		cond = ~f['route_color'].map(v)
		msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_color')
	
	# Check route_text_color
	if 'route_text_color' in f.columns:
		cond = ~f['route_text_color'].map(v)
		msgs = check_table(msgs, f, cond, 'route_id', 'Invalid route_text_color')

	return build_errors('routes', msgs)

def validate(feed):
	errors = []

	# Halt if the following critical checks reveal errors
	ops = [
   	  'check_for_required_tables',
	  'check_for_required_fields',
	  ]
	for op in ops:
		errors.extend(globals()[op](feed))
		if errors:
			return errors

	# Carry on assuming that all the required tables and fields are present
	ops = [
	  'check_routes',
	  ]
	for op in ops:
		errors.extend(globals()[op](feed))

	return errors

def errors_to_df(errors):
	return pd.DataFrame(errors, columns=['table', 'error'])