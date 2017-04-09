"""
GTFS validator.
Work in progress.
"""
import pandas as pd

from . import constants as cs


def check_for_required_tables(feed):
	errors = []
	required_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
	for table in required_tables:
		if getattr(feed, table) is None:
			errors.append([
				table,
				'missing file',
				])
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
				errors.append([
					table,
					'missing field {!s}'.format(field),
					])
	return errors 

def check_routes(feed):
	"""
	"""
	f = feed.routes.copy()
	errors = []
	pass

def validate(feed):
	errors = []
	ops = [
		'check_for_required_tables',
		'check_for_required_fields',
	]
	for op in ops:
		errors.extend(globals()[op](feed))

	report = pd.DataFrame(errors, columns=['table', 'message'])
	return report

