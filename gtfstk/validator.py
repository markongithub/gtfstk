"""
GTFS validator.
Work in progress.
"""
from . import constants as cs


def check_for_required_tables(feed):
	errors = []
	required_tables = cs.GTFS_REF.loc[cs.GTFS_REF['table_required'], 'table']
	for table in required_tables:
		if getattr(feed, table) is None:
			msg = '{!s}.txt is missing'.format(table)
			errors.append(msg)
	return errors 

def check_agency(feed):
	errors = []
	# Check for required fields



