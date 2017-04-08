"""
GTFS validator.
Work in progress.
"""
from . import constants as cs


def check_for_required_files(feed):
	errors = []
	for table, value in cs.GTFS_TABLES.items():
		if value == 'required' and getattr(feed, table) is None:
			msg = '{!s}.txt is missing'.format(table)
			errors.append(msg)
	return errors 

def check_agency(feed):
	errors = []
	# Check for required fields



