#:
REQUIRED_GTFS_FILES = [
  'agency',  
  'stops',   
  'routes',
  'trips',
  'stop_times',
  'calendar',
  ]

#:
OPTIONAL_GTFS_FILES = [
  'calendar_dates',  
  'fare_attributes',    
  'fare_rules',  
  'shapes',  
  'frequencies',     
  'transfers',   
  'feed_info',
  ]

#:
DTYPE = {
  'stop_id': str, 
  'stop_code': str,
  'route_id': str, 
  'route_short_name': str,
  'trip_id': str, 
  'service_id': str, 
  'shape_id': str, 
  'start_date': str, 
  'end_date': str,
  'date': str,
}

# Columns that must be formatted as integers when outputting GTFS
#:
INT_COLS = [
  'location_type',
  'wheelchair_boarding',
  'route_type',
  'direction_id',
  'stop_sequence',
  'wheelchair_accessible',
  'bikes_allowed',
  'pickup_type',
  'drop_off_type',
  'timepoint',
  'monday',
  'tuesday',
  'wednesday',
  'thursday',
  'friday',
  'saturday',
  'sunday',
  'exception_type',
  'payment_method',
  'transfers',
  'shape_pt_sequence',
  'exact_times',
  'transfer_type',
  'transfer_duration',
  'min_transfer_time',
]

#:
DISTANCE_UNITS = ['ft', 'mi', 'm', 'km']

#:
CRS_WGS84 = {'no_defs': True, 'ellps': 'WGS84', 'datum': 
  'WGS84', 'proj': 'longlat'}

#:
BIG = 1000000
