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
  'agency_id': str,
  'stop_id': str, 
  'stop_code': str,
  'zone_id': str,
  'parent_station': str,
  'route_id': str, 
  'route_short_name': str,
  'trip_id': str, 
  'service_id': str, 
  'shape_id': str, 
  'start_date': str, 
  'end_date': str,
  'date': str,
  'fare_id': str,
  'origin_id': str,
  'destination_id': str,
  'contains_id': str,
  'from_stop_id': str,
  'to_stop_id': str,
}

# From the GTFS reference at https://developers.google.com/transit/gtfs/reference/
VALID_COLUMNS_BY_TABLE = {
    'agency': [
        'agency_id',  
        'agency_name',    
        'agency_url', 
        'agency_timezone', 
        'agency_lang',
        'agency_phone',   
        'agency_fare_url',
        'agency_email',   
        ],
    'calendar': [
        'service_id',
        'monday',
        'tuesday',
        'wednesday',
        'thursday',
        'friday',
        'saturday',
        'sunday',
        'start_date',
        'end_date',
        ],
    'calendar_dates': [
        'service_id',
        'date',
        'exception_type',
        ],
    'fare_attributes': [
        'fare_id',
        'price',
        'currency_type',
        'payment_method',
        'transfers',
        'transfer_duration',
        ],
    'fare_rules': [
        'fare_id',
        'route_id',
        'origin_id',
        'destination_id',
        'contains_id',
        ],
    'feed_info': [
        'feed_publisher_name',
        'feed_publisher_url',
        'feed_lang',
        'feed_start_date',
        'feed_end_date',
        'feed_version',
        ],
    'frequencies': [
        'trip_id',
        'start_time',
        'end_time',
        'headway_secs',
        'exact_times',
        ],
    'routes': [
        'route_id',
        'agency_id',
        'route_short_name',
        'route_long_name',
        'route_desc',
        'route_type',
        'route_url',
        'route_color',
        'route_text_color',
        ],
    'shapes': [
        'shape_id',
        'shape_pt_lat',
        'shape_pt_lon',
        'shape_pt_sequence',
        'shape_dist_traveled',
        ], 
    'stops': [
        'stop_id',
        'stop_code',
        'stop_name',
        'stop_desc',
        'stop_lat',
        'stop_lon',
        'zone_id',
        'stop_url',
        'location_type',
        'parent_station',
        'stop_timezone',
        'wheelchair_boarding',
        ],
    'stop_times': [
        'trip_id',
        'arrival_time',
        'departure_time',
        'stop_id',
        'stop_sequence',
        'stop_headsign',
        'pickup_type',
        'drop_off_type',
        'shape_dist_traveled',
        'timepoint',
        ],
    'transfers': [
        'from_stop_id',
        'to_stop_id',
        'transfer_type',
        'min_transfer_time',
        ],
    'trips': [
        'route_id',
        'service_id',
        'trip_id',
        'trip_headsign',
        'trip_short_name',
        'direction_id',
        'block_id',
        'shape_id',
        'wheelchair_accessible',
        'bikes_allowed',
        ],
    }

#:
DIST_UNITS = ['ft', 'mi', 'm', 'km']

#:
FEED_ATTRS_PUBLIC = [
  'agency', 
  'stops', 
  'routes', 
  'trips', 
  'stop_times', 
  'calendar', 
  'calendar_dates', 
  'fare_attributes', 
  'fare_rules', 
  'shapes', 
  'frequencies', 
  'transfers', 
  'feed_info',
  'dist_units',
  ]

#:
FEED_ATTRS_PRIVATE = [
  '_trips_i', 
  '_calendar_i', 
  '_calendar_dates_g',
  ]
  
FEED_ATTRS = FEED_ATTRS_PUBLIC + FEED_ATTRS_PRIVATE

# Columns that must be formatted as integers when outputting GTFS
#:
INT_COLUMNS = [
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
CRS_WGS84 = {'no_defs': True, 'ellps': 'WGS84', 'datum': 
  'WGS84', 'proj': 'longlat'}
