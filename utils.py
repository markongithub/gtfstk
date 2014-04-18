def get_trip_activity(dates=None, ndigits=3):
    merged = pd.merge(trips[['trip_id', 'service_id']], calendar)
    if dates is None:
        # Use the full range of dates for which the feed is valid
        feed_start = merged['start_date'].min()
        feed_end = merged['end_date'].max()
        num_days =  (feed_end - feed_start).days
        dates = [feed_start + rd.relativedelta(days=+d) for d in range(num_days + 1)]
        activity_header = 'fraction of dates active during %s--%s' %\
          (date_to_str(feed_start), date_to_str(feed_end))
    else:
        activity_header = 'fraction of dates active among %s' % '-'.join([date_to_str(d) for d in dates])
    merged[activity_header] = pd.Series()

    n = len(dates)
    for index, row in merged.iterrows():    
        trip = row['trip_id']
        start_date = row['start_date']
        end_date = row['end_date']
        weekdays_running = [i for i in range(7) if row[weekday_to_str(i)] == 1]
        count = 0
        for date in dates:
            if not start_date <= date <= end_date:
                continue
            if date.weekday() in weekdays_running:
                count += 1
        merged.ix[index, activity_header] = round(count/n, ndigits)
    return merged
  
def get_shapes_with_shape_dist_traveled():
    t1 = dt.datetime.now()
    print(t1, 'Adding shape_dist_traveled field to %s shape points...' %\
      shapes['shape_id'].count())

    new_shapes = shapes.copy()
    new_shapes['shape_dist_traveled'] = pd.Series()
    for shape_id, group in shapes.groupby('shape_id'):
        group.sort('shape_pt_sequence')
        lons = group['shape_pt_lon'].values
        lats = group['shape_pt_lat'].values
        xys = [utm.from_latlon(lat, lon)[:2] 
          for lat, lon in izip(lats, lons)]
        xy_prev = xys[0]
        d_prev = 0
        index = group.index[0]
        for (i, xy) in enumerate(xys):
            p = Point(xy_prev)
            q = Point(xy)
            d = p.distance(q)
            new_shapes.ix[index + i, 'shape_dist_traveled'] =\
              round(d_prev + d, 2)
            xy_prev = xy
            d_prev += d

    t2 = dt.datetime.now()
    minutes = (t2 - t1).seconds/60
    print(t2, 'Finished in %.2f min' % minutes)    
    return new_shapes

def get_stop_times_with_shape_dist_traveled():
    t1 = dt.datetime.now()
    print(t1, 'Adding shape_dist_traveled field to %s stop times...' %\
      stop_times['trip_id'].count())

    linestring_by_shape = get_linestring_by_shape()
    xy_by_stop = get_xy_by_stop()

    failures = []

    # Initialize data frame
    merged = pd.merge(trips[['trip_id', 'shape_id']], stop_times)
    merged['shape_dist_traveled'] = pd.Series()

    # Compute shape_dist_traveled column entries
    dist_by_stop_by_shape = {shape: {} for shape in linestring_by_shape}
    for shape, group in merged.groupby('shape_id'):
        linestring = linestring_by_shape[shape]
        for trip, subgroup in group.groupby('trip_id'):
            # Compute the distances of the stops along this trip
            subgroup.sort('stop_sequence')
            stops = subgroup['stop_id'].values
            distances = []
            for stop in stops:
                if stop in dist_by_stop_by_shape[shape]:
                    d = dist_by_stop_by_shape[shape][stop]
                else:
                    d = round(
                      get_segment_length(linestring, xy_by_stop[stop]), 2)
                    dist_by_stop_by_shape[shape][stop] = d
                distances.append(d)
            if distances[0] > distances[1]:
                # This happens when the shape linestring direction is the
                # opposite of the trip direction. Reverse the distances.
                distances = distances[::-1]
            if distances != sorted(distances):
                # Uh oh
                failures.append(trip)
            # Insert stop distances
            index = subgroup.index[0]
            for (i, d) in enumerate(distances):
                merged.ix[index + i, 'shape_dist_traveled'] = d
    
    # Clean up
    del merged['shape_id']
    t2 = dt.datetime.now()
    minutes = (t2 - t1).seconds/60
    print(t2, 'Finished in %.2f min' % minutes)
    print('%s failures on these trips: %s' % (len(failures), failures))    
    return merged