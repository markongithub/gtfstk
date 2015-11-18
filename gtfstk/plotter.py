"""
Functions for plotting various graphs related to GTFS feeds.
Optional.
Requires Matplotlib.
"""
import pandas as pd
import importlib


# Import matplotlib if it is installed
loader = importlib.find_loader('matplotlib')
if loader is None:
    print("Warning: matplotlib is not installed, "\
      "so the plotting functions (those in module gtfstk.plotter) "\
      "will not work.")
else:
    import matplotlib.pyplot as plt


def plot_headways(stats, max_headway_limit=60):
    """
    Given a stops or routes stats data frame, 
    return bar charts of the max and mean headways as a MatplotLib figure.
    Only include the stops/routes with max headways at most 
    ``max_headway_limit`` minutes.
    If ``max_headway_limit is None``, then include them all in a giant plot. 
    If there are no stops/routes within the max headway limit, then return 
    ``None``.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    # Set Pandas plot style
    pd.options.display.mpl_style = 'default'

    if 'stop_id' in stats.columns:
        index = 'stop_id'
    elif 'route_id' in stats.columns:
        index = 'route_id'
    split_directions = 'direction_id' in stats.columns
    if split_directions:
        # Move the direction_id column to a hierarchical column,
        # select the headway columns, and convert from seconds to minutes
        f = stats.pivot(index=index, columns='direction_id')[['max_headway', 
          'mean_headway']]
        # Only take the stops/routes within the max headway limit
        if max_headway_limit is not None:
            f = f[(f[('max_headway', 0)] <= max_headway_limit) |
              (f[('max_headway', 1)] <= max_headway_limit)]
        # Sort by max headway
        f = f.sort_values(by=[('max_headway', 0)], ascending=False)
    else:
        f = stats.set_index(index)[['max_headway', 'mean_headway']]
        if max_headway_limit is not None:
            f = f[f['max_headway'] <= max_headway_limit]
        f = f.sort_values('max_headway', ascending=False)
    if f.empty:
        return

    # Plot max and mean headway separately
    n = f.shape[0]
    data_frames = [f['max_headway'], f['mean_headway']]
    titles = ['Max Headway','Mean Headway']
    ylabels = [index, index]
    xlabels = ['minutes', 'minutes']
    fig, axes = plt.subplots(nrows=1, ncols=2)
    for (i, f) in enumerate(data_frames):
        f.plot(kind='barh', ax=axes[i], figsize=(10, max(n/9, 10)),
          colors=['lightblue', 'blue'])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel(xlabels[i])
        axes[i].set_ylabel(ylabels[i])
    return fig

def plot_routes_time_series(routes_time_series):
    """
    Given a routes time series data frame,
    sum each time series indicator over all routes, 
    plot each series indicator using MatplotLib, 
    and return the resulting figure of subplots.

    NOTES:

    Take the resulting figure ``f`` and do ``f.tight_layout()``
    for a nice-looking plot.
    """
    rts = routes_time_series
    if 'route_id' not in rts.columns.names:
        return

    # Aggregate time series
    f = compute_feed_time_series(rts)

    # Reformat time periods
    f.index = [t.time().strftime('%H:%M') 
      for t in rts.index.to_datetime()]
    
    #split_directions = 'direction_id' in rts.columns.names

    # Split time series by into its component time series by indicator type
    # stats = rts.columns.levels[0].tolist()
    stats = [
      'num_trip_starts',
      'num_trips',
      'service_distance',
      'service_duration',
      'service_speed',
      ]
    ts_dict = {stat: f[stat] for stat in stats}

    # Create plots  
    pd.options.display.mpl_style = 'default'
    titles = [stat.capitalize().replace('_', ' ') for stat in stats]
    units = ['','','km','h', 'kph']
    alpha = 1
    fig, axes = plt.subplots(nrows=len(stats), ncols=1)
    for (i, stat) in enumerate(stats):
        if stat == 'service_speed':
            stacked = False
        else:
            stacked = True
        ts_dict[stat].plot(ax=axes[i], alpha=alpha, 
          kind='bar', figsize=(8, 10), stacked=stacked, width=1)
        axes[i].set_title(titles[i])
        axes[i].set_ylabel(units[i])

    return fig