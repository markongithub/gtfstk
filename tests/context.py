import os
import sys
from pathlib import Path
import importlib
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np 

import gtfstk
import pytest

# Decorator to mark slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)

# Check if GeoPandas is installed
loader = importlib.find_loader('geopandas')
if loader is None:
    HAS_GEOPANDAS = False
else:
    HAS_GEOPANDAS = True

# Load/create test feeds
DATA_DIR = Path('data')
sample = gtfstk.read_gtfs(DATA_DIR/'sample_gtfs.zip', dist_units='km')
cairns = gtfstk.read_gtfs(DATA_DIR/'cairns_gtfs.zip', dist_units='km')
cairns_shapeless = cairns.copy()
cairns_shapeless.shapes = None
t = cairns_shapeless.trips
t['shape_id'] = np.nan
cairns_shapeless.trips = t
cairns_date = cairns.get_first_week()[0]
cairns_trip_stats = pd.read_csv(DATA_DIR/'cairns_trip_stats.csv', dtype=gtfstk.DTYPE)
