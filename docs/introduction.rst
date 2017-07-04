GTFSTK is a Python 3.4+ tool kit that analyzes `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It uses Pandas and Shapely to do the heavy lifting.


Installation
=============
Create a Python 3.4+ virtual environment and ``pip install gtfstk``.


Examples
========
See the Jupyter notebook ``ipynb/examples.ipynb``.


Conventions
============
- In conformance with GTFS, dates are encoded as YYYYMMDD date strings, and times are encoded as HH:MM:SS time strings with the possibility that HH > 24. **Watch out** for that possibility, because it has counterintuitive consequences; see e.g. :func:`.trips.is_active_trip`, which is used in :func:`.routes.compute_route_stats`,  :func:`.stops.compute_stop_stats`, and :func:`.miscellany.compute_feed_stats`.
- 'DataFrame' and 'Series' refer to Pandas DataFrame and Series objects,
  respectively
