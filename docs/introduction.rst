This is a Python 3 tool kit for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It's mostly for computing network-level statistics, such as mean daily service distance per route and mean daily number of vehicles per stop.
Uses Pandas and Shapely to do the heavy lifting.


Installation
-------------
``pip install gtfstk``


Examples
--------
Play with ``ipynb/examples.ipynb`` in a Jupyter notebook


Conventions
------------
In conformance with GTFS and unless specified otherwise, 
dates are encoded as date strings of 
the form YYMMDD and times are encoded as time strings of the form HH:MM:SS
with the possibility that the hour is greater than 24.
Unless specified otherwise, 'data frame' and 'series' refer to
Pandas data frames and series, respectively.
