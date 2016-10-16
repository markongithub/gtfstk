GTFSTK is a Python 3.5 tool kit for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It is mostly for computing statistics, such as daily service distance per route and daily number of trips per stop.
It uses Pandas and Shapely to do the heavy lifting.


Installation
=============
Create a Python 3.5 virtual environment and ``pip install gtfstk``.


Examples
========
You can play with ``ipynb/examples.ipynb`` in a Jupyter notebook


Conventions
============
In conformance with GTFS and unless specified otherwise, dates are encoded as date strings of the form YYMMDD and times are encoded as time strings of the form HH:MM:SS with the possibility that the hour is greater than 24.
Unless specified otherwise, 'data frame' and 'series' refer to Pandas data frames and series, respectively.
