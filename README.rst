GTFS Toolkit
============
This is a set of Python 2.7 tools for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data.
It's mostly for collecting routes and stops stats, such as mean daily service distance per route and mean daily number of vehicles per stop.
Uses Pandas and Shapely to do the heavy lifting.

Warning
--------
This package is experimental at this point, so use it at your own risk.
It needs more testing.

Installation
-------------
``pip install gtfs-toolkit``

Examples
--------
Play with ``examples/gtfs_toolkit_examples.ipynb`` in an iPython notebook

Todo
----
- Add error check and workaround for feeds missing``shapes.txt``
- Add more tests
