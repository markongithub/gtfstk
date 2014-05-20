GTFS Toolkit
============
Python tools for processing `General Transit Feed Specification <https://en.wikipedia.org/wiki/GTFS>`_ data.
Mostly for collecting routes and stops stats, such as mean daily service distance per route and mean daily number of vehicles per stop.
Uses Pandas and Shapely for most of the heavy lifting.

Experimental at this point, so use at your own risk.g

Requirements
------------
See ``requirements.txt``

Examples
--------
Play with ``gtfs_toolkit_examples.ipynb`` in an iPython notebook

Todo
----
- Add some IO functionality, e.g. unzipping feeds
- Add more tests
- Add error check and workaround for feeds missing``shapes.txt``