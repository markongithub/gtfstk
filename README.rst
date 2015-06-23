GTFS-TK
============
This is a Python 3.4 toolkit for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It's mostly for computing statistics, such as daily service distance per route and daily number of trips per stop.
It uses Pandas and Shapely to do the heavy lifting.

Warning
--------
This is an alpha release.
It needs more testing and the API might change.

Installation
-------------
``pip install gtfs-tk``

Examples
--------
You can play with ``examples/gtfs_tk_examples.ipynb`` in an iPython notebook

Documentation
--------------
Documentation is in ``docs/`` and also `here <https://rawgit.com/araichev/gtfs-tk/master/docs/_build/html/index.html>`_.

Comments
------------
Constructive comments are welcome and are best filed in this repository's issue section with an appropriate label, e.g. 'enhancement'.