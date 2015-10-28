GTFSTK
========
This is a Python 3.4 tool kit for processing `General Transit Feed Specification (GTFS) <https://en.wikipedia.org/wiki/GTFS>`_ data in memory without a database.
It's mostly for computing statistics, such as daily service distance per route and daily number of trips per stop.
It uses Pandas and Shapely to do the heavy lifting.


Installation
-------------
``pip install gtfstk``


Examples
--------
You can play with ``ipynb/examples.ipynb`` in a Jupyter notebook


Documentation
--------------
Documentation is in ``docs/`` and also on RawGit `here <https://rawgit.com/araichev/gtfstk/master/docs/_build/html/index.html>`_.


Notes
--------
- Development Status is Beta
- This project uses semantic versioning (major.minor.micro), where each breaking feature or API change is considered a major release, at least after version 0.12.3. 
  So the version code reflects the project's change history, rather than its development status. 
  In particular, a high major version number, does not imply a mature development status.


Comments
------------
Constructive comments are welcome and are best filed in this repository's issue section with an appropriate label, e.g. 'enhancement'.


Authors
---------
- Alex Raichev (2014-05)