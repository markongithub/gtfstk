# Profile some function calls
from gtfs_toolkit.utils import time_it
from gtfs_toolkit.feed import *
import profile

feed = Feed('gtfs_toolkit/tests/portland_gtfs.zip')

def main():
    profile.run(feed.get_trips_stats)

if __name__ == '__main__':
    main()