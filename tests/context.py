import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import gtfstk
import pytest

# Decorator to mark slow tests
slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
)