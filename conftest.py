"""
Pytest configuration file for SAM2 tests.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Add tests directory to Python path
tests_dir = os.path.join(project_root, 'tests')
sys.path.insert(0, tests_dir) 