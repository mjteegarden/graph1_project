# graph1_project/tests/conftest.py
import os
import sys

# Add the project root (the parent of tests/) to sys.path so `from graph1 import Graph` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)