"""Ensure the repo root is importable when running pytest from data_view/."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
