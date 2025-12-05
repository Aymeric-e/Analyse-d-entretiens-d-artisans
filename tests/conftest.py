"""
Conftest for pytest fixture configuration.

This module configures the Python path to ensure imports from src/ work correctly
when running tests. It adds the project's src/ directory to sys.path, allowing
test files to import modules using clean relative paths like:
    from preprocessing.text_cleaning import InterviewCleaner
    from utils.logger_config import setup_logger
"""

import os
import sys

# Ensure the project's src/ directory is on sys.path for imports like
# `from preprocessing.text_cleaning import InterviewCleaner`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
