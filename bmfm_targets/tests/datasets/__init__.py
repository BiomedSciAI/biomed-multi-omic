"""
Tests that work specifically with the dataset samples.

These are tests that do not cover generic package logic but are primarily to verify
that the sample data in `tests/resources` is in fact being loaded correctly.
"""
import pytest
import os

if os.environ.get("BMFM_TARGETS_TESTS_SKIP_DATASET_TESTS"):
    pytest.skip("Skipping dataset tests based on environment", allow_module_level=True)
