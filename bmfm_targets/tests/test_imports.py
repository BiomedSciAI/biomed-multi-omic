# tests/test_import_packages_only_subprocess.py
import pkgutil
import subprocess
import sys

import pytest

import bmfm_targets


def _iter_subpackages(package):
    for finder, name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            yield name


@pytest.mark.parametrize("modname", list(_iter_subpackages(bmfm_targets)))
def test_import_subpackages_isolated(modname):
    """
    Try to import each package in a *fresh Python subprocess* to detect circular
    imports or other initialization errors that may be hidden by sys.modules caching.
    """
    cmd = [sys.executable, "-c", f"import {modname}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(
            f"Failed to import {modname} in isolated process:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
