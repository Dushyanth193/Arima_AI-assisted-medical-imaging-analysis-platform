"""
Test Runner CLI Script
Executes pytest test suite.
"""
import sys
import pytest

if __name__ == "__main__":
    ret = pytest.main(["tests/", "-v"])
    sys.exit(ret)
