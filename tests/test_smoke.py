# tests/test_smoke.py
import sys
import pathlib

class TestSmoke:
    """Simple smoke tests to confirm environment & repo structure."""

    def test_python_version(self):
        # Course machines sometimes vary; anything 3.9+ is fine
        major, minor = sys.version_info[:2]
        assert (major, minor) >= (3, 9)

    def test_repo_layout(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for p in ["src", "results", "tests", "README.md", "requirements.txt"]:
            assert (root / p).exists(), f"Missing {p}"
