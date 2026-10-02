import os
from pathlib import Path


# Keep import-time SQLite initialization out of the repository checkout.
TEST_DB = Path("/tmp/repo-analyzer-pytest.sqlite3")
if TEST_DB.exists():
    TEST_DB.unlink()
os.environ["REPO_ANALYZER_DB"] = str(TEST_DB)
