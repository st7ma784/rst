"""
pytest configuration and fixtures for backend comparison tests.

Environment variables:
  BACKEND_URLS   comma-separated list of backend base URLs to test against.
                 Default: http://localhost:8001,http://localhost:8002,http://localhost:8003
  TEST_DATA_DIR  path to directory with rawacf test files.
                 Default: <repo-root>/test_data/rawacf_samples
  TEST_TIMEOUT   seconds to wait for a processing job to finish. Default: 120
"""

import os
import sys
import types
from pathlib import Path
import pytest
import httpx

_REPO_ROOT = Path(__file__).parent.parent.parent

# Make pythonv2 importable as 'superdarn_gpu' without installing.
_PYTHONV2 = _REPO_ROOT / "pythonv2"
if str(_PYTHONV2) not in sys.path:
    sys.path.insert(0, str(_PYTHONV2))

# Stub optional heavy dependencies so superdarn_gpu can be imported
# in a minimal test environment that lacks them.
for _stub in ("netCDF4", "ipywidgets", "ipympl"):
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)

DEFAULT_URLS = "http://localhost:8001,http://localhost:8002,http://localhost:8003"


def pytest_addoption(parser):
    parser.addoption(
        "--backend-urls",
        default=os.environ.get("BACKEND_URLS", DEFAULT_URLS),
        help="Comma-separated backend base URLs"
    )
    parser.addoption(
        "--test-data-dir",
        default=os.environ.get("TEST_DATA_DIR",
                               str(_REPO_ROOT / "test_data" / "rawacf_samples")),
        help="Directory containing rawacf test files"
    )
    parser.addoption(
        "--timeout",
        default=int(os.environ.get("TEST_TIMEOUT", "120")),
        type=int,
        help="Seconds to wait for job completion"
    )


@pytest.fixture(scope="session")
def backend_urls(request) -> list[str]:
    raw = request.config.getoption("--backend-urls")
    return [u.strip() for u in raw.split(",") if u.strip()]


@pytest.fixture(scope="session")
def test_data_dir(request) -> Path:
    return Path(request.config.getoption("--test-data-dir"))


@pytest.fixture(scope="session")
def job_timeout(request) -> int:
    return request.config.getoption("--timeout")


@pytest.fixture(scope="session")
def small_rawacf(test_data_dir) -> Path:
    p = test_data_dir / "test_small.rawacf"
    if not p.exists():
        pytest.skip(f"Test file not found: {p}")
    return p


@pytest.fixture(scope="session")
def medium_rawacf(test_data_dir) -> Path:
    p = test_data_dir / "test_medium.rawacf"
    if not p.exists():
        pytest.skip(f"Test file not found: {p}")
    return p


@pytest.fixture(scope="session")
def http_client():
    with httpx.Client(timeout=30) as client:
        yield client
