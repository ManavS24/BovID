"""pytest configuration — bovid is installed (pip install -e .), so no sys.path manipulation.

Shared resource-skip guards live in `_bovid_reqs.py` (imported bare by the tests that need them),
not here: ultralytics installs its own `tests` package into site-packages, so importing helpers
via `tests.conftest` would resolve to the wrong module.
"""
