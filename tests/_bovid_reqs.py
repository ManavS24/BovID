"""Shared skip guards for resources not guaranteed to be present in every environment.

(Named distinctively rather than living in conftest.py because `ultralytics` installs its own
`tests` package into site-packages, so `from tests.conftest import ...` resolves to the wrong
module; this file is imported bare as `_bovid_reqs`, which pytest's prepend import mode makes
available from the tests directory.)

Resources that may be absent:

  * the **server detector** (yolov8x.pt, 137 MB) exceeds GitHub's 100 MB file limit, so a CI
    checkout will not have it unless weights are fetched separately;
  * the **deployed classifier** (best.pt) and the **dataset** may likewise be absent on CI.

Tests that need these declare it with the marks below and SKIP cleanly when the resource is
missing, so `pytest -m "not slow"` is green both locally (full run) and on CI (logic-only run)
without maintaining a separate list of which files CI may execute.
"""

import os
from importlib.util import find_spec

import pytest

from bovid import config

_DEPLOYED_MODEL = config.DEFAULT_MODEL_PATH
_DATASET = os.path.join(config.RAW_DATA_DIR, "test", "_classes.csv")


def valid_breeds():
    """The set of known breed names, read from the test split's one-hot column headers. Used to
    assert predictions name a real breed."""
    import pandas as pd
    return set(pd.read_csv(_DATASET).columns[1:])


def _tflite_runtime_available():
    """Either TFLite runtime works. ai-edge-litert is the standalone edge runtime and has a wheel
    on Pythons where TensorFlow does not (e.g. 3.14, the dev venv), so gating the edge tests on
    *tensorflow* alone made them skip on 3.14 even though the edge path runs there. The tests need
    *a* TFLite interpreter, not TensorFlow specifically."""
    return find_spec("ai_edge_litert") is not None or find_spec("tensorflow") is not None


requires_tflite = pytest.mark.skipif(
    not _tflite_runtime_available(),
    reason="no TFLite runtime (install ai-edge-litert or tensorflow)",
)

# A stricter gate: tests that load a .tflite *through Ultralytics* (to compare our torch-free
# runner against Ultralytics' own TFLite inference) need TensorFlow specifically -- Ultralytics'
# TFLite backend imports tensorflow, and ai-edge-litert does not satisfy it. Most edge tests only
# need `requires_tflite`; only the Ultralytics-reference parity checks need this.
requires_tensorflow = pytest.mark.skipif(
    find_spec("tensorflow") is None,
    reason="tensorflow needed for Ultralytics to load a .tflite reference (no wheel on Py 3.14)",
)

requires_detector = pytest.mark.skipif(
    not os.path.exists(config.DETECTOR_MODEL),
    reason=f"server detector not present ({config.DETECTOR_MODEL}); "
           "137 MB, not in a CI checkout",
)

requires_deployed_model = pytest.mark.skipif(
    not os.path.exists(_DEPLOYED_MODEL),
    reason=f"deployed classifier not present ({_DEPLOYED_MODEL})",
)

requires_dataset = pytest.mark.skipif(
    not os.path.exists(_DATASET),
    reason=f"labelled dataset not present ({_DATASET})",
)
