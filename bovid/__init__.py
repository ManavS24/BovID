"""bovid — Indian bovine breed classification (detect → crop → classify)."""

import os

# Set before torch imports: lets ops with no MPS kernel fall back to CPU instead of aborting.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "1.0.0"
