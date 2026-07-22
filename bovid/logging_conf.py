"""Logging setup for entrypoints. Diagnostics go to stderr (logging), results to stdout (print),
so `bovid predict img.jpg > out.txt` stays clean. Library modules never configure logging.
"""

import logging
import os


def setup_logging(level=None):
    """Configure root logging for an entrypoint. Level may be overridden with BOVID_LOG_LEVEL."""
    level = level or os.environ.get("BOVID_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Ultralytics is extremely chatty at INFO; keep its noise out of our logs.
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    return logging.getLogger("bovid")
