import os

_VERBOSE = False

def is_verbose() -> bool:
    return _VERBOSE or os.environ.get("KL_VERBOSE", "0") in ("1", "true", "True")

def set_verbose(val: bool = True):
    global _VERBOSE
    _VERBOSE = val
