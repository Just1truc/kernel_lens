import os

_VERBOSE = False

def is_debug() -> bool:
    return _VERBOSE or os.environ.get("KERNEL_LENS_DEBUG", "0").lower() in ("1", "true", "yes", "on") or os.environ.get("KL_VERBOSE", "0").lower() in ("1", "true", "yes", "on")

def is_verbose() -> bool:
    return is_debug()

def set_verbose(val: bool = True):
    global _VERBOSE
    _VERBOSE = val

def debug_print(*args, **kwargs):
    if is_debug():
        print(*args, **kwargs)
