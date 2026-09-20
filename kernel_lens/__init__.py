from .compiler.core import compile
from .utils.deployment import extract_libs
from .compiler.core import load
from .config import set_verbose, is_verbose





__version__ = "1.1.7"

__all__ = [
    "compile",
    "load",
    "extract_libs",
    "set_verbose",
    "is_verbose",
    "__version__",
]