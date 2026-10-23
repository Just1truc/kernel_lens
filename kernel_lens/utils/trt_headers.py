"""Provision TensorRT headers matching the installed runtime.

Why this exists
---------------
KernelLens compiles its plugins against a set of TensorRT headers, and the pip
``tensorrt`` wheels ship none -- verified: the package contains only
``__init__.py`` and ``plugin/``. So the only headers on a normal install are the
ones this package bundles, and those are frozen at whatever version was vendored
(currently TensorRT 10.14).

A plugin built against 10.x headers and loaded into an 11.x runtime is not
guaranteed to work, and when it fails it segfaults inside TensorRT with no
message. Rather than refusing every runtime that does not match the bundled
headers, we fetch the matching ones.

Where headers come from
-----------------------
In order of preference:

1. ``TENSORRT_INCLUDE_DIR`` -- explicit, always wins.
2. The installed TensorRT's own include directory, if the wheel happens to ship
   one (some distributions do).
3. A local cache, ``$KERNEL_LENS_HEADER_CACHE`` or ``~/.kernel_lens_headers``,
   keyed by ``<major>.<minor>``.
4. Download from NVIDIA's TensorRT OSS repository, branch ``release/<major>.<minor>``.
   The public API headers live there and the branches track releases.
5. The bundled headers, which is only correct when the major versions agree.

Fetching is best-effort and cached: it happens once per version, and if the
network is unavailable we fall back to the bundled set so the existing
header/runtime mismatch check can report the problem clearly.
"""
import os
import shutil
import urllib.error
import urllib.request

#: The files a plugin build needs. Every one of these is present in the OSS
#: repository's ``include/`` directory for each release branch.
HEADER_FILES = (
    "NvInfer.h",
    "NvInferImpl.h",
    "NvInferLegacyDims.h",
    "NvInferPluginBase.h",
    "NvInferPlugin.h",
    "NvInferPluginUtils.h",
    "NvInferRuntime.h",
    "NvInferRuntimeBase.h",
    "NvInferRuntimeCommon.h",
    "NvInferRuntimePlugin.h",
    "NvInferVersion.h",
)

_RAW = "https://raw.githubusercontent.com/NVIDIA/TensorRT/release/{ver}/include/{name}"
_FETCH_TIMEOUT = 20


def installed_version():
    """The installed TensorRT version string, or None."""
    try:
        import tensorrt as trt
        return str(trt.__version__)
    except Exception:
        return None


def major_minor(version=None):
    """``"11.3"`` from ``"11.3.0.99"``, or None."""
    v = version or installed_version()
    if not v:
        return None
    parts = v.split(".")
    if len(parts) < 2:
        return None
    return f"{parts[0]}.{parts[1]}"


def _bundled_dir():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "include", "tensorrt")


def header_major(directory):
    """Major version declared by ``NvInferVersion.h`` in ``directory``."""
    import re
    try:
        with open(os.path.join(directory, "NvInferVersion.h"), errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None
    m = re.search(r"#define\s+TRT_MAJOR_ENTERPRISE\s+(\d+)", txt)
    if m:
        return int(m.group(1))
    m = re.search(r"#define\s+NV_TENSORRT_MAJOR\s+(\d+)", txt)
    return int(m.group(1)) if m else None


def _is_complete(directory):
    return all(os.path.exists(os.path.join(directory, f)) for f in HEADER_FILES)


def cache_root():
    return os.environ.get(
        "KERNEL_LENS_HEADER_CACHE",
        os.path.join(os.path.expanduser("~"), ".kernel_lens_headers"),
    )


def cached_dir(mm):
    return os.path.join(cache_root(), mm)


def fetch(mm, dest=None, timeout=_FETCH_TIMEOUT):
    """Download the header set for ``<major>.<minor>``. Returns the dir or None.

    Downloads into a temporary directory and moves it into place only when every
    file arrived, so an interrupted fetch never leaves a half-set behind that a
    later run would treat as valid.
    """
    dest = dest or cached_dir(mm)
    tmp = dest + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    try:
        for name in HEADER_FILES:
            url = _RAW.format(ver=mm, name=name)
            with urllib.request.urlopen(url, timeout=timeout) as r:
                data = r.read()
            if not data:
                raise RuntimeError(f"empty response for {name}")
            with open(os.path.join(tmp, name), "wb") as f:
                f.write(data)
        # Sanity: the set must declare the version we asked for.
        got = header_major(tmp)
        if got is None or f"{got}." not in mm + ".":
            raise RuntimeError(f"fetched headers declare TensorRT {got}, expected {mm}")
        shutil.rmtree(dest, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(tmp, dest)
        return dest
    except (urllib.error.URLError, OSError, RuntimeError):
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def resolve(version=None, allow_fetch=None):
    """Return ``(directory, reason)`` for the headers to compile against.

    ``directory`` is None only if nothing usable exists, which the caller treats
    as "fall back to the bundled set and let the mismatch check decide".
    """
    import sys

    if allow_fetch is None:
        allow_fetch = os.environ.get("KERNEL_LENS_FETCH_HEADERS", "1").lower() not in (
            "0", "false", "no", "off",
        )

    # 1. explicit
    explicit = os.environ.get("TENSORRT_INCLUDE_DIR")
    if explicit and os.path.exists(os.path.join(explicit, "NvInfer.h")):
        return explicit, "TENSORRT_INCLUDE_DIR"

    user = os.path.expanduser("~/tensorrt_headers")
    if os.path.exists(os.path.join(user, "NvInfer.h")):
        return user, "~/tensorrt_headers"

    mm = major_minor(version)
    if mm is None:
        return None, "no TensorRT detected"

    # 2. the installed distribution's own headers, if it has any
    for cand in (
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "local", "include"),
    ):
        if os.path.exists(os.path.join(cand, "NvInfer.h")) and header_major(cand) == int(mm.split(".")[0]):
            return cand, "installed TensorRT headers"

    bundled = _bundled_dir()
    bundled_major = header_major(bundled)

    # Nothing to fetch when the bundled set already matches the runtime.
    if bundled_major is not None and bundled_major == int(mm.split(".")[0]):
        return bundled, f"bundled headers already match TensorRT {mm}"

    # 3. cache
    hit = cached_dir(mm)
    if _is_complete(hit) and header_major(hit) == int(mm.split(".")[0]):
        return hit, f"cached headers for TensorRT {mm}"

    # 4. fetch
    if allow_fetch:
        got = fetch(mm)
        if got:
            return got, f"fetched headers for TensorRT {mm} from the TensorRT OSS repository"
        return None, f"could not fetch headers for TensorRT {mm}"

    return None, f"fetching disabled and no headers for TensorRT {mm}"
