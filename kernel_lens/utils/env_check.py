from __future__ import annotations
import os
import shutil
import importlib.util

def find_nvcc() -> str | None:
    path = shutil.which("nvcc")
    if path and os.path.exists(path):
        return path
    
    candidates = []
    if os.environ.get("CUDA_HOME"):
        candidates.append(os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"))
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc"))
    
    import sys
    candidates.extend([
        os.path.join(sys.prefix, "bin", "nvcc"),
        os.path.join(sys.exec_prefix, "bin", "nvcc"),
        os.path.join(os.path.dirname(sys.executable), "nvcc"),
    ])
    
    candidates.extend([
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-12.4/bin/nvcc",
        "/usr/local/cuda-12.2/bin/nvcc",
        "/usr/local/cuda-12.1/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
        "/usr/bin/nvcc",
    ])
    
    home = os.path.expanduser("~")
    for sub in ["miniconda3", "miniconda", "anaconda3", "anaconda", "TRELLIS.2/miniconda3"]:
        candidates.append(os.path.join(home, sub, "bin", "nvcc"))
        
    for p in candidates:
        if os.path.exists(p):
            nvcc_dir = os.path.dirname(p)
            path_env = os.environ.get("PATH", "")
            if nvcc_dir not in path_env.split(os.pathsep):
                os.environ["PATH"] = f"{nvcc_dir}:{path_env}"
            cuda_home = os.path.dirname(nvcc_dir)
            if "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = cuda_home
            return p
    return None

def tensorrt_major_version():
    """Major version of the installed TensorRT runtime, or None."""
    try:
        import tensorrt as trt
        return int(str(trt.__version__).split(".")[0])
    except Exception:
        return None


def bundled_headers_have_v3() -> bool:
    """Does the header set we compile against actually declare IPluginV3?

    A capability check rather than a version guess: the plugins are compiled
    against the headers bundled with the package (the pip TensorRT wheels ship
    none), so this is the only thing that decides whether a V3 plugin can be
    emitted at all.
    """
    import os as _os
    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    path = _os.path.join(here, "include", "tensorrt", "NvInferPluginBase.h")
    try:
        with open(path, errors="ignore") as f:
            return "class IPluginV3 " in f.read()
    except Exception:
        return False


def active_trt_headers_dir():
    """The TensorRT header directory the builder will compile against.

    Same resolution the builder uses, so the mismatch check and the compile
    agree: an explicit TENSORRT_INCLUDE_DIR, then provisioned/cached headers for
    the installed runtime, then the bundled set.
    """
    import os as _os
    try:
        from .trt_headers import resolve as _resolve
        d, _reason = _resolve()
        if d:
            return d
    except Exception:
        pass
    if _os.environ.get("TENSORRT_INCLUDE_DIR"):
        return _os.environ["TENSORRT_INCLUDE_DIR"]
    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    bundled = _os.path.join(here, "include", "tensorrt")
    return bundled if _os.path.exists(_os.path.join(bundled, "NvInfer.h")) else None


def headers_major_version():
    """Major version of the TensorRT headers we compile against, or None."""
    import os as _os
    import re as _re
    d = active_trt_headers_dir()
    if not d:
        return None
    try:
        with open(_os.path.join(d, "NvInferVersion.h"), errors="ignore") as f:
            txt = f.read()
    except Exception:
        return None
    m = _re.search(r"#define\s+TRT_MAJOR_ENTERPRISE\s+(\d+)", txt)
    if m:
        return int(m.group(1))
    m = _re.search(r"#define\s+NV_TENSORRT_MAJOR\s+(\d+)", txt)
    return int(m.group(1)) if m else None


def check_header_runtime_match():
    """Raise if the headers and the runtime are different TensorRT majors.

    A plugin compiled against 10.x headers and loaded into an 11.x runtime
    segfaults inside TensorRT with no message, which is a bad way to find out.
    """
    hdr = headers_major_version()
    rt = tensorrt_major_version()
    if hdr is None or rt is None or hdr == rt:
        return
    d = active_trt_headers_dir()
    raise RuntimeError(
        f"TensorRT version mismatch: the plugins would be compiled against "
        f"TensorRT {hdr}.x headers ({d}) but loaded into a TensorRT {rt}.x "
        f"runtime. The plugin ABI is not guaranteed across majors, and the "
        f"mismatch crashes inside TensorRT rather than failing cleanly.\n\n"
        f"Fix: point TENSORRT_INCLUDE_DIR at the headers matching the runtime "
        f"(for TensorRT {rt}.x, the SDK's include directory), or install the "
        f"TensorRT {hdr}.x Python bindings so the runtime matches the headers."
    )


def resolve_plugin_api(requested=None):
    """Pick the plugin interface to emit. Returns ``(api, reason)``.

    The matrix this covers:

    ==================  ================  ==================  ============
    TensorRT            IPluginV2*        IPluginV3           emitted
    ==================  ================  ==================  ============
    < 10 (8.6, 9.x)     present           absent              v2
    10.x                deprecated        present             v3
    11+                 **removed**       present             v3
    ==================  ================  ==================  ============

    ``requested`` is ``"auto"`` (the default), ``"v2"`` or ``"v3"``; the
    ``KERNEL_LENS_PLUGIN_API`` environment variable supplies it when the caller
    passes nothing. An explicit request is honoured unless it cannot work, in
    which case the error says why.
    """
    import os as _os
    if requested is None:
        requested = _os.environ.get("KERNEL_LENS_PLUGIN_API", "auto")
    requested = str(requested).strip().lower()
    if requested not in ("auto", "v2", "v3"):
        raise ValueError(
            f"plugin_api must be 'auto', 'v2' or 'v3', got {requested!r}"
        )

    have_v3 = bundled_headers_have_v3()
    major = tensorrt_major_version()

    if requested == "v3":
        if not have_v3:
            raise RuntimeError(
                "plugin_api='v3' was requested, but the bundled TensorRT headers "
                "do not declare IPluginV3. They are older than TensorRT 10."
            )
        return "v3", "requested"

    if requested == "v2":
        if major is not None and major >= 11:
            raise RuntimeError(
                f"plugin_api='v2' was requested, but TensorRT {major} removed the "
                "IPluginV2 interfaces. Use plugin_api='v3' (or 'auto')."
            )
        return "v2", "requested"

    # auto
    if major is None:
        return ("v3" if have_v3 else "v2"), "auto: TensorRT not importable"
    if major >= 11:
        if not have_v3:
            raise RuntimeError(
                f"TensorRT {major} needs an IPluginV3 plugin, but the bundled "
                "headers are too old to emit one. Upgrade kernel-lens."
            )
        return "v3", f"auto: TensorRT {major} has no IPluginV2 interfaces"
    if major >= 10 and have_v3:
        return "v3", f"auto: TensorRT {major} supports IPluginV3 (IPluginV2 is deprecated)"
    return "v2", f"auto: TensorRT {major} predates IPluginV3"


def check_environment(backends: list[str], plugin_api: str = "v2"):
    """
    Validates the system environment for the requested backends before compilation begins.
    Fails fast with actionable advice if dependencies are missing.
    """
    # 1. Base C++ Compilation Requirements
    if not find_nvcc():
        raise EnvironmentError(
            "'nvcc' not found. The CUDA toolkit must be installed and in your PATH to compile Triton PTX."
        )

    if not shutil.which("g++") and not os.path.exists("/usr/bin/g++"):
        raise EnvironmentError(
            "'g++' not found. A C++ compiler is required to link the shared libraries."
        )

    # 2. ONNX Runtime Checks
    if "onnx" in backends:
        if not importlib.util.find_spec("onnxruntime"):
            raise EnvironmentError(
                "'onnxruntime' is not installed. \n"
                "Fix: Run `pip install kernel-lens[ort]` or `pip install onnxruntime-gpu`"
            )
        
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            from ..config import debug_print
            debug_print("[Warning] ONNX Runtime is installed, but CUDAExecutionProvider is missing. Inference will fall back to CPU.")

    # 3. TensorRT Checks
    if "tensorrt" in backends:
        check_header_runtime_match()
        if not importlib.util.find_spec("tensorrt"):
            raise EnvironmentError(
                "'tensorrt' Python bindings are not installed. \n"
                "Fix: Run `pip install kernel-lens[trt]` or `pip install tensorrt`"
            )

        import tensorrt as trt
        try:
            trt_major = int(trt.__version__.split(".")[0])
            # Only a problem when we would emit IPluginV2 plugins. With the V3
            # emitter selected, TensorRT 11 is supported rather than refused.
            if trt_major >= 11 and plugin_api != "v3":
                raise RuntimeError(
                    "\n" + "=" * 80 + "\n"
                    f"TENSORRT COMPILATION ERROR: TensorRT {trt.__version__} is installed, but TensorRT 11+ removed legacy IPluginV2 C++ plugin interfaces.\n\n"
                    "To resolve this issue:\n"
                    "1. Install TensorRT 10.x Python bindings:\n"
                    "   pip install 'tensorrt<11' 'tensorrt-cu12-libs<11'\n\n"
                    "2. Or emit IPluginV3 plugins instead, which TensorRT 11 "
                    "still supports:\n"
                    "   kl.compile(..., plugin_api='v3')   # or plugin_api='auto'\n\n"
                    "3. Or use the ONNX Runtime backend by passing "
                    "backends=['onnx'] to kl.compile().\n"
                    + "=" * 80
                )
        except (ValueError, IndexError):
            pass

        # Check if TRT C++ libraries are likely in the system path (simple heuristic)
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "tensorrt" not in ld_lib_path.lower() and not os.path.exists("/usr/lib/x86_64-linux-gnu/libnvinfer.so"):
            from ..config import debug_print
            debug_print("⚠️ [Warning] libnvinfer.so not explicitly found in standard paths or LD_LIBRARY_PATH. The g++ linking phase may fail.")