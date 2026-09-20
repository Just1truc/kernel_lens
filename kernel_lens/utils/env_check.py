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

def check_environment(backends: list[str]):
    """
    Validates the system environment for the requested backends before compilation begins.
    Fails fast with actionable advice if dependencies are missing.
    """
    # 1. Base C++ Compilation Requirements
    if not find_nvcc():
        raise EnvironmentError(
            "❌ 'nvcc' not found. The CUDA toolkit must be installed and in your PATH to compile Triton PTX."
        )

    if not shutil.which("g++") and not os.path.exists("/usr/bin/g++"):
        raise EnvironmentError(
            "❌ 'g++' not found. A C++ compiler is required to link the shared libraries."
        )

    # 2. ONNX Runtime Checks
    if "onnx" in backends:
        if not importlib.util.find_spec("onnxruntime"):
            raise EnvironmentError(
                "❌ 'onnxruntime' is not installed. \n"
                "💡 Fix: Run `pip install kernel-lens[ort]` or `pip install onnxruntime-gpu`"
            )
        
        import onnxruntime as ort
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            from ..config import debug_print
            debug_print("⚠️ [Warning] ONNX Runtime is installed, but CUDAExecutionProvider is missing. Inference will fall back to CPU.")

    # 3. TensorRT Checks
    if "tensorrt" in backends:
        if not importlib.util.find_spec("tensorrt"):
            raise EnvironmentError(
                "❌ 'tensorrt' Python bindings are not installed. \n"
                "💡 Fix: Run `pip install kernel-lens[trt]` or `pip install tensorrt`"
            )

        import tensorrt as trt
        try:
            trt_major = int(trt.__version__.split(".")[0])
            if trt_major >= 11:
                raise RuntimeError(
                    "\n" + "=" * 80 + "\n"
                    f"❌ TENSORRT COMPILATION ERROR: TensorRT {trt.__version__} is installed, but TensorRT 11+ removed legacy IPluginV2 C++ plugin interfaces.\n\n"
                    "To resolve this issue:\n"
                    "1. Install TensorRT 10.x Python bindings:\n"
                    "   pip install 'tensorrt<11' 'tensorrt-cu12-libs<11'\n\n"
                    "2. Or use the ONNX Runtime backend by passing backends=['onnx'] to kl.compile().\n"
                    + "=" * 80
                )
        except (ValueError, IndexError):
            pass

        # Check if TRT C++ libraries are likely in the system path (simple heuristic)
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "tensorrt" not in ld_lib_path.lower() and not os.path.exists("/usr/lib/x86_64-linux-gnu/libnvinfer.so"):
            from ..config import debug_print
            debug_print("⚠️ [Warning] libnvinfer.so not explicitly found in standard paths or LD_LIBRARY_PATH. The g++ linking phase may fail.")