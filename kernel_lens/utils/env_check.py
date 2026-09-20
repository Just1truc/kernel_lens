import os
import shutil
import importlib.util

def check_environment(backends: list[str]):
    """
    Validates the system environment for the requested backends before compilation begins.
    Fails fast with actionable advice if dependencies are missing.
    """
    # 1. Base C++ Compilation Requirements
    if not shutil.which("nvcc"):
        cuda_paths = [
            "/usr/local/cuda/bin/nvcc",
            "/usr/local/cuda-12/bin/nvcc",
            "/usr/local/cuda-12.4/bin/nvcc",
            "/usr/local/cuda-12.2/bin/nvcc",
            "/usr/local/cuda-12.1/bin/nvcc",
            "/usr/local/cuda-11/bin/nvcc",
            "/usr/bin/nvcc",
        ]
        if os.environ.get("CUDA_HOME"):
            cuda_paths.insert(0, os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"))
        
        found_nvcc = None
        for p in cuda_paths:
            if os.path.exists(p):
                found_nvcc = p
                nvcc_dir = os.path.dirname(p)
                os.environ["PATH"] = f"{nvcc_dir}:{os.environ.get('PATH', '')}"
                break
        
        if not found_nvcc:
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

        # Check if TRT C++ libraries are likely in the system path (simple heuristic)
        ld_lib_path = os.environ.get("LD_LIBRARY_PATH", "")
        if "tensorrt" not in ld_lib_path.lower() and not os.path.exists("/usr/lib/x86_64-linux-gnu/libnvinfer.so"):
            from ..config import debug_print
            debug_print("⚠️ [Warning] libnvinfer.so not explicitly found in standard paths or LD_LIBRARY_PATH. The g++ linking phase may fail.")