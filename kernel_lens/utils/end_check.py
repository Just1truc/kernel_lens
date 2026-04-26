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
        raise EnvironmentError(
            "❌ 'nvcc' not found. The CUDA toolkit must be installed and in your PATH to compile Triton PTX."
        )
    if not shutil.which("g++"):
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
            print("⚠️ [Warning] ONNX Runtime is installed, but CUDAExecutionProvider is missing. Inference will fall back to CPU.")

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
            print("⚠️ [Warning] libnvinfer.so not explicitly found in standard paths or LD_LIBRARY_PATH. The g++ linking phase may fail.")