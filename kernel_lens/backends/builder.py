import os
import subprocess
import urllib.request
import tarfile
import onnxruntime

def build_ort_plugin(ort_plugins_dir: str, cache_dir: str):
    """
    Natively compiles the generated C++ files into a Shared Library (.so)
    replacing the need for an external bash script.
    """
    # 1. Get exact ORT version from the current Python environment
    ort_version = onnxruntime.__version__.split('+')[0]
    # print(f"     [Builder] Detected ONNX Runtime v{ort_version}")
    
    ort_release_dir = os.path.join(cache_dir, f"onnxruntime-linux-x64-gpu-{ort_version}")
    ort_tgz = f"{ort_release_dir}.tgz"
    
    # 2. Download exact matching C++ Developer Release if missing
    if not os.path.exists(ort_release_dir):
        url = f"https://github.com/microsoft/onnxruntime/releases/download/v{ort_version}/onnxruntime-linux-x64-gpu-{ort_version}.tgz"
        # print(f"     [Builder] Downloading ORT C++ headers from {url}...")
        urllib.request.urlretrieve(url, ort_tgz)
        with tarfile.open(ort_tgz, "r:gz") as tar:
            tar.extractall(path=cache_dir)
            
    ort_inc = os.path.join(ort_release_dir, "include")
    ort_lib = os.path.join(ort_release_dir, "lib")
    
    # # 3. Dynamically find CUDA paths via nvcc
    # print("     [Builder] Querying system for CUDA configuration...")
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        # Fallback to standard Linux path
        cuda_home = "/usr/local/cuda"
        
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    # print(f"     [Builder] Detected CUDA at {cuda_home}")
    
    # 4. Compile the .cu files into object files
    cu_files = [f for f in os.listdir(ort_plugins_dir) if f.endswith(".cu")]
    obj_files = []
    
    for cu_file in cu_files:
        cu_path = os.path.join(ort_plugins_dir, cu_file)
        obj_path = os.path.join(ort_plugins_dir, cu_file.replace(".cu", ".o"))
        obj_files.append(obj_path)
        
        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", "-Xcompiler", "-fPIC",
            f"-I{ort_inc}", f"-I{cuda_inc}"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # 5. Compile the registrar (register_ops.cpp)
    reg_cpp = os.path.join(ort_plugins_dir, "register_ops.cpp")
    reg_obj = os.path.join(ort_plugins_dir, "register_ops.o")
    obj_files.append(reg_obj)
    
    cmd = [
        "g++", "-c", reg_cpp, "-o", reg_obj, "-O3", "-fPIC",
        f"-I{ort_inc}", f"-I{cuda_inc}"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 6. Link everything into the final .so library (with RPATH baked in)
    so_path = os.path.join(ort_plugins_dir, "libtriton_ort_plugins.so")
    abs_ort_lib = os.path.abspath(ort_lib)
    
    cmd = [
        "g++", "-shared", "-o", so_path
    ] + obj_files + [
        f"-L{ort_lib}", "-lonnxruntime",
        f"-L{cuda_lib}", "-lcuda", "-lcudart",
        f"-Wl,-rpath,{abs_ort_lib}"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # print(f"     [Builder] Compilation successful! Plugin saved to {so_path}")

def build_trt_plugin(trt_plugins_dir: str, cache_dir: str):
    """
    Natively compiles the generated C++ files into a TensorRT Shared Library (.so).
    """
    # print("     [Builder] Querying system for CUDA configuration...")
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
    except Exception:
        cuda_home = "/usr/local/cuda"
        
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")
    
    # In TRT 8.6+, the library is often split into nvinfer and nvinfer_plugin
    # We will assume standard system paths for TRT (/usr/lib/x86_64-linux-gnu or LD_LIBRARY_PATH)
    
    cu_files = [f for f in os.listdir(trt_plugins_dir) if f.endswith(".cu")]
    obj_files = []
    
    for cu_file in cu_files:
        cu_path = os.path.join(trt_plugins_dir, cu_file)
        obj_path = os.path.join(trt_plugins_dir, cu_file.replace(".cu", ".o"))
        obj_files.append(obj_path)
        
        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", "-Xcompiler", "-fPIC",
            f"-I{cuda_inc}", "-Wno-deprecated-gpu-targets"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # subprocess.run(cmd, check=True)
        
    so_path = os.path.join(trt_plugins_dir, "libtriton_trt_plugins.so")
    
    cmd = [
        "g++", "-shared", "-o", so_path
    ] + obj_files + [
        f"-L{cuda_lib}", "-lcuda", "-lcudart", "-lnvinfer"
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] TensorRT compilation failed. Ensure TensorRT is installed and in your LD_LIBRARY_PATH.")
        raise e
        
    # print(f"     [Builder] TRT Compilation successful! Plugin saved to {so_path}")