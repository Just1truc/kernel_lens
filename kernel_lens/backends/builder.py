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
    # 1. Prepare ORT release include/lib directory using stable C++ developer package
    ort_release_dir = os.path.join(cache_dir, "onnxruntime-linux-x64-gpu-1.20.1")
    ort_inc = os.path.join(ort_release_dir, "include")
    ort_lib = os.path.join(ort_release_dir, "lib")
    
    if not os.path.exists(os.path.join(ort_inc, "onnxruntime_cxx_api.h")):
        tgz_path = os.path.join(cache_dir, "ort_1.20.1.tgz")
        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz"
        try:
            subprocess.run(["curl", "-sL", url, "-o", tgz_path], check=True)
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=cache_dir)
        except Exception as e:
            print(f"[Builder Warning] Failed to download ORT headers: {e}")



    
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
        
        try:
            import torch
            cap = torch.cuda.get_device_capability(0)
            major, minor = cap[0], cap[1]
            if major >= 10:
                major, minor = 9, 0
            arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        except Exception:
            arch_flag = "-gencode=arch=compute_75,code=sm_75"

        user_trt_inc = os.path.expanduser("~/tensorrt_headers")
        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", arch_flag, "-Xcompiler", "-fPIC",
            f"-I{ort_inc}", f"-I{cuda_inc}", "-I/usr/include", "-Wno-deprecated-gpu-targets"
        ]

        if os.path.exists(user_trt_inc):
            cmd.insert(-1, f"-I{user_trt_inc}")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and "Unsupported gpu architecture" in res.stderr:
            # Fallback to compute_80 / sm_80
            cmd = [c.replace(arch_flag, "-gencode=arch=compute_80,code=sm_80") for c in cmd]
            res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[NVCC ERROR] {res.stderr}")
            res.check_returncode()

        
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
        
        try:
            import torch
            cap = torch.cuda.get_device_capability(0)
            major, minor = cap[0], cap[1]
            if major >= 10:
                major, minor = 9, 0
            arch_flag = f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"
        except Exception:
            arch_flag = "-gencode=arch=compute_75,code=sm_75"

        user_trt_inc = os.path.expanduser("~/tensorrt_headers")
        inc_flags = []
        if os.path.exists(user_trt_inc):
            inc_flags.append(f"-I{user_trt_inc}")
        inc_flags.extend([f"-I{cuda_inc}", "-I/usr/include"])

        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", arch_flag, "-Xcompiler", "-fPIC",
            "-Wno-deprecated-gpu-targets"
        ] + inc_flags

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and "Unsupported gpu architecture" in res.stderr:
            cmd = [c.replace(arch_flag, "-gencode=arch=compute_80,code=sm_80") for c in cmd]
            res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[NVCC ERROR] {res.stderr}")
            res.check_returncode()


    so_path = os.path.join(trt_plugins_dir, "libtriton_trt_plugins.so")
    
    trt_lib_dirs = []
    try:
        import tensorrt
        trt_dir = os.path.dirname(tensorrt.__file__)
        for candidate in [trt_dir, os.path.join(os.path.dirname(trt_dir), "tensorrt_libs"), os.path.join(os.path.dirname(trt_dir), "tensorrt_cu12_libs"), os.path.join(os.path.dirname(trt_dir), "tensorrt_cu13_libs")]:
            if os.path.exists(candidate):
                trt_lib_dirs.append(candidate)
    except Exception:
        pass



    extra_link_args = []
    for d in trt_lib_dirs:
        extra_link_args.extend([f"-L{d}", f"-Wl,-rpath,{d}"])

    cmd = [
        "g++", "-shared", "-o", so_path
    ] + obj_files + [
        f"-L{cuda_lib}", "-lcuda", "-lcudart"
    ] + extra_link_args + ["-lnvinfer"]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("\n[ERROR] TensorRT compilation failed. Link command:", " ".join(cmd))
        raise e
        
    # print(f"     [Builder] TRT Compilation successful! Plugin saved to {so_path}")