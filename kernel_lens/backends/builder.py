import os
import subprocess
import urllib.request
import tarfile
import shutil

from ..utils.env_check import find_nvcc

def get_cuda_home() -> str:
    nvcc_path = find_nvcc()
    if nvcc_path:
        return os.path.dirname(os.path.dirname(nvcc_path))
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    if os.environ.get("CONDA_PREFIX"):
        return os.environ["CONDA_PREFIX"]
    import sys
    return sys.prefix

def ensure_cuda_compat():
    """
    Checks if Conda's CUDA toolkit crt/math_functions.h has the known glibc 2.38+ incompatibility bug
    (where cospi, sinpi, rsqrt, cospif, sinpif, rsqrtf lack __THROW/noexcept).
    If present and writable, patches crt/math_functions.h in place.
    """
    try:
        search_dirs = []
        nvcc_path = find_nvcc()
        if nvcc_path:
            search_dirs.append(os.path.dirname(os.path.dirname(nvcc_path)))
        if os.environ.get("CUDA_HOME"):
            search_dirs.append(os.environ["CUDA_HOME"])
        if os.environ.get("CONDA_PREFIX"):
            search_dirs.append(os.environ["CONDA_PREFIX"])
        import sys
        search_dirs.append(sys.prefix)
        home = os.path.expanduser("~")
        for sub in ["miniconda3", "miniconda", "anaconda3", "anaconda", "TRELLIS.2/miniconda3"]:
            search_dirs.append(os.path.join(home, sub))
        
        for base_dir in search_dirs:
            math_hdr = os.path.join(base_dir, "include", "crt", "math_functions.h")
            if os.path.exists(math_hdr) and os.access(math_hdr, os.W_OK):
                with open(math_hdr, "r") as f:
                    content = f.read()
                if "double                 cospi(double x);" in content:
                    content = content.replace("double                 cospi(double x);", "double                 cospi(double x) __THROW;")
                    content = content.replace("double                 sinpi(double x);", "double                 sinpi(double x) __THROW;")
                    content = content.replace("double                 rsqrt(double x);", "double                 rsqrt(double x) __THROW;")
                    content = content.replace("float                  cospif(float x);", "float                  cospif(float x) __THROW;")
                    content = content.replace("float                  sinpif(float x);", "float                  sinpif(float x) __THROW;")
                    content = content.replace("float                  rsqrtf(float x);", "float                  rsqrtf(float x) __THROW;")
                    with open(math_hdr, "w") as f:
                        f.write(content)
    except Exception:
        pass

def build_ort_plugin(ort_plugins_dir: str, cache_dir: str):
    """
    Natively compiles the generated C++ files into a Shared Library (.so)
    replacing the need for an external bash script.
    """
    ensure_cuda_compat()
    # 1. Prepare ORT release include/lib directory using stable C++ developer package
    parent_cache = os.path.dirname(cache_dir)
    ort_release_dir = os.path.join(parent_cache, "onnxruntime-linux-x64-gpu-1.20.1")
    ort_inc = os.path.join(ort_release_dir, "include")
    ort_lib = os.path.join(ort_release_dir, "lib")
    
    if not os.path.exists(os.path.join(ort_inc, "onnxruntime_cxx_api.h")):
        tgz_path = os.path.join(parent_cache, "ort_1.20.1.tgz")
        url = "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz"
        try:
            if not os.path.exists(tgz_path):
                subprocess.run(["curl", "-sL", url, "-o", tgz_path], check=True)
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(path=parent_cache)
        except Exception as e:
            print(f"[Builder Warning] Failed to download ORT headers: {e}")

    cuda_inc = "/usr/local/cuda/include"
    try:
        import triton
        t_inc = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "include")
        if os.path.exists(os.path.join(t_inc, "cuda.h")):
            cuda_inc = t_inc
    except Exception:
        pass

    cuda_home = get_cuda_home()
    if cuda_inc == "/usr/local/cuda/include":
        cand = os.path.join(cuda_home, "include")
        if os.path.exists(os.path.join(cand, "cuda.h")) and not os.path.exists(os.path.join(cand, "crt", "math_functions.h")):
            cuda_inc = cand

    cuda_lib = "/usr/local/cuda/lib64"
    for cand in [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "lib"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64"
    ]:
        if os.path.exists(cand) and any(f.startswith("libcudart") for f in os.listdir(cand)):
            cuda_lib = cand
            break

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
        cpp_compiler = "g++-13" if os.path.exists("/usr/bin/g++-13") else "g++"
        ccbin_flag = ["-ccbin", cpp_compiler] if os.path.exists(f"/usr/bin/{cpp_compiler}") else []

        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", arch_flag, "-Xcompiler", "-fPIC", "-Xcompiler", "-D_GNU_SOURCE",
            "-allow-unsupported-compiler",
            f"-I{ort_inc}", "-Wno-deprecated-gpu-targets"
        ] + ccbin_flag

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
        cpp_compiler, "-c", reg_cpp, "-o", reg_obj, "-O3", "-fPIC",
        f"-I{ort_inc}", f"-I{cuda_inc}"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # 6. Link everything into the final .so library (with RPATH baked in)
    so_path = os.path.join(ort_plugins_dir, "libtriton_ort_plugins.so")
    abs_ort_lib = os.path.abspath(ort_lib)
    
    cmd = [
        cpp_compiler, "-shared", "-o", so_path
    ] + obj_files + [
        f"-L{ort_lib}", "-lonnxruntime",
        f"-L{cuda_lib}", "-lcuda", "-lcudart",
        f"-Wl,-rpath,{abs_ort_lib}", f"-Wl,-rpath,{cuda_lib}"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_trt_plugin(trt_plugins_dir: str, cache_dir: str):
    """
    Natively compiles the generated C++ files into a TensorRT Shared Library (.so).
    """
    ensure_cuda_compat()
    cuda_home = get_cuda_home()
    cuda_inc = os.path.join(cuda_home, "include")
    cuda_lib = "/usr/local/cuda/lib64"
    for cand in [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "lib"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "lib"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64"
    ]:
        if os.path.exists(cand) and any(f.startswith("libcudart") for f in os.listdir(cand)):
            cuda_lib = cand
            break
            
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

        trt_inc_dirs = []
        if os.environ.get("TENSORRT_INCLUDE_DIR"):
            trt_inc_dirs.append(os.environ["TENSORRT_INCLUDE_DIR"])
        user_trt_inc = os.path.expanduser("~/tensorrt_headers")
        if os.path.exists(user_trt_inc):
            trt_inc_dirs.append(user_trt_inc)

        import sys
        pkg_inc = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "include", "tensorrt"))
        search_candidates = [
            pkg_inc,
            os.path.join(sys.prefix, "include"),
            os.path.join(sys.prefix, "local", "include"),
            os.path.join(cuda_home, "include"),
            "/usr/local/cuda/include",
            "/usr/local/cuda-12/include",
            "/usr/local/cuda-12.4/include",
            "/usr/local/cuda-12.2/include",
            "/usr/local/cuda-12.1/include",
            "/usr/local/cuda-11/include",
            "/usr/include",
            "/usr/local/include",
            "/usr/include/x86_64-linux-gnu",
            "/opt/tensorrt/include",
            "/usr/local/tensorrt/include",
            "/usr/include/tensorrt",
        ]

        for site_p in sys.path:
            if "site-packages" in site_p or "dist-packages" in site_p:
                search_candidates.extend([
                    os.path.join(site_p, "tensorrt", "include"),
                    os.path.join(site_p, "tensorrt_libs", "include"),
                    os.path.join(site_p, "tensorrt_cu12_libs", "include"),
                    os.path.join(site_p, "tensorrt_cu13_libs", "include"),
                    os.path.join(site_p, "tensorrt_cu11_libs", "include"),
                    os.path.join(site_p, "tensorrt_include"),
                ])

        try:
            import tensorrt
            trt_pkg_dir = os.path.dirname(tensorrt.__file__)
            parent_dir = os.path.dirname(trt_pkg_dir)
            search_candidates.extend([
                os.path.join(trt_pkg_dir, "include"),
                os.path.join(parent_dir, "tensorrt_libs", "include"),
                os.path.join(parent_dir, "tensorrt_cu12_libs", "include"),
                os.path.join(parent_dir, "tensorrt_cu13_libs", "include"),
            ])
        except Exception:
            pass

        for c in search_candidates:
            if os.path.exists(c) and c not in trt_inc_dirs:
                trt_inc_dirs.append(c)

        has_nvinfer = any(os.path.exists(os.path.join(d, "NvInfer.h")) for d in trt_inc_dirs)
        has_nvinfer_plugin = any(os.path.exists(os.path.join(d, "NvInferPlugin.h")) for d in trt_inc_dirs)

        if not (has_nvinfer and has_nvinfer_plugin):
            missing = []
            if not has_nvinfer: missing.append("NvInfer.h")
            if not has_nvinfer_plugin: missing.append("NvInferPlugin.h")
            missing_str = " and ".join(missing)
            raise RuntimeError(
                "\n" + "=" * 80 + "\n"
                f"TENSORRT COMPILATION ERROR: TensorRT C++ header file(s) ({missing_str}) not found!\n\n"
                "KernelLens requires TensorRT C++ headers (NvInfer.h & NvInferPlugin.h) to compile TensorRT plugins.\n\n"
                "To resolve this issue:\n"
                "1. If TensorRT C++ headers are installed on your system, specify their location via:\n"
                "   export TENSORRT_INCLUDE_DIR=/path/to/tensorrt/include\n\n"
                "2. Or install TensorRT Python libraries with headers:\n"
                "   pip install tensorrt-cu12-libs  (or pip install tensorrt_libs)\n\n"
                "3. Or copy NvInfer.h and NvInferPlugin.h into ~/tensorrt_headers/\n\n"
                "4. If you only wish to use ONNX Runtime backend, pass backends=['onnx'] to kl.compile().\n"
                + "=" * 80
            )

        inc_flags = [
            f"-I{d}" for d in trt_inc_dirs
            if (os.path.exists(os.path.join(d, "NvInfer.h")) or os.path.exists(os.path.join(d, "NvInferPlugin.h")))
            and not os.path.exists(os.path.join(d, "crt", "math_functions.h"))
        ]

        cpp_compiler = "g++-13" if os.path.exists("/usr/bin/g++-13") else "g++"
        ccbin_flag = ["-ccbin", cpp_compiler] if os.path.exists(f"/usr/bin/{cpp_compiler}") else []

        cmd = [
            "nvcc", "-c", cu_path, "-o", obj_path, "-O3", arch_flag, "-Xcompiler", "-fPIC", "-Xcompiler", "-D_GNU_SOURCE",
            "-allow-unsupported-compiler",
            "-Wno-deprecated-gpu-targets",
            "-diag-suppress", "997"
        ] + ccbin_flag + inc_flags

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and "Unsupported gpu architecture" in res.stderr:
            cmd = [c.replace(arch_flag, "-gencode=arch=compute_80,code=sm_80") for c in cmd]
            res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[NVCC ERROR] {res.stderr}")
            if "No such file or directory" in res.stderr and ("NvInfer" in res.stderr or "NvInferPlugin" in res.stderr):
                raise RuntimeError(
                    "\n" + "=" * 80 + "\n"
                    "TENSORRT COMPILATION ERROR: NvInferPlugin.h or NvInfer.h header file not found during compilation!\n\n"
                    "To resolve this issue:\n"
                    "1. Set environment variable: export TENSORRT_INCLUDE_DIR=/path/to/tensorrt/include\n"
                    "2. Or install: pip install tensorrt-cu12-libs\n"
                    "3. Or copy NvInfer.h and NvInferPlugin.h to ~/tensorrt_headers/\n"
                    + "=" * 80
                )
            raise RuntimeError(f"NVCC Compilation Failed:\n{res.stderr}")

    # Compile register_plugins.cpp
    reg_cpp = os.path.join(trt_plugins_dir, "register_plugins.cpp")
    if os.path.exists(reg_cpp):
        reg_obj = os.path.join(trt_plugins_dir, "register_plugins.o")
        inc_flags_cpp = [
            f"-I{d}" for d in trt_inc_dirs
            if (os.path.exists(os.path.join(d, "NvInfer.h")) or os.path.exists(os.path.join(d, "NvInferPlugin.h")))
            and not os.path.exists(os.path.join(d, "crt", "math_functions.h"))
        ]
        cmd_cpp = [
            cpp_compiler, "-c", reg_cpp, "-o", reg_obj, "-O3", "-fPIC", f"-I{cuda_inc}"
        ] + inc_flags_cpp
        subprocess.run(cmd_cpp, check=True)
        obj_files.append(reg_obj)

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
    nvinfer_link_flag = ["-lnvinfer"]
    for d in trt_lib_dirs:
        extra_link_args.extend([f"-L{d}", f"-Wl,-rpath,{d}"])
        if not os.path.exists(os.path.join(d, "libnvinfer.so")):
            for f in os.listdir(d):
                if f.startswith("libnvinfer.so."):
                    nvinfer_link_flag = [os.path.join(d, f)]
                    break

    cpp_compiler = "g++-13" if os.path.exists("/usr/bin/g++-13") else "g++"
    cmd = [
        cpp_compiler, "-shared", "-o", so_path
    ] + obj_files + [
        f"-L{cuda_lib}", "-lcuda", "-lcudart", f"-Wl,-rpath,{cuda_lib}"
    ] + extra_link_args + nvinfer_link_flag
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("\n[ERROR] TensorRT compilation failed. Link command:", " ".join(cmd))
        print("LINK STDOUT:", res.stdout)
        print("LINK STDERR:", res.stderr)
        raise RuntimeError(f"TensorRT linking failed: {res.stderr}")
        
    # print(f"     [Builder] TRT Compilation successful! Plugin saved to {so_path}")