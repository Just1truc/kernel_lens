import os
import torch
import triton
from torch.fx.experimental.proxy_tensor import make_fx
from typing import Optional
import warnings

from .tracer import extract_manifests
from .ast_analyzer import analyze_grid_asts
from .interaction import InteractionHandler
from .onnx_exporter import TritonGlobalONNXExporter

# Backend Generators
from ..backends.ort_gen import generate_ort_bindings
from ..backends.trt_gen import generate_trt_bindings
from ..utils.env_check import check_environment

# Builders
from ..backends.builder import build_ort_plugin, build_trt_plugin

# Runtime
from ..runtime.engine import CompiledModel

_orig_next_power_of_2 = triton.next_power_of_2

def _dynamic_next_power_of_2(n):
    # Check if we are dealing with a PyTorch Symbolic Variable during a trace
    if "SymInt" in str(type(n)) or isinstance(n, torch.Tensor):
        # 2 ^ ceil(log2(n)) -> This creates dynamic nodes in the ONNX graph!
        n_tensor = torch.as_tensor(n, dtype=torch.float32)
        power = torch.ceil(torch.log2(n_tensor))
        return torch.pow(2, power).to(torch.int64)
    
    # Normal execution fallback
    return _orig_next_power_of_2(n)

triton.next_power_of_2 = _dynamic_next_power_of_2

def _get_cache_dir(model_name: str, inputs: tuple) -> str:
    home_dir = os.path.expanduser("~")
    cache_path = os.path.join(home_dir, ".kernel_lens_cache", model_name)
    os.makedirs(cache_path, exist_ok=True)
    return cache_path

def compile(
    model: torch.nn.Module, 
    inputs: tuple, 
    name: str = "custom_model", 
    backends: list[str] = ["onnx", "tensorrt"],
    interaction_handler: Optional[InteractionHandler] = None
) -> CompiledModel:
    
    check_environment(backends)
    
    cache_dir = _get_cache_dir(name, inputs)
    
    manifests = extract_manifests(model, inputs)
    
    if not manifests:
        return CompiledModel(cache_dir, name, backends)

    manifests = analyze_grid_asts(manifests, handler=interaction_handler)
    
    # 1. Base ONNX Export (Needed by BOTH ORT and TRT)
    onnx_path = os.path.join(cache_dir, f"{name}.onnx")
    if not os.path.exists(onnx_path):
        _export_to_onnx(model, inputs, onnx_path, manifests)
    
    # 2. Compile ONNX Runtime Plugins
    if "onnx" in backends:
        ort_plugins_dir = os.path.join(cache_dir, "ort_plugins")
        generate_ort_bindings(manifests, ort_plugins_dir)
        build_ort_plugin(ort_plugins_dir, cache_dir)

    # 3. Compile TensorRT Plugins
    if "tensorrt" in backends:
        trt_plugins_dir = os.path.join(cache_dir, "trt_plugins")
        generate_trt_bindings(manifests, trt_plugins_dir)
        build_trt_plugin(trt_plugins_dir, cache_dir)

    return CompiledModel(cache_dir, name, backends)

def load(name: str) -> CompiledModel:
    """
    Loads a previously compiled model from the cache without recompiling.
    """
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".kernel_lens_cache", name)
    
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Model '{name}' not found in cache. Did you compile it?")
        
    backends = []
    if os.path.exists(os.path.join(cache_dir, "ort_plugins", "libtriton_ort_plugins.so")):
        backends.append("onnx")
    if os.path.exists(os.path.join(cache_dir, "trt_plugins", "libtriton_trt_plugins.so")):
        backends.append("tensorrt")
        
    if not backends:
        raise RuntimeError(f"Cache for '{name}' exists, but no compiled backend plugins were found.")
        
    return CompiledModel(cache_dir, name, backends)

def _export_to_onnx(model, inputs, output_path, manifests):
    """
    Exports the PyTorch model to ONNX. 
    Uses the TritonGlobalONNXExporter to bypass Triton and inject custom nodes.
    """
    # Dry-run the model to dynamically count the outputs
    with torch.no_grad():
        dummy_out = model(*inputs)
        
    if isinstance(dummy_out, torch.Tensor):
        out_names = ["output_0"]
    elif isinstance(dummy_out, (list, tuple)):
        out_names = [f"output_{i}" for i in range(len(dummy_out))]
    elif isinstance(dummy_out, dict):
        out_names = list(dummy_out.keys())
    else:
        out_names = ["output_0"]

    with TritonGlobalONNXExporter(manifests):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python.*")
            
            torch.onnx.export(
                model,
                inputs,
                output_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=False, 
                input_names=[f"input_{i}" for i in range(len(inputs))],
                output_names=out_names # <--- DYNAMIC MULTI-OUTPUT MAPPING
            )

# def _export_to_onnx(model, inputs, output_path, manifests):
#     """
#     Exports the PyTorch model to ONNX. 
#     Uses the TritonGlobalONNXExporter to bypass Triton and inject custom nodes.
#     """
#     # Wrap the export in our protective patch!
#     with TritonGlobalONNXExporter(manifests):
#         with warnings.catch_warnings():
#             warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
#             warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python.*")
            
#             torch.onnx.export(
#                 model,
#                 inputs,
#                 output_path,
#                 export_params=True,
#                 opset_version=17,
#                 do_constant_folding=False, 
#                 input_names=[f"input_{i}" for i in range(len(inputs))],
#                 output_names=["output_0"] 
#             )