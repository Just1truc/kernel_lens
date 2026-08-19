import os
import re
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
from ..config import debug_print

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

def is_nhwc(tensor_shape, tensor_strides):
    # For a 4D tensor (N, C, H, W)
    # NHWC strides should be: (C*H*W, 1, W*C, C)
    N, C, H, W = tensor_shape
    expected_nhwc = (C*H*W, 1, W*C, C)
    # Allow for some padding/alignment variations if necessary
    return tensor_strides[1] == 1

def validate_manifests(manifests):
    debug_print("DEBUG: Validating manifests...")
    for m in manifests:
        for arg in m.arguments:
            if hasattr(arg, 'strides') and arg.strides and len(arg.shape) == 4:
                # 1. THE STRIDE CHECK (Now with Layout Intelligence)
                N, C, H, W = [int(d) for d in arg.shape]
                actual_strides = tuple(arg.strides)
                
                # Standard NCHW (Contiguous)
                # Strides: (C*H*W, H*W, W, 1)
                expected_nchw = (C*H*W, H*W, W, 1)
                
                # Channels Last NHWC (Contiguous)
                # Strides: (H*W*C, 1, W*C, C)
                expected_nhwc = (H*W*C, 1, W*C, C)
                
                is_standard = actual_strides == expected_nchw
                is_channels_last = actual_strides == expected_nhwc
                
                if not (is_standard or is_channels_last):
                    raise ValueError(
                        f"❌ [Layout Error] Tensor '{arg.name}' has non-contiguous strides {actual_strides}. "
                        f"Expected NCHW {expected_nchw} or NHWC {expected_nhwc}.\n"
                        f"Action: If you are using custom views, call .contiguous() or .to(memory_format=torch.channels_last)."
                    )
            
            elif hasattr(arg, 'strides') and arg.strides:
                # Fallback for non-4D tensors (1D, 2D, 3D, 5D)
                expected_strides = []
                current_stride = 1
                for d in reversed(arg.shape):
                    expected_strides.insert(0, current_stride)
                    try:
                        current_stride *= int(d)
                    except: break
                
                if expected_strides and tuple(arg.strides) != tuple(expected_strides):
                    # We still allow the 1-element scalar case which can have weird strides
                    if len(arg.shape) > 0 and any(d > 1 for d in arg.shape):
                        raise ValueError(f"❌ [Layout Error] Tensor '{arg.name}' has invalid strides {arg.strides}.")

            if hasattr(arg, 'strides') and arg.strides and hasattr(arg, 'shape'):
                # Find which dimension has stride 1 (the contiguous inner-most dim)
                try:
                    # Get the index of the dimension that is physically contiguous
                    inner_dim_idx = arg.strides.index(1)
                    inner_dim_size = int(arg.shape[inner_dim_idx])
                    
                    # 2. PERFORM ALIGNMENT CHECK ON THE PHYSICAL INNER DIM
                    if inner_dim_size % 8 != 0:
                        layout_type = "NHWC" if inner_dim_idx == 1 else "Standard"
                        raise ValueError(
                            f"❌ [Alignment Error] Kernel '{m.kernel_name}' uses tensor '{arg.name}'\n"
                            f"Physical inner dimension (index {inner_dim_idx}, size {inner_dim_size}) is not aligned.\n"
                            f"Detected Layout: {layout_type}. Requires multiple of 8 for vectorization."
                        )
                except ValueError:
                    # If no stride is 1, it's a non-contiguous mess we already caught
                    pass
                except Exception as e:
                    print(f"Warning during validation: {e}")

def compile(
    model: torch.nn.Module, 
    inputs: tuple, 
    name: str = "custom_model", 
    backends: list[str] = ["onnx", "tensorrt"],
    interaction_handler: Optional[InteractionHandler] = None,
    verbose: bool = False
) -> CompiledModel:
    if verbose:
        from ..config import set_verbose
        set_verbose(True)
        
    check_environment(backends)
    
    cache_dir = _get_cache_dir(name, inputs)
    
    manifests = extract_manifests(model, inputs)
    
    if not manifests:
        return CompiledModel(cache_dir, name, backends)

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    for m in manifests:
        if not m.kernel_name.endswith(f"_{safe_name}"):
            m.kernel_name = f"{m.kernel_name}_{safe_name}"

    manifests = analyze_grid_asts(manifests, handler=interaction_handler)

    validate_manifests(manifests)
    
    # 1. Base ONNX Export (Needed by BOTH ORT and TRT)
    onnx_path = os.path.join(cache_dir, f"{name}.onnx")
    saved_args = _export_to_onnx(model, inputs, onnx_path, manifests)
    
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
        engine_file = os.path.join(cache_dir, f"{name}.engine")
        if os.path.exists(engine_file):
            try:
                os.remove(engine_file)
            except Exception:
                pass
    with torch.no_grad():
        dummy_out = model(*inputs)
    out_list = list(dummy_out) if isinstance(dummy_out, (tuple, list)) else [dummy_out]
    output_shapes = [tuple(t.shape) for t in out_list if isinstance(t, torch.Tensor)]
    output_strides = [tuple(t.stride()) for t in out_list if isinstance(t, torch.Tensor)]
    output_dtypes = [t.dtype for t in out_list if isinstance(t, torch.Tensor)]

    if saved_args:
        try:
            torch.save(saved_args, os.path.join(cache_dir, "saved_args.pt"))
        except Exception:
            pass

    return CompiledModel(cache_dir, name, backends, output_shapes=output_shapes, output_strides=output_strides, output_dtypes=output_dtypes, saved_args=saved_args)

def load(name: str) -> CompiledModel:
    """
    Loads a previously compiled model from the cache without recompiling.
    """
    cache_dir = os.path.join(os.path.expanduser("~"), ".kernel_lens_cache", name)
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Model '{name}' not found in cache. Did you compile it?")
        
    backends = []
    if os.path.exists(os.path.join(cache_dir, "ort_plugins", "libtriton_ort_plugins.so")):
        backends.append("onnx")
    if os.path.exists(os.path.join(cache_dir, "trt_plugins", "libtriton_trt_plugins.so")):
        backends.append("tensorrt")
        
    if not backends:
        raise RuntimeError(f"Cache for '{name}' exists, but no compiled backend plugins were found.")
        
    saved_args = None
    saved_args_path = os.path.join(cache_dir, "saved_args.pt")
    if os.path.exists(saved_args_path):
        try:
            saved_args = torch.load(saved_args_path)
        except Exception:
            pass

    return CompiledModel(cache_dir, name, backends, saved_args=saved_args)

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

    exporter = TritonGlobalONNXExporter(manifests)
    with exporter:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", message=".*Converting a tensor to a Python.*")
            
            export_kwargs = dict(
                export_params=True,
                opset_version=17,
                do_constant_folding=False, 
                input_names=[f"input_{i}" for i in range(len(inputs))],
                output_names=out_names
            )
            # Ensure classic TorchScript tracing in PyTorch 2.6+ when onnxscript is installed
            try:
                import inspect
                sig = inspect.signature(torch.onnx.export)
                if 'dynamo' in sig.parameters:
                    export_kwargs['dynamo'] = False
            except Exception:
                pass

            torch.onnx.export(
                model,
                inputs,
                output_path,
                **export_kwargs
            )
            _clean_onnx_graph(output_path)
    return getattr(exporter, 'saved_args', None)

def _clean_onnx_graph(onnx_path):
    try:
        import onnx
        model = onnx.load(onnx_path)
        graph = model.graph

        producer = {}
        for node in graph.node:
            for out in node.output:
                producer[out] = node

        nodes_to_remove = set()
        for node in list(graph.node):
            if node.op_type in ('Expand', 'Reshape'):
                inp_tensor = node.input[0]
                out_tensor = node.output[0]
                parent = producer.get(inp_tensor)
                if parent and ('fused_' in parent.op_type or 'triton' in parent.op_type or 'kernel' in parent.op_type):
                    for idx, p_out in enumerate(parent.output):
                        if p_out == inp_tensor:
                            parent.output[idx] = out_tensor
                            nodes_to_remove.add(node.name)

        if nodes_to_remove:
            new_nodes = [n for n in graph.node if n.name not in nodes_to_remove]
            graph.ClearField('node')
            graph.node.extend(new_nodes)
            onnx.save(model, onnx_path)
    except Exception:
        pass

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