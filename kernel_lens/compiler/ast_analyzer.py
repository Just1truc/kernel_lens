import re
import torch
from typing import List, Optional, Dict, Any
from ..config import debug_print

# REMOVED the circular import: from .manifest import KernelManifest
# REMOVED the circular import: from .interaction import InteractionHandler

def translate_symint_to_cxx(sym_int: Any, sym_map: Dict[str, str]) -> str:
    """
    Converts a PyTorch SymInt into a TensorRT/ORT C++ string.
    """
    if not isinstance(sym_int, torch.SymInt):
        return str(sym_int)
        
    expr_str = str(sym_int.node.expr)
    sorted_map = sorted(sym_map.items(), key=lambda x: len(x[0]), reverse=True)
    
    for sym_var, cxx_code in sorted_map:
        if sym_var.isalnum():
            expr_str = re.sub(rf'\b{sym_var}\b', cxx_code, expr_str)
        else:
            expr_str = expr_str.replace(sym_var, f"({cxx_code})")
            
    expr_str = expr_str.replace("//", "/") 
    expr_str = expr_str.replace("**", "^") 
    return expr_str

def analyze_grid_asts(
    manifests: List[Any], 
    handler: Optional[Any] = None
) -> List[Any]:
    """
    Analyzes symbolic ASTs and classifies tensor arguments deterministically via tl.store inspection,
    recursively traversing called helper functions if necessary.
    """
    import inspect, ast

    for manifest in manifests:
        debug_print(f"\n{'='*50}\nConfiguring I/O for: {manifest.kernel_name}\n{'='*50}")
        
        output_arg_names = set()
        input_arg_names = set()
        
        if hasattr(manifest, 'fn') and manifest.fn is not None:
            try:
                visited_fns = set()
                
                def inspect_fn_ast(fn_obj):
                    if fn_obj in visited_fns:
                        return
                    visited_fns.add(fn_obj)
                    
                    actual_fn = fn_obj.fn if hasattr(fn_obj, 'fn') else fn_obj
                    if not callable(actual_fn):
                        return
                        
                    try:
                        source = inspect.getsource(actual_fn)
                        tree = ast.parse(source)
                    except Exception:
                        return

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func_name = ""
                            if isinstance(node.func, ast.Attribute): 
                                func_name = node.func.attr
                            elif isinstance(node.func, ast.Name): 
                                func_name = node.func.id

                            if func_name == 'store' and node.args:
                                target_str = ast.unparse(node.args[0])
                                for arg in manifest.arguments:
                                    if arg.shape and (arg.name in target_str or target_str.startswith(arg.name) or re.search(rf'\b{arg.name}\b', target_str)):
                                        output_arg_names.add(arg.name)
                            elif func_name == 'load' and node.args:
                                target_str = ast.unparse(node.args[0])
                                for arg in manifest.arguments:
                                    if arg.shape and (arg.name in target_str or target_str.startswith(arg.name) or re.search(rf'\b{arg.name}\b', target_str)):
                                        input_arg_names.add(arg.name)
                            else:
                                # Check if calling another triton jit or helper function
                                try:
                                    glob = getattr(actual_fn, '__globals__', {})
                                    if func_name in glob and callable(glob[func_name]):
                                        inspect_fn_ast(glob[func_name])
                                except Exception:
                                    pass

                fn_obj = manifest.fn
                inspect_fn_ast(fn_obj)
            except Exception as e:
                debug_print(f"Warning during AST analysis: {e}")

        # Heuristic fallback if output_arg_names is empty but tensor args exist
        tensor_args = [a for a in manifest.arguments if a.shape]
        if not output_arg_names and tensor_args:
            for arg in tensor_args:
                lname = arg.name.lower()
                if any(kw in lname for kw in ['out', 'dst', 'res', 'output', 'result', 'y']) and not lname.startswith('in'):
                    output_arg_names.add(arg.name)
            if not output_arg_names and len(tensor_args) > 1:
                # Default the last tensor argument as output if no output detected
                output_arg_names.add(tensor_args[-1].name)

        for arg in manifest.arguments:
            if arg.shape:
                if arg.name in output_arg_names and arg.name in input_arg_names:
                    arg.kind = 'inplace'
                    debug_print(f"[AST-Analysis] '{arg.name}' statically resolved to INPLACE via tl.load and tl.store analysis.")
                elif arg.name in output_arg_names:
                    arg.kind = 'output'
                    debug_print(f"[AST-Analysis] '{arg.name}' statically resolved to OUTPUT via tl.store analysis.")
                elif handler is not None:
                    arg.kind = handler.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
                else:
                    from .interaction import AutoInteractionHandler
                    h = AutoInteractionHandler()
                    arg.kind = h.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
            else:
                arg.kind = 'scalar'
                debug_print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")
                
    return manifests