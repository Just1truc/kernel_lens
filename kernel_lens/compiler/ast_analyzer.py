import re
import torch
from typing import List, Optional, Dict, Any

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
    Analyzes symbolic ASTs and classifies tensor arguments deterministically via tl.store inspection.
    """
    for manifest in manifests:
        print(f"\n{'='*50}\nConfiguring I/O for: {manifest.kernel_name}\n{'='*50}")
        
        output_arg_names = set()
        if hasattr(manifest, 'fn') and manifest.fn is not None:
            try:
                import inspect, ast
                fn_obj = manifest.fn.fn if hasattr(manifest.fn, 'fn') else manifest.fn
                source = inspect.getsource(fn_obj)
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func_name = ""
                        if isinstance(node.func, ast.Attribute): func_name = node.func.attr
                        elif isinstance(node.func, ast.Name): func_name = node.func.id
                        if func_name == 'store' and node.args:
                            target_str = ast.unparse(node.args[0])
                            for arg in manifest.arguments:
                                if arg.shape and (arg.name in target_str or target_str.startswith(arg.name)):
                                    output_arg_names.add(arg.name)
            except Exception:
                pass

        for arg in manifest.arguments:
            if arg.shape:
                if arg.name in output_arg_names:
                    arg.kind = 'output'
                    print(f"[AST-Analysis] '{arg.name}' statically resolved to OUTPUT via tl.store analysis.")
                elif handler is not None:
                    arg.kind = handler.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
                else:
                    from .interaction import AutoInteractionHandler
                    h = AutoInteractionHandler()
                    arg.kind = h.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
            else:
                arg.kind = 'scalar'
                print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")
                
    return manifests