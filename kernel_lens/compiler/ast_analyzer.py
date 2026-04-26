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
    Analyzes symbolic ASTs and classifies tensor arguments using the interaction handler.
    """
    if handler is None:
        # Lazy import to avoid circular dependency
        from .interaction import TerminalInteractionHandler
        handler = TerminalInteractionHandler()

    for manifest in manifests:
        print(f"\n{'='*50}\nConfiguring I/O for: {manifest.kernel_name}\n{'='*50}")
        
        for arg in manifest.arguments:
            if arg.shape:
                arg.kind = handler.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
            else:
                arg.kind = 'scalar'
                print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")
                
    return manifests