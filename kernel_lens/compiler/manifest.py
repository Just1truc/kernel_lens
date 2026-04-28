from dataclasses import dataclass, field
from typing import Any, List, Tuple
import torch
from .ast_analyzer import translate_symint_to_cxx

@dataclass
class ArgumentDef:
    name: str
    kind: str  # Starts as "unknown" for tensors, "scalar" for primitives
    shape: tuple = ()
    strides: tuple = ()
    dtype: str = ""
    value: Any = None
    _sym_ast: Any = None
    cxx_expr: str = ""

@dataclass
class KernelManifest:
    kernel_name: str
    ptx: str
    shared_memory_bytes: int
    num_warps: int
    arguments: List[ArgumentDef]
    
    grid_cxx_exprs: List[str] = field(default_factory=list)
    output_dims_cxx_code: str = ""
    
    _sym_grid_asts: Tuple[Any, ...] = ()
    _sym_out_asts: Tuple[Any, ...] = ()
    
    def confirm_and_compile(self):
        """Asks the user for missing I/O context, then evaluates C++ bindings."""
        print(f"\n{'='*50}\nConfiguring: {self.kernel_name}\n{'='*50}")
        
        for arg in self.arguments:
            if arg.shape:
                while True:
                    choice = input(f"Tensor '{arg.name}' shape={arg.shape}. Is it an [i]nput or [o]utput? ").strip().lower()
                    if choice in ['i', 'o']:
                        arg.kind = 'input' if choice == 'i' else 'output'
                        break
                    print("Invalid choice. Please type 'i' or 'o'.")
            else:
                print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")
                
        print(f"\n[+] Compiling SymInt ASTs into C++ bindings...")
        self._transpile_symints_to_cxx()

    def _transpile_symints_to_cxx(self):
        sym_map = {}
        input_idx = 0
        
        for arg in self.arguments:
            if arg.kind == "input":
                for d_idx, dim in enumerate(arg.shape):
                    if isinstance(dim, torch.SymInt):
                        # Agnostic mapping: dim_values comes from ORT, inputDesc from TRT
                        sym_map[str(dim.node.expr)] = f"dim_values[{d_idx}]" 
                input_idx += 1
            elif arg.kind == "scalar":
                sym_map[str(arg.name)] = f"m_{arg.name}"
                
        for arg in self.arguments:
            if arg.kind == 'scalar' and arg._sym_ast is not None:
                arg.cxx_expr = translate_symint_to_cxx(arg._sym_ast, sym_map)
            elif arg.kind == 'scalar':
                arg.cxx_expr = f"m_{arg.name}"

        self.grid_cxx_exprs = [
            translate_symint_to_cxx(g, sym_map) for g in self._sym_grid_asts
        ]
        while len(self.grid_cxx_exprs) < 3:
            self.grid_cxx_exprs.append("1")