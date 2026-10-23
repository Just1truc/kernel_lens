import re
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import torch
from .ast_analyzer import translate_symint_to_cxx, raw_sym_expr
from ..config import debug_print

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
    # Pass 2's shape for this argument: SymInt where the dim is dynamic, plain
    # int where it is fixed. Pass 1's `shape` stays concrete for dtype/size work.
    sym_shape: tuple = ()
    is_constexpr: bool = False

@dataclass
class KernelManifest:
    kernel_name: str
    ptx: str
    shared_memory_bytes: int
    num_warps: int
    arguments: List[ArgumentDef]
    
    grid_cxx_exprs: List[str] = field(default_factory=list)
    sym_map: Any = None
    output_dims_cxx_code: str = ""

    # True when the kernel addresses its tensors as a flat buffer (no strides),
    # so it is layout-agnostic and can accept channel-last plugin formats.
    layout_agnostic: bool = False

    # Dim indices that must be read from the tensor at run time rather than
    # baked from the traced shape. Set by compile(dynamic_shapes=True).
    dynamic_axes: Tuple[int, ...] = ()
    
    _sym_grid_asts: Tuple[Any, ...] = ()
    _sym_out_asts: Tuple[Any, ...] = ()
    fn: Any = None
    
    def confirm_and_compile(self):
        """Asks the user for missing I/O context, then evaluates C++ bindings."""
        debug_print(f"\n{'='*50}\nConfiguring: {self.kernel_name}\n{'='*50}")
        
        for arg in self.arguments:
            if arg.shape:
                while True:
                    choice = input(f"Tensor '{arg.name}' shape={arg.shape}. Is it an [i]nput or [o]utput? ").strip().lower()
                    if choice in ['i', 'o']:
                        arg.kind = 'input' if choice == 'i' else 'output'
                        break
                    print("Invalid choice. Please type 'i' or 'o'.")
            else:
                debug_print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")
                
        debug_print(f"\n[+] Compiling SymInt ASTs into C++ bindings...")
        self._transpile_symints_to_cxx()

    def _transpile_symints_to_cxx(self):
        # The symbolic shapes captured by Pass 2, not the concrete ones from
        # Pass 1: only Pass 2's dims are SymInts, so only they carry the symbol
        # names that a runtime grid has to be expressed in terms of.
        sym_map = {}
        for arg in self.arguments:
            if arg.kind == "input":
                shape = getattr(arg, "sym_shape", None) or arg.shape
                for d_idx, dim in enumerate(shape):
                    if isinstance(dim, torch.SymInt):
                        # Agnostic mapping: dim_values comes from ORT, inputDesc from TRT
                        sym_map[str(raw_sym_expr(dim))] = f"dim_values[{d_idx}]"
            elif arg.kind == "scalar":
                sym_map[str(arg.name)] = f"m_{arg.name}"
                
        for arg in self.arguments:
            if arg.kind == 'scalar' and arg._sym_ast is not None:
                arg.cxx_expr = translate_symint_to_cxx(arg._sym_ast, sym_map)
            elif arg.kind == 'scalar':
                arg.cxx_expr = f"m_{arg.name}"

        self.sym_map = dict(sym_map)
        self.grid_cxx_exprs = self.grid_exprs()
        while len(self.grid_cxx_exprs) < 3:
            self.grid_cxx_exprs.append("1")

    def shape_scalar_specs(self):
        """Scalars that follow the input shape, plus what is needed to
        recompute them at run time.

        `n = x.numel()` is injected into the ONNX graph as a scalar input, and
        the runtime fed it from the value captured while tracing -- so a
        different batch got the traced element count. Pass 2 records the
        expression (`s75*s96`) and which input dim each symbol stands for, so
        the value can be recomputed from the live shapes instead.
        """
        sym_dims = {}
        for arg in self.arguments:
            if arg.kind == "input":
                for d_idx, dim in enumerate(getattr(arg, "sym_shape", None) or arg.shape):
                    if isinstance(dim, torch.SymInt):
                        sym_dims[str(raw_sym_expr(dim))] = d_idx
                break

        specs = []
        for arg in self.arguments:
            if arg.kind != "scalar" or arg._sym_ast is None:
                continue
            expr = str(raw_sym_expr(arg._sym_ast))
            if not re.fullmatch(r"[\w\s*+()\-/]+", expr):
                continue
            if not any(re.search(rf"\b{re.escape(sym)}\b", expr) for sym in sym_dims):
                continue
            specs.append({
                "name": arg.name,
                "expr": expr,
                "sym_dims": dict(sym_dims),
                "is_float": isinstance(arg.value, float),
                # The value captured while tracing. The runtime matches a graph
                # input slot against this before substituting, because the
                # session's scalar inputs are not in the same order as this
                # list -- they include every scalar, not just the shape-derived
                # ones, so indexing by position would feed n into alpha.
                "traced": arg.value,
            })
        return specs

    def sym_map_for(self, dim_accessor=None):
        """symbol -> C++ that reads that dimension at run time.

        ``dim_accessor`` maps a dimension index to the C++, so one translation
        serves both backends: ONNX Runtime reads ``dim_values[i]``, TensorRT
        reads ``inputDesc[0].dims.d[i]``.
        """
        if dim_accessor is None:
            dim_accessor = lambda i: f"dim_values[{i}]"  # noqa: E731
        sym_map = {}
        for arg in self.arguments:
            if arg.kind == "input":
                shape = getattr(arg, "sym_shape", None) or arg.shape
                for d_idx, dim in enumerate(shape):
                    if isinstance(dim, torch.SymInt):
                        sym_map[str(raw_sym_expr(dim))] = dim_accessor(d_idx)
        return sym_map

    def grid_exprs(self, dim_accessor=None):
        """Launch-grid expressions resolved against the runtime input."""
        sym_map = self.sym_map_for(dim_accessor)
        return [translate_symint_to_cxx(g, sym_map) for g in self._sym_grid_asts]

    def scalar_exprs(self, dim_accessor=None):
        """Runtime expressions for scalars that follow the input shape.

        Returns {arg name: C++ expression}. A scalar with no symbolic form
        keeps its baked plugin member, so this degrades to the old behaviour.
        """
        sym_map = self.sym_map_for(dim_accessor)
        out = {}
        for arg in self.arguments:
            if arg.kind != "scalar":
                continue
            ast = arg._sym_ast
            if isinstance(ast, torch.SymInt):
                out[arg.name] = translate_symint_to_cxx(ast, sym_map)
            else:
                out[arg.name] = f"m_{arg.name}"
        return out
