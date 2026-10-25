import ast
import inspect
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional, Set

import torch

from ..config import debug_print

# REMOVED the circular import: from .manifest import KernelManifest
# REMOVED the circular import: from .interaction import InteractionHandler

def raw_sym_expr(sym_int: Any) -> Any:
    """The *unsubstituted* symbolic expression behind a SymInt.

    ``SymNode.expr`` runs ``shape_env.replace()`` with the values the env has
    recorded, so once tracing has bound s75 -> 8 and s96 -> 512 it folds
    ``(s75*s96 + 255)//256`` to the constant 16. That is the right answer for a
    shape-specialised engine and exactly the wrong one for a dynamic engine.
    ``SymNode._expr`` still holds the symbol, so read that first.
    """
    node = getattr(sym_int, "node", None)
    if node is None:
        return sym_int
    raw = getattr(node, "_expr", None)
    if raw is not None and not isinstance(raw, int):
        return raw
    return getattr(node, "expr", raw)


def _cxx_pow(expr_str: str) -> str:
    """sympy renders powers as ``a**b``; C++ has no ``**``.

    Small integer powers are expanded by hand -- ``std::pow`` would drag in
    <cmath> and go through a double, and a launch grid is exact integer maths.
    """
    for _ in range(4):
        m = re.search(r'\b(\w+)\*\*(\d+)\b', expr_str)
        if not m:
            break
        base, exp = m.group(1), int(m.group(2))
        expr_str = expr_str[:m.start()] + "(" + "*".join([base] * exp) + ")" + expr_str[m.end():]
    if "**" in expr_str:
        # Only reachable for a non-integer or large exponent; keep it valid C++.
        expr_str = re.sub(r'\b(\w+)\*\*(\w+)\b', r'((int64_t)std::pow((double)\1, (double)\2))', expr_str)
    return expr_str


def translate_symint_to_cxx(sym_int: Any, sym_map: Dict[str, str]) -> str:
    """
    Converts a PyTorch SymInt into a TensorRT/ORT C++ string.
    """
    if not isinstance(sym_int, torch.SymInt):
        return str(sym_int)

    raw = raw_sym_expr(sym_int)
    # Expand sympy powers BEFORE substituting: at this point the base is a bare
    # symbol (s76**2), which is easy to rewrite. Afterwards the base is a
    # parenthesised C++ expression and the rewrite gets ambiguous.
    expr_str = _cxx_pow(str(raw))

    sorted_map = sorted(sym_map.items(), key=lambda x: len(x[0]), reverse=True)

    for sym_var, cxx_code in sorted_map:
        if sym_var.isalnum():
            expr_str = re.sub(rf'\b{re.escape(sym_var)}\b', lambda _m, c=cxx_code: c, expr_str)
        else:
            expr_str = expr_str.replace(sym_var, f"({cxx_code})")

    if re.search(r'\bs\d+\b', expr_str):
        # A symbol survived: the map did not cover it. Emitting it would be a
        # C++ identifier error, so fall back to the value the shape env
        # resolved -- the old behaviour. Static, but valid.
        expr_str = str(sym_int)

    expr_str = _cxx_pow(expr_str)
    expr_str = expr_str.replace("//", "/")
    return expr_str

# ---------------------------------------------------------------------------
# Argument taxonomy
# ---------------------------------------------------------------------------

def warn(msg: str) -> None:
    """Taxonomy diagnostics go to stderr so stdout stays parseable."""
    sys.stderr.write(f"[kernel_lens] {msg}\n")


def _source_ast(fn_obj: Any) -> Optional[ast.FunctionDef]:
    """The FunctionDef behind a kernel or helper, or None if it is unavailable."""
    actual = getattr(fn_obj, "fn", fn_obj)
    if not callable(actual):
        return None
    try:
        source = inspect.getsource(actual)
    except Exception:
        return None
    # A nested function's source comes back indented; ast.parse needs it flush.
    for candidate in (source, textwrap.dedent(source)):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node
    return None


def _definitions(fn_node: ast.FunctionDef) -> Dict[str, List[Any]]:
    """Local name -> the expressions assigned to it.

    ``p += off`` is treated as ``p = p + off`` (self-referential; the resolver's
    ``seen`` set terminates it), and a ``for`` target takes its iterable.
    """
    defs: Dict[str, List[Any]] = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            defs.setdefault(node.targets[0].id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            defs.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            defs.setdefault(node.target.id, []).append(
                ast.BinOp(left=node.target, op=node.op, right=node.value))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            defs.setdefault(node.target.id, []).append(node.iter)
    return defs


class TargetResolver:
    """Resolves load/store targets back to the arguments they derive from.

    This replaces a string test -- "does this argument's name occur in the
    unparsed target?" -- which misses every pointer renamed through an
    intermediate (``q_ptrs = Q_ptr + offs`` names neither ``Q`` nor ``Q_ptr``)
    and over-matches short names (``o`` is a substring of both ``row_x`` and
    ``row_out``, which turns a plain output into an in-place argument).
    """

    _MAX_DEPTH = 12

    def __init__(self, fn_node, env, globals_map, visited):
        self.fn = fn_node
        self.env = env                 # this scope's parameter -> kernel arguments
        self.globals = globals_map
        self.visited = visited         # (qualname, bound env) recursion guard
        self.defs = _definitions(fn_node)
        self.loads: Set[str] = set()
        self.stores: Set[str] = set()

    def resolve(self, node, seen=frozenset(), depth: int = 0) -> Set[str]:
        """Kernel arguments reachable by walking ``node`` back through assignments."""
        out: Set[str] = set()
        if depth > self._MAX_DEPTH:
            return out
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in self.env:
                    out |= self.env[child.id]
                elif child.id in self.defs and child.id not in seen:
                    for rhs in self.defs[child.id]:
                        out |= self.resolve(rhs, seen | {child.id}, depth + 1)
        return out

    def scan(self):
        for node in ast.walk(self.fn):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if callee in ("load", "store") and node.args:
                reached = self.resolve(node.args[0])
                (self.loads if callee == "load" else self.stores).update(reached)
            elif callee:
                self._descend(callee, node)
        return self.loads, self.stores

    def _descend(self, callee_name: str, call_node: ast.Call) -> None:
        """Follow a call into a helper, binding its parameters to the call site.

        The binding is what makes nested helpers work. Attributing the helper's
        own parameter names, as a bare recursion does, only lines up with the
        kernel's arguments when the two happen to share a name.
        """
        target = self.globals.get(callee_name)
        if target is None or not callable(target):
            return
        helper = _source_ast(target)
        if helper is None:
            return
        actual = getattr(target, "fn", target)
        try:
            bound = inspect.signature(actual).bind_partial(
                *call_node.args,
                **{kw.arg: kw.value for kw in call_node.keywords if kw.arg})
        except Exception:
            warn(f"could not bind the arguments of '{callee_name}'; "
                 "its loads and stores are not attributed")
            return
        env = {}
        for pname, pval in bound.arguments.items():
            env[pname] = self.resolve(pval) if isinstance(pval, ast.AST) else set()
        key = (getattr(actual, "__qualname__", callee_name),
               frozenset((k, frozenset(v)) for k, v in env.items()))
        if key in self.visited:
            return
        self.visited.add(key)
        sub = TargetResolver(helper, env, getattr(actual, "__globals__", self.globals), self.visited)
        loads, stores = sub.scan()
        self.loads |= loads
        self.stores |= stores


# A tensor argument the resolver could not reach is guessed from its name.
# These hints only ever decide arguments the resolution left undetermined.
_OUTPUT_HINTS = ['out', 'dst', 'res', 'output', 'result', 'y']


def analyze_grid_asts(
    manifests: List[Any],
    handler: Optional[Any] = None
) -> List[Any]:
    """Classify each tensor argument as input, output or in-place.

    Every ``tl.load`` / ``tl.store`` target is resolved backwards through the
    function's own assignments to the arguments it derives from. Only arguments
    that resolution does not reach are guessed, and each guess is reported on
    stderr so a silent misclassification cannot pass unnoticed.
    """
    for manifest in manifests:
        debug_print(f"\n{'='*50}\nConfiguring I/O for: {manifest.kernel_name}\n{'='*50}")

        load_reached: Set[str] = set()
        store_reached: Set[str] = set()

        if hasattr(manifest, 'fn') and manifest.fn is not None:
            fn_node = _source_ast(manifest.fn)
            if fn_node is not None:
                actual = getattr(manifest.fn, "fn", manifest.fn)
                params = [a.arg for a in fn_node.args.args] + [a.arg for a in fn_node.args.kwonlyargs]
                resolver = TargetResolver(
                    fn_node,
                    {p: {p} for p in params},
                    getattr(actual, "__globals__", {}),
                    set(),
                )
                load_reached, store_reached = resolver.scan()
            else:
                warn(f"source unavailable for '{manifest.kernel_name}'; "
                     "argument roles fall back to name heuristics")

        tensor_args = [a for a in manifest.arguments if a.shape]
        tensor_names = {a.name for a in tensor_args}
        input_arg_names = load_reached & tensor_names
        output_arg_names = store_reached & tensor_names

        undetermined = tensor_names - input_arg_names - output_arg_names
        if undetermined and tensor_args:
            warn(f"'{manifest.kernel_name}': no load/store target resolved to "
                 f"{sorted(undetermined)}; guessing from names")
            for name in sorted(undetermined):
                lname = name.lower()
                if any(kw in lname for kw in _OUTPUT_HINTS) and not lname.startswith('in'):
                    output_arg_names.add(name)
            if not output_arg_names and len(tensor_args) > 1:
                last = tensor_args[-1].name
                output_arg_names.add(last)
                warn(f"'{manifest.kernel_name}': defaulting the last tensor "
                     f"argument '{last}' to output")

        for arg in manifest.arguments:
            if arg.shape:
                if arg.name in output_arg_names and arg.name in input_arg_names:
                    arg.kind = 'inplace'
                    debug_print(f"[AST-Analysis] '{arg.name}' resolved to INPLACE (loaded and stored).")
                elif arg.name in output_arg_names:
                    arg.kind = 'output'
                    debug_print(f"[AST-Analysis] '{arg.name}' resolved to OUTPUT.")
                elif arg.name in input_arg_names:
                    arg.kind = 'input'
                    debug_print(f"[AST-Analysis] '{arg.name}' resolved to INPUT.")
                elif handler is not None:
                    arg.kind = handler.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
                else:
                    from .interaction import AutoInteractionHandler
                    h = AutoInteractionHandler()
                    arg.kind = h.ask_tensor_kind(manifest.kernel_name, arg.name, arg.shape)
            else:
                arg.kind = 'scalar'
                debug_print(f"[Auto] Mapped scalar constant: {arg.name} = {arg.value}")

        # A kernel with no stride arguments addresses a flat buffer, so it is
        # agnostic to the physical layout and can accept channel-last formats.
        manifest.layout_agnostic = not any(
            'stride' in a.name.lower() for a in manifest.arguments
        )
        debug_print(f"[Layout] '{manifest.kernel_name}' layout_agnostic={manifest.layout_agnostic}")

    return manifests
