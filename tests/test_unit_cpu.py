import pytest
import torch
from kernel_lens.compiler.manifest import ArgumentDef, KernelManifest
from kernel_lens.compiler.ast_analyzer import translate_symint_to_cxx
from kernel_lens.compiler.core import validate_manifests, is_nhwc


def test_argument_def_initialization():
    arg = ArgumentDef(
        name="x_ptr",
        kind="input",
        shape=(2, 64),
        strides=(64, 1),
        dtype="float32",
        is_constexpr=False
    )
    assert arg.name == "x_ptr"
    assert arg.kind == "input"
    assert arg.shape == (2, 64)
    assert arg.strides == (64, 1)
    assert arg.dtype == "float32"
    assert not arg.is_constexpr


def test_kernel_manifest_structure():
    arg1 = ArgumentDef(name="x", kind="input", shape=(10, 24), strides=(24, 1), dtype="float32")
    arg2 = ArgumentDef(name="y", kind="output", shape=(10, 24), strides=(24, 1), dtype="float32")
    
    manifest = KernelManifest(
        kernel_name="test_kernel",
        ptx="// dummy ptx",
        shared_memory_bytes=1024,
        num_warps=4,
        arguments=[arg1, arg2],
        grid_cxx_exprs=["1", "1", "1"]
    )
    
    assert manifest.kernel_name == "test_kernel"
    assert manifest.shared_memory_bytes == 1024
    assert manifest.num_warps == 4
    assert len(manifest.arguments) == 2
    assert manifest.grid_cxx_exprs == ["1", "1", "1"]


def test_translate_symint_to_cxx_primitives():
    res = translate_symint_to_cxx(128, {})
    assert res == "128"
    
    res_str = translate_symint_to_cxx("batch_size", {})
    assert res_str == "batch_size"


def test_is_nhwc_layout_check():
    shape = (1, 128, 64, 64)
    # NHWC format has stride 1 on channel dim (dim index 1)
    strides_nhwc = (128 * 64 * 64, 1, 64 * 128, 128)
    assert is_nhwc(shape, strides_nhwc) is True


def test_alignment_validation_failure():
    # Inner dim size 7 is not aligned to multiple of 8
    unaligned_arg = ArgumentDef(
        name="x",
        kind="input",
        shape=(1, 7),
        strides=(7, 1),
        dtype="float32"
    )
    manifest = KernelManifest(
        kernel_name="unaligned_kernel",
        ptx="",
        shared_memory_bytes=0,
        num_warps=4,
        arguments=[unaligned_arg]
    )
    
    with pytest.raises(ValueError, match="Alignment Error"):
        validate_manifests([manifest])
