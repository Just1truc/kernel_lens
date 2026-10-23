from typing import Dict, List
import textwrap
import json
import re
import os
from ..compiler.manifest import KernelManifest

def element_size_bytes(dtype_val) -> float:
    """Best-effort element size (in bytes) for a PyTorch/triton dtype string."""
    s = str(dtype_val).lower()
    if any(k in s for k in ('float8', 'e4m3', 'e5m2', 'fp8')):
        return 1.0
    if 'bool' in s or 'uint8' in s or 'int8' in s:
        return 1.0
    if 'uint4' in s or 'int4' in s:
        return 0.5
    if 'int16' in s or 'short' in s:
        return 2.0
    if ('float16' in s or 'fp16' in s or 'half' in s or 'bfloat16' in s or 'bf16' in s):
        return 2.0
    if 'int64' in s or 'long' in s or 'double' in s or 'float64' in s:
        return 8.0
    return 4.0

class TensorRTPluginGenerator:
    def __init__(self, manifests: List[KernelManifest], plugin_namespace: str = "triton_custom", plugin_version: str = "1"):
        self.manifests = manifests
        self.plugin_namespace = plugin_namespace
        self.plugin_version = plugin_version
        
    def _generate_kernel_h(self, manifest: KernelManifest) -> str:
        plugin_name = f"{manifest.kernel_name}Plugin"

        constant_decls = []
        for arg in manifest.arguments:
            if arg.kind == 'scalar':
                ctype = "float" if isinstance(arg.value, float) else "int"
                constant_decls.append(f"{ctype} m_{arg.name};")
        dynamic_members_cpp = "\n    ".join(constant_decls)

        def get_trt_datatype(dtype_obj):
            dtype_str = str(dtype_obj).lower()
            if 'float8' in dtype_str or 'fp8' in dtype_str or 'e4m3' in dtype_str or 'e5m2' in dtype_str:
                return "nvinfer1::DataType::kFP8"
            elif 'bool' in dtype_str:
                return "nvinfer1::DataType::kBOOL"
            elif 'uint8' in dtype_str or 'int8' in dtype_str or 'int4' in dtype_str or 'uint4' in dtype_str:
                return "nvinfer1::DataType::kINT8"
            elif 'bfloat16' in dtype_str or 'bf16' in dtype_str:
                return "nvinfer1::DataType::kBF16"
            elif 'float16' in dtype_str or 'half' in dtype_str:
                return "nvinfer1::DataType::kHALF"
            elif 'float' in dtype_str:
                return "nvinfer1::DataType::kFLOAT"
            elif 'int64' in dtype_str or 'long' in dtype_str:
                return "nvinfer1::DataType::kINT64"
            elif 'int' in dtype_str:
                return "nvinfer1::DataType::kINT32"
            else:
                return "nvinfer1::DataType::kFLOAT"

        in_args = [a for a in manifest.arguments if a.kind in ('input', 'inplace')]

        # Enforce the dtype the kernel was compiled for, position by position.
        # Without this TensorRT is free to run the plugin in fp32 (or promote it),
        # which silently disables the fp16/quantized path.
        _ordered_in = [a for a in manifest.arguments if a.kind in ('input', 'inplace')]
        _ordered_sc = [a for a in manifest.arguments if a.kind == 'scalar' and not getattr(a, 'is_constexpr', False)]
        _ordered_out = [a for a in manifest.arguments if a.kind in ('output', 'inplace')]
        _pos_types = [get_trt_datatype(a.dtype) for a in (_ordered_in + _ordered_sc + _ordered_out)]
        _pos_is_tensor = ([True] * len(_ordered_in) + [False] * len(_ordered_sc) + [True] * len(_ordered_out))
        _types_cpp = ", ".join(_pos_types) if _pos_types else "nvinfer1::DataType::kFLOAT"
        _is_tensor_cpp = ", ".join("true" if t else "false" for t in _pos_is_tensor) if _pos_is_tensor else "true"
        type_guard_cxx = (
            "static const nvinfer1::DataType kExpectedTypes[] = { " + _types_cpp + " };\n"
            "        static const bool kIsTensor[] = { " + _is_tensor_cpp + " };\n"
            "        const int kNTypes = (int)(sizeof(kExpectedTypes)/sizeof(kExpectedTypes[0]));\n"
            "        if (pos < kNTypes && kIsTensor[pos] && inOut[pos].type != kExpectedTypes[pos]) return false;\n"
        )

        if getattr(manifest, 'layout_agnostic', False):
            # A pointwise kernel reads a flat buffer, so the physical layout is
            # irrelevant. Accepting channel-last lets the engine keep the
            # surrounding convolutions in their native format instead of
            # inserting reformats around this plugin.
            supports_format_cxx = type_guard_cxx + (
                "// Pointwise kernel: layout-agnostic, so channel-last is safe too.\n"
                "        auto f = inOut[pos].format;\n"
                "        if (f == nvinfer1::TensorFormat::kLINEAR) return true;\n"
                "        if (f == nvinfer1::TensorFormat::kHWC8) {\n"
                "            auto d = inOut[pos].dims;\n"
                "            int32_t c = (d.nbDims == 4) ? d.d[1] : (d.nbDims > 0 ? d.d[d.nbDims - 1] : 1);\n"
                "            return (c % 8) == 0;  // require an unpadded channel dim\n"
                "        }\n"
                "        return false;"
            )
        else:
            supports_format_cxx = type_guard_cxx + "return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;"
        
        output_dim_cases = []
        out_args = [a for a in manifest.arguments if a.kind in ('output', 'inplace')]
        _dyn_axes = tuple(getattr(manifest, "dynamic_axes", ()) or ())
        for out_idx, out_arg in enumerate(out_args):
            output_dim_cases.append(f"if (outputIndex == {out_idx}) {{")
            output_dim_cases.append(f"    nvinfer1::DimsExprs res;")
            output_dim_cases.append(f"    res.nbDims = {len(out_arg.shape)};")
            for d_idx, d in enumerate(out_arg.shape):
                if d_idx in _dyn_axes:
                    # Declared dynamic: take the dim from the input at run time
                    # instead of freezing the traced value.
                    output_dim_cases.append(f"    res.d[{d_idx}] = inputs[0].d[{d_idx}];")
                    continue
                try:
                    d_int = int(d)
                    output_dim_cases.append(f"    res.d[{d_idx}] = exprBuilder.constant({d_int});")
                except Exception:
                    output_dim_cases.append(f"    res.d[{d_idx}] = inputs[0].d[{d_idx}];")
            output_dim_cases.append("    return res;")
            output_dim_cases.append("}")
        
        get_out_dims_cxx = "\n        ".join(output_dim_cases) if output_dim_cases else "return inputs[0];"

        tpl = f'''
#ifndef {manifest.kernel_name.upper()}_PLUGIN_H
#define {manifest.kernel_name.upper()}_PLUGIN_H

#include "NvInferPlugin.h"
#include <cuda.h>
#include <string>
#include <vector>
#include <mutex>
#include <cstdio>
#include <cstdlib>


namespace {self.plugin_namespace}_{manifest.kernel_name} {{


class {plugin_name} : public nvinfer1::IPluginV2DynamicExt {{
public:
    {plugin_name}();
    {plugin_name}(const void* data, size_t length);
    
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    
    int32_t getNbOutputs() const noexcept override {{ 
        return {len([a for a in manifest.arguments if a.kind in ('output', 'inplace')])}; 
    }}
    
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
    using nvinfer1::IPluginV2Ext::configurePlugin;

    nvinfer1::DimsExprs getOutputDimensions(int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {{
        {get_out_dims_cxx}
        return inputs[0]; 
    }}


    
    int32_t enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    
    bool supportsFormatCombination(int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override {{ 
        {supports_format_cxx}
    }}
    
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override {{}}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override {{ return 0; }}
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override {{ delete this; }}
    void setPluginNamespace(const char* pluginNamespace) noexcept override {{ mNamespace = pluginNamespace; }}
    const char* getPluginNamespace() const noexcept override {{ return mNamespace.c_str(); }}



    
private:
    std::string mNamespace;
    int mNbOutputs; 
    
    // --- DYNAMIC CONSTANTS ---
    {dynamic_members_cpp}
    static const char* PTX_CODE;
    
    CUmodule mModule{{nullptr}};
    CUfunction mKernel{{nullptr}};
}};

class {plugin_name}Creator : public nvinfer1::IPluginCreator {{
public:
    {plugin_name}Creator();
    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
}};

}} // namespace {self.plugin_namespace}_{manifest.kernel_name}


#endif
'''
        return textwrap.dedent(tpl).strip()

    def _generate_kernel_cu(self, manifest: KernelManifest) -> str:
        plugin_name = f"{manifest.kernel_name}Plugin"
        ptx_str = manifest.ptx
        ptx_encoded = json.dumps(ptx_str)
        block_size = (manifest.num_warps if manifest.num_warps > 0 else 4) * 32
        
        scalars = [arg for arg in manifest.arguments if arg.kind == 'scalar']
        nb_outputs = sum(1 for arg in manifest.arguments if arg.kind == 'output')
        nb_outputs = max(1, nb_outputs)
        
        # --- LOCAL TRT GRID AST EVALUATION ---
        
        grid_strs = []
        if hasattr(manifest, '_sym_grid_asts') and manifest._sym_grid_asts:
            for g in manifest._sym_grid_asts:
                expr = str(g.node.expr) if hasattr(g, 'node') else str(g)
                expr = re.sub(r'([a-zA-Z0-9_]+)\*\*([a-zA-Z0-9_]+)', r'std::pow(\1, \2)', expr)
                expr = expr.replace("//", "/")
                expr = re.sub(r'floor\((.*?)\)', r'(\1)', expr)
                
                # Replace any s(\d+) symbol dynamically using regex
                expr = re.sub(r'\bs(\d+)\b', r"(int64_t)(inputDesc[0].dims.nbDims > \1 ? inputDesc[0].dims.d[\1] : 1)", expr)
                grid_strs.append(expr)
                
        while len(grid_strs) < 3:
            grid_strs.append("1")
            
        grid_x_cxx = grid_strs[0]
        grid_y_cxx = grid_strs[1]
        grid_z_cxx = grid_strs[2]

        fn_match = re.search(r'\.entry\s+([a-zA-Z0-9_]+)', manifest.ptx)
        ptx_entry_name = fn_match.group(1) if fn_match else manifest.kernel_name

        init_lines = [f"m_{arg.name}({arg.value})" for arg in scalars]
        init_str = (", " + ", ".join(init_lines)) if init_lines else ""

        deserialize_lines = []
        for arg in scalars:
            ctype = "float" if isinstance(arg.value, float) else "int"
            deserialize_lines.append(f"m_{arg.name} = *reinterpret_cast<const {ctype}*>(d);")
            deserialize_lines.append(f"d += sizeof({ctype});")
        deserialize_cpp = "\n    ".join(deserialize_lines)

        size_additions = "".join([f" + sizeof({'float' if isinstance(arg.value, float) else 'int'})" for arg in scalars])

        serialize_lines = []
        for arg in scalars:
            ctype = "float" if isinstance(arg.value, float) else "int"
            serialize_lines.append(f"*reinterpret_cast<{ctype}*>(d) = m_{arg.name};")
            serialize_lines.append(f"d += sizeof({ctype});")
        serialize_cpp = "\n    ".join(serialize_lines)

        clone_lines = [f"plugin->m_{arg.name} = this->m_{arg.name};" for arg in scalars]
        clone_cpp = "\n    ".join(clone_lines)

        entry_match = re.search(r'\.entry\s+[a-zA-Z0-9_]+\s*\((.*?)\)\s*(?:\.reqntid|\{)', manifest.ptx, re.DOTALL)
        entry_sig = entry_match.group(1) if entry_match else manifest.ptx

        ptx_params = []
        for p in entry_sig.split('.param'):
            p = p.strip().rstrip(',').rstrip(')').strip()
            if not p:
                continue
            parts = p.split()
            if parts:
                p_ident = parts[-1]
                p_decl = " ".join(parts[:-1])
                ptx_params.append((p_decl, p_ident))
        active_args = [arg for arg in manifest.arguments if not getattr(arg, 'is_constexpr', False)]
        ptr_args = [arg for arg in active_args if arg.kind in ('input', 'output', 'inplace')]
        scalar_args = [arg for arg in active_args if arg.kind == 'scalar']
        num_ptr_args = len(ptr_args)

        ptx_ordered_slots = []
        for s_idx, (p_decl, p_ident) in enumerate(ptx_params):
            match = re.search(r'_param_(\d+)$', p_ident)
            param_num = int(match.group(1)) if match else s_idx
            arg = active_args[param_num] if param_num < len(active_args) else (active_args[s_idx] if s_idx < len(active_args) else None)
            if arg:
                is_ptr = arg.kind in ('input', 'output', 'inplace')
            else:
                is_ptr = ("ptr" in p_decl)

            if is_ptr: c_type = "void*"
            elif "32" in p_decl and "f" not in p_decl: c_type = "int32_t"
            elif "64" in p_decl and "f" not in p_decl: c_type = "int64_t"
            elif "f32" in p_decl: c_type = "float"
            elif "f64" in p_decl: c_type = "double"
            else: c_type = "int32_t"
            ptx_ordered_slots.append({"type": c_type, "ident": p_ident, "is_ptr": is_ptr, "param_num": param_num})

        num_ptx_slots = len(ptx_ordered_slots)
        in_counter = 0
        out_counter = 0
        arg_setup_lines = []

        for slot_idx, slot in enumerate(ptx_ordered_slots):
            c_type = slot["type"]
            param_num = slot["param_num"]
            arg = active_args[param_num] if param_num < len(active_args) else (active_args[slot_idx] if slot_idx < len(active_args) else None)

            if slot["is_ptr"]:
                if arg and arg.kind == 'input':
                    _esz = element_size_bytes(arg.dtype)
                    arg_setup_lines.append(f"const void* raw_ptr_{slot_idx} = (const void*)inputs[{in_counter}];")
                    arg_setup_lines.append(f"static thread_local const void* tmp_ptr_{slot_idx};")
                    arg_setup_lines.append(f"static thread_local void* align_buf_{slot_idx} = nullptr;")
                    arg_setup_lines.append(f"static thread_local size_t align_cap_{slot_idx} = 0;")
                    arg_setup_lines.append(f"if ((reinterpret_cast<uintptr_t>(raw_ptr_{slot_idx}) & 15u) != 0) {{")
                    arg_setup_lines.append(f"    size_t align_bytes_{slot_idx} = 1; for (int dd = 0; dd < inputDesc[{in_counter}].dims.nbDims; ++dd) align_bytes_{slot_idx} *= inputDesc[{in_counter}].dims.d[dd];")
                    if _esz == 0.5:
                        arg_setup_lines.append(f"    align_bytes_{slot_idx} = (align_bytes_{slot_idx} + 1) / 2;")
                    else:
                        arg_setup_lines.append(f"    align_bytes_{slot_idx} *= {int(_esz)};")
                    arg_setup_lines.append(f"    if (align_cap_{slot_idx} < align_bytes_{slot_idx}) {{ if (align_buf_{slot_idx}) cudaFree(align_buf_{slot_idx}); cudaMalloc(&align_buf_{slot_idx}, align_bytes_{slot_idx} + 16); align_cap_{slot_idx} = align_bytes_{slot_idx}; }}")
                    arg_setup_lines.append(f"    cudaMemcpyAsync(align_buf_{slot_idx}, raw_ptr_{slot_idx}, align_bytes_{slot_idx}, cudaMemcpyDeviceToDevice, stream);")
                    arg_setup_lines.append(f"    tmp_ptr_{slot_idx} = align_buf_{slot_idx};")
                    arg_setup_lines.append(f"}} else {{ tmp_ptr_{slot_idx} = raw_ptr_{slot_idx}; }}")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_ptr_{slot_idx};")
                    in_counter += 1
                elif arg and arg.kind == 'output':
                    arg_setup_lines.append(f"static thread_local void* tmp_ptr_{slot_idx}; tmp_ptr_{slot_idx} = (void*)outputs[{out_counter}];")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_ptr_{slot_idx};")
                    out_counter += 1
                elif arg and arg.kind == 'inplace':
                    arg_setup_lines.append(f"const void* in_ptr_{slot_idx} = inputs[{in_counter}];")
                    arg_setup_lines.append(f"void* out_ptr_{slot_idx} = outputs[{out_counter}];")
                    arg_setup_lines.append(f"size_t num_bytes_{slot_idx} = 1;")
                    arg_setup_lines.append(f"for (int d = 0; d < inputDesc[{in_counter}].dims.nbDims; ++d) num_bytes_{slot_idx} *= inputDesc[{in_counter}].dims.d[d];")
                    _esz = element_size_bytes(arg.dtype)
                    if _esz == 0.5:
                        arg_setup_lines.append(f"num_bytes_{slot_idx} = (num_bytes_{slot_idx} + 1) / 2;")
                    else:
                        arg_setup_lines.append(f"num_bytes_{slot_idx} *= {int(_esz)};")
                    arg_setup_lines.append(f"if (in_ptr_{slot_idx} != out_ptr_{slot_idx}) {{ cudaMemcpyAsync(out_ptr_{slot_idx}, in_ptr_{slot_idx}, num_bytes_{slot_idx}, cudaMemcpyDeviceToDevice, stream); }}")
                    arg_setup_lines.append(f"static thread_local void* tmp_ptr_{slot_idx}; tmp_ptr_{slot_idx} = out_ptr_{slot_idx};")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_ptr_{slot_idx};")
                    in_counter += 1
                    out_counter += 1
                else:
                    arg_setup_lines.append(f"static thread_local int64_t tmp_null_{slot_idx} = 1;")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_null_{slot_idx};")
            else:
                if arg and arg.kind == 'scalar':
                    expr = getattr(arg, 'cxx_expr', '') or f"m_{arg.name}"
                    scalar_ctype = "int64_t" if "64" in str(arg.dtype) else ("float" if ("float" in str(arg.dtype) or "float" in c_type or "double" in c_type) else "int32_t")
                    arg_setup_lines.append(f"static thread_local {scalar_ctype} tmp_scalar_{slot_idx}; tmp_scalar_{slot_idx} = ({scalar_ctype})({expr});")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_scalar_{slot_idx};")
                else:
                    arg_setup_lines.append(f"static thread_local {c_type} tmp_scalar_{slot_idx} = 1;")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_scalar_{slot_idx};")

        # --- DYNAMIC OUTPUT DATATYPES ---
        output_type_lines = []
        out_args = [a for a in manifest.arguments if a.kind in ('output', 'inplace')]
        for i, out_arg in enumerate(out_args):
            dtype_str = str(out_arg.dtype).lower()
            if 'float8' in dtype_str or 'fp8' in dtype_str or 'e4m3' in dtype_str or 'e5m2' in dtype_str:
                trt_type = "nvinfer1::DataType::kFP8"
            elif 'bool' in dtype_str:
                trt_type = "nvinfer1::DataType::kBOOL"
            elif 'uint8' in dtype_str or 'int8' in dtype_str or 'int4' in dtype_str or 'uint4' in dtype_str:
                trt_type = "nvinfer1::DataType::kINT8"
            elif 'float16' in dtype_str or 'half' in dtype_str:
                trt_type = "nvinfer1::DataType::kHALF"
            elif 'float' in dtype_str:
                trt_type = "nvinfer1::DataType::kFLOAT"
            elif 'int64' in dtype_str or 'long' in dtype_str:
                trt_type = "nvinfer1::DataType::kINT64"
            elif 'int' in dtype_str:
                trt_type = "nvinfer1::DataType::kINT32"
            else:
                trt_type = "inputTypes[0]"
                
            output_type_lines.append(f"if (index == {i}) return {trt_type};")
            
        dynamic_output_types = "\n    ".join(output_type_lines)

        dynamic_args_cpp = "\n    ".join(arg_setup_lines)
        
        tpl = f'''
#include "{manifest.kernel_name}Plugin.h"
#include <cuda.h>
#include <cstring>
#include <cstdlib>
#include <iostream>

namespace {self.plugin_namespace}_{manifest.kernel_name} {{


const char* {plugin_name}::PTX_CODE = {ptx_encoded};

{plugin_name}::{plugin_name}() : mNbOutputs({len([a for a in manifest.arguments if a.kind in ('output', 'inplace')])}){init_str} {{
    mNamespace = "{self.plugin_namespace}";
}}

{plugin_name}::{plugin_name}(const void* data, size_t length) {{
    const char* d = reinterpret_cast<const char*>(data);
    
    size_t nsLength = *reinterpret_cast<const size_t*>(d);
    d += sizeof(size_t);
    mNamespace = std::string(d, nsLength);
    d += nsLength;
    
    mNbOutputs = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    
    {deserialize_cpp}
}}

const char* {plugin_name}::getPluginType() const noexcept {{ return "{manifest.kernel_name}"; }}
const char* {plugin_name}::getPluginVersion() const noexcept {{ return "{self.plugin_version}"; }}

int32_t {plugin_name}::initialize() noexcept {{
    if (mModule == nullptr) {{
        cuInit(0);
        CUcontext ctx = nullptr;
        cuCtxGetCurrent(&ctx);
        if (ctx == nullptr) {{
            CUdevice cuDev;
            cuDeviceGet(&cuDev, 0);
            cuDevicePrimaryCtxRetain(&ctx, cuDev);
            cuCtxSetCurrent(ctx);
        }}
        CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) {{
            // Some CUDA driver JITs reject the newest PTX ISA version. Retry with a
            // clamped .version, then with an older .target, before giving up. The
            // original (unclamped) PTX is preserved so modern drivers JIT it natively.
            std::string ptx(PTX_CODE);
            size_t vp = ptx.find(".version");
            if (vp != std::string::npos) {{
                size_t nl = ptx.find('\\n', vp);
                if (nl != std::string::npos) ptx.replace(vp, nl - vp, ".version 9.0");
            }}
            res = cuModuleLoadDataEx(&mModule, ptx.c_str(), 0, nullptr, nullptr);
            if (res != CUDA_SUCCESS) {{
                size_t tp = ptx.find(".target");
                if (tp != std::string::npos) {{
                    size_t nl = ptx.find('\\n', tp);
                    if (nl != std::string::npos) ptx.replace(tp, nl - tp, ".target sm_90");
                }}
                res = cuModuleLoadDataEx(&mModule, ptx.c_str(), 0, nullptr, nullptr);
            }}
            if (res != CUDA_SUCCESS) {{
                fprintf(stderr, "TRT_INIT_ERROR: cuModuleLoadDataEx failed with code %d\\n", (int)res);
                return -1;
            }}
        }}
        res = cuModuleGetFunction(&mKernel, mModule, "{ptx_entry_name}");
        if (res != CUDA_SUCCESS) {{
            fprintf(stderr, "TRT_INIT_ERROR: cuModuleGetFunction failed with code %d for entry %s\\n", (int)res, "{ptx_entry_name}");
            return -1;
        }}
        if ({manifest.shared_memory_bytes} > 0) {{
            res = cuFuncSetAttribute(mKernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {manifest.shared_memory_bytes});
            if (res != CUDA_SUCCESS) {{
                fprintf(stderr, "TRT_INIT_ERROR: cuFuncSetAttribute failed with code %d\\n", (int)res);
                return -1;
            }}
        }}
    }}
    return 0;
}}


void {plugin_name}::terminate() noexcept {{
    mModule = nullptr;
    mKernel = nullptr;
}}


int32_t {plugin_name}::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {{

    if (!mKernel) {{
        int32_t res = this->initialize();
        if (res != 0 || !mKernel) {{
            fprintf(stderr, "TRT_ENQUEUE_ERROR: initialize failed or mKernel is null in enqueue\\n");
            return -1;
        }}
    }}
    if (getenv("KL_DBG")) fprintf(stderr, "[ENQ] in type=%d fmt=%d dims=%d | out type=%d fmt=%d\\n", (int)inputDesc[0].type, (int)inputDesc[0].format, inputDesc[0].dims.nbDims, (int)outputDesc[0].type, (int)outputDesc[0].format);

    unsigned int grid_x = std::max(1u, (unsigned int)({grid_x_cxx}));
    unsigned int grid_y = std::max(1u, (unsigned int)({grid_y_cxx}));
    unsigned int grid_z = std::max(1u, (unsigned int)({grid_z_cxx}));
    unsigned int block_x = std::max(1u, (unsigned int)({block_size}));

    void* kernelParams[{num_ptx_slots}];
    {dynamic_args_cpp}

    CUresult launch_res = cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, {manifest.shared_memory_bytes}, stream, kernelParams, nullptr);
    if (launch_res != CUDA_SUCCESS) {{
        fprintf(stderr, "TRT_ENQUEUE_ERROR: cuLaunchKernel failed with code %d\\n", (int)launch_res);
        return -1;
    }}

    return 0;
}}

size_t {plugin_name}::getSerializationSize() const noexcept {{ 
    return sizeof(size_t) + mNamespace.size() + sizeof(int){size_additions}; 
}}

void {plugin_name}::serialize(void* buffer) const noexcept {{
    char* d = reinterpret_cast<char*>(buffer);
    
    size_t nsLength = mNamespace.size();
    *reinterpret_cast<size_t*>(d) = nsLength;
    d += sizeof(size_t);
    
    std::memcpy(d, mNamespace.data(), nsLength);
    d += nsLength;
    
    *reinterpret_cast<int*>(d) = mNbOutputs;
    d += sizeof(int);
    
    {serialize_cpp}
}}

nvinfer1::IPluginV2DynamicExt* {plugin_name}::clone() const noexcept {{ 
    auto* plugin = new {plugin_name}(); 
    plugin->setPluginNamespace(mNamespace.c_str());
    plugin->mNbOutputs = this->mNbOutputs;
    
    {clone_cpp}
    
    return plugin;
}}

nvinfer1::DataType {plugin_name}::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept {{

    {dynamic_output_types}
    return inputTypes[0];
}}

// Plugin creator methods




const char* {plugin_name}Creator::getPluginName() const noexcept {{ return "{manifest.kernel_name}"; }}
const char* {plugin_name}Creator::getPluginVersion() const noexcept {{ return "{self.plugin_version}"; }}

{plugin_name}Creator::{plugin_name}Creator() {{
    mPluginAttributes.clear();
    mFC.nbFields = 0;
    mFC.fields = nullptr;
    mNamespace = "{self.plugin_namespace}";
}}




const nvinfer1::PluginFieldCollection* {plugin_name}Creator::getFieldNames() noexcept {{ return &mFC; }}

nvinfer1::IPluginV2* {plugin_name}Creator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept {{
    auto* plugin = new {plugin_name}();
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}}

nvinfer1::IPluginV2* {plugin_name}Creator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept {{
    auto* plugin = new {plugin_name}(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}}

void {plugin_name}Creator::setPluginNamespace(const char* pluginNamespace) noexcept {{ mNamespace = pluginNamespace; }}
const char* {plugin_name}Creator::getPluginNamespace() const noexcept {{ return mNamespace.c_str(); }}


}} // namespace {self.plugin_namespace}_{manifest.kernel_name}
'''
        return textwrap.dedent(tpl).strip()
    
    def generate(self) -> Dict[str, str]:
        files = {}
        seen_kernels = set()
        unique_manifests = []
        for m in self.manifests:
            if m.kernel_name not in seen_kernels:
                seen_kernels.add(m.kernel_name)
                unique_manifests.append(m)
                files[f"{m.kernel_name}Plugin.h"] = self._generate_kernel_h(m)
                files[f"{m.kernel_name}Plugin.cu"] = self._generate_kernel_cu(m)
        
        reg_cpp = '#include <NvInferRuntime.h>\n'
        for m in unique_manifests:
            reg_cpp += f'#include "{m.kernel_name}Plugin.h"\n'
        
        reg_cpp += '''
extern "C" {
    __attribute__((visibility("default"))) bool register_triton_plugins_explicit() {
        static bool g_registered = false;
        if (g_registered) return true;
        auto* registry = ::getPluginRegistry();
        if (registry != nullptr) {
'''
        for m in unique_manifests:
            plugin_name = f"{m.kernel_name}Plugin"
            ns = f"{self.plugin_namespace}_{m.kernel_name}"
            reg_cpp += f'            auto* creator_{m.kernel_name} = new {ns}::{plugin_name}Creator();\n'
            reg_cpp += f'            creator_{m.kernel_name}->setPluginNamespace("triton_custom");\n'
            reg_cpp += f'            registry->registerCreator(*creator_{m.kernel_name}, "triton_custom");\n'

        reg_cpp += '''
            g_registered = true;
            return true;
        }
        return false;
    }

    __attribute__((visibility("default"))) bool initLibNvInferPlugins(void* logger, const char* libNamespace) {
        return register_triton_plugins_explicit();
    }
}
'''
        files["register_plugins.cpp"] = reg_cpp
        return files

def generate_trt_bindings(manifests: List[KernelManifest], output_dir: str):
    gen = TensorRTPluginGenerator(manifests)
    files = gen.generate()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(content)