from typing import Dict, List
import textwrap
import json
import re
import os
from ..compiler.manifest import KernelManifest

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

        supports_format_cxx = "return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;"
        
        output_dim_cases = []
        out_args = [a for a in manifest.arguments if a.kind == 'output']
        for out_idx, out_arg in enumerate(out_args):
            output_dim_cases.append(f"if (outputIndex == {out_idx}) {{")
            output_dim_cases.append(f"    nvinfer1::DimsExprs res;")
            output_dim_cases.append(f"    res.nbDims = {len(out_arg.shape)};")
            for d_idx, d in enumerate(out_arg.shape):
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


namespace {self.plugin_namespace}_{manifest.kernel_name} {{


class {plugin_name} : public nvinfer1::IPluginV2DynamicExt {{
public:
    {plugin_name}();
    {plugin_name}(const void* data, size_t length);
    
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    
    int32_t getNbOutputs() const noexcept override {{ 
        return {len([a for a in manifest.arguments if a.kind == 'output'])}; 
    }}
    
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
        ptx_encoded = json.dumps(manifest.ptx)
        block_size = (manifest.num_warps if manifest.num_warps > 0 else 4) * 32
        
        scalars = [arg for arg in manifest.arguments if arg.kind == 'scalar']
        nb_outputs = sum(1 for arg in manifest.arguments if arg.kind == 'output')
        nb_outputs = max(1, nb_outputs)
        
        # --- LOCAL TRT GRID AST EVALUATION ---
        import re
        
        # --- LOCAL TRT GRID AST EVALUATION ---
        grid_strs = []
        if hasattr(manifest, '_sym_grid_asts') and manifest._sym_grid_asts:
            for g in manifest._sym_grid_asts:
                expr = str(g.node.expr) if hasattr(g, 'node') else str(g)
                expr = re.sub(r'([a-zA-Z0-9_]+)\*\*([a-zA-Z0-9_]+)', r'std::pow(\1, \2)', expr)
                expr = expr.replace("//", "/")
                expr = re.sub(r'floor\((.*?)\)', r'(\1)', expr)
                
                # Replace SymPy symbols with TRT C++
                expr = re.sub(r'\bs0\b', "(int64_t)inputDesc[0].dims.d[0]", expr)
                expr = re.sub(r'\bs1\b', "(int64_t)(inputDesc[0].dims.nbDims > 1 ? inputDesc[0].dims.d[1] : 1)", expr)
                expr = re.sub(r'\bs2\b', "(int64_t)(inputDesc[0].dims.nbDims > 2 ? inputDesc[0].dims.d[2] : 1)", expr)
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
        ptr_args = [arg for arg in active_args if arg.kind in ('input', 'output')]
        scalar_args = [arg for arg in active_args if arg.kind == 'scalar']
        num_ptr_args = len(ptr_args)

        ptx_ordered_slots = []
        for s_idx, (p_decl, p_ident) in enumerate(ptx_params):
            match = re.search(r'_param_(\d+)$', p_ident)
            if match:
                p_ident = match.group(1)
            arg = active_args[s_idx] if s_idx < len(active_args) else None
            if arg:
                is_ptr = arg.kind in ('input', 'output')
            else:
                is_ptr = ("ptr" in p_decl)

            if is_ptr: c_type = "void*"
            elif "32" in p_decl and "f" not in p_decl: c_type = "int32_t"
            elif "64" in p_decl and "f" not in p_decl: c_type = "int64_t"
            elif "f32" in p_decl: c_type = "float"
            elif "f64" in p_decl: c_type = "double"
            else: c_type = "int32_t"
            ptx_ordered_slots.append({"type": c_type, "ident": p_ident, "is_ptr": is_ptr})

        num_ptx_slots = len(ptx_ordered_slots)
        in_counter = 0
        out_counter = 0
        arg_setup_lines = []

        for slot_idx, slot in enumerate(ptx_ordered_slots):
            c_type = slot["type"]
            arg = active_args[slot_idx] if slot_idx < len(active_args) else None

            if slot["is_ptr"]:
                if arg and arg.kind == 'input':
                    arg_setup_lines.append(f"static thread_local const void* tmp_ptr_{slot_idx}; tmp_ptr_{slot_idx} = (const void*)inputs[{in_counter}];")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_ptr_{slot_idx};")
                    in_counter += 1
                elif arg and arg.kind == 'output':
                    arg_setup_lines.append(f"static thread_local void* tmp_ptr_{slot_idx}; tmp_ptr_{slot_idx} = (void*)outputs[{out_counter}];")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_ptr_{slot_idx};")
                    out_counter += 1
                else:
                    arg_setup_lines.append(f"static thread_local void* tmp_null_{slot_idx} = nullptr;")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_null_{slot_idx};")
            else:
                if arg and arg.kind == 'scalar':
                    expr = getattr(arg, 'cxx_expr', '') or f"m_{arg.name}"
                    scalar_ctype = "int64_t" if "64" in str(arg.dtype) else ("float" if ("float" in str(arg.dtype) or "float" in c_type or "double" in c_type) else "int32_t")
                    arg_setup_lines.append(f"static thread_local {scalar_ctype} tmp_scalar_{slot_idx}; tmp_scalar_{slot_idx} = ({scalar_ctype})({expr});")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_scalar_{slot_idx};")
                else:
                    arg_setup_lines.append(f"static thread_local {c_type} tmp_scalar_{slot_idx} = 0;")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&tmp_scalar_{slot_idx};")


        # --- DYNAMIC OUTPUT DATATYPES ---
        output_type_lines = []
        out_args = [a for a in manifest.arguments if a.kind == 'output']
        for i, out_arg in enumerate(out_args):
            # Map PyTorch/Triton dtypes to TensorRT Enums
            if 'float16' in out_arg.dtype or 'half' in out_arg.dtype:
                trt_type = "nvinfer1::DataType::kHALF"
            elif 'float' in out_arg.dtype:
                trt_type = "nvinfer1::DataType::kFLOAT"
            elif 'int64' in out_arg.dtype or 'long' in out_arg.dtype:
                trt_type = "nvinfer1::DataType::kINT64"
            elif 'int' in out_arg.dtype:
                trt_type = "nvinfer1::DataType::kINT32"
            elif 'bool' in out_arg.dtype:
                trt_type = "nvinfer1::DataType::kBOOL"
            else:
                trt_type = "inputTypes[0]" # Fallback
                
            output_type_lines.append(f"if (index == {i}) return {trt_type};")
            
        dynamic_output_types = "\n    ".join(output_type_lines)

        dynamic_args_cpp = "\n    ".join(arg_setup_lines)
        
        tpl = f'''
#include "{manifest.kernel_name}Plugin.h"
#include <cuda.h>
#include <cstring>
#include <iostream>

namespace {self.plugin_namespace}_{manifest.kernel_name} {{


const char* {plugin_name}::PTX_CODE = {ptx_encoded};

{plugin_name}::{plugin_name}() : mNbOutputs({len([a for a in manifest.arguments if a.kind == 'output'])}){init_str} {{
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
        CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) return -1;
        
        res = cuModuleGetFunction(&mKernel, mModule, "{ptx_entry_name}");
        if (res != CUDA_SUCCESS) return -1;
        cuFuncSetAttribute(mKernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {manifest.shared_memory_bytes});
    }}
    return 0;
}}


void {plugin_name}::terminate() noexcept {{
    mModule = nullptr;
    mKernel = nullptr;
}}


int32_t {plugin_name}::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {{

    if (!mKernel) this->initialize();

    unsigned int grid_x = std::max(1u, (unsigned int)({grid_x_cxx}));
    unsigned int grid_y = std::max(1u, (unsigned int)({grid_y_cxx}));
    unsigned int grid_z = std::max(1u, (unsigned int)({grid_z_cxx}));
    unsigned int block_x = std::max(1u, (unsigned int)({block_size}));

    void* kernelParams[{num_ptx_slots}];
    {dynamic_args_cpp}

    cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, {manifest.shared_memory_bytes}, stream, kernelParams, nullptr);

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

extern "C" {{
    __attribute__((visibility("default"))) nvinfer1::IPluginCreator* const* getPluginCreators(int32_t& nbCreators) {{
        static auto* creator = new {self.plugin_namespace}_{manifest.kernel_name}::{plugin_name}Creator();
        static nvinfer1::IPluginCreator* const creators[] = {{ creator }};
        nbCreators = 1;
        return creators;
    }}

    __attribute__((visibility("default"))) bool register_triton_plugins_explicit() {{
        static bool g_registered = false;
        if (g_registered) return true;
        auto* registry = ::getPluginRegistry();
        if (registry != nullptr) {{
            auto* creator = new {self.plugin_namespace}_{manifest.kernel_name}::{plugin_name}Creator();
            creator->setPluginNamespace("triton_custom");
            registry->registerCreator(*creator, "triton_custom");
            g_registered = true;
            return true;
        }}
        return false;
    }}




    __attribute__((visibility("default"))) bool initLibNvInferPlugins(void* logger, const char* libNamespace) {{
        return register_triton_plugins_explicit();
    }}
}}









































'''
        return textwrap.dedent(tpl).strip()
    
    def generate(self) -> Dict[str, str]:
        files = {}
        for m in self.manifests:
            files[f"{m.kernel_name}Plugin.h"] = self._generate_kernel_h(m)
            files[f"{m.kernel_name}Plugin.cu"] = self._generate_kernel_cu(m)
        return files

def generate_trt_bindings(manifests: List[KernelManifest], output_dir: str):
    gen = TensorRTPluginGenerator(manifests)
    files = gen.generate()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename, content in files.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            f.write(content)