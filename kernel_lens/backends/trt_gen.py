from typing import Dict, List
import textwrap
import json
import re
import os
from ..compiler.manifest import KernelManifest

class TensorRTPluginGenerator:
    def __init__(self, manifests: List[KernelManifest], plugin_namespace: str = "custom", plugin_version: str = "1"):
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
        
        tpl = f'''
#ifndef {manifest.kernel_name.upper()}_PLUGIN_H
#define {manifest.kernel_name.upper()}_PLUGIN_H

#include "NvInferPlugin.h"
#include <cuda.h>
#include <string>
#include <vector>

namespace {self.plugin_namespace} {{

class {plugin_name} : public nvinfer1::IPluginV2DynamicExt {{
public:
    {plugin_name}();
    {plugin_name}(const void* data, size_t length);
    
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    
    int getNbOutputs() const noexcept override {{ 
        return {len([a for a in manifest.arguments if a.kind == 'output'])}; 
    }}
    
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override {{
        // Fallback: Safely mirror the first input's dimensions
        return inputs[0]; 
    }}
    
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept {{ 
        {supports_format_cxx}
    }}
    
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;
    
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
REGISTER_TENSORRT_PLUGIN({plugin_name}Creator);

}} // namespace {self.plugin_namespace}

#endif
'''
        return textwrap.dedent(tpl).strip()

    def _generate_kernel_cu(self, manifest: KernelManifest) -> str:
        plugin_name = f"{manifest.kernel_name}Plugin"
        ptx_encoded = json.dumps(manifest.ptx)
        block_size = manifest.num_warps * 32
        
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

        ptx_params = re.findall(r'\.param\s+\.([a-z0-9]+).*?([a-zA-Z0-9_]+)(?:,|\s*\))', manifest.ptx)
        
        ptx_ordered_slots = []
        for p_type, p_ident in ptx_params:
            match = re.search(r'_param_(\d+)$', p_ident)
            if match:
                p_ident = match.group(1)
                
            if "32" in p_type and "f" not in p_type: c_type = "int32_t"
            elif "64" in p_type and "f" not in p_type: c_type = "int64_t"
            elif "f32" in p_type: c_type = "float"
            elif "f64" in p_type: c_type = "double"
            else: c_type = "int32_t"
            
            ptx_ordered_slots.append({"type": c_type, "ident": p_ident})

        arg_setup_lines = []
        num_ptx_slots = len(ptx_ordered_slots)
        
        input_list = [arg for arg in manifest.arguments if arg.kind == 'input']
        output_list = [arg for arg in manifest.arguments if arg.kind == 'output']
        scalar_list = [arg for arg in manifest.arguments if arg.kind == 'scalar']
        
        input_idx, output_idx, scalar_idx = 0, 0, 0
        
        for slot_idx, slot in enumerate(ptx_ordered_slots):
            c_type = slot["type"]
            
            if input_idx < len(input_list):
                arg_setup_lines.append(f"const void* arg_{slot_idx} = inputs[{input_idx}];")
                arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&arg_{slot_idx};")
                input_idx += 1
                continue
            elif output_idx < len(output_list):
                arg_setup_lines.append(f"void* arg_{slot_idx} = outputs[{output_idx}];")
                arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&arg_{slot_idx};")
                output_idx += 1
                continue
                
            if scalar_idx < len(scalar_list):
                while scalar_idx < len(scalar_list) and scalar_list[scalar_idx].value == 1 and c_type in ["int32_t", "int64_t"]:
                    scalar_idx += 1
                    
                if scalar_idx < len(scalar_list):
                    arg = scalar_list[scalar_idx]
                    expr = getattr(arg, 'cxx_expr', '') or f"m_{arg.name}"
                    arg_setup_lines.append(f"{c_type} arg_{slot_idx} = {expr};")
                    arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&arg_{slot_idx};")
                    scalar_idx += 1
                    continue
                    
            arg_setup_lines.append(f"{c_type} implicit_pad_{slot_idx} = 0;")
            arg_setup_lines.append(f"kernelParams[{slot_idx}] = (void*)&implicit_pad_{slot_idx};")

        # --- KERNEL PARAMS PACKING (TRITON DCE AVOIDANCE) ---
        # --- KERNEL PARAMS PACKING (TRITON DCE AVOIDANCE) ---
        args_packing = []
        in_idx = 0
        out_idx = 0
        
        # --- KERNEL PARAMS PACKING (TRITON DCE AVOIDANCE) ---
        args_packing = []
        # On crée une vraie variable en mémoire contenant un pointeur nul
        args_packing.append("void* dummy_ptr = nullptr;")
        in_idx = 0
        out_idx = 0
        
        for arg in manifest.arguments:
            if arg.kind == 'input':
                args_packing.append(f"kernelParams_vec.push_back((void*)&inputs[{in_idx}]);")
                in_idx += 1
            elif arg.kind == 'output':
                args_packing.append(f"kernelParams_vec.push_back((void*)&outputs[{out_idx}]);")
                out_idx += 1
            elif arg.kind == 'scalar':
                val = arg.value
                # L'heuristique Triton equal_to_1 supprime l'argument du PTX
                if val == 1:
                    args_packing.append(f"// Triton equal_to_1 DCE: Skipped {arg.name}")
                else:
                    args_packing.append(f"kernelParams_vec.push_back((void*)&m_{arg.name});")
                    
        # On remplit la fin de la signature avec l'ADRESSE de notre pointeur nul
        pad_str = f"while(kernelParams_vec.size() < {len(manifest.arguments)}) kernelParams_vec.push_back((void*)&dummy_ptr);"
        args_packing.append(pad_str)
        
        kernel_params_cpp = "\n    ".join(args_packing)

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

namespace {self.plugin_namespace} {{

const char* {plugin_name}::PTX_CODE = {ptx_encoded};

{plugin_name}::{plugin_name}() : mNbOutputs({len([a for a in manifest.arguments if a.kind == 'output'])}){init_str} {{
    mNamespace = "";
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

int {plugin_name}::initialize() noexcept {{
    if (mModule == nullptr) {{
        CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) return -1;
        
        res = cuModuleGetFunction(&mKernel, mModule, "{manifest.kernel_name}");
        if (res != CUDA_SUCCESS) return -1;
    }}
    return 0;
}}

void {plugin_name}::terminate() noexcept {{
    if (mModule) {{
        cuModuleUnload(mModule);
        mModule = nullptr;
    }}
}}

int {plugin_name}::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {{
    if (!mKernel) this->initialize();
    
    unsigned int grid_x = std::max(1u, (unsigned int)({grid_x_cxx}));
    unsigned int grid_y = std::max(1u, (unsigned int)({grid_y_cxx}));
    unsigned int grid_z = std::max(1u, (unsigned int)({grid_z_cxx}));
    unsigned int block_x = std::max(1u, (unsigned int)({block_size}));
    
    void* kernelParams[{num_ptx_slots}];
    {dynamic_args_cpp}
    
    std::vector<void*> kernelParams_vec;
    /*for(int i = 0; i < {num_ptx_slots}; i++) {{
        kernelParams_vec.push_back(kernelParams[i]);
    }}*/
    {kernel_params_cpp}
    
    CUresult launch_status = cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, {manifest.shared_memory_bytes}, stream, kernelParams_vec.data(), nullptr);
    /*printf("=======================================\\n");
    printf("[TRT C++ DEBUG] Kernel Enqueue Fired!\\n");
    for (size_t i = 0; i < kernelParams_vec.size(); i++) {{
        void* actual_ptr = *(void**)kernelParams_vec[i]; 
        printf("[TRT C++ DEBUG] Argument %zu VRAM Address: %p\\n", i, actual_ptr);
    }}
    printf("=======================================\\n");*/
    
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

void {plugin_name}::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {{}}
size_t {plugin_name}::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs, const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {{ return 0; }}

//nvinfer1::DataType {plugin_name}::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {{ 
//    return inputTypes[0]; 
//}}

nvinfer1::DataType {plugin_name}::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept {{ 
    {dynamic_output_types}
    return inputTypes[0]; // Ultimate fallback
}}

void {plugin_name}::destroy() noexcept {{ delete this; }}
void {plugin_name}::setPluginNamespace(const char* pluginNamespace) noexcept {{ mNamespace = pluginNamespace; }}
const char* {plugin_name}::getPluginNamespace() const noexcept {{ return mNamespace.c_str(); }}

// Plugin creator methods
const char* {plugin_name}Creator::getPluginName() const noexcept {{ return "{manifest.kernel_name}"; }}
const char* {plugin_name}Creator::getPluginVersion() const noexcept {{ return "{self.plugin_version}"; }}

{plugin_name}Creator::{plugin_name}Creator() {{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    mNamespace = "";
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

}} // namespace {self.plugin_namespace}
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