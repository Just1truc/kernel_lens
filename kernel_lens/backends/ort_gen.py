import os
import textwrap
import re

class ORTGenerator:
    def __init__(self, manifests, ops_package="triton_custom"):
        self.manifests = manifests
        self.ops_package = ops_package

    def _generate_kernel_h(self, manifest) -> str:
        op_name = f"{manifest.kernel_name}Op"
        kernel_name = f"{manifest.kernel_name}Kernel"
        
        inputs_to_node = manifest.arguments
        outputs_from_node = [a for a in manifest.arguments if a.kind == 'output']
        
        input_types_cpp = []
        mem_types_cpp = []
        
        for a in inputs_to_node:
            if a.kind == 'scalar':
                # --- NEW: Keep Scalars on the CPU so we can safely read them! ---
                mem_types_cpp.append("OrtMemTypeCPUInput")
                if 'float' in a.dtype.lower():
                    input_types_cpp.append("ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE")
                else:
                    input_types_cpp.append("ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64")
            else:
                # --- Keep Tensors on the GPU ---
                mem_types_cpp.append("OrtMemTypeDefault")
                input_types_cpp.append("ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT")
                
        input_types_str = ",\n            ".join(input_types_cpp)
        mem_types_str = ",\n            ".join(mem_types_cpp)
        
        tpl = f'''#pragma once
// Unlock the Custom Op Initialization API
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <cuda.h>

namespace custom {{

struct {kernel_name} {{
    void Compute(OrtKernelContext* context);
}};

struct {op_name} : Ort::CustomOpBase<{op_name}, {kernel_name}> {{
    {op_name}() = default;

    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {{
        return new {kernel_name}();
    }}

    const char* GetName() const {{ return "{manifest.kernel_name}"; }}
    const char* GetExecutionProviderType() const {{ return "CUDAExecutionProvider"; }}

    size_t GetInputTypeCount() const {{ return {len(inputs_to_node)}; }}
    ONNXTensorElementDataType GetInputType(size_t index) const {{
        static const ONNXTensorElementDataType types[] = {{
            {input_types_str}
        }};
        return types[index];
    }}

    size_t GetOutputTypeCount() const {{ return {max(1, len(outputs_from_node))}; }}
    ONNXTensorElementDataType GetOutputType(size_t index) const {{
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }}

    // --- THE MAGIC FIX: Dynamic Memory Placement ---
    OrtMemType GetInputMemoryType(size_t index) const {{
        static const OrtMemType types[] = {{
            {mem_types_str}
        }};
        return types[index];
    }}
    
    OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {{
        return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
    }}
}};

}} // namespace custom
'''
        return textwrap.dedent(tpl).strip()

    def _generate_kernel_cu(self, manifest) -> str:
        kernel_name = f"{manifest.kernel_name}Kernel"
        
        arg_setup_lines = []
        arg_setup_lines.append("Ort::KernelContext ctx(context);")
        arg_setup_lines.append("auto ref_in = ctx.GetInput(0);")
        arg_setup_lines.append("auto info = ref_in.GetTensorTypeAndShapeInfo();")
        arg_setup_lines.append("std::vector<int64_t> dim_values = info.GetShape();")
        
        import re
        
        # --- ROBUST GRID EVALUATION ---
        grid_strs = []
        if hasattr(manifest, '_sym_grid_asts') and manifest._sym_grid_asts:
            for g in manifest._sym_grid_asts:
                # Always extract pure SymPy string FIRST
                expr = str(g.node.expr) if hasattr(g, 'node') else str(g)
                expr = re.sub(r'([a-zA-Z0-9_]+)\*\*([a-zA-Z0-9_]+)', r'std::pow(\1, \2)', expr)
                expr = expr.replace("//", "/")
                expr = re.sub(r'floor\((.*?)\)', r'(\1)', expr)
                
                # Replace SymPy symbols with ORT C++
                # Use regex to avoid replacing s0 inside s01
                expr = re.sub(r'\bs0\b', "(int64_t)dim_values[0]", expr)
                expr = re.sub(r'\bs1\b', "(int64_t)(dim_values.size() > 1 ? dim_values[1] : 1)", expr)
                expr = re.sub(r'\bs2\b', "(int64_t)(dim_values.size() > 2 ? dim_values[2] : 1)", expr)
                grid_strs.append(expr)
                
        while len(grid_strs) < 3:
            grid_strs.append("1")
            
        grid_eval_lines = [
            f"unsigned int grid_x = std::max(1u, (unsigned int)({grid_strs[0]}));",
            f"unsigned int grid_y = std::max(1u, (unsigned int)({grid_strs[1]}));",
            f"unsigned int grid_z = std::max(1u, (unsigned int)({grid_strs[2]}));"
        ]

        arg_setup_lines.append("// Stable memory addresses for CUDA kernel launch")
        for slot_idx, arg in enumerate(manifest.arguments):
            if arg.kind == 'input':
                arg_setup_lines.append(f"const void* arg_{slot_idx} = nullptr;")
            elif arg.kind == 'scalar':
                if 'float' in arg.dtype.lower():
                    arg_setup_lines.append(f"float arg_{slot_idx} = 0.0f;")
                else:
                    arg_setup_lines.append(f"int32_t arg_{slot_idx} = 0;")
            elif arg.kind == 'output':
                arg_setup_lines.append(f"void* arg_{slot_idx} = nullptr;")
                
        padding_idx = len(manifest.arguments)
        arg_setup_lines.append(f"int64_t pad_{padding_idx} = 0;")

        arg_setup_lines.append("std::vector<void*> kp;")

        input_idx = 0
        output_idx = 0
        
        for slot_idx, arg in enumerate(manifest.arguments):
            if arg.kind == 'input':
                arg_setup_lines.append(f"auto input_{slot_idx} = ctx.GetInput({input_idx});")
                arg_setup_lines.append(f"arg_{slot_idx} = (const void*)input_{slot_idx}.GetTensorData<float>();")
                arg_setup_lines.append(f"kp.push_back(&arg_{slot_idx});")
                input_idx += 1
            elif arg.kind == 'scalar':
                arg_setup_lines.append(f"auto input_{slot_idx} = ctx.GetInput({input_idx});")
                # NOW THIS IS 100% SAFE because ONNX stored it in CPU memory!
                if 'float' in arg.dtype.lower():
                    arg_setup_lines.append(f"arg_{slot_idx} = (float)(*input_{slot_idx}.GetTensorData<double>());")
                else:
                    arg_setup_lines.append(f"arg_{slot_idx} = (int32_t)(*input_{slot_idx}.GetTensorData<int64_t>());")
                
                arg_name = getattr(arg, 'name', '')
                if 'stride' in arg_name.lower():
                    arg_setup_lines.append(f"if (arg_{slot_idx} != 1) kp.push_back(&arg_{slot_idx});")
                else:
                    arg_setup_lines.append(f"kp.push_back(&arg_{slot_idx});")
                input_idx += 1
            elif arg.kind == 'output':
                arg_setup_lines.append(f"// Skip PyTorch's dummy input")
                arg_setup_lines.append(f"auto dummy_in_{slot_idx} = ctx.GetInput({input_idx});")
                input_idx += 1
                
                arg_setup_lines.append(f"auto output_{slot_idx} = ctx.GetOutput({output_idx}, dim_values.data(), dim_values.size());")
                arg_setup_lines.append(f"arg_{slot_idx} = (void*)output_{slot_idx}.GetTensorMutableData<float>();")
                arg_setup_lines.append(f"kp.push_back(&arg_{slot_idx});")
                output_idx += 1
                
        arg_setup_lines.append(f"kp.push_back(&pad_{padding_idx});")

        dynamic_args_cpp = "\n    ".join(arg_setup_lines)
        grid_cpp = "\n    ".join(grid_eval_lines)

        tpl = f'''#include "{manifest.kernel_name}Op.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace custom {{

static const char* PTX_CODE = R"ptx(
{manifest.ptx}
)ptx";

void {kernel_name}::Compute(OrtKernelContext* context) {{
    {dynamic_args_cpp}

    static CUmodule mModule = nullptr;
    static CUfunction mKernel = nullptr;
    
    if (mModule == nullptr) {{
        CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) throw std::runtime_error("Failed to load PTX module");
        res = cuModuleGetFunction(&mKernel, mModule, "{manifest.kernel_name}");
        if (res != CUDA_SUCCESS) throw std::runtime_error("Failed to extract function");
    }}

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
    
    {grid_cpp}
    unsigned int block_x = 128; 

    cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, {manifest.shared_memory_bytes}, stream, kp.data(), nullptr);
}}

}} // namespace custom
'''
        return textwrap.dedent(tpl).strip()

    def generate(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for m in self.manifests:
            base_path = os.path.join(output_dir, f"{m.kernel_name}Op")
            with open(f"{base_path}.h", "w") as f:
                f.write(self._generate_kernel_h(m))
            with open(f"{base_path}.cu", "w") as f:
                f.write(self._generate_kernel_cu(m))

        registrar_code = '''#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

'''
        for m in self.manifests:
            registrar_code += f'#include "{m.kernel_name}Op.h"\n'

        registrar_code += '''
#ifndef ORT_EXPORT
#ifdef _WIN32
#define ORT_EXPORT __declspec(dllexport)
#else
#define ORT_EXPORT __attribute__((visibility("default")))
#endif
#endif

extern "C" {
    ORT_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
        Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
        static Ort::CustomOpDomain custom_domain("triton_custom");
'''
        for m in self.manifests:
            registrar_code += f'        static custom::{m.kernel_name}Op c_{m.kernel_name};\n'
            registrar_code += f'        custom_domain.Add(&c_{m.kernel_name});\n'

        registrar_code += '''
        Ort::UnownedSessionOptions sess_options(options);
        sess_options.Add(custom_domain);
        return nullptr;
    }
}
'''
        with open(os.path.join(output_dir, "register_ops.cpp"), "w") as f:
            f.write(registrar_code)

def generate_ort_bindings(manifests, output_path: str):
    gen = ORTGenerator(manifests)
    gen.generate(output_path)