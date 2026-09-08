import os
import textwrap
import re
from ..config import is_verbose

class ORTGenerator:
    def __init__(self, manifests, ops_package="triton_custom"):
        self.manifests = manifests
        self.ops_package = ops_package

    def _generate_kernel_h(self, manifest) -> str:
        op_name = f"{manifest.kernel_name}Op"
        kernel_name = f"{manifest.kernel_name}Kernel"
        
        inputs_to_node = [a for a in manifest.arguments if (a.kind in ('input', 'inplace') or (a.kind == 'scalar' and not getattr(a, 'is_constexpr', False)))]
        outputs_from_node = [a for a in manifest.arguments if a.kind in ('output', 'inplace')]
        
        input_types_cpp = []
        mem_types_cpp = []
        
        def torch_dtype_to_onnx_str(dtype_val) -> str:
            dtype_str = str(dtype_val).lower()
            if 'bool' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL"
            elif 'uint8' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8"
            elif 'int8' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8"
            elif 'int16' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16"
            elif 'int32' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32"
            elif 'int64' in dtype_str or 'int' in dtype_str or 'long' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64"
            elif 'float16' in dtype_str or 'fp16' in dtype_str or 'half' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16"
            elif 'bfloat16' in dtype_str or 'bf16' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16"
            elif 'double' in dtype_str or 'float64' in dtype_str:
                return "ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE"
            return "ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"

        for a in inputs_to_node:
            if a.kind == 'scalar':
                mem_types_cpp.append("OrtMemTypeCPUInput")
            else:
                mem_types_cpp.append("OrtMemTypeDefault")
            input_types_cpp.append(torch_dtype_to_onnx_str(a.dtype))

        output_types_cpp = [torch_dtype_to_onnx_str(a.dtype) for a in outputs_from_node]
        if not output_types_cpp:
            output_types_cpp = ["ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT"]
            
        output_types_str = ",\n            ".join(output_types_cpp)
                
        input_types_str = ",\n            ".join(input_types_cpp)
        mem_types_str = ",\n            ".join(mem_types_cpp)
        
        tpl = f'''#pragma once
// Unlock the Custom Op Initialization API
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#include <cuda.h>
#include <iostream>
#include <cstdio>

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
        static const ONNXTensorElementDataType types[] = {{
            {output_types_str}
        }};
        return types[index];
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
        fn_match = re.search(r'\.entry\s+([a-zA-Z0-9_]+)', manifest.ptx)
        ptx_entry_name = fn_match.group(1) if fn_match else manifest.kernel_name
        
        # --- ROBUST GRID EVALUATION WITH GENERIC REGEX SUBSTITUTION ---
        grid_strs = []
        if hasattr(manifest, '_sym_grid_asts') and manifest._sym_grid_asts:
            for g in manifest._sym_grid_asts:
                expr = str(g.node.expr) if hasattr(g, 'node') else str(g)
                expr = re.sub(r'([a-zA-Z0-9_]+)\*\*([a-zA-Z0-9_]+)', r'std::pow(\1, \2)', expr)
                expr = expr.replace("//", "/")
                expr = re.sub(r'floor\((.*?)\)', r'(\1)', expr)
                
                # Replace any s(\d+) symbol dynamically using regex
                expr = re.sub(r'\bs(\d+)\b', r"(int64_t)(dim_values.size() > \1 ? dim_values[\1] : 1)", expr)
                grid_strs.append(expr)
                
        while len(grid_strs) < 3:
            grid_strs.append("1")
            
        grid_eval_lines = [
            f"unsigned int grid_x = std::max(1u, (unsigned int)({grid_strs[0]}));",
            f"unsigned int grid_y = std::max(1u, (unsigned int)({grid_strs[1]}));",
            f"unsigned int grid_z = std::max(1u, (unsigned int)({grid_strs[2]}));"
        ]

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

        active_args = [a for a in manifest.arguments if not getattr(a, 'is_constexpr', False)]
        ptx_ordered_slots = []
        for p_idx, (p_decl, p_ident) in enumerate(ptx_params):
            match = re.search(r'_param_(\d+)$', p_ident)
            param_num = int(match.group(1)) if match else p_idx
            arg = active_args[param_num] if param_num < len(active_args) else (active_args[p_idx] if p_idx < len(active_args) else None)
            if arg:
                is_ptr = arg.kind in ('input', 'output', 'inplace')
            else:
                is_ptr = ("ptr" in p_decl)

            if is_ptr:
                c_type = "void*"
            elif "64" in p_decl or ".u64" in p_decl or ".s64" in p_decl:
                c_type = "int64_t"
            elif "f32" in p_decl:
                c_type = "float"
            elif "f64" in p_decl:
                c_type = "double"
            else:
                c_type = "int32_t"
            
            ptx_ordered_slots.append({"type": c_type, "ident": str(param_num), "is_ptr": is_ptr, "decl": p_decl, "param_num": param_num})

        arg_setup_lines.append("// Extract inputs/outputs/scalars from ORT context")
        onnx_input_counter = 0
        ort_output_counter = 0
        
        for slot_idx, arg in enumerate(manifest.arguments):
            if arg.kind == 'input':
                arg_setup_lines.append(f"auto in_tensor_{slot_idx} = ctx.GetInput({onnx_input_counter});")
                arg_setup_lines.append(f"static thread_local const void* arg_ptr_{slot_idx};")
                arg_setup_lines.append(f"arg_ptr_{slot_idx} = (const void*)in_tensor_{slot_idx}.GetTensorData<float>();")
                onnx_input_counter += 1
            elif arg.kind == 'output':
                out_dim_exprs = []
                for dim_idx, d in enumerate(arg.shape):
                    try:
                        d_int = int(d)
                        out_dim_exprs.append(str(d_int))
                    except Exception:
                        out_dim_exprs.append(f"(int64_t)(dim_values.size() > {dim_idx} ? dim_values[{dim_idx}] : 1)")
                out_shape_str = "{" + ", ".join(out_dim_exprs) + "}" if out_dim_exprs else "dim_values"
                
                arg_setup_lines.append(f"std::vector<int64_t> out_dims_{ort_output_counter} = {out_shape_str};")
                arg_setup_lines.append(f"auto out_tensor_{slot_idx} = ctx.GetOutput({ort_output_counter}, out_dims_{ort_output_counter}.data(), out_dims_{ort_output_counter}.size());")
                arg_setup_lines.append(f"static thread_local void* arg_ptr_{slot_idx};")
                arg_setup_lines.append(f"arg_ptr_{slot_idx} = (void*)out_tensor_{slot_idx}.GetTensorMutableData<float>();")
                ort_output_counter += 1
            elif arg.kind == 'inplace':
                out_dim_exprs = []
                for dim_idx, d in enumerate(arg.shape):
                    try:
                        d_int = int(d)
                        out_dim_exprs.append(str(d_int))
                    except Exception:
                        out_dim_exprs.append(f"(int64_t)(dim_values.size() > {dim_idx} ? dim_values[{dim_idx}] : 1)")
                out_shape_str = "{" + ", ".join(out_dim_exprs) + "}" if out_dim_exprs else "dim_values"
                
                arg_setup_lines.append(f"auto in_tensor_{slot_idx} = ctx.GetInput({onnx_input_counter});")
                arg_setup_lines.append(f"std::vector<int64_t> out_dims_{ort_output_counter} = {out_shape_str};")
                arg_setup_lines.append(f"auto out_tensor_{slot_idx} = ctx.GetOutput({ort_output_counter}, out_dims_{ort_output_counter}.data(), out_dims_{ort_output_counter}.size());")
                arg_setup_lines.append(f"const void* in_ptr_{slot_idx} = (const void*)in_tensor_{slot_idx}.GetTensorData<float>();")
                arg_setup_lines.append(f"void* out_ptr_{slot_idx} = (void*)out_tensor_{slot_idx}.GetTensorMutableData<float>();")
                arg_setup_lines.append(f"size_t num_bytes_{slot_idx} = in_tensor_{slot_idx}.GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float);")
                arg_setup_lines.append(f"if (in_ptr_{slot_idx} != out_ptr_{slot_idx}) {{ cudaMemcpyAsync(out_ptr_{slot_idx}, in_ptr_{slot_idx}, num_bytes_{slot_idx}, cudaMemcpyDeviceToDevice, (cudaStream_t)ctx.GetGPUComputeStream()); }}")
                arg_setup_lines.append(f"static thread_local void* arg_ptr_{slot_idx};")
                arg_setup_lines.append(f"arg_ptr_{slot_idx} = out_ptr_{slot_idx};")
                onnx_input_counter += 1
                ort_output_counter += 1
            elif arg.kind == 'scalar':
                if not getattr(arg, 'is_constexpr', False):
                    arg_setup_lines.append(f"auto scalar_tensor_{slot_idx} = ctx.GetInput({onnx_input_counter});")
                    arg_setup_lines.append(f"double scalar_val_{slot_idx} = 0;")
                    arg_setup_lines.append(f"if (scalar_tensor_{slot_idx}.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {{ scalar_val_{slot_idx} = (double)(scalar_tensor_{slot_idx}.GetTensorData<int64_t>()[0]); }}")
                    arg_setup_lines.append(f"else if (scalar_tensor_{slot_idx}.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {{ scalar_val_{slot_idx} = (double)(scalar_tensor_{slot_idx}.GetTensorData<int32_t>()[0]); }}")
                    arg_setup_lines.append(f"else {{ scalar_val_{slot_idx} = (double)(scalar_tensor_{slot_idx}.GetTensorData<float>()[0]); }}")
                    onnx_input_counter += 1
                else:
                    s_val = arg.value if getattr(arg, 'value', None) is not None else 0.0
                    arg_setup_lines.append(f"double scalar_val_{slot_idx} = {s_val};")

        arg_setup_lines.append("static thread_local const void* null_ptr_arg = nullptr;")
        arg_setup_lines.append("static thread_local std::vector<void*> kp;")
        arg_setup_lines.append("kp.clear();")

        for slot_idx, slot in enumerate(ptx_ordered_slots):
            c_type = slot["type"]
            param_num = slot["param_num"]
            arg = active_args[param_num] if param_num < len(active_args) else (active_args[slot_idx] if slot_idx < len(active_args) else None)

            if slot["is_ptr"] and arg and arg.kind in ('input', 'output', 'inplace'):
                orig_idx = manifest.arguments.index(arg)
                arg_setup_lines.append(f"kp.push_back((void*)&arg_ptr_{orig_idx});")
            elif slot["is_ptr"] and (not arg or arg.kind not in ('input', 'output', 'inplace')):
                arg_setup_lines.append(f"static thread_local int64_t dummy_stride_{slot_idx} = 1;")
                arg_setup_lines.append(f"kp.push_back((void*)&dummy_stride_{slot_idx});")
            else:
                if arg and arg.kind == 'scalar':
                    orig_idx = manifest.arguments.index(arg)
                    arg_setup_lines.append(f"static thread_local {c_type} ptx_scalar_{slot_idx};")
                    arg_setup_lines.append(f"ptx_scalar_{slot_idx} = ({c_type})scalar_val_{orig_idx};")
                    arg_setup_lines.append(f"kp.push_back((void*)&ptx_scalar_{slot_idx});")
                else:
                    arg_setup_lines.append(f"static thread_local {c_type} dummy_scalar_{slot_idx} = 1;")
                    arg_setup_lines.append(f"kp.push_back((void*)&dummy_scalar_{slot_idx});")

        dynamic_args_cpp = "\n    ".join(arg_setup_lines)
        grid_cpp = "\n    ".join(grid_eval_lines)

        tpl = f'''#include "{manifest.kernel_name}Op.h"
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdint>

namespace custom {{

static const char* PTX_CODE = R"ptx(
{manifest.ptx}
)ptx";

void {kernel_name}::Compute(OrtKernelContext* context) {{
    {dynamic_args_cpp}

    static thread_local CUmodule mModule = nullptr;
    static thread_local CUfunction mKernel = nullptr;
    
    if (mModule == nullptr) {{
        cuInit(0);
        CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) throw std::runtime_error("Failed to load PTX module");
        res = cuModuleGetFunction(&mKernel, mModule, "{ptx_entry_name}");
        if (res != CUDA_SUCCESS) throw std::runtime_error("Failed to extract function");
        cuFuncSetAttribute(mKernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, {manifest.shared_memory_bytes});
    }}

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(ctx.GetGPUComputeStream());
    
    {grid_cpp}
    unsigned int block_x = {(manifest.num_warps if manifest.num_warps > 0 else 4) * 32};

    CUresult res = cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, {manifest.shared_memory_bytes}, stream, kp.data(), nullptr);
    if (res != CUDA_SUCCESS) {{
        printf("[CPP ERROR] cuLaunchKernel returned %d\\n", (int)res); fflush(stdout);
        throw std::runtime_error("cuLaunchKernel failed");
    }}

    cudaStreamSynchronize(stream);
    {"printf(\"[CPP DEBUG] Kernel sync complete successfully!\\\\n\"); fflush(stdout);" if is_verbose() else ""}
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