"""``IPluginV3`` emitter.

TensorRT 10 introduced ``IPluginV3`` and deprecated the whole ``IPluginV2*``
hierarchy; TensorRT 11 removed the V2 interfaces outright. This emitter produces
the same plugin semantics as ``trt_gen`` (``IPluginV2DynamicExt``) expressed
against the modern interfaces, so a plugin can be built for a TensorRT that no
longer ships the legacy API.

The two emitters share the entire execution path. The PTX entry name, the
``kernelParams`` assembly and the output-type dispatch all come from
``TensorRTPluginGenerator.build_shared_body``, and ``supportsFormatCombination``
from ``TensorRTPluginGenerator.supports_format_body`` -- so a plugin emitted here
and one emitted by ``trt_gen`` run identical code. What differs is the class
hierarchy and how the object is reconstructed:

*   V2 serialises a raw byte buffer it owns and is rebuilt by ``deserialize()``.
*   V3 returns a ``PluginFieldCollection`` from ``getFieldsToSerialize()`` and is
    rebuilt by the creator's ``createPlugin()``, which TensorRT calls with those
    fields on engine load.

Interface notes, because the multiple inheritance here is deliberately not
virtual and that matters:

*   ``IPluginV3`` and each of the three capability interfaces derive separately
    from ``IVersionedInterface``, so the concrete class holds several
    subobjects. We therefore never call ``getInterfaceInfo()`` on the concrete
    type -- TensorRT calls it through a specific interface pointer, which
    resolves to that interface's own override and is what lets it tell the
    capabilities apart. Overriding it once in the derived class would collapse
    all three to the same tag and break capability detection.
*   ``getCapabilityInterface`` casts via the specific interface
    (``static_cast<IPluginV3OneBuild*>(this)``) rather than via
    ``IPluginCapability``, which would be ambiguous.
"""
import json
import re
import textwrap
from typing import Dict, List, Tuple

from ..compiler.manifest import KernelManifest
from .trt_gen import TensorRTPluginGenerator


class TensorRTV3PluginGenerator:
    """Emit ``IPluginV3`` plugins for a set of manifests."""

    def __init__(self, manifests: List[KernelManifest],
                 plugin_namespace: str = "triton_custom",
                 plugin_version: str = "1"):
        self.manifests = manifests
        self.plugin_namespace = plugin_namespace
        self.plugin_version = plugin_version

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _scalars(manifest: KernelManifest) -> List[Tuple[str, str, str, str]]:
        """``(name, C++ type, PluginFieldType, default literal)`` per scalar."""
        out = []
        for a in manifest.arguments:
            if a.kind != 'scalar':
                continue
            ds = str(a.dtype).lower()
            if 'float' in ds or 'double' in ds:
                out.append((a.name, 'float', 'nvinfer1::PluginFieldType::kFLOAT32',
                            f"{float(a.value)!r}f"))
            elif 'int64' in ds or 'long' in ds:
                out.append((a.name, 'int64_t', 'nvinfer1::PluginFieldType::kINT64',
                            f"{int(a.value)}LL"))
            else:
                out.append((a.name, 'int32_t', 'nvinfer1::PluginFieldType::kINT32',
                            str(int(a.value))))
        return out

    @staticmethod
    def _outputs(manifest: KernelManifest) -> List:
        return [a for a in manifest.arguments if a.kind in ('output', 'inplace')]

    @staticmethod
    def _output_type_body(manifest: KernelManifest) -> str:
        """``getOutputDataTypes`` cases.

        The V2 emitter returns the DataType from a per-index accessor; V3 hands
        in an array to fill and returns a status, so the values are written
        rather than returned.
        """
        outs = TensorRTV3PluginGenerator._outputs(manifest)
        cases = "\n            ".join(
            f"case {i}: outputTypes[i] = {TensorRTPluginGenerator.trt_datatype(a.dtype)}; break;"
            for i, a in enumerate(outs)
        )
        return cases + "\n            default: outputTypes[i] = nvinfer1::DataType::kFLOAT; break;"

    @staticmethod
    def _output_shape_body(manifest: KernelManifest) -> str:
        """``getOutputShapes`` body.

        Same dimension rules as the V2 emitter's ``getOutputDimensions`` (dims
        listed in ``manifest.dynamic_axes`` are read from the input at run time;
        the rest are constants), but V3 writes into the caller's ``outputs``
        array and returns a status instead of returning a ``DimsExprs`` by value.
        """
        dyn = tuple(getattr(manifest, "dynamic_axes", ()) or ())
        out = []
        for idx, arg in enumerate(TensorRTV3PluginGenerator._outputs(manifest)):
            out.append(f"if (outputIndex == {idx}) {{")
            out.append("    nvinfer1::DimsExprs res;")
            out.append(f"    res.nbDims = {len(arg.shape)};")
            for d_idx, d in enumerate(arg.shape):
                if d_idx in dyn:
                    out.append(f"    res.d[{d_idx}] = inputs[0].d[{d_idx}];")
                    continue
                try:
                    out.append(f"    res.d[{d_idx}] = exprBuilder.constant({int(d)});")
                except Exception:
                    out.append(f"    res.d[{d_idx}] = inputs[0].d[{d_idx}];")
            out.append(f"    outputs[{idx}] = res;")
            out.append("    continue;")
            out.append("}")
        # Only reached when no case matched; otherwise the chain already
        # assigned. Without the `continue` above this would overwrite a
        # correctly computed output with the input's shape.
        out.append("outputs[outputIndex] = inputs[0];")
        return "\n        ".join(out)

    # ------------------------------------------------------------------- header
    H_TEMPLATE = '''
#ifndef @@UPPER@@_V3_PLUGIN_H
#define @@UPPER@@_V3_PLUGIN_H

#include "NvInferPlugin.h"
#include <cuda.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cstdlib>

namespace @@NS_KERNEL@@ {

// IPluginV3 plus the three capability interfaces. The inheritance is
// deliberately not virtual -- see the module docstring for why.
class @@PLUGIN@@ : public nvinfer1::IPluginV3,
                   public nvinfer1::IPluginV3OneCore,
                   public nvinfer1::IPluginV3OneBuild,
                   public nvinfer1::IPluginV3OneRuntime
{
public:
    @@PLUGIN@@();
    explicit @@PLUGIN@@(nvinfer1::PluginFieldCollection const* fc);
    ~@@PLUGIN@@() noexcept override = default;

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                               nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                            nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
                            nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
                                   int32_t nbInputs, int32_t nbOutputs) noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
                          nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    int32_t initialize();

    std::string mNamespace;
@@MEMBERS@@
    CUmodule mModule{nullptr};
    CUfunction mKernel{nullptr};
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mFields;
};

class @@PLUGIN@@Creator : public nvinfer1::IPluginCreatorV3One
{
public:
    @@PLUGIN@@Creator();
    ~@@PLUGIN@@Creator() noexcept override = default;

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const* name,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::AsciiChar const* getPluginName() const noexcept override;
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override;
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mFields;
};

} // namespace @@NS_KERNEL@@

#endif
'''

    @staticmethod
    def _fill(tpl: str, values: dict) -> str:
        for k, v in values.items():
            tpl = tpl.replace("@@%s@@" % k, str(v))
        return tpl

    def _generate_kernel_h(self, manifest: KernelManifest) -> str:
        plugin_name = f"{manifest.kernel_name}Plugin"
        scalars = self._scalars(manifest)
        members = ("\n".join(f"    {ct} m_{n};" for n, ct, _, _ in scalars) + "\n") if scalars else ""
        return self._fill(self.H_TEMPLATE, {
            "UPPER": manifest.kernel_name.upper(),
            "NS_KERNEL": f"{self.plugin_namespace}_{manifest.kernel_name}",
            "PLUGIN": plugin_name,
            "MEMBERS": members,
        }).strip() + "\n"

    # ------------------------------------------------------------------- source
    CU_TEMPLATE = '''
#include "@@KERNEL@@Plugin.h"
#include <cuda.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace @@NS_KERNEL@@ {

// ---------------------------------------------------------------- plugin

@@PLUGIN@@::@@PLUGIN@@() : mNamespace("@@NS@@")@@CTOR_DEFAULTS@@ {}

@@PLUGIN@@::@@PLUGIN@@(nvinfer1::PluginFieldCollection const* fc) : @@PLUGIN@@() {
    if (fc == nullptr) return;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        nvinfer1::PluginField const& f = fc->fields[i];
        if (f.name == nullptr) continue;
        std::string name(f.name);
        if (false) {}
        @@PARSE@@
        else if (name == "namespace") {
            if (f.data && f.length > 0) mNamespace.assign(static_cast<char const*>(f.data), f.length);
        }
    }
}

nvinfer1::IPluginCapability* @@PLUGIN@@::getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept {
    switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:    return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:   return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME: return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
    }
    return nullptr;
}

nvinfer1::IPluginV3* @@PLUGIN@@::clone() noexcept {
    auto* p = new (std::nothrow) @@PLUGIN@@();
    if (p == nullptr) return nullptr;
    p->mNamespace = mNamespace;
@@CLONE_COPY@@
    // The CUDA module and function are deliberately not copied: they belong to
    // the context that created them, and initialize() re-loads the PTX on demand.
    return p;
}

// ------------------------------------------------------------ IPluginV3OneCore

nvinfer1::AsciiChar const* @@PLUGIN@@::getPluginName() const noexcept { return "@@KERNEL@@"; }
nvinfer1::AsciiChar const* @@PLUGIN@@::getPluginVersion() const noexcept { return "@@VERSION@@"; }
nvinfer1::AsciiChar const* @@PLUGIN@@::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

// ----------------------------------------------------------- IPluginV3OneBuild

int32_t @@PLUGIN@@::getNbOutputs() const noexcept { return @@NB_OUTPUTS@@; }

int32_t @@PLUGIN@@::getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
                                      nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept {
    (void)nbInputs;
    for (int32_t i = 0; i < nbOutputs; ++i) {
        switch (i) {
            @@DYN_TYPES@@
        }
    }
    return 0;
}

int32_t @@PLUGIN@@::getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
                                   nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                                   nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
                                   nvinfer1::IExprBuilder& exprBuilder) noexcept {
    (void)nbInputs; (void)shapeInputs; (void)nbShapeInputs; (void)nbOutputs;
    for (int32_t outputIndex = 0; outputIndex < nbOutputs; ++outputIndex) {
        @@SHAPE_BODY@@
    }
    return 0;
}

bool @@PLUGIN@@::supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut,
                                           int32_t nbInputs, int32_t nbOutputs) noexcept {
    (void)nbInputs; (void)nbOutputs;
    @@SUPPORTS@@
}

int32_t @@PLUGIN@@::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
                                    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
    return 0;
}

size_t @@PLUGIN@@::getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                    nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept {
    (void)inputs; (void)nbInputs; (void)outputs; (void)nbOutputs;
    return 0;
}

// --------------------------------------------------------- IPluginV3OneRuntime

int32_t @@PLUGIN@@::onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs,
                                  nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept {
    (void)in; (void)nbInputs; (void)out; (void)nbOutputs;
    return 0;
}

int32_t @@PLUGIN@@::enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
                            void const* const* inputs, void* const* outputs,
                            void* workspace, cudaStream_t stream) noexcept {
    (void)outputDesc; (void)workspace;

    if (mKernel == nullptr) {
        int32_t res = initialize();
        if (res != 0 || mKernel == nullptr) {
            fprintf(stderr, "TRT_V3_ENQUEUE_ERROR: initialize failed or mKernel is null\\n");
            return -1;
        }
    }


    unsigned int grid_x = std::max(1u, (unsigned int)(@@GRID_X@@));
    unsigned int grid_y = std::max(1u, (unsigned int)(@@GRID_Y@@));
    unsigned int grid_z = std::max(1u, (unsigned int)(@@GRID_Z@@));
    unsigned int block_x = std::max(1u, (unsigned int)(@@BLOCK_SIZE@@));

    void* kernelParams[@@NUM_SLOTS@@];
    @@DYN_ARGS@@

    CUresult launch_res = cuLaunchKernel(mKernel, grid_x, grid_y, grid_z, block_x, 1, 1, @@SHARED_MEM@@, stream, kernelParams, nullptr);
    if (launch_res != CUDA_SUCCESS) {
        fprintf(stderr, "TRT_V3_ENQUEUE_ERROR: cuLaunchKernel failed with code %d\\n", (int)launch_res);
        return -1;
    }

    return 0;
}

nvinfer1::IPluginV3* @@PLUGIN@@::attachToContext(nvinfer1::IPluginResourceContext* context) noexcept {
    (void)context;
    // No per-context resources to move: the CUDA module is loaded lazily by
    // initialize() against whichever context runs first.
    return clone();
}

nvinfer1::PluginFieldCollection const* @@PLUGIN@@::getFieldsToSerialize() noexcept {
    mFields.clear();
    @@SERIALIZE@@
    mFields.emplace_back("namespace", mNamespace.data(), nvinfer1::PluginFieldType::kCHAR, mNamespace.size());
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
    return &mFC;
}

// ------------------------------------------------------------------ CUDA

int32_t @@PLUGIN@@::initialize() {
    if (mModule != nullptr && mKernel != nullptr) return 0;

    static const char* PTX_CODE = @@PTX@@;

    // TensorRT can hand a plugin over before any context is current; the module
    // has to load into the primary context.
    CUcontext ctx = nullptr;
    cuCtxGetCurrent(&ctx);
    if (ctx == nullptr) {
        CUdevice cuDev;
        cuDeviceGet(&cuDev, 0);
        cuDevicePrimaryCtxRetain(&ctx, cuDev);
        cuCtxSetCurrent(ctx);
    }

    CUresult res = cuModuleLoadDataEx(&mModule, PTX_CODE, 0, nullptr, nullptr);
    if (res != CUDA_SUCCESS) {
        // Some CUDA driver JITs reject the newest PTX ISA version. Retry with a
        // clamped .version, then an older .target, before giving up. The
        // original PTX is preserved so modern drivers JIT it natively.
        std::string ptx(PTX_CODE);
        size_t vp = ptx.find(".version");
        if (vp != std::string::npos) {
            size_t nl = ptx.find('\\n', vp);
            if (nl != std::string::npos) ptx.replace(vp, nl - vp, ".version 9.0");
        }
        res = cuModuleLoadDataEx(&mModule, ptx.c_str(), 0, nullptr, nullptr);
        if (res != CUDA_SUCCESS) {
            size_t tp = ptx.find(".target");
            if (tp != std::string::npos) {
                size_t nl = ptx.find('\\n', tp);
                if (nl != std::string::npos) ptx.replace(tp, nl - tp, ".target sm_90");
            }
            res = cuModuleLoadDataEx(&mModule, ptx.c_str(), 0, nullptr, nullptr);
        }
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "TRT_V3_INIT_ERROR: cuModuleLoadDataEx failed with code %d\\n", (int)res);
            return -1;
        }
    }

    res = cuModuleGetFunction(&mKernel, mModule, "@@PTX_ENTRY@@");
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "TRT_V3_INIT_ERROR: cuModuleGetFunction failed with code %d for entry %s\\n", (int)res, "@@PTX_ENTRY@@");
        return -1;
    }
    if (@@SHARED_MEM@@ > 0) {
        res = cuFuncSetAttribute(mKernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, @@SHARED_MEM@@);
        if (res != CUDA_SUCCESS) {
            fprintf(stderr, "TRT_V3_INIT_ERROR: cuFuncSetAttribute failed with code %d\\n", (int)res);
            return -1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------- creator

@@PLUGIN@@Creator::@@PLUGIN@@Creator() : mNamespace("@@NS@@") {}

nvinfer1::IPluginV3* @@PLUGIN@@Creator::createPlugin(nvinfer1::AsciiChar const* name,
                                                    nvinfer1::PluginFieldCollection const* fc,
                                                    nvinfer1::TensorRTPhase phase) noexcept {
    (void)name; (void)phase;
    // Called at build time with the graph's fields, and on engine load with
    // what getFieldsToSerialize() returned -- which is how the plugin is
    // reconstructed without a hand-written deserialiser.
    return new (std::nothrow) @@PLUGIN@@(fc);
}

nvinfer1::PluginFieldCollection const* @@PLUGIN@@Creator::getFieldNames() noexcept {
    mFields.clear();
    @@FIELD_NAMES@@
    mFields.emplace_back("namespace", nullptr, nvinfer1::PluginFieldType::kCHAR, 0);
    mFC.nbFields = static_cast<int32_t>(mFields.size());
    mFC.fields = mFields.data();
    return &mFC;
}

nvinfer1::AsciiChar const* @@PLUGIN@@Creator::getPluginName() const noexcept { return "@@KERNEL@@"; }
nvinfer1::AsciiChar const* @@PLUGIN@@Creator::getPluginVersion() const noexcept { return "@@VERSION@@"; }
nvinfer1::AsciiChar const* @@PLUGIN@@Creator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }

} // namespace @@NS_KERNEL@@
'''

    def _generate_kernel_cu(self, manifest: KernelManifest) -> str:
        plugin_name = f"{manifest.kernel_name}Plugin"
        block_size = (manifest.num_warps if manifest.num_warps > 0 else 4) * 32
        scalars = self._scalars(manifest)
        shared = TensorRTPluginGenerator.build_shared_body(manifest, block_size)

        ctor_defaults = "".join(f", m_{n}({d})" for n, _, _, d in scalars)
        parse = "\n        ".join(
            f'else if (name == "{n}") {{ if (f.data && f.length >= 1) m_{n} = *static_cast<{ct} const*>(f.data); }}'
            for n, ct, _, _ in scalars) or "// no scalar fields"
        clone_copy = "\n".join(f"    p->m_{n} = m_{n};" for n, _, _, _ in scalars)
        serialize = "\n    ".join(
            f'mFields.emplace_back("{n}", &m_{n}, {pft}, 1);' for n, _, pft, _ in scalars) or "// no scalar fields"
        field_names = "\n    ".join(
            f'mFields.emplace_back("{n}", nullptr, {pft}, 1);' for n, _, pft, _ in scalars) or "// no scalar fields"

        return self._fill(self.CU_TEMPLATE, {
            "KERNEL": manifest.kernel_name,
            "NS_KERNEL": f"{self.plugin_namespace}_{manifest.kernel_name}",
            "NS": self.plugin_namespace,
            "VERSION": self.plugin_version,
            "PLUGIN": plugin_name,
            "CTOR_DEFAULTS": ctor_defaults,
            "PARSE": parse,
            "CLONE_COPY": clone_copy,
            "NB_OUTPUTS": max(1, len(self._outputs(manifest))),
            "DYN_TYPES": self._output_type_body(manifest),
            "SHAPE_BODY": self._output_shape_body(manifest),
            "SUPPORTS": TensorRTPluginGenerator.supports_format_body(manifest, ".desc"),
            "GRID_X": shared["grid_x_cxx"],
            "GRID_Y": shared["grid_y_cxx"],
            "GRID_Z": shared["grid_z_cxx"],
            "BLOCK_SIZE": block_size,
            "NUM_SLOTS": shared["num_ptx_slots"],
            "DYN_ARGS": shared["dynamic_args_cpp"],
            "SHARED_MEM": manifest.shared_memory_bytes,
            "FIRST_SCALAR": (f"m_{scalars[0][0]}" if scalars else "0"),
            "SERIALIZE": serialize,
            "FIELD_NAMES": field_names,
            "PTX": json.dumps(manifest.ptx),
            "PTX_ENTRY": shared["ptx_entry_name"],
        }).strip() + "\n"

    # -------------------------------------------------------------------- output
    def _generate_register_cpp(self) -> str:
        includes = "\n".join(f'#include "{m.kernel_name}Plugin.h"' for m in self.manifests)
        decls, regs = [], []
        for i, m in enumerate(self.manifests):
            ns = f"{self.plugin_namespace}_{m.kernel_name}"
            creator = f"{m.kernel_name}PluginCreator"
            decls.append(f"        static {ns}::{creator} c{i};")
            regs.append(f"        if (!registry->registerCreator(c{i}, c{i}.getPluginNamespace())) return false;")
        return textwrap.dedent(f'''
        #include <NvInferPlugin.h>
        {includes}
        #include <cstdio>

        // Registration entry point for the IPluginV3 plugins KernelLens emits.
        // IPluginCreatorV3One derives from IPluginCreatorInterface, so this binds
        // to the non-deprecated registerCreator() overload and produces no
        // deprecation warning (the IPluginV2 path does emit one).
        extern "C" __attribute__((visibility("default"))) bool register_triton_v3_plugins_explicit() {{
            static bool g_registered = false;
            if (g_registered) return true;
            auto* registry = ::getPluginRegistry();
            if (registry == nullptr) return false;
        {chr(10).join(decls)}
        {chr(10).join(regs)}
            g_registered = true;
            return true;
        }}
        ''').strip() + "\n"

    def generate(self) -> Dict[str, str]:
        files: Dict[str, str] = {}
        for m in self.manifests:
            files[f"{m.kernel_name}Plugin.h"] = self._generate_kernel_h(m)
            files[f"{m.kernel_name}Plugin.cu"] = self._generate_kernel_cu(m)
        files["register_plugins.cpp"] = self._generate_register_cpp()
        return files


def generate_trt_v3_bindings(manifests: List[KernelManifest], output_dir: str,
                             plugin_namespace: str = "triton_custom") -> None:
    """Write the IPluginV3 plugin sources into ``output_dir``."""
    import os as _os
    gen = TensorRTV3PluginGenerator(manifests, plugin_namespace=plugin_namespace)
    files = gen.generate()
    _os.makedirs(output_dir, exist_ok=True)
    for filename, content in files.items():
        with open(_os.path.join(output_dir, filename), "w") as f:
            f.write(content)
