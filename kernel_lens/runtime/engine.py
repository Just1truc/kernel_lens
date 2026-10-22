import os
import numpy as np
import torch


def _graph_has_int8_custom_inputs(onnx_path: str) -> bool:
    """True if any triton_custom node consumes an INT8/UINT8 tensor.

    TensorRT treats INT8 network tensors as *quantized* and its INT8 path
    corrupts custom plugins that use tensor-core `tl.dot` (verified on
    sm_120a: identical PTX/args/geometry, yet divergent results). For such
    graphs KernelLens executes the numerically-exact ONNX Runtime plugin.
    """
    try:
        import onnx
        from onnx import TensorProto
        m = onnx.load(onnx_path)
        int8_types = {TensorProto.INT8, TensorProto.UINT8}
        has_custom = any(n.domain == "triton_custom" for n in m.graph.node)
        if not has_custom:
            return False
        # Intermediate value_info is often absent from torch.onnx exports, so treat
        # any INT8/UINT8 graph input or initializer as a potential raw-int8 source.
        for vi in m.graph.input:
            if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type in int8_types:
                return True
        for init in m.graph.initializer:
            if init.data_type in int8_types:
                return True
    except Exception:
        return False
    return False

def _make_stream_reader(trt, path):
    """File-backed ``IStreamReaderV2``.

    Engines above roughly 2 GB fail to deserialize from a memory buffer; the
    stream-reader path reads them straight from disk instead. TensorRT fills a
    caller-provided capsule, so the bytes have to be memmove'd into it.
    """
    import ctypes

    _get_ptr = ctypes.pythonapi.PyCapsule_GetPointer
    _get_ptr.restype = ctypes.c_void_p
    _get_ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]

    _whence = {getattr(trt.SeekPosition, "kSET", 0): os.SEEK_SET,
               getattr(trt.SeekPosition, "kCUR", 1): os.SEEK_CUR,
               getattr(trt.SeekPosition, "kEND", 2): os.SEEK_END}

    class _FileStreamReader(trt.IStreamReaderV2):
        def __init__(self):
            super().__init__()
            self._f = open(path, "rb")

        def read(self, destination, num_bytes, stream):
            data = self._f.read(int(num_bytes))
            if not data:
                return 0
            ctypes.memmove(_get_ptr(destination, None), data, len(data))
            return len(data)

        def seek(self, offset, where):
            self._f.seek(int(offset), _whence.get(where, os.SEEK_SET))
            return True

    return _FileStreamReader()


def _deserialize_engine(trt, runtime, engine_path):
    if os.path.getsize(engine_path) > (1 << 31) and hasattr(trt, "IStreamReaderV2"):
        try:
            return runtime.deserialize_cuda_engine(_make_stream_reader(trt, engine_path))
        except Exception:
            pass
    with open(engine_path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


class CompiledModel:
    def __init__(self, cache_dir: str, model_name: str, backends: list[str], output_shapes: list = None, output_strides: list = None, output_dtypes: list = None, saved_args: list = None):
        """
        Represents a compiled PyTorch model ready for deployment.
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.backends = backends
        self.output_shapes = output_shapes
        self.output_strides = output_strides
        self.output_dtypes = output_dtypes
        self.saved_args = saved_args
        self._ort_session = None

    def run(self, inputs: tuple, backend: str = "onnx"):
        """
        Executes the compiled model using the specified backend.
        """
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' was not requested during compilation. Available: {self.backends}")

        if backend == "onnx":
            return self._run_ort(inputs)
        elif backend == "tensorrt":
            return self._run_trt(inputs)
        else:
            raise NotImplementedError(f"Execution for backend '{backend}' is not implemented yet.")

    # def _run_ort(self, inputs: tuple):
    #     import onnxruntime as ort
    #     import torch
    #     import numpy as np

    #     if self._ort_session is None:
    #         so_path = os.path.join(self.cache_dir, "ort_plugins", "libtriton_ort_plugins.so")
    #         onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")

    #         if not os.path.exists(so_path):
    #             raise FileNotFoundError(f"ORT Plugin missing at: {so_path}")
            
    #         session_options = ort.SessionOptions()
    #         session_options.register_custom_ops_library(so_path)
            
    #         # On force CUDA
    #         providers = ['CUDAExecutionProvider']
    #         self._ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
    #         # Cache pour l'IO Binding
    #         self._ort_io_binding = self._ort_session.io_binding()

    #     # 1. Préparation des tenseurs (contigus et sur GPU)
    #     trt_inputs = []
    #     session_inputs = self._ort_session.get_inputs()
        
    #     for i, sess_in in enumerate(session_inputs):
    #         inp = inputs[i]
    #         # On regarde ce que le graphe ONNX attend pour cet index
    #         expected_type = sess_in.type # ex: 'tensor(float)' ou 'tensor(double)'
            
    #         if isinstance(inp, torch.Tensor):
    #             t = inp if inp.is_cuda else inp.cuda()
    #             trt_inputs.append(t.contiguous())
    #         elif isinstance(inp, (float, int)):
    #             # ADAPTATION DYNAMIQUE AU TYPE DU GRAPHE
    #             dtype = torch.float32
    #             if "double" in expected_type: dtype = torch.float64
    #             if "int64" in expected_type:  dtype = torch.int64
    #             if "int32" in expected_type:  dtype = torch.int32
                
    #             trt_inputs.append(torch.tensor(inp, device='cuda', dtype=dtype))
    #         else:
    #             trt_inputs.append(torch.as_tensor(inp, device='cuda'))

    #     # 2. Binding des Entrées (Utilise le type exact du tenseur préparé)
    #     for i, sess_in in enumerate(session_inputs):
    #         t = trt_inputs[i]
    #         # Mapping numpy/onnx type
    #         onnx_type = np.float32
    #         if t.dtype == torch.float64: onnx_type = np.float64
    #         if t.dtype == torch.int64:   onnx_type = np.int64
            
    #         self._ort_io_binding.bind_input(
    #             name=sess_in.name,
    #             device_type='cuda',
    #             device_id=0,
    #             element_type=onnx_type,
    #             shape=tuple(t.shape),
    #             buffer_ptr=t.data_ptr()
    #         )

    #     # 3. Binding des Sorties (On pré-alloue dans PyTorch pour garder la main sur la VRAM)
    #     session_outputs = self._ort_session.get_outputs()
    #     torch_outputs = []
    #     for sess_out in session_outputs:
    #         # Note: Si shapes dynamiques complexes, il faut parfois appeler 
    #         # self._ort_io_binding.bind_output(name=sess_out.name, device_type='cuda')
    #         # Mais ici on alloue proprement :
    #         shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in sess_out.shape]
    #         # Pour RoPE/Attention, on force les shapes connues du premier input si ORT renvoie None
    #         if any(isinstance(d, str) or d is None for d in sess_out.shape):
    #              shape = trt_inputs[0].shape # Heuristique simple
                 
    #         out_tensor = torch.empty(tuple(shape), device='cuda', dtype=torch.float32)
    #         torch_outputs.append(out_tensor)
            
    #         self._ort_io_binding.bind_output(
    #             name=sess_out.name,
    #             device_type='cuda',
    #             device_id=0,
    #             element_type=np.float32,
    #             shape=tuple(shape),
    #             buffer_ptr=out_tensor.data_ptr()
    #         )

    #     # 4. Exécution
    #     self._ort_session.run_with_iobinding(self._ort_io_binding)
        
    #     return torch_outputs
    def _run_ort(self, inputs: tuple):
        import onnxruntime as ort
        import torch
        import numpy as np

        if self._ort_session is None:
            so_path = os.path.join(self.cache_dir, "ort_plugins", "libtriton_ort_plugins.so")
            onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")

            session_options = ort.SessionOptions()
            session_options.register_custom_ops_library(so_path)
            self._ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=['CUDAExecutionProvider'])
            self._ort_inputs = self._ort_session.get_inputs()
            self._ort_outputs = self._ort_session.get_outputs()
            self._ort_binding = self._ort_session.io_binding()
            self._ort_out_tensors = {}   # (index, shape, dtype) -> cached tensor
            self._ort_dtype_map = {
                torch.float32: np.float32, torch.float64: np.float64, torch.float16: np.float16,
                torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
                torch.int8: np.int8, torch.uint8: np.uint8,
            }
            if hasattr(torch, 'float8_e4m3fn'):
                self._ort_dtype_map[torch.float8_e4m3fn] = np.uint8
            if hasattr(torch, 'float8_e5m2'):
                self._ort_dtype_map[torch.float8_e5m2] = np.uint8
            try:
                self._ort_dtype_map[torch.bfloat16] = np.dtype("bfloat16")
            except Exception:
                pass

        io_binding = self._ort_binding
        session_inputs = self._ort_inputs
        from ..config import is_verbose, debug_print
        if is_verbose():
            debug_print(f"[ENGINE DEBUG] session_inputs count: {len(session_inputs)}")
            for idx, s in enumerate(session_inputs):
                debug_print(f"  sess_in {idx}: name='{s.name}', type='{s.type}', shape={s.shape}")

        trt_inputs = []
        for i, sess_in in enumerate(session_inputs):
            if i < len(inputs):
                inp = inputs[i]
            elif self.saved_args and i < len(self.saved_args):
                inp = self.saved_args[i]
            else:
                inp = inputs[-1]

            expected_type = sess_in.type
            is_scalar_input = (i >= len(inputs))
            if isinstance(inp, torch.Tensor):
                t = inp.cpu() if is_scalar_input else (inp if inp.is_cuda else inp.cuda())
                trt_inputs.append(t)
            elif isinstance(inp, (float, int)):
                dtype = torch.float32
                if "double" in expected_type: dtype = torch.float64
                elif "int64" in expected_type: dtype = torch.int64
                elif "int32" in expected_type: dtype = torch.int32
                trt_inputs.append(torch.tensor([inp], device='cpu', dtype=dtype))
            else:
                t = torch.as_tensor(inp)
                trt_inputs.append(t.cpu() if is_scalar_input else t.cuda())

        # 2. BIND INPUTS
        for i, sess_in in enumerate(session_inputs):
            t = trt_inputs[i]
            onnx_type = self._ort_dtype_map.get(t.dtype, np.float32)
            dev_type = 'cpu' if (t.device.type == 'cpu' or i >= len(inputs)) else 'cuda'
            io_binding.bind_input(
                name=sess_in.name, device_type=dev_type, device_id=0,
                element_type=onnx_type, shape=tuple(t.shape), buffer_ptr=t.data_ptr()
            )

        # 3. OUTPUT ALLOCATION (cached, zero-copy re-bind)
        ort_output_tensors = []
        for i, sess_out in enumerate(self._ort_outputs):
            if self.output_shapes and i < len(self.output_shapes) and tuple(self.output_shapes[i]) == tuple(trt_inputs[0].shape):
                out_shape = tuple(self.output_shapes[i])
            else:
                out_shape = tuple([d if (isinstance(d, int) and d > 0) else trt_inputs[0].shape[j] for j, d in enumerate(sess_out.shape)])
            out_dtype = self.output_dtypes[i] if (self.output_dtypes and i < len(self.output_dtypes)) else torch.float32
            key = (i, out_shape, out_dtype)
            t = self._ort_out_tensors.get(key)
            if t is None:
                t = torch.empty(out_shape, device='cuda', dtype=out_dtype)
                self._ort_out_tensors[key] = t
            ort_output_tensors.append(t)

        for i, (sess_out, t) in enumerate(zip(self._ort_outputs, ort_output_tensors)):
            np_dtype = self._ort_dtype_map.get(t.dtype, np.float32)
            io_binding.bind_output(
                name=sess_out.name, device_type='cuda', device_id=0,
                element_type=np_dtype, shape=tuple(t.shape), buffer_ptr=t.data_ptr()
            )

        try:
            io_binding.synchronize_inputs()
            self._ort_session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
        except Exception as e:
            raise RuntimeError(f"ORT Execution failed: {e}")

        return ort_output_tensors[0] if len(ort_output_tensors) == 1 else tuple(ort_output_tensors)

    def _run_trt(self, inputs: tuple):
        import ctypes
        import tensorrt as trt
        import torch
        import os
        
        if torch.cuda.is_available():
            torch.cuda.init()
            
        # =======================================================
        # 1. COLD START CACHE (Ne s'exécute qu'une seule fois)
        # =======================================================
        if not hasattr(self, '_trt_engine'):
            onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")
            use_ort = (os.path.exists(os.path.join(self.cache_dir, "trt_use_ort"))
                       or _graph_has_int8_custom_inputs(onnx_path))
            if use_ort:
                from ..config import debug_print
                debug_print("[Engine] INT8 tensors or in-place buffers detected; executing the exact "
                            "ONNX Runtime plugin (TensorRT's tensor path corrupts these patterns).")
                return self._run_ort(inputs)
            so_path = os.path.join(self.cache_dir, "trt_plugins", "libtriton_trt_plugins.so")
            engine_path = os.path.join(self.cache_dir, f"{self.model_name}.engine")
            
            import sys
            for site_p in sys.path:
                for lib_folder in ["tensorrt_libs", "tensorrt", "tensorrt_cu12_libs", "tensorrt_cu13_libs", "tensorrt_cu11_libs"]:
                    p = os.path.join(site_p, lib_folder)
                    if os.path.exists(p):
                        for f in sorted(os.listdir(p)):
                            if f.startswith("libnvinfer") or f.startswith("libnvonnxparser"):
                                try:
                                    ctypes.CDLL(os.path.join(p, f), mode=ctypes.RTLD_GLOBAL)
                                except Exception:
                                    pass

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            if hasattr(trt, "init_libnvinfer_plugins"):
                trt.init_libnvinfer_plugins(TRT_LOGGER, "")

            dll = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            if hasattr(dll, "register_triton_plugins_explicit"):
                dll.register_triton_plugins_explicit.restype = ctypes.c_bool
                dll.register_triton_plugins_explicit()
            
            # --- CONSTRUCTION DU MOTEUR (S'il n'existe pas) ---
            if not os.path.exists(engine_path):
                from ..config import debug_print
                debug_print("     [Engine] Building TensorRT Engine... (This takes a moment)")
                builder = trt.Builder(TRT_LOGGER)
                if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
                    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                else:
                    network = builder.create_network()
                parser = trt.OnnxParser(network, TRT_LOGGER)

                # Parse from the file path rather than a bytes buffer. Graphs
                # above the 2 GB protobuf limit spill their weights to external
                # files beside the model, and the parser can only resolve those
                # relative locations when it knows which directory that is.
                if hasattr(parser, "parse_from_file"):
                    ok = parser.parse_from_file(onnx_path)
                else:
                    with open(onnx_path, 'rb') as model:
                        ok = parser.parse(model.read())
                if not ok:
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX graph for TensorRT.")
                        
                config = builder.create_builder_config()
                if hasattr(config, "set_memory_pool_limit"):
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

                # TensorRT keeps plugin layers in FP32 unless the matching
                # precision flag is enabled. Without it the builder inserts a
                # cast on the plugin's inputs, and the plugin's dtype guard
                # (which mirrors the kernel's signature) then rejects them, so
                # the engine build fails with "could not find any supported
                # formats consistent with input/output data types".
                try:
                    import onnx as _onnx
                    _g = _onnx.load(onnx_path, load_external_data=False).graph
                    # TensorProto: 10 = FLOAT16, 16 = BFLOAT16. An LLM's graph
                    # *inputs* are token ids (int64), so the precision has to be
                    # read from the initializers and the internal tensors too.
                    _elem = {t.type.tensor_type.elem_type
                             for t in list(_g.input) + list(_g.output) + list(_g.value_info)}
                    _elem |= {t.data_type for t in _g.initializer}
                    if 10 in _elem and hasattr(trt.BuilderFlag, "FP16"):
                        config.set_flag(trt.BuilderFlag.FP16)
                    if 16 in _elem and hasattr(trt.BuilderFlag, "BF16"):
                        config.set_flag(trt.BuilderFlag.BF16)
                except Exception:
                    pass
                
                profile = builder.create_optimization_profile()
                has_dynamic = False
                for i in range(network.num_inputs):
                    t = network.get_input(i)
                    shape = tuple(t.shape)
                    if any(s is None or s < 0 for s in shape):
                        has_dynamic = True
                        actual_shape = tuple(inputs[i].shape) if (inputs and i < len(inputs) and hasattr(inputs[i], 'shape')) else shape
                        min_s = tuple(1 if (s is None or s < 0) else s for s in shape)
                        opt_s = tuple(actual_shape[j] if (s is None or s < 0) else s for j, s in enumerate(shape))
                        max_s = tuple(max(8192, (actual_shape[j] if j < len(actual_shape) else 1) * 4) if (s is None or s < 0) else s for j, s in enumerate(shape))
                        profile.set_shape(t.name, min_s, opt_s, max_s)

                if has_dynamic:
                    config.add_optimization_profile(profile)

                # Set dynamic range ONLY for network inputs/outputs with DataType Int8
                if not os.environ.get("KL_TRT_NO_DR"):
                    for i in range(network.num_inputs):
                        t = network.get_input(i)
                        if t.dtype == trt.DataType.INT8:
                            try:
                                t.set_dynamic_range(-128.0, 127.0)
                            except Exception:
                                pass
                    for i in range(network.num_outputs):
                        t = network.get_output(i)
                        if t.dtype == trt.DataType.INT8:
                            try:
                                t.set_dynamic_range(-128.0, 127.0)
                            except Exception:
                                pass
                
                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    raise RuntimeError("TensorRT Engine Building Error: builder.build_serialized_network returned None.")
                with open(engine_path, "wb") as f:
                    f.write(serialized_engine)
            # ---------------------------------------------------
            
            # On lit le SSD et on désérialise UNE SEULE FOIS
            with trt.Runtime(TRT_LOGGER) as runtime:
                self._trt_engine = _deserialize_engine(trt, runtime, engine_path)
                
            self._trt_context = self._trt_engine.create_execution_context()
            self._output_tensors = {}   # (name, shape, dtype) -> cached CUDA tensor


        trt_inputs = []
        raw_inputs = list(inputs)
        if self.saved_args and len(raw_inputs) < len(self.saved_args):
            raw_inputs.extend(self.saved_args[len(raw_inputs):])

        for inp in raw_inputs:
            if isinstance(inp, torch.Tensor):
                t = inp if inp.is_cuda else inp.cuda()
                trt_inputs.append(t)
            elif isinstance(inp, (float, int)):
                trt_inputs.append(torch.as_tensor(inp, device='cuda'))

        current_input_ptrs = tuple(t.data_ptr() for t in trt_inputs)

        # # Assignation des Inputs
        # input_idx = 0
        # for i in range(self._trt_engine.num_io_tensors):
        #     name = self._trt_engine.get_tensor_name(i)
        #     if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        #         tensor = trt_inputs[input_idx]
        #         self._trt_context.set_input_shape(name, tuple(tensor.shape))
        #         self._trt_context.set_tensor_address(name, tensor.data_ptr())
        #         input_idx += 1

        # # Allocation dynamique des Outputs
        # torch_outputs = []
        # for i in range(self._trt_engine.num_io_tensors):
        #     name = self._trt_engine.get_tensor_name(i)
        #     if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
        #         shape = self._trt_context.get_tensor_shape(name)
        #         resolved_shape = tuple(s if s >= 0 else 1 for s in shape)
        #         out_tensor = torch.empty(resolved_shape, device='cuda', dtype=torch.float32)
        #         self._trt_context.set_tensor_address(name, out_tensor.data_ptr())
        #         torch_outputs.append(out_tensor)

        # # Lancement Asynchrone sur le Stream Dédié
        # self._trt_context.execute_async_v3(stream_handle=self._trt_stream.cuda_stream)
        
        # # Sécurité mémoire sans bloquer le CPU !
        # torch.cuda.current_stream().wait_stream(self._trt_stream)
        
        # return torch_outputs
        # 2. Mise à jour des bindings (Seulement si nécessaire)
        in_idx = 0
        for i in range(self._trt_engine.num_io_tensors):
            name = self._trt_engine.get_tensor_name(i)
            if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if in_idx < len(trt_inputs):
                    t = trt_inputs[in_idx]
                    self._trt_context.set_input_shape(name, t.shape)
                    self._trt_context.set_tensor_address(name, t.data_ptr())
                    in_idx += 1

        torch_outputs = []
        out_idx = 0
        for i in range(self._trt_engine.num_io_tensors):
            name = self._trt_engine.get_tensor_name(i)
            if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                raw_shape = self._trt_context.get_tensor_shape(name)
                if any(s < 0 for s in raw_shape):
                    if self.output_shapes and out_idx < len(self.output_shapes):
                        shape = tuple(self.output_shapes[out_idx])
                    else:
                        shape = tuple(s if s >= 0 else 1 for s in raw_shape)
                else:
                    shape = tuple(raw_shape)
                out_dtype = self.output_dtypes[out_idx] if (self.output_dtypes and out_idx < len(self.output_dtypes)) else torch.float32
                key = (name, tuple(shape), out_dtype)
                out_t = self._output_tensors.get(key)
                if out_t is None:
                    out_t = torch.empty(shape, device='cuda', dtype=out_dtype)
                    self._output_tensors[key] = out_t
                self._trt_context.set_tensor_address(name, out_t.data_ptr())
                torch_outputs.append(out_t)
                out_idx += 1

        # Execute on the *current* torch stream so it is ordered after any
        # producer ops (and after the cached-output allocation) with no race.
        stream = torch.cuda.current_stream().cuda_stream
        self._trt_context.execute_async_v3(stream_handle=stream)
        # Guarantee completion before returning: the caller may free the input
        # tensors immediately, and a raw async launch is not tracked by torch's
        # caching allocator.
        torch.cuda.current_stream().synchronize()
        return torch_outputs[0] if len(torch_outputs) == 1 else tuple(torch_outputs)