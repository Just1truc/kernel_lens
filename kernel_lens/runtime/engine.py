import os
import onnxruntime as ort
import numpy as np
import torch

class CompiledModel:
    def __init__(self, cache_dir: str, model_name: str, backends: list[str]):
        """
        Represents a compiled PyTorch model ready for deployment.
        
        Args:
            cache_dir: The hidden directory where artifacts are stored (e.g., ~/.kernel_lens_cache/model_hash)
            model_name: The base name of the model (e.g., "my_model")
            backends: List of successfully compiled backends (e.g., ["onnx", "tensorrt"])
        """
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.backends = backends
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

    def _run_ort(self, inputs: tuple):
        import onnxruntime as ort
        import torch
        import numpy as np

        if self._ort_session is None:
            so_path = os.path.join(self.cache_dir, "ort_plugins", "libtriton_ort_plugins.so")
            onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")

            if not os.path.exists(so_path):
                raise FileNotFoundError(f"ORT Plugin missing at: {so_path}")
            
            session_options = ort.SessionOptions()
            session_options.register_custom_ops_library(so_path)
            
            # On force CUDA
            providers = ['CUDAExecutionProvider']
            self._ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
            # Cache pour l'IO Binding
            self._ort_io_binding = self._ort_session.io_binding()

        # 1. Préparation des tenseurs (contigus et sur GPU)
        trt_inputs = []
        session_inputs = self._ort_session.get_inputs()
        
        for i, sess_in in enumerate(session_inputs):
            inp = inputs[i]
            # On regarde ce que le graphe ONNX attend pour cet index
            expected_type = sess_in.type # ex: 'tensor(float)' ou 'tensor(double)'
            
            if isinstance(inp, torch.Tensor):
                t = inp if inp.is_cuda else inp.cuda()
                trt_inputs.append(t.contiguous())
            elif isinstance(inp, (float, int)):
                # ADAPTATION DYNAMIQUE AU TYPE DU GRAPHE
                dtype = torch.float32
                if "double" in expected_type: dtype = torch.float64
                if "int64" in expected_type:  dtype = torch.int64
                if "int32" in expected_type:  dtype = torch.int32
                
                trt_inputs.append(torch.tensor(inp, device='cuda', dtype=dtype))
            else:
                trt_inputs.append(torch.as_tensor(inp, device='cuda'))

        # 2. Binding des Entrées (Utilise le type exact du tenseur préparé)
        for i, sess_in in enumerate(session_inputs):
            t = trt_inputs[i]
            # Mapping numpy/onnx type
            onnx_type = np.float32
            if t.dtype == torch.float64: onnx_type = np.float64
            if t.dtype == torch.int64:   onnx_type = np.int64
            
            self._ort_io_binding.bind_input(
                name=sess_in.name,
                device_type='cuda',
                device_id=0,
                element_type=onnx_type,
                shape=tuple(t.shape),
                buffer_ptr=t.data_ptr()
            )

        # 3. Binding des Sorties (On pré-alloue dans PyTorch pour garder la main sur la VRAM)
        session_outputs = self._ort_session.get_outputs()
        torch_outputs = []
        for sess_out in session_outputs:
            # Note: Si shapes dynamiques complexes, il faut parfois appeler 
            # self._ort_io_binding.bind_output(name=sess_out.name, device_type='cuda')
            # Mais ici on alloue proprement :
            shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in sess_out.shape]
            # Pour RoPE/Attention, on force les shapes connues du premier input si ORT renvoie None
            if any(isinstance(d, str) or d is None for d in sess_out.shape):
                 shape = trt_inputs[0].shape # Heuristique simple
                 
            out_tensor = torch.empty(tuple(shape), device='cuda', dtype=torch.float32)
            torch_outputs.append(out_tensor)
            
            self._ort_io_binding.bind_output(
                name=sess_out.name,
                device_type='cuda',
                device_id=0,
                element_type=np.float32,
                shape=tuple(shape),
                buffer_ptr=out_tensor.data_ptr()
            )

        # 4. Exécution
        self._ort_session.run_with_iobinding(self._ort_io_binding)
        
        return torch_outputs

    def _run_trt(self, inputs: tuple):
        import ctypes
        import tensorrt as trt
        import torch
        import os
        
        # =======================================================
        # 1. COLD START CACHE (Ne s'exécute qu'une seule fois)
        # =======================================================
        if not hasattr(self, '_trt_engine'):
            so_path = os.path.join(self.cache_dir, "trt_plugins", "libtriton_trt_plugins.so")
            onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")
            engine_path = os.path.join(self.cache_dir, f"{self.model_name}.engine")
            
            ctypes.CDLL(so_path)
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(TRT_LOGGER, "")
            
            # --- CONSTRUCTION DU MOTEUR (S'il n'existe pas) ---
            if not os.path.exists(engine_path):
                print("     [Engine] Building TensorRT Engine... (This takes a moment)")
                builder = trt.Builder(TRT_LOGGER)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, TRT_LOGGER)
                
                with open(onnx_path, 'rb') as model:
                    if not parser.parse(model.read()):
                        for error in range(parser.num_errors):
                            print(parser.get_error(error))
                        raise RuntimeError("Failed to parse ONNX graph for TensorRT.")
                        
                config = builder.create_builder_config()
                if hasattr(config, "set_memory_pool_limit"):
                    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                
                serialized_engine = builder.build_serialized_network(network, config)
                with open(engine_path, "wb") as f:
                    f.write(serialized_engine)
            # ---------------------------------------------------
            
            # On lit le SSD et on désérialise UNE SEULE FOIS
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self._trt_engine = runtime.deserialize_cuda_engine(f.read())
                
            self._trt_context = self._trt_engine.create_execution_context()
            self._trt_stream = torch.cuda.Stream() # Résout les warnings TRT !
            self._output_tensors = {} # Cache pour éviter cudaMalloc
            self._last_input_ptrs = None


        trt_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                trt_inputs.append(inp.cuda().contiguous() if not inp.is_cuda else inp.contiguous())
            elif isinstance(inp, (float, int)):
                trt_inputs.append(torch.as_tensor(inp, device='cuda').contiguous())

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
        if current_input_ptrs != self._last_input_ptrs:
            in_idx = 0
            for i in range(self._trt_engine.num_io_tensors):
                name = self._trt_engine.get_tensor_name(i)
                if self._trt_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    t = trt_inputs[in_idx]
                    self._trt_context.set_input_shape(name, t.shape)
                    self._trt_context.set_tensor_address(name, t.data_ptr())
                    in_idx += 1
                else:
                    # Réutilisation ou allocation des sorties
                    shape = tuple(self._trt_context.get_tensor_shape(name))
                    if name not in self._output_tensors or self._output_tensors[name].shape != shape:
                        self._output_tensors[name] = torch.empty(shape, device='cuda', dtype=torch.float32)
                    
                    self._trt_context.set_tensor_address(name, self._output_tensors[name].data_ptr())
            
            self._last_input_ptrs = current_input_ptrs

        # 3. Lancement 100% asynchrone
        self._trt_context.execute_async_v3(self._trt_stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self._trt_stream)
        
        # Retourne les tenseurs du cache
        return [self._output_tensors[self._trt_engine.get_tensor_name(i)] 
                for i in range(self._trt_engine.num_io_tensors) 
                if self._trt_engine.get_tensor_mode(self._trt_engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]