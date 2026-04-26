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
        """
        Loads the Custom Op .so plugin and executes the ONNX graph using ONNX Runtime.
        """
        if self._ort_session is None:
            so_path = os.path.join(self.cache_dir, "ort_plugins", "libtriton_ort_plugins.so")
            onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")

            if not os.path.exists(so_path):
                raise FileNotFoundError(f"ORT Plugin missing at: {so_path}")
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX Graph missing at: {onnx_path}")

            session_options = ort.SessionOptions()
            session_options.log_severity_level = 3 
            session_options.register_custom_ops_library(so_path)
            
            providers = ['CUDAExecutionProvider']
            self._ort_session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)

        ort_inputs = {}
        session_inputs = self._ort_session.get_inputs()
        
        for i, session_input in enumerate(session_inputs):
            if i < len(inputs):
                tensor = inputs[i]
                if isinstance(tensor, torch.Tensor):
                    ort_inputs[session_input.name] = tensor.detach().cpu().numpy()
                elif isinstance(tensor, (int, float)):
                    if isinstance(tensor, int):
                         ort_inputs[session_input.name] = np.array(tensor, dtype=np.int64)
                    else:
                         ort_inputs[session_input.name] = np.array(tensor, dtype=np.float64) 
                else:
                    ort_inputs[session_input.name] = tensor
            else:
                print(f"Warning: Input '{session_input.name}' missing. Supplying dummy data.")
                ort_inputs[session_input.name] = np.zeros(session_input.shape, dtype=np.float32)

        return self._ort_session.run(None, ort_inputs)

    def _run_trt(self, inputs: tuple):
        import ctypes
        import tensorrt as trt
        
        so_path = os.path.join(self.cache_dir, "trt_plugins", "libtriton_trt_plugins.so")
        onnx_path = os.path.join(self.cache_dir, f"{self.model_name}.onnx")
        engine_path = os.path.join(self.cache_dir, f"{self.model_name}.engine")
        
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"TRT Plugin missing at: {so_path}")
            
        ctypes.CDLL(so_path)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")

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
                
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        context = engine.create_execution_context()
        torch_outputs = []
        bindings = []
        trt_stream = torch.cuda.Stream()
        
        # Safely handle both V2 and V3 Python TRT Wrappers
        if hasattr(engine, 'num_bindings'):
            # Older TRT Python API
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    tensor = inputs[i]
                    if not tensor.is_cuda: tensor = tensor.cuda().contiguous()
                    else: tensor = tensor.contiguous()
                    context.set_binding_shape(i, tuple(tensor.shape))
                    bindings.append(tensor.data_ptr())
                else:
                    shape = context.get_binding_shape(i)
                    resolved_shape = tuple(s if s >= 0 else 1 for s in shape)
                    out_tensor = torch.empty(resolved_shape, device='cuda', dtype=torch.float32)
                    torch_outputs.append(out_tensor)
                    bindings.append(out_tensor.data_ptr())
                    
            # stream = torch.cuda.current_stream().cuda_stream
            stream = trt_stream.cuda_stream
            context.execute_async_v2(bindings=bindings, stream_handle=stream)
            
        else:
            # Modern TRT 8.5+ Python API
            input_idx = 0
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                mode = engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    tensor = inputs[input_idx]
                    input_idx += 1
                    if not tensor.is_cuda: tensor = tensor.cuda().contiguous()
                    else: tensor = tensor.contiguous()
                    context.set_input_shape(name, tuple(tensor.shape))
                    context.set_tensor_address(name, tensor.data_ptr())
                else:
                    shape = engine.get_tensor_shape(name)
                    resolved_shape = tuple(s if s >= 0 else 1 for s in shape)
                    out_tensor = torch.empty(resolved_shape, device='cuda', dtype=torch.float32)
                    torch_outputs.append(out_tensor)
                    context.set_tensor_address(name, out_tensor.data_ptr())
                    
            # stream = torch.cuda.current_stream().cuda_stream
            stream = trt_stream.cuda_stream
            context.execute_async_v3(stream_handle=stream)
            
        # torch.cuda.synchronize()
        trt_stream.synchronize()
        return torch_outputs