import os
import shutil

def extract_libs(compiled_model, dest_dir: str, libs: list = None):
    """
    Extracts compiled artifacts from the hidden cache into a user-facing directory for deployment.
    
    Args:
        compiled_model: The CompiledModel object returned by kl.compile().
        dest_dir: The target directory to copy the files into.
        libs: A list of artifacts to extract (e.g., ["onnx", "tensorrt"]). 
              Defaults to ["all"] to extract everything generated.
    """
    if libs is None:
        libs = ["all"]
        
    print(f"\n📦 Extracting deployment artifacts for '{compiled_model.model_name}' to '{dest_dir}'...")
    os.makedirs(dest_dir, exist_ok=True)
    
    cache_dir = compiled_model.cache_dir
    base_name = compiled_model.model_name
    
    artifacts = []
    
    # Base ONNX is needed by almost everything
    if "all" in libs or "onnx" in libs or "tensorrt" in libs:
         artifacts.append((os.path.join(cache_dir, f"{base_name}.onnx"), f"{base_name}.onnx"))
    
    # ORT Plugins
    if "all" in libs or "onnx" in libs:
        artifacts.append((os.path.join(cache_dir, "ort_plugins", "libtriton_ort_plugins.so"), "libtriton_ort_plugins.so"))
        
    # TensorRT Engine and Plugins
    if "all" in libs or "tensorrt" in libs:
        artifacts.extend([
            (os.path.join(cache_dir, f"{base_name}.engine"), f"{base_name}.engine"),
            (os.path.join(cache_dir, "trt_plugins", "libtriton_trt_plugins.so"), "libtriton_trt_plugins.so")
        ])
        
    extracted_count = 0
    # Use a set to prevent copying the base ONNX file twice if it overlaps
    processed_files = set()
    
    for src_path, filename in artifacts:
        if filename in processed_files:
            continue
            
        if os.path.exists(src_path):
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(src_path, dest_path)
            print(f"  -> Extracted: {filename}")
            processed_files.add(filename)
            extracted_count += 1
        elif "all" not in libs:
            # Only warn if the user explicitly asked for a specific backend artifact that isn't found
            print(f"  [Warning] Artifact not found in cache: {src_path}")
            
    if extracted_count == 0:
        print("  [Warning] No artifacts were found to extract. Did compilation complete successfully?")
    else:
        print(f"✅ Successfully extracted {extracted_count} files to {dest_dir}/")