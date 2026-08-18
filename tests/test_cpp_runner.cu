#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

__global__ void nchw_to_nhwc_cuda_kernel(const float* __restrict__ src, float* __restrict__ dst, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int tmp2 = tmp / H;
    int c = tmp2 % C;
    int n = tmp2 / C;
    int nhwc_idx = n * (H * W * C) + h * (W * C) + w * C + c;
    dst[nhwc_idx] = src[idx];
}

__global__ void oihw_to_ohwi_cuda_kernel(const float* __restrict__ src, float* __restrict__ dst, int O, int I, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = O * I * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int tmp2 = tmp / H;
    int i = tmp2 % I;
    int o = tmp2 / I;
    int ohwi_idx = o * (H * W * I) + h * (W * I) + w * I + i;
    dst[ohwi_idx] = src[idx];
}

int main() {
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);

    std::ifstream file("/home/ostentatoire/.kernel_lens_cache/NHWC_Conv_SOTA/ort_plugins/_fused_seq_conv_nhwc_kernelOp.cu");
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string full_code = buffer.str();

    std::string marker_start = "static const char* PTX_CODE = R\"ptx(";
    std::string marker_end = ")ptx\";";
    size_t start = full_code.find(marker_start) + marker_start.length();
    size_t end = full_code.find(marker_end, start);
    std::string ptx_code = full_code.substr(start, end - start);

    CUmodule mod;
    cuModuleLoadData(&mod, ptx_code.c_str());
    CUfunction kernel;
    cuModuleGetFunction(&kernel, mod, "_fused_seq_conv_nhwc_kernel");

    float *d_x_nchw, *d_w_oihw;
    float *d_x_nhwc, *d_w_ohwi, *d_out_nhwc;

    size_t x_sz = 1 * 128 * 64 * 64 * sizeof(float);
    size_t w_sz = 512 * 128 * 3 * 3 * sizeof(float);
    size_t out_sz = 1 * 512 * 64 * 64 * sizeof(float);

    cudaMalloc(&d_x_nchw, x_sz);
    cudaMalloc(&d_w_oihw, w_sz);

    cudaMalloc(&d_x_nhwc, x_sz);
    cudaMalloc(&d_w_ohwi, w_sz);
    cudaMalloc(&d_out_nhwc, out_sz);

    std::vector<float> h_x(1 * 128 * 64 * 64, 1.0f);
    std::vector<float> h_w(512 * 128 * 3 * 3, 1.0f);
    cudaMemcpy(d_x_nchw, h_x.data(), x_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_oihw, h_w.data(), w_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_out_nhwc, 0, out_sz);

    nchw_to_nhwc_cuda_kernel<<<(1*128*64*64 + 255)/256, 256>>>(d_x_nchw, d_x_nhwc, 1, 128, 64, 64);
    oihw_to_ohwi_cuda_kernel<<<(512*128*3*3 + 255)/256, 256>>>(d_w_oihw, d_w_ohwi, 512, 128, 3, 3);
    cudaDeviceSynchronize();

    void* tmp_x = d_x_nhwc;
    void* tmp_w = d_w_ohwi;
    void* tmp_out = d_out_nhwc;

    // SKIPPING batch=1 because Triton specialized batch=1 as constexpr in PTX!
    int32_t channels = 128, H = 64, W = 64;
    int32_t sxn = 524288, sxh = 8192, sxw = 128;
    int32_t swn = 1152, swh = 384, sww = 128;
    int32_t son = 2097152, soh = 32768, sow = 512;

    void* params[] = {
        &tmp_x, &tmp_w, &tmp_out,
        &channels, &H, &W,
        &sxn, &sxh, &sxw,
        &swn, &swh, &sww,
        &son, &soh, &sow
    };

    cuFuncSetAttribute(kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 20480);
    CUresult res = cuLaunchKernel(kernel, 1, 32, 4, 128, 1, 1, 20480, 0, params, nullptr);
    std::cout << "cuLaunchKernel res: " << res << std::endl;
    cudaDeviceSynchronize();

    std::vector<float> h_out(1 * 512 * 64 * 64);
    cudaMemcpy(h_out.data(), d_out_nhwc, out_sz, cudaMemcpyDeviceToHost);

    size_t nonzeros = 0;
    for (float v : h_out) if (v != 0.0f) nonzeros++;
    std::cout << "d_out_nhwc nonzeros: " << nonzeros << " / " << h_out.size() << std::endl;
    std::cout << "d_out_nhwc[:5]: " << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << ", " << h_out[3] << ", " << h_out[4] << std::endl;

    return 0;
}
