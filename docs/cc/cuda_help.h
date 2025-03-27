#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2*>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162*>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4*>(&(value))[0])

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                 \
    if (((T).options().dtype() != (th_type))) {              \
        std::cout << "Tensor Info:" << (T).options() << std::endl; \
        throw std::runtime_error("Tensor dtype must be " #th_type); \
    }

// 定义 TORCH_BINDING_COMMON_EXTENSION 宏
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func) \
    m.def(STRINGFY(func), &func, STRINGFY(func));


// PyTorch 绑定代码
#define TORCH_BINDING_ELU(packed_type, th_type, element_type, n_elements)      \
void elu_##packed_type(torch::Tensor x, torch::Tensor y) {                     \
  CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                     \
  CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                     \
  const int ndim = x.dim();                                                  \
  if (ndim != 2) {                                                           \
    int N = 1;                                                             \
    for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                     \
    dim3 block(256 / (n_elements));                                        \
    dim3 grid((N + 256 - 1) / 256);                                        \
    elu_##packed_type##_kernel<<<grid, block>>>(                           \
        reinterpret_cast<element_type*>(x.data_ptr()),                     \
        reinterpret_cast<element_type*>(y.data_ptr()), N);                 \
  } else {                                                                   \
    const int S = x.size(0);                                               \
    const int K = x.size(1);                                               \
    const int N = S * K;                                                   \
    if ((K/(n_elements)) <= 1024) {                                        \
        dim3 block(K/(n_elements));                                        \
        dim3 grid(S);                                                      \
        elu_##packed_type##_kernel<<<grid, block>>>(                       \
            reinterpret_cast<element_type*>(x.data_ptr()),                 \
            reinterpret_cast<element_type*>(y.data_ptr()), N);             \
    } else {                                                               \
        int N = 1;                                                         \
        for (int i = 0; i < ndim; ++i) { N *= x.size(i); }                 \
        dim3 block(256 / (n_elements));                                    \
        dim3 grid((N + 256 - 1) / 256);                                    \
        elu_##packed_type##_kernel<<<grid, block>>>(                       \
        reinterpret_cast<element_type*>(x.data_ptr()),                 \
        reinterpret_cast<element_type*>(y.data_ptr()), N);             \
        }                                                                      \
    }                                                                          \
}


TORCH_BINDING_ELU(f32,        torch::kFloat32,    float,    1)
TORCH_BINDING_ELU(f32x4,      torch::kFloat32,    float,    4)
TORCH_BINDING_ELU(f16,        torch::kHalf,       half,     1)
TORCH_BINDING_ELU(f16x2,      torch::kHalf,       half,     2)
TORCH_BINDING_ELU(f16x8,      torch::kHalf,       half,     8)
TORCH_BINDING_ELU(f16x8_pack, torch::kHalf,       half,     8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(elu_f32)
  TORCH_BINDING_COMMON_EXTENSION(elu_f32x4)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x2)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8)
  TORCH_BINDING_COMMON_EXTENSION(elu_f16x8_pack)
}