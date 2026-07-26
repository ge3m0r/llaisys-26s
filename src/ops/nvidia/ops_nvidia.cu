#include "ops_nvidia.cuh"

#include "../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaisys::ops::nvidia {
namespace {

constexpr size_t BLOCK_SIZE = 256;

void checkLaunch(const char *operation) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
__device__ float toFloat(T value) {
    return static_cast<float>(value);
}

template <>
__device__ float toFloat(fp16_t value) {
    return __half2float(__ushort_as_half(value._v));
}

template <>
__device__ float toFloat(bf16_t value) {
    return __bfloat162float(__ushort_as_bfloat16(value._v));
}

template <typename T>
__device__ T fromFloat(float value) {
    return static_cast<T>(value);
}

template <>
__device__ fp16_t fromFloat(float value) {
    return fp16_t{__half_as_ushort(__float2half_rn(value))};
}

template <>
__device__ bf16_t fromFloat(float value) {
    return bf16_t{__bfloat16_as_ushort(__float2bfloat16_rn(value))};
}

template <typename T>
__global__ void addKernel(T *out, const T *a, const T *b, size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        out[index] = fromFloat<T>(toFloat(a[index]) + toFloat(b[index]));
    }
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx,
                             T *max_val,
                             const T *vals,
                             size_t numel) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    size_t best_index = 0;
    float best_value = toFloat(vals[0]);
    for (size_t index = 1; index < numel; ++index) {
        const float value = toFloat(vals[index]);
        if (value > best_value) {
            best_index = index;
            best_value = value;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_index);
    max_val[0] = vals[best_index];
}

template <typename T>
__global__ void embeddingKernel(T *out,
                                const int64_t *index,
                                const T *weight,
                                size_t nindex,
                                size_t embedding_dim) {
    const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = nindex * embedding_dim;
    if (element < numel) {
        const size_t row = element / embedding_dim;
        const size_t col = element % embedding_dim;
        out[element] = weight[static_cast<size_t>(index[row]) * embedding_dim + col];
    }
}

template <typename T>
__global__ void linearKernel(T *out,
                             const T *in,
                             const T *weight,
                             const T *bias,
                             size_t rows,
                             size_t in_features,
                             size_t out_features) {
    const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = rows * out_features;
    if (element >= numel) {
        return;
    }
    const size_t row = element / out_features;
    const size_t out_col = element % out_features;
    float value = bias == nullptr ? 0.0f : toFloat(bias[out_col]);
    for (size_t in_col = 0; in_col < in_features; ++in_col) {
        value += toFloat(in[row * in_features + in_col])
                 * toFloat(weight[out_col * in_features + in_col]);
    }
    out[element] = fromFloat<T>(value);
}

template <typename T>
__global__ void rmsNormKernel(T *out,
                              const T *in,
                              const T *weight,
                              size_t rows,
                              size_t width,
                              float eps) {
    const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float square_sum = 0.0f;
    for (size_t col = 0; col < width; ++col) {
        const float value = toFloat(in[row * width + col]);
        square_sum += value * value;
    }
    const float norm = rsqrtf(square_sum / static_cast<float>(width) + eps);
    for (size_t col = 0; col < width; ++col) {
        const size_t index = row * width + col;
        out[index] =
            fromFloat<T>(toFloat(in[index]) * norm * toFloat(weight[col]));
    }
}

template <typename T>
__global__ void ropeKernel(T *out,
                           const T *in,
                           const int64_t *pos_ids,
                           size_t seq_len,
                           size_t n_heads,
                           size_t head_dim,
                           float theta) {
    const size_t half_dim = head_dim / 2;
    const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t numel = seq_len * n_heads * half_dim;
    if (element >= numel) {
        return;
    }
    const size_t dim = element % half_dim;
    const size_t head_element = element / half_dim;
    const size_t head = head_element % n_heads;
    const size_t seq = head_element / n_heads;
    const size_t base = (seq * n_heads + head) * head_dim;
    const float exponent =
        2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
    const float angle =
        static_cast<float>(pos_ids[seq]) / powf(theta, exponent);
    const float sin_value = sinf(angle);
    const float cos_value = cosf(angle);
    const float a = toFloat(in[base + dim]);
    const float b = toFloat(in[base + half_dim + dim]);
    out[base + dim] = fromFloat<T>(a * cos_value - b * sin_value);
    out[base + half_dim + dim] =
        fromFloat<T>(b * cos_value + a * sin_value);
}

template <typename T>
__global__ void attentionKernel(T *out,
                                const T *q,
                                const T *k,
                                const T *v,
                                size_t q_len,
                                size_t kv_len,
                                size_t n_heads,
                                size_t n_kv_heads,
                                size_t qk_dim,
                                size_t value_dim,
                                float scale) {
    const size_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t query_heads = q_len * n_heads;
    if (element >= query_heads) {
        return;
    }
    const size_t query_pos = element / n_heads;
    const size_t head = element % n_heads;
    const size_t kv_head = head / (n_heads / n_kv_heads);
    const size_t attended_len = kv_len - q_len + query_pos + 1;
    const T *query = q + (query_pos * n_heads + head) * qk_dim;

    float max_score = -CUDART_INF_F;
    for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
        const T *key = k + (key_pos * n_kv_heads + kv_head) * qk_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qk_dim; ++dim) {
            score += toFloat(query[dim]) * toFloat(key[dim]);
        }
        max_score = fmaxf(max_score, score * scale);
    }

    float exp_sum = 0.0f;
    for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
        const T *key = k + (key_pos * n_kv_heads + kv_head) * qk_dim;
        float score = 0.0f;
        for (size_t dim = 0; dim < qk_dim; ++dim) {
            score += toFloat(query[dim]) * toFloat(key[dim]);
        }
        exp_sum += expf(score * scale - max_score);
    }

    T *output = out + (query_pos * n_heads + head) * value_dim;
    for (size_t dim = 0; dim < value_dim; ++dim) {
        float result = 0.0f;
        for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
            const T *key = k + (key_pos * n_kv_heads + kv_head) * qk_dim;
            float score = 0.0f;
            for (size_t qk_index = 0; qk_index < qk_dim; ++qk_index) {
                score += toFloat(query[qk_index]) * toFloat(key[qk_index]);
            }
            const T *value =
                v + (key_pos * n_kv_heads + kv_head) * value_dim;
            result += expf(score * scale - max_score) * toFloat(value[dim]);
        }
        output[dim] = fromFloat<T>(result / exp_sum);
    }
}

template <typename T>
__global__ void swigluKernel(T *out,
                             const T *gate,
                             const T *up,
                             size_t numel) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numel) {
        const float gate_value = toFloat(gate[index]);
        out[index] = fromFloat<T>(
            toFloat(up[index]) * gate_value / (1.0f + expf(-gate_value)));
    }
}

template <typename T>
void launchAdd(std::byte *out,
               const std::byte *a,
               const std::byte *b,
               size_t numel) {
    addKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
}

template <typename T>
void launchArgmax(std::byte *max_idx,
                  std::byte *max_val,
                  const std::byte *vals,
                  size_t numel) {
    argmaxKernel<<<1, 1>>>(reinterpret_cast<int64_t *>(max_idx),
                           reinterpret_cast<T *>(max_val),
                           reinterpret_cast<const T *>(vals),
                           numel);
}

template <typename T>
void launchEmbedding(std::byte *out,
                     const std::byte *index,
                     const std::byte *weight,
                     size_t nindex,
                     size_t embedding_dim) {
    const size_t numel = nindex * embedding_dim;
    embeddingKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        nindex,
        embedding_dim);
}

template <typename T>
void launchLinear(std::byte *out,
                  const std::byte *in,
                  const std::byte *weight,
                  const std::byte *bias,
                  size_t rows,
                  size_t in_features,
                  size_t out_features) {
    const size_t numel = rows * out_features;
    linearKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        bias == nullptr ? nullptr : reinterpret_cast<const T *>(bias),
        rows,
        in_features,
        out_features);
}

template <typename T>
void launchRmsNorm(std::byte *out,
                   const std::byte *in,
                   const std::byte *weight,
                   size_t rows,
                   size_t width,
                   float eps) {
    rmsNormKernel<<<(rows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        rows,
        width,
        eps);
}

template <typename T>
void launchRope(std::byte *out,
                const std::byte *in,
                const std::byte *pos_ids,
                size_t seq_len,
                size_t n_heads,
                size_t head_dim,
                float theta) {
    const size_t numel = seq_len * n_heads * (head_dim / 2);
    ropeKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        seq_len,
        n_heads,
        head_dim,
        theta);
}

template <typename T>
void launchAttention(std::byte *out,
                     const std::byte *q,
                     const std::byte *k,
                     const std::byte *v,
                     size_t q_len,
                     size_t kv_len,
                     size_t n_heads,
                     size_t n_kv_heads,
                     size_t qk_dim,
                     size_t value_dim,
                     float scale) {
    const size_t numel = q_len * n_heads;
    attentionKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        q_len,
        kv_len,
        n_heads,
        n_kv_heads,
        qk_dim,
        value_dim,
        scale);
}

template <typename T>
void launchSwiglu(std::byte *out,
                  const std::byte *gate,
                  const std::byte *up,
                  size_t numel) {
    swigluKernel<<<(numel + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
}

#define DISPATCH_FLOAT_TYPES(DTYPE, FUNCTION, ...)       \
    switch (DTYPE) {                                     \
    case LLAISYS_DTYPE_F32:                              \
        FUNCTION<float>(__VA_ARGS__);                    \
        break;                                           \
    case LLAISYS_DTYPE_F16:                              \
        FUNCTION<fp16_t>(__VA_ARGS__);                   \
        break;                                           \
    case LLAISYS_DTYPE_BF16:                             \
        FUNCTION<bf16_t>(__VA_ARGS__);                   \
        break;                                           \
    default:                                             \
        EXCEPTION_UNSUPPORTED_DATATYPE(DTYPE);           \
    }

} // namespace

void add(std::byte *out,
         const std::byte *a,
         const std::byte *b,
         llaisysDataType_t dtype,
         size_t numel) {
    if (numel == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(dtype, launchAdd, out, a, b, numel);
    checkLaunch("CUDA add");
}

void argmax(std::byte *max_idx,
            std::byte *max_val,
            const std::byte *vals,
            llaisysDataType_t dtype,
            size_t numel) {
    DISPATCH_FLOAT_TYPES(dtype, launchArgmax, max_idx, max_val, vals, numel);
    checkLaunch("CUDA argmax");
}

void embedding(std::byte *out,
               const std::byte *index,
               const std::byte *weight,
               llaisysDataType_t dtype,
               size_t nindex,
               size_t embedding_dim) {
    if (nindex == 0 || embedding_dim == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(
        dtype, launchEmbedding, out, index, weight, nindex, embedding_dim);
    checkLaunch("CUDA embedding");
}

void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t dtype,
            size_t rows,
            size_t in_features,
            size_t out_features) {
    if (rows == 0 || out_features == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(dtype,
                         launchLinear,
                         out,
                         in,
                         weight,
                         bias,
                         rows,
                         in_features,
                         out_features);
    checkLaunch("CUDA linear");
}

void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t dtype,
              size_t rows,
              size_t width,
              float eps) {
    if (rows == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(
        dtype, launchRmsNorm, out, in, weight, rows, width, eps);
    checkLaunch("CUDA rms_norm");
}

void rope(std::byte *out,
          const std::byte *in,
          const std::byte *pos_ids,
          llaisysDataType_t dtype,
          size_t seq_len,
          size_t n_heads,
          size_t head_dim,
          float theta) {
    if (seq_len == 0 || n_heads == 0 || head_dim == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(dtype,
                         launchRope,
                         out,
                         in,
                         pos_ids,
                         seq_len,
                         n_heads,
                         head_dim,
                         theta);
    checkLaunch("CUDA rope");
}

void self_attention(std::byte *out,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t dtype,
                    size_t q_len,
                    size_t kv_len,
                    size_t n_heads,
                    size_t n_kv_heads,
                    size_t qk_dim,
                    size_t value_dim,
                    float scale) {
    if (q_len == 0 || n_heads == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(dtype,
                         launchAttention,
                         out,
                         q,
                         k,
                         v,
                         q_len,
                         kv_len,
                         n_heads,
                         n_kv_heads,
                         qk_dim,
                         value_dim,
                         scale);
    checkLaunch("CUDA self_attention");
}

void swiglu(std::byte *out,
            const std::byte *gate,
            const std::byte *up,
            llaisysDataType_t dtype,
            size_t numel) {
    if (numel == 0) {
        return;
    }
    DISPATCH_FLOAT_TYPES(dtype, launchSwiglu, out, gate, up, numel);
    checkLaunch("CUDA swiglu");
}

} // namespace llaisys::ops::nvidia
