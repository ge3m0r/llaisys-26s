#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops {
namespace {

template <typename T>
void self_attention_cpu(T *attn_val,
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
    const size_t group_size = n_heads / n_kv_heads;
    const size_t cache_len = kv_len - q_len;
    std::vector<float> scores(kv_len);

    for (size_t query_pos = 0; query_pos < q_len; ++query_pos) {
        const size_t attended_len = cache_len + query_pos + 1;
        for (size_t head = 0; head < n_heads; ++head) {
            const size_t kv_head = head / group_size;
            const T *query = q + (query_pos * n_heads + head) * qk_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                const T *key = k + (key_pos * n_kv_heads + kv_head) * qk_dim;
                float dot = 0.0f;
                for (size_t dim = 0; dim < qk_dim; ++dim) {
                    dot += cpu::to_float(query[dim]) * cpu::to_float(key[dim]);
                }
                scores[key_pos] = dot * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float exp_sum = 0.0f;
            for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                exp_sum += scores[key_pos];
            }

            T *output =
                attn_val + (query_pos * n_heads + head) * value_dim;
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float value = 0.0f;
                for (size_t key_pos = 0; key_pos < attended_len; ++key_pos) {
                    const T *value_row =
                        v + (key_pos * n_kv_heads + kv_head) * value_dim;
                    value += scores[key_pos] * cpu::to_float(value_row[dim]);
                }
                output[dim] = cpu::from_float<T>(value / exp_sum);
            }
        }
    }
}

template <typename T>
void dispatch_self_attention(tensor_t attn_val,
                             tensor_t q,
                             tensor_t k,
                             tensor_t v,
                             float scale) {
    self_attention_cpu(reinterpret_cast<T *>(attn_val->data()),
                       reinterpret_cast<const T *>(q->data()),
                       reinterpret_cast<const T *>(k->data()),
                       reinterpret_cast<const T *>(v->data()),
                       q->shape()[0],
                       k->shape()[0],
                       q->shape()[1],
                       k->shape()[1],
                       q->shape()[2],
                       v->shape()[2],
                       scale);
}

} // namespace

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3
                       && k->ndim() == 3 && v->ndim() == 3,
                   "self_attention tensors must be three-dimensional");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0]
                       && k->shape()[1] == v->shape()[1],
                   "self_attention key and value sequence/head dimensions must match");
    CHECK_ARGUMENT(k->shape()[2] == q->shape()[2],
                   "self_attention query and key dimensions must match");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0]
                       && attn_val->shape()[1] == q->shape()[1]
                       && attn_val->shape()[2] == v->shape()[2],
                   "self_attention output shape is invalid");
    CHECK_ARGUMENT(k->shape()[0] >= q->shape()[0],
                   "self_attention KV length must be at least query length");
    CHECK_ARGUMENT(k->shape()[1] > 0
                       && q->shape()[1] % k->shape()[1] == 0,
                   "self_attention query heads must be divisible by KV heads");
    CHECK_ARGUMENT(attn_val->dtype() == q->dtype()
                       && q->dtype() == k->dtype() && q->dtype() == v->dtype(),
                   "self_attention tensor dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(q->dtype()),
                   "self_attention only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(attn_val->isContiguous() && q->isContiguous()
                       && k->isContiguous() && v->isContiguous(),
                   "self_attention tensors must be contiguous");
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            return dispatch_self_attention<float>(attn_val, q, k, v, scale);
        case LLAISYS_DTYPE_F16:
            return dispatch_self_attention<fp16_t>(attn_val, q, k, v, scale);
        case LLAISYS_DTYPE_BF16:
            return dispatch_self_attention<bf16_t>(attn_val, q, k, v, scale);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
    }

    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (attn_val->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::self_attention(attn_val->data(),
                                      q->data(),
                                      k->data(),
                                      v->data(),
                                      attn_val->dtype(),
                                      q->shape()[0],
                                      k->shape()[0],
                                      q->shape()[1],
                                      k->shape()[1],
                                      q->shape()[2],
                                      v->shape()[2],
                                      scale);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
