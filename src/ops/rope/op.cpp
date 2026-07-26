#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cmath>
#include <cstdint>

namespace llaisys::ops {
namespace {

template <typename T>
void rope_cpu(T *out,
              const T *in,
              const int64_t *pos_ids,
              size_t seq_len,
              size_t n_heads,
              size_t head_dim,
              float theta) {
    const size_t half_dim = head_dim / 2;
    for (size_t seq = 0; seq < seq_len; ++seq) {
        for (size_t dim = 0; dim < half_dim; ++dim) {
            const float exponent =
                2.0f * static_cast<float>(dim) / static_cast<float>(head_dim);
            const float angle =
                static_cast<float>(pos_ids[seq]) / std::pow(theta, exponent);
            const float sin_value = std::sin(angle);
            const float cos_value = std::cos(angle);
            for (size_t head = 0; head < n_heads; ++head) {
                const size_t base = (seq * n_heads + head) * head_dim;
                const float a = cpu::to_float(in[base + dim]);
                const float b = cpu::to_float(in[base + half_dim + dim]);
                out[base + dim] = cpu::from_float<T>(a * cos_value - b * sin_value);
                out[base + half_dim + dim] =
                    cpu::from_float<T>(b * cos_value + a * sin_value);
            }
        }
    }
}

template <typename T>
void dispatch_rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    rope_cpu(reinterpret_cast<T *>(out->data()),
             reinterpret_cast<const T *>(in->data()),
             reinterpret_cast<const int64_t *>(pos_ids->data()),
             in->shape()[0],
             in->shape()[1],
             in->shape()[2],
             theta);
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3,
                   "rope input and output must be three-dimensional");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rope output shape must match input");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0,
                   "rope head dimension must be even");
    CHECK_ARGUMENT(pos_ids->ndim() == 1
                       && pos_ids->shape()[0] == in->shape()[0],
                   "rope position IDs must match sequence length");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
                   "rope position IDs must have int64 dtype");
    CHECK_ARGUMENT(out->dtype() == in->dtype(), "rope input and output dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(out->dtype()),
                   "rope only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous()
                       && pos_ids->isContiguous(),
                   "rope tensors must be contiguous");
    CHECK_ARGUMENT(theta > 0.0f, "rope theta must be positive");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return dispatch_rope<float>(out, in, pos_ids, theta);
        case LLAISYS_DTYPE_F16:
            return dispatch_rope<fp16_t>(out, in, pos_ids, theta);
        case LLAISYS_DTYPE_BF16:
            return dispatch_rope<bf16_t>(out, in, pos_ids, theta);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rope(out->data(),
                            in->data(),
                            pos_ids->data(),
                            out->dtype(),
                            in->shape()[0],
                            in->shape()[1],
                            in->shape()[2],
                            theta);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
