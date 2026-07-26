#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cstdint>

namespace llaisys::ops {
namespace {

template <typename T>
void argmax_cpu(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best_idx = 0;
    float best_val = cpu::to_float(vals[0]);
    for (size_t i = 1; i < numel; ++i) {
        const float value = cpu::to_float(vals[i]);
        if (value > best_val) {
            best_idx = i;
            best_val = value;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = vals[best_idx];
}

} // namespace

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_ARGUMENT(vals->ndim() == 1, "argmax input must be one-dimensional");
    CHECK_ARGUMENT(vals->numel() > 0, "argmax input must not be empty");
    CHECK_ARGUMENT(max_idx->shape() == std::vector<size_t>{1},
                   "argmax index output must have shape [1]");
    CHECK_ARGUMENT(max_val->shape() == std::vector<size_t>{1},
                   "argmax value output must have shape [1]");
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64,
                   "argmax index output must have int64 dtype");
    CHECK_ARGUMENT(max_val->dtype() == vals->dtype(),
                   "argmax value output dtype must match input dtype");
    CHECK_ARGUMENT(cpu::is_float_dtype(vals->dtype()),
                   "argmax only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(max_idx->isContiguous() && max_val->isContiguous()
                       && vals->isContiguous(),
                   "argmax tensors must be contiguous");
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        auto *idx = reinterpret_cast<int64_t *>(max_idx->data());
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu(idx,
                              reinterpret_cast<float *>(max_val->data()),
                              reinterpret_cast<const float *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_F16:
            return argmax_cpu(idx,
                              reinterpret_cast<fp16_t *>(max_val->data()),
                              reinterpret_cast<const fp16_t *>(vals->data()),
                              vals->numel());
        case LLAISYS_DTYPE_BF16:
            return argmax_cpu(idx,
                              reinterpret_cast<bf16_t *>(max_val->data()),
                              reinterpret_cast<const bf16_t *>(vals->data()),
                              vals->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }

    core::context().setDevice(vals->deviceType(), vals->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (vals->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::argmax(max_idx->data(),
                              max_val->data(),
                              vals->data(),
                              vals->dtype(),
                              vals->numel());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
