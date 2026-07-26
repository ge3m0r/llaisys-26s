#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

namespace llaisys::ops {
namespace {

template <typename T>
void linear_cpu(T *out,
                const T *in,
                const T *weight,
                const T *bias,
                size_t rows,
                size_t in_features,
                size_t out_features) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t out_col = 0; out_col < out_features; ++out_col) {
            float value = bias == nullptr ? 0.0f : cpu::to_float(bias[out_col]);
            const T *in_row = in + row * in_features;
            const T *weight_row = weight + out_col * in_features;
            for (size_t in_col = 0; in_col < in_features; ++in_col) {
                value += cpu::to_float(in_row[in_col])
                         * cpu::to_float(weight_row[in_col]);
            }
            out[row * out_features + out_col] = cpu::from_float<T>(value);
        }
    }
}

template <typename T>
void dispatch_linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    linear_cpu(reinterpret_cast<T *>(out->data()),
               reinterpret_cast<const T *>(in->data()),
               reinterpret_cast<const T *>(weight->data()),
               bias == nullptr ? nullptr : reinterpret_cast<const T *>(bias->data()),
               in->shape()[0],
               in->shape()[1],
               weight->shape()[0]);
}

} // namespace

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "linear input, weight, and output must be two-dimensional");
    CHECK_ARGUMENT(weight->shape()[1] == in->shape()[1],
                   "linear input and weight feature dimensions must match");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0]
                       && out->shape()[1] == weight->shape()[0],
                   "linear output shape is invalid");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
                   "linear tensor dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(out->dtype()),
                   "linear only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous()
                       && weight->isContiguous(),
                   "linear tensors must be contiguous");
    if (bias != nullptr) {
        CHECK_ARGUMENT(bias->ndim() == 1
                           && bias->shape()[0] == weight->shape()[0],
                       "linear bias shape is invalid");
        CHECK_ARGUMENT(bias->dtype() == out->dtype(),
                       "linear bias dtype must match output dtype");
        CHECK_ARGUMENT(bias->isContiguous(), "linear bias must be contiguous");
    }
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return dispatch_linear<float>(out, in, weight, bias);
        case LLAISYS_DTYPE_F16:
            return dispatch_linear<fp16_t>(out, in, weight, bias);
        case LLAISYS_DTYPE_BF16:
            return dispatch_linear<bf16_t>(out, in, weight, bias);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias == nullptr ? nullptr : bias->data(),
            out->dtype(),
            in->shape()[0],
            in->shape()[1],
            weight->shape()[0]);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
