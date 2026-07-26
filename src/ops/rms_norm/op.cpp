#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cmath>

namespace llaisys::ops {
namespace {

template <typename T>
void rms_norm_cpu(T *out,
                  const T *in,
                  const T *weight,
                  size_t rows,
                  size_t width,
                  float eps) {
    for (size_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        const T *in_row = in + row * width;
        T *out_row = out + row * width;
        for (size_t col = 0; col < width; ++col) {
            const float value = cpu::to_float(in_row[col]);
            square_sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(square_sum / static_cast<float>(width) + eps);
        for (size_t col = 0; col < width; ++col) {
            const float value =
                cpu::to_float(in_row[col]) * scale * cpu::to_float(weight[col]);
            out_row[col] = cpu::from_float<T>(value);
        }
    }
}

template <typename T>
void dispatch_rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    rms_norm_cpu(reinterpret_cast<T *>(out->data()),
                 reinterpret_cast<const T *>(in->data()),
                 reinterpret_cast<const T *>(weight->data()),
                 in->shape()[0],
                 in->shape()[1],
                 eps);
}

} // namespace

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2,
                   "rms_norm input and output must be two-dimensional");
    CHECK_ARGUMENT(weight->ndim() == 1, "rms_norm weight must be one-dimensional");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rms_norm output shape must match input");
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1],
                   "rms_norm weight length must match the last input dimension");
    CHECK_ARGUMENT(out->dtype() == in->dtype() && out->dtype() == weight->dtype(),
                   "rms_norm tensor dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(out->dtype()),
                   "rms_norm only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(out->isContiguous() && in->isContiguous()
                       && weight->isContiguous(),
                   "rms_norm tensors must be contiguous");
    CHECK_ARGUMENT(in->shape()[1] > 0, "rms_norm last dimension must not be empty");
    CHECK_ARGUMENT(eps >= 0.0f, "rms_norm epsilon must be non-negative");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return dispatch_rms_norm<float>(out, in, weight, eps);
        case LLAISYS_DTYPE_F16:
            return dispatch_rms_norm<fp16_t>(out, in, weight, eps);
        case LLAISYS_DTYPE_BF16:
            return dispatch_rms_norm<bf16_t>(out, in, weight, eps);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::rms_norm(out->data(),
                                in->data(),
                                weight->data(),
                                out->dtype(),
                                in->shape()[0],
                                in->shape()[1],
                                eps);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
