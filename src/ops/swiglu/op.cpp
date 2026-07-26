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
void swiglu_cpu(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; ++i) {
        const float gate_value = cpu::to_float(gate[i]);
        const float result =
            cpu::to_float(up[i]) * gate_value / (1.0f + std::exp(-gate_value));
        out[i] = cpu::from_float<T>(result);
    }
}

template <typename T>
void dispatch_swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    swiglu_cpu(reinterpret_cast<T *>(out->data()),
               reinterpret_cast<const T *>(gate->data()),
               reinterpret_cast<const T *>(up->data()),
               out->numel());
}

} // namespace

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_ARGUMENT(out->shape() == gate->shape() && out->shape() == up->shape(),
                   "swiglu tensor shapes must match");
    CHECK_ARGUMENT(out->dtype() == gate->dtype() && out->dtype() == up->dtype(),
                   "swiglu tensor dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(out->dtype()),
                   "swiglu only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(out->isContiguous() && gate->isContiguous()
                       && up->isContiguous(),
                   "swiglu tensors must be contiguous");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return dispatch_swiglu<float>(out, gate, up);
        case LLAISYS_DTYPE_F16:
            return dispatch_swiglu<fp16_t>(out, gate, up);
        case LLAISYS_DTYPE_BF16:
            return dispatch_swiglu<bf16_t>(out, gate, up);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::swiglu(out->data(),
                              gate->data(),
                              up->data(),
                              out->dtype(),
                              out->numel());
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
