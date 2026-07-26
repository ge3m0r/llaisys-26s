#include "op.hpp"

#include "../../utils.hpp"
#include "../cpu_common.hpp"
#ifdef ENABLE_NVIDIA_API
#include "../nvidia/ops_nvidia.cuh"
#endif

#include <cstring>

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "embedding indices must be one-dimensional");
    CHECK_ARGUMENT(weight->ndim() == 2, "embedding weight must be two-dimensional");
    CHECK_ARGUMENT(out->ndim() == 2, "embedding output must be two-dimensional");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64,
                   "embedding indices must have int64 dtype");
    CHECK_ARGUMENT(out->dtype() == weight->dtype(),
                   "embedding output and weight dtypes must match");
    CHECK_ARGUMENT(cpu::is_float_dtype(weight->dtype()),
                   "embedding only supports float32, float16, and bfloat16");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0]
                       && out->shape()[1] == weight->shape()[1],
                   "embedding output shape is invalid");
    CHECK_ARGUMENT(out->isContiguous() && index->isContiguous()
                       && weight->isContiguous(),
                   "embedding tensors must be contiguous");
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const auto *indices = reinterpret_cast<const int64_t *>(index->data());
        const size_t row_bytes = weight->shape()[1] * weight->elementSize();
        for (size_t row = 0; row < index->numel(); ++row) {
            CHECK_ARGUMENT(
                indices[row] >= 0
                    && static_cast<size_t>(indices[row]) < weight->shape()[0],
                "embedding index is out of range");
            std::memcpy(out->data() + row * row_bytes,
                        weight->data()
                            + static_cast<size_t>(indices[row]) * row_bytes,
                        row_bytes);
        }
        return;
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
#ifdef ENABLE_NVIDIA_API
    if (out->deviceType() == LLAISYS_DEVICE_NVIDIA) {
        return nvidia::embedding(out->data(),
                                 index->data(),
                                 weight->data(),
                                 out->dtype(),
                                 index->numel(),
                                 weight->shape()[1]);
    }
#endif
    EXCEPTION_UNSUPPORTED_DEVICE;
}
} // namespace llaisys::ops
