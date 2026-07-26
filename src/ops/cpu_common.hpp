#pragma once

#include "../utils.hpp"

#include <type_traits>

namespace llaisys::ops::cpu {

template <typename T>
float to_float(T value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(value);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
T from_float(float value) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<T>(value);
    } else {
        return static_cast<T>(value);
    }
}

inline bool is_float_dtype(llaisysDataType_t dtype) {
    return dtype == LLAISYS_DTYPE_F32
           || dtype == LLAISYS_DTYPE_F16
           || dtype == LLAISYS_DTYPE_BF16;
}

} // namespace llaisys::ops::cpu
