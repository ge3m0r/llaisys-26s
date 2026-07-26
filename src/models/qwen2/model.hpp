#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace llaisys::models {

struct Qwen2LayerWeights {
    tensor_t attn_norm;
    tensor_t q_weight;
    tensor_t q_bias;
    tensor_t k_weight;
    tensor_t k_bias;
    tensor_t v_weight;
    tensor_t v_bias;
    tensor_t o_weight;
    tensor_t mlp_norm;
    tensor_t gate_weight;
    tensor_t up_weight;
    tensor_t down_weight;
};

class Qwen2Model {
private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    tensor_t _in_embed;
    tensor_t _out_embed;
    tensor_t _out_norm;
    std::vector<Qwen2LayerWeights> _layers;

    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    size_t _cache_length;

    tensor_t tensorForWeight(const std::string &name);

public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device,
               int device_id);

    const LlaisysQwen2Meta &meta() const;
    const tensor_t &inEmbed() const;
    const tensor_t &outEmbed() const;
    const tensor_t &outNorm() const;
    const std::vector<Qwen2LayerWeights> &layers() const;

    void loadWeight(const std::string &name, const void *data, size_t nbytes);
    void reset();
    int64_t infer(const int64_t *token_ids, size_t ntoken);
};

} // namespace llaisys::models
