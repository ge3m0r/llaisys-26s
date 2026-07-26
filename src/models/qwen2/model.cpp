#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

namespace llaisys::models {
namespace {

tensor_t make_tensor(const std::vector<size_t> &shape,
                     llaisysDataType_t dtype,
                     llaisysDeviceType_t device,
                     int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device,
                       int device_id)
    : _meta(meta),
      _device(device),
      _device_id(device_id),
      _cache_length(0) {
    CHECK_ARGUMENT(_device == LLAISYS_DEVICE_CPU
                       || _device == LLAISYS_DEVICE_NVIDIA,
                   "Qwen2 device is unsupported");
    CHECK_ARGUMENT(_meta.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2 currently supports bfloat16 weights only");
    CHECK_ARGUMENT(_meta.nlayer > 0 && _meta.hs > 0 && _meta.nh > 0
                       && _meta.nkvh > 0 && _meta.dh > 0 && _meta.di > 0
                       && _meta.voc > 0,
                   "Qwen2 metadata dimensions must be positive");
    CHECK_ARGUMENT(_meta.nh * _meta.dh == _meta.hs,
                   "Qwen2 hidden size must equal attention heads times head dimension");
    CHECK_ARGUMENT(_meta.nh % _meta.nkvh == 0,
                   "Qwen2 attention heads must be divisible by KV heads");

    _in_embed = make_tensor({_meta.voc, _meta.hs}, _meta.dtype, _device, _device_id);
    _out_embed = make_tensor({_meta.voc, _meta.hs}, _meta.dtype, _device, _device_id);
    _out_norm = make_tensor({_meta.hs}, _meta.dtype, _device, _device_id);

    _layers.reserve(_meta.nlayer);
    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        Qwen2LayerWeights weights{
            make_tensor({_meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.hs, _meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.nkvh * _meta.dh, _meta.hs},
                        _meta.dtype,
                        _device,
                        _device_id),
            make_tensor({_meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.nkvh * _meta.dh, _meta.hs},
                        _meta.dtype,
                        _device,
                        _device_id),
            make_tensor({_meta.nkvh * _meta.dh}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.hs, _meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.di, _meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.di, _meta.hs}, _meta.dtype, _device, _device_id),
            make_tensor({_meta.hs, _meta.di}, _meta.dtype, _device, _device_id)};
        _layers.push_back(std::move(weights));
    }

    _key_cache.resize(_meta.nlayer);
    _value_cache.resize(_meta.nlayer);
}

const LlaisysQwen2Meta &Qwen2Model::meta() const {
    return _meta;
}

const tensor_t &Qwen2Model::inEmbed() const {
    return _in_embed;
}

const tensor_t &Qwen2Model::outEmbed() const {
    return _out_embed;
}

const tensor_t &Qwen2Model::outNorm() const {
    return _out_norm;
}

const std::vector<Qwen2LayerWeights> &Qwen2Model::layers() const {
    return _layers;
}

tensor_t Qwen2Model::tensorForWeight(const std::string &name) {
    if (name == "model.embed_tokens.weight") {
        return _in_embed;
    }
    if (name == "lm_head.weight") {
        return _out_embed;
    }
    if (name == "model.norm.weight") {
        return _out_norm;
    }

    const std::string prefix = "model.layers.";
    CHECK_ARGUMENT(name.compare(0, prefix.size(), prefix) == 0,
                   "unknown Qwen2 weight name");
    const size_t layer_end = name.find('.', prefix.size());
    CHECK_ARGUMENT(layer_end != std::string::npos, "invalid Qwen2 layer weight name");
    const size_t layer =
        static_cast<size_t>(std::stoul(name.substr(prefix.size(),
                                                   layer_end - prefix.size())));
    CHECK_ARGUMENT(layer < _layers.size(), "Qwen2 weight layer is out of range");
    const std::string suffix = name.substr(layer_end + 1);
    auto &weights = _layers[layer];

    if (suffix == "input_layernorm.weight") {
        return weights.attn_norm;
    }
    if (suffix == "self_attn.q_proj.weight") {
        return weights.q_weight;
    }
    if (suffix == "self_attn.q_proj.bias") {
        return weights.q_bias;
    }
    if (suffix == "self_attn.k_proj.weight") {
        return weights.k_weight;
    }
    if (suffix == "self_attn.k_proj.bias") {
        return weights.k_bias;
    }
    if (suffix == "self_attn.v_proj.weight") {
        return weights.v_weight;
    }
    if (suffix == "self_attn.v_proj.bias") {
        return weights.v_bias;
    }
    if (suffix == "self_attn.o_proj.weight") {
        return weights.o_weight;
    }
    if (suffix == "post_attention_layernorm.weight") {
        return weights.mlp_norm;
    }
    if (suffix == "mlp.gate_proj.weight") {
        return weights.gate_weight;
    }
    if (suffix == "mlp.up_proj.weight") {
        return weights.up_weight;
    }
    if (suffix == "mlp.down_proj.weight") {
        return weights.down_weight;
    }

    CHECK_ARGUMENT(false, "unknown Qwen2 layer weight name");
    return nullptr;
}

void Qwen2Model::loadWeight(const std::string &name,
                            const void *data,
                            size_t nbytes) {
    tensor_t tensor = tensorForWeight(name);
    CHECK_ARGUMENT(tensor != nullptr, "Qwen2 weight tensor is missing");
    CHECK_ARGUMENT(nbytes == tensor->numel() * tensor->elementSize(),
                   "Qwen2 weight byte size does not match model metadata");
    tensor->load(data);
}

void Qwen2Model::reset() {
    for (auto &cache : _key_cache) {
        cache.reset();
    }
    for (auto &cache : _value_cache) {
        cache.reset();
    }
    _cache_length = 0;
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr, "Qwen2 token IDs must not be null");
    CHECK_ARGUMENT(ntoken > 0, "Qwen2 inference requires at least one token");
    CHECK_ARGUMENT(_cache_length + ntoken <= _meta.maxseq,
                   "Qwen2 sequence exceeds maximum position embeddings");

    auto hidden =
        make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    std::vector<int64_t> input_tokens(ntoken);
    for (size_t token = 0; token < ntoken; ++token) {
        CHECK_ARGUMENT(token_ids[token] >= 0
                           && static_cast<size_t>(token_ids[token]) < _meta.voc,
                       "Qwen2 token ID is out of range");
        input_tokens[token] = token_ids[token];
    }
    auto input_token_tensor =
        make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    input_token_tensor->load(input_tokens.data());
    ops::embedding(hidden, input_token_tensor, _in_embed);

    auto normed =
        make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    auto q = make_tensor({ntoken, _meta.nh, _meta.dh},
                         _meta.dtype,
                         _device,
                         _device_id);
    auto current_k = make_tensor({ntoken, _meta.nkvh, _meta.dh},
                                 _meta.dtype,
                                 _device,
                                 _device_id);
    auto current_v = make_tensor({ntoken, _meta.nkvh, _meta.dh},
                                 _meta.dtype,
                                 _device,
                                 _device_id);
    auto attention =
        make_tensor({ntoken, _meta.nh, _meta.dh},
                    _meta.dtype,
                    _device,
                    _device_id);
    auto projected =
        make_tensor({ntoken, _meta.hs}, _meta.dtype, _device, _device_id);
    auto gate =
        make_tensor({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
    auto up = make_tensor({ntoken, _meta.di}, _meta.dtype, _device, _device_id);
    auto positions =
        make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);

    std::vector<int64_t> position_values(ntoken);
    for (size_t token = 0; token < ntoken; ++token) {
        position_values[token] = static_cast<int64_t>(_cache_length + token);
    }
    positions->load(position_values.data());

    const size_t total_length = _cache_length + ntoken;
    core::context().setDevice(_device, _device_id);
    const llaisysMemcpyKind_t cache_copy_kind =
        _device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H
                                      : LLAISYS_MEMCPY_D2D;

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        const auto &weights = _layers[layer];

        ops::rms_norm(normed, hidden, weights.attn_norm, _meta.epsilon);
        ops::linear(q->view({ntoken, _meta.hs}),
                    normed,
                    weights.q_weight,
                    weights.q_bias);
        ops::linear(current_k->view({ntoken, _meta.nkvh * _meta.dh}),
                    normed,
                    weights.k_weight,
                    weights.k_bias);
        ops::linear(current_v->view({ntoken, _meta.nkvh * _meta.dh}),
                    normed,
                    weights.v_weight,
                    weights.v_bias);

        ops::rope(q, q, positions, _meta.theta);
        ops::rope(current_k, current_k, positions, _meta.theta);

        const size_t current_cache_bytes =
            current_k->numel() * current_k->elementSize();
        const size_t previous_cache_bytes =
            _cache_length * _meta.nkvh * _meta.dh * current_k->elementSize();
        auto next_key_cache =
            make_tensor({total_length, _meta.nkvh, _meta.dh},
                        _meta.dtype,
                        _device,
                        _device_id);
        auto next_value_cache =
            make_tensor({total_length, _meta.nkvh, _meta.dh},
                        _meta.dtype,
                        _device,
                        _device_id);
        if (_key_cache[layer] != nullptr) {
            core::context().runtime().api()->memcpy_sync(
                next_key_cache->data(),
                _key_cache[layer]->data(),
                previous_cache_bytes,
                cache_copy_kind);
            core::context().runtime().api()->memcpy_sync(
                next_value_cache->data(),
                _value_cache[layer]->data(),
                previous_cache_bytes,
                cache_copy_kind);
        }
        core::context().runtime().api()->memcpy_sync(
            next_key_cache->data() + previous_cache_bytes,
            current_k->data(),
            current_cache_bytes,
            cache_copy_kind);
        core::context().runtime().api()->memcpy_sync(
            next_value_cache->data() + previous_cache_bytes,
            current_v->data(),
            current_cache_bytes,
            cache_copy_kind);
        _key_cache[layer] = std::move(next_key_cache);
        _value_cache[layer] = std::move(next_value_cache);

        ops::self_attention(attention,
                            q,
                            _key_cache[layer],
                            _value_cache[layer],
                            1.0f / std::sqrt(static_cast<float>(_meta.dh)));
        ops::linear(projected,
                    attention->view({ntoken, _meta.hs}),
                    weights.o_weight,
                    nullptr);
        ops::add(hidden, hidden, projected);

        ops::rms_norm(normed, hidden, weights.mlp_norm, _meta.epsilon);
        ops::linear(gate, normed, weights.gate_weight, nullptr);
        ops::linear(up, normed, weights.up_weight, nullptr);
        ops::swiglu(gate, gate, up);
        ops::linear(projected, gate, weights.down_weight, nullptr);
        ops::add(hidden, hidden, projected);
    }

    _cache_length = total_length;

    auto final_hidden =
        make_tensor({1, _meta.hs}, _meta.dtype, _device, _device_id);
    ops::rms_norm(final_hidden,
                  hidden->slice(0, ntoken - 1, ntoken),
                  _out_norm,
                  _meta.epsilon);
    auto logits =
        make_tensor({1, _meta.voc}, _meta.dtype, _device, _device_id);
    ops::linear(logits, final_hidden, _out_embed, nullptr);
    auto max_index =
        make_tensor({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_value = make_tensor({1}, _meta.dtype, _device, _device_id);
    ops::argmax(max_index, max_value, logits->view({_meta.voc}));
    int64_t result = 0;
    core::context().runtime().api()->memcpy_sync(
        &result,
        max_index->data(),
        sizeof(result),
        _device == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H
                                      : LLAISYS_MEMCPY_D2H);
    return result;
}

} // namespace llaisys::models
