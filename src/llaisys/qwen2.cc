#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/model.hpp"
#include "../utils.hpp"

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;
};

namespace {

llaisysTensor_t add_handle(LlaisysQwen2Model *model,
                           const llaisys::tensor_t &tensor) {
    model->tensor_handles.push_back(
        std::make_unique<LlaisysTensor>(LlaisysTensor{tensor}));
    return model->tensor_handles.back().get();
}

void initialize_weight_handles(LlaisysQwen2Model *model) {
    const size_t nlayer = model->model->meta().nlayer;
    model->weights.in_embed = add_handle(model, model->model->inEmbed());
    model->weights.out_embed = add_handle(model, model->model->outEmbed());
    model->weights.out_norm_w = add_handle(model, model->model->outNorm());

    model->weights.attn_norm_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_w = new llaisysTensor_t[nlayer];
    model->weights.attn_q_b = new llaisysTensor_t[nlayer];
    model->weights.attn_k_w = new llaisysTensor_t[nlayer];
    model->weights.attn_k_b = new llaisysTensor_t[nlayer];
    model->weights.attn_v_w = new llaisysTensor_t[nlayer];
    model->weights.attn_v_b = new llaisysTensor_t[nlayer];
    model->weights.attn_o_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_norm_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_gate_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_up_w = new llaisysTensor_t[nlayer];
    model->weights.mlp_down_w = new llaisysTensor_t[nlayer];

    const auto &layers = model->model->layers();
    for (size_t layer = 0; layer < nlayer; ++layer) {
        model->weights.attn_norm_w[layer] =
            add_handle(model, layers[layer].attn_norm);
        model->weights.attn_q_w[layer] =
            add_handle(model, layers[layer].q_weight);
        model->weights.attn_q_b[layer] =
            add_handle(model, layers[layer].q_bias);
        model->weights.attn_k_w[layer] =
            add_handle(model, layers[layer].k_weight);
        model->weights.attn_k_b[layer] =
            add_handle(model, layers[layer].k_bias);
        model->weights.attn_v_w[layer] =
            add_handle(model, layers[layer].v_weight);
        model->weights.attn_v_b[layer] =
            add_handle(model, layers[layer].v_bias);
        model->weights.attn_o_w[layer] =
            add_handle(model, layers[layer].o_weight);
        model->weights.mlp_norm_w[layer] =
            add_handle(model, layers[layer].mlp_norm);
        model->weights.mlp_gate_w[layer] =
            add_handle(model, layers[layer].gate_weight);
        model->weights.mlp_up_w[layer] =
            add_handle(model, layers[layer].up_weight);
        model->weights.mlp_down_w[layer] =
            add_handle(model, layers[layer].down_weight);
    }
}

void destroy_weight_arrays(LlaisysQwen2Weights &weights) {
    delete[] weights.attn_norm_w;
    delete[] weights.attn_q_w;
    delete[] weights.attn_q_b;
    delete[] weights.attn_k_w;
    delete[] weights.attn_k_b;
    delete[] weights.attn_v_w;
    delete[] weights.attn_v_b;
    delete[] weights.attn_o_w;
    delete[] weights.mlp_norm_w;
    delete[] weights.mlp_gate_w;
    delete[] weights.mlp_up_w;
    delete[] weights.mlp_down_w;
}

void report_error(const char *function, const char *message) noexcept {
    std::fprintf(stderr, "[ERROR] %s: %s\n", function, message);
}

void report_unknown_error(const char *function) noexcept {
    report_error(function, "unknown C++ exception");
}

} // namespace

__C {
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta,
                                                llaisysDeviceType_t device,
                                                int *device_ids,
                                                int ndevice) {
        LlaisysQwen2Model *result = nullptr;
        try {
            CHECK_ARGUMENT(meta != nullptr, "Qwen2 metadata must not be null");
            CHECK_ARGUMENT(
                ndevice == 1
                    || (device == LLAISYS_DEVICE_CPU && ndevice == 0),
                "Qwen2 currently supports one device");
            CHECK_ARGUMENT(ndevice == 0 || device_ids != nullptr,
                           "Qwen2 device IDs must not be null");
            const int device_id = ndevice == 0 ? 0 : device_ids[0];
            result = new LlaisysQwen2Model;
            result->model =
                std::make_unique<llaisys::models::Qwen2Model>(
                    *meta, device, device_id);
            initialize_weight_handles(result);
            return result;
        } catch (const std::exception &error) {
            if (result != nullptr) {
                destroy_weight_arrays(result->weights);
                delete result;
            }
            report_error("llaisysQwen2ModelCreate", error.what());
            return nullptr;
        } catch (...) {
            if (result != nullptr) {
                destroy_weight_arrays(result->weights);
                delete result;
            }
            report_unknown_error("llaisysQwen2ModelCreate");
            return nullptr;
        }
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        if (model == nullptr) {
            return;
        }
        destroy_weight_arrays(model->weights);
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            return &model->weights;
        } catch (const std::exception &error) {
            report_error("llaisysQwen2ModelWeights", error.what());
            return nullptr;
        } catch (...) {
            report_unknown_error("llaisysQwen2ModelWeights");
            return nullptr;
        }
    }

    uint8_t llaisysQwen2ModelLoadWeight(LlaisysQwen2Model *model,
                                        const char *name,
                                        const void *data,
                                        size_t nbytes) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            CHECK_ARGUMENT(name != nullptr, "Qwen2 weight name must not be null");
            model->model->loadWeight(std::string(name), data, nbytes);
            return 1;
        } catch (const std::exception &error) {
            report_error("llaisysQwen2ModelLoadWeight", error.what());
            return 0;
        } catch (...) {
            report_unknown_error("llaisysQwen2ModelLoadWeight");
            return 0;
        }
    }

    uint8_t llaisysQwen2ModelReset(LlaisysQwen2Model *model) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            model->model->reset();
            return 1;
        } catch (const std::exception &error) {
            report_error("llaisysQwen2ModelReset", error.what());
            return 0;
        } catch (...) {
            report_unknown_error("llaisysQwen2ModelReset");
            return 0;
        }
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model,
                                   int64_t *token_ids,
                                   size_t ntoken) {
        try {
            CHECK_ARGUMENT(model != nullptr, "Qwen2 model must not be null");
            return model->model->infer(token_ids, ntoken);
        } catch (const std::exception &error) {
            report_error("llaisysQwen2ModelInfer", error.what());
            return -1;
        } catch (...) {
            report_unknown_error("llaisysQwen2ModelInfer");
            return -1;
        }
    }
}
