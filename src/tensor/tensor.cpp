#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>

namespace llaisys {
namespace {

size_t checked_numel(const std::vector<size_t> &shape) {
    size_t result = 1;
    for (size_t size : shape) {
        if (size != 0) {
            CHECK_ARGUMENT(result <= std::numeric_limits<size_t>::max() / size,
                           "tensor shape is too large");
        }
        result *= size;
    }
    return result;
}

std::vector<ptrdiff_t> contiguous_strides(const std::vector<size_t> &shape) {
    std::vector<ptrdiff_t> strides(shape.size());
    size_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        CHECK_ARGUMENT(stride <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                       "tensor stride is too large");
        strides[i - 1] = static_cast<ptrdiff_t>(stride);
        if (shape[i - 1] != 0) {
            CHECK_ARGUMENT(stride <= std::numeric_limits<size_t>::max() / shape[i - 1],
                           "tensor shape is too large");
        }
        stride *= shape[i - 1];
    }
    return strides;
}

} // namespace

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    if (this->numel() == 0) {
        return true;
    }

    ptrdiff_t expected_stride = 1;
    for (size_t i = this->ndim(); i > 0; --i) {
        const size_t dim = i - 1;
        if (_meta.shape[dim] == 1) {
            continue;
        }
        if (_meta.strides[dim] != expected_stride) {
            return false;
        }
        CHECK_ARGUMENT(
            _meta.shape[dim] <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max())
                                     / static_cast<size_t>(expected_stride),
            "tensor stride is too large");
        expected_stride *= static_cast<ptrdiff_t>(_meta.shape[dim]);
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == this->ndim(),
                   "permutation must contain one entry for every tensor dimension");

    std::vector<bool> seen(this->ndim(), false);
    TensorMeta meta{this->dtype(),
                    std::vector<size_t>(this->ndim()),
                    std::vector<ptrdiff_t>(this->ndim())};
    for (size_t i = 0; i < order.size(); ++i) {
        CHECK_ARGUMENT(order[i] < this->ndim(), "permutation dimension is out of range");
        CHECK_ARGUMENT(!seen[order[i]], "permutation dimensions must be unique");
        seen[order[i]] = true;
        meta.shape[i] = _meta.shape[order[i]];
        meta.strides[i] = _meta.strides[order[i]];
    }

    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    const size_t old_numel = checked_numel(_meta.shape);
    const size_t new_numel = checked_numel(shape);
    CHECK_ARGUMENT(old_numel == new_numel,
                   "view shape must contain the same number of elements");

    std::vector<ptrdiff_t> new_strides(shape.size());

    // Empty tensors have no observable memory layout, so a canonical contiguous
    // layout is valid for every shape with zero elements.
    if (old_numel == 0 || _meta.shape.empty()) {
        new_strides = contiguous_strides(shape);
    } else {
        ptrdiff_t view_dim = static_cast<ptrdiff_t>(shape.size()) - 1;
        size_t view_numel = 1;
        size_t tensor_numel = 1;
        ptrdiff_t chunk_base_stride = _meta.strides.back();

        for (size_t tensor_i = _meta.shape.size(); tensor_i > 0; --tensor_i) {
            const size_t tensor_dim = tensor_i - 1;
            CHECK_ARGUMENT(
                _meta.shape[tensor_dim] == 0
                    || tensor_numel <= std::numeric_limits<size_t>::max()
                                            / _meta.shape[tensor_dim],
                "tensor shape is too large");
            tensor_numel *= _meta.shape[tensor_dim];

            const bool chunk_ends =
                tensor_dim == 0
                || (_meta.shape[tensor_dim - 1] != 1
                    && _meta.strides[tensor_dim - 1]
                           != static_cast<ptrdiff_t>(tensor_numel) * chunk_base_stride);
            if (!chunk_ends) {
                continue;
            }

            while (view_dim >= 0
                   && (view_numel < tensor_numel
                       || shape[static_cast<size_t>(view_dim)] == 1)) {
                CHECK_ARGUMENT(
                    view_numel
                        <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                    "tensor stride is too large");
                new_strides[static_cast<size_t>(view_dim)] =
                    static_cast<ptrdiff_t>(view_numel) * chunk_base_stride;
                const size_t dim_size = shape[static_cast<size_t>(view_dim)];
                CHECK_ARGUMENT(
                    dim_size == 0
                        || view_numel <= std::numeric_limits<size_t>::max() / dim_size,
                    "tensor shape is too large");
                view_numel *= dim_size;
                --view_dim;
            }

            CHECK_ARGUMENT(view_numel == tensor_numel,
                           "view shape is incompatible with the tensor's strides");
            if (tensor_dim > 0) {
                chunk_base_stride = _meta.strides[tensor_dim - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }

        CHECK_ARGUMENT(view_dim == -1,
                       "view shape is incompatible with the tensor's strides");
    }

    TensorMeta meta{this->dtype(), shape, std::move(new_strides)};
    return std::shared_ptr<Tensor>(new Tensor(std::move(meta), _storage, _offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < this->ndim(), "slice dimension is out of range");
    CHECK_ARGUMENT(start <= end, "slice start must not be greater than slice end");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice end is out of range");
    CHECK_ARGUMENT(_meta.strides[dim] >= 0, "negative strides are not supported");

    TensorMeta meta = _meta;
    meta.shape[dim] = end - start;

    const size_t stride = static_cast<size_t>(_meta.strides[dim]);
    CHECK_ARGUMENT(start == 0 || stride <= std::numeric_limits<size_t>::max() / start,
                   "slice offset is too large");
    const size_t element_offset = start * stride;
    CHECK_ARGUMENT(element_offset == 0
                       || this->elementSize()
                              <= std::numeric_limits<size_t>::max() / element_offset,
                   "slice offset is too large");
    const size_t byte_offset = element_offset * this->elementSize();
    CHECK_ARGUMENT(_offset <= std::numeric_limits<size_t>::max() - byte_offset,
                   "slice offset is too large");

    return std::shared_ptr<Tensor>(
        new Tensor(std::move(meta), _storage, _offset + byte_offset));
}

void Tensor::load(const void *src_) {
    CHECK_ARGUMENT(src_ != nullptr || this->numel() == 0,
                   "source pointer must not be null");
    CHECK_ARGUMENT(this->isContiguous(), "load requires a contiguous tensor");

    const size_t bytes = this->numel() * this->elementSize();
    if (bytes == 0) {
        return;
    }

    core::context().setDevice(this->deviceType(), this->deviceId());
    const llaisysMemcpyKind_t kind =
        this->deviceType() == LLAISYS_DEVICE_CPU ? LLAISYS_MEMCPY_H2H
                                                 : LLAISYS_MEMCPY_H2D;
    core::context().runtime().api()->memcpy_sync(this->data(), src_, bytes, kind);
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
