from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
from ..libllaisys import DataType
from ..libllaisys import LlaisysQwen2Meta

from ctypes import byref, c_int, c_int64, c_size_t, c_void_p
import json
import numpy as np
from pathlib import Path


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        with (model_path / "config.json").open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        dtype_name = config.get("torch_dtype", "bfloat16")
        dtype_map = {
            "bfloat16": DataType.BF16,
            "float16": DataType.F16,
            "float32": DataType.F32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported Qwen2 dtype: {dtype_name}")

        hidden_size = int(config["hidden_size"])
        n_heads = int(config["num_attention_heads"])
        self._end_token = int(config["eos_token_id"])
        self._meta = LlaisysQwen2Meta(
            dtype=int(dtype_map[dtype_name]),
            nlayer=int(config["num_hidden_layers"]),
            hs=hidden_size,
            nh=n_heads,
            nkvh=int(config["num_key_value_heads"]),
            dh=hidden_size // n_heads,
            di=int(config["intermediate_size"]),
            maxseq=int(config["max_position_embeddings"]),
            voc=int(config["vocab_size"]),
            epsilon=float(config["rms_norm_eps"]),
            theta=float(config["rope_theta"]),
            end_token=self._end_token,
        )

        device_ids = (c_int * 1)(0)
        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self._meta), int(device), device_ids, 1
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        for file in sorted(model_path.glob("*.safetensors")):
            self._load_safetensors_file(file)

    def __del__(self):
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def _load_safetensors_file(self, path: Path):
        with path.open("rb") as file:
            header_size = int.from_bytes(file.read(8), "little")
            header = json.loads(file.read(header_size))

        data_start = 8 + header_size
        mapped = np.memmap(path, mode="r", dtype=np.uint8)
        try:
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                if info["dtype"] != "BF16":
                    raise ValueError(
                        f"Unsupported weight dtype {info['dtype']} for {name}"
                    )
                start, end = info["data_offsets"]
                weight_data = mapped[data_start + start : data_start + end]
                loaded = LIB_LLAISYS.llaisysQwen2ModelLoadWeight(
                    self._model,
                    name.encode("utf-8"),
                    c_void_p(int(weight_data.ctypes.data)),
                    c_size_t(weight_data.nbytes),
                )
                if not loaded:
                    raise RuntimeError(f"Failed to load Qwen2 weight: {name}")
        finally:
            del mapped

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        del top_k, top_p, temperature
        if len(inputs) == 0:
            raise ValueError("Qwen2 generation requires at least one input token")
        if max_new_tokens is None:
            max_new_tokens = 128
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        output = [int(token) for token in inputs]
        if max_new_tokens == 0:
            return output

        if not LIB_LLAISYS.llaisysQwen2ModelReset(self._model):
            raise RuntimeError("Failed to reset Qwen2 model")
        pending = output
        for _ in range(max_new_tokens):
            token_buffer = (c_int64 * len(pending))(*pending)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model, token_buffer, c_size_t(len(pending))
                )
            )
            if next_token < 0:
                raise RuntimeError("Qwen2 inference failed")
            output.append(next_token)
            if next_token == self._end_token:
                break
            pending = [next_token]

        return output
