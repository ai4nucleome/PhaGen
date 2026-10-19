# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_loader/loader.py

from abc import ABC

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..utils import logging
from ..utils.import_utils import is_torch_npu_available, is_vescale_available
from .module_utils import init_empty_weights, load_model_weights
from .registry import get_registry


logger = logging.get_logger(__name__)


class BaseModelLoader(ABC):
    def __init__(self):
        pass

    def load_model(self, model_config, **kwargs):
        raise NotImplementedError


class HuggingfaceLoader(BaseModelLoader):
    def __init__(self):
        super().__init__()

    def load_model(self, init_kwargs: dict, **kwargs):
        model_config = init_kwargs["config"]
        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():  # assume built-in models
            load_class = AutoModelForVision2Seq
        else:
            load_class = AutoModelForCausalLM

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from Huggingface modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            with torch.device(init_device):
                model = load_class.from_config(**init_kwargs)
        else:
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init():
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                with init_empty_weights(), no_init_weights():
                    model = load_class.from_config(**init_kwargs)
            if not empty_init:
                load_model_weights(model, weights_path, init_device)

        # we should tie embeddings after loading weights because to_empty() leads to untied weights,
        # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
        if isinstance(model, PreTrainedModel) and getattr(model.config, "tie_word_embeddings", True):
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]

        return model


class CustomizedModelingLoader(BaseModelLoader):
    def __init__(self, model_cls: PreTrainedModel):
        super().__init__()
        self.model_cls = model_cls

    def load_model(self, init_kwargs: dict, **kwargs):
        init_kwargs.pop("trust_remote_code", True)

        init_device = kwargs.pop("init_device", "cuda")
        weights_path = kwargs.pop("weights_path", None)
        empty_init = kwargs.pop("empty_init", False)

        logger.info_rank0(
            f"Loading model from customized modeling.\n"
            f"init_device: {init_device}\n"
            f"empty_init: {empty_init}\n"
            f"weights_path: {weights_path}"
        )

        if weights_path is None:  # init empty model from config
            if is_torch_npu_available() and init_device == "cuda":
                init_device = "npu"
            with torch.device(init_device):
                model = self.model_cls._from_config(**init_kwargs)
        else:
            if is_vescale_available() and init_device == "meta":
                from vescale.initialize.meta_init import meta_device_init

                with meta_device_init(): 
                    model = self.model_cls._from_config(**init_kwargs)
            else:
                #! 在这里初始化模型，使用 init_empty_weights 和 no_init_weights 来初始化模型
                with init_empty_weights(), no_init_weights():
                    model = self.model_cls._from_config(**init_kwargs)
            #! 在这里加载模型权重，如果是空模型，则不加载模型权重
            if not empty_init:
                load_model_weights(model, weights_path, init_device)

        # we should tie embeddings after loading weights because to_empty() leads to untied weights,
        # except for fsdp1 (custom init) and fsdp2 (swap tensor) contexts.
        # 我们应该在加载权重后绑定嵌入层，因为 to_empty() 会导致权重解绑，
        # 除了 fsdp1（自定义初始化）和 fsdp2（张量交换）上下文之外。
        if isinstance(model, PreTrainedModel) and getattr(model.config, "tie_word_embeddings", True):
            input_embeddings = model.get_input_embeddings()
            output_embeddings = model.get_output_embeddings()
            output_embeddings._parameters["weight"] = input_embeddings._parameters["weight"]

        return model


def _get_model_arch_from_config(model_config):
    """
    获取要加载的模型的架构类型
    """
    arch_name = model_config.architectures
    if isinstance(arch_name, list):
        arch_name = arch_name[0]
    return arch_name


def get_loader(model_config):
    # 获取要加载的模型的架构类型
    model_arch = _get_model_arch_from_config(model_config) 
    # 获取模型注册表
    model_registry = get_registry()
    # 如果模型架构在模型注册表中，则使用模型注册表中的模型类
    if model_arch in model_registry.supported_models:
        model_cls = model_registry.get_model_cls_from_model_arch(model_arch)
        # 返回一个加载模型的对象（这里Qwen2.5-Coder-0.5B的model_arch是Qwen2ForCausalLM，因此使用的是 modelling_qwen2.py 中的 Qwen2ForCausalLM 类）
        loader = CustomizedModelingLoader(model_cls=model_cls)
    else:
    # 如果模型架构不在模型注册表中，则使用 HuggingfaceLoader 加载模型
        loader = HuggingfaceLoader()

    return loader
