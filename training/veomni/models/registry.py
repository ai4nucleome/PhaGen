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


# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py
"""Discover model classes exposed through the ModelClass module attribute."""
import importlib
import pkgutil
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Type, Union

import torch.nn as nn

from ..utils import logging


logger = logging.get_logger(__name__)

# 模型注册所在的文件夹路径，这里注册了两个文件夹，分别是 transformers 和 seed_omni
MODELING_PATH = ["veomni.models.transformers", "veomni.models.seed_omni"]


@dataclass
class _ModelRegistry:
    # Keyed by model_arch
    # 储存要扫描的文件夹路径
    modeling_path: List[str] = field(default_factory=list)
    # 模型架构到类名的映射
    model_arch_name_to_cls: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)

    def __post_init__(self):
        for modeling_path in self.modeling_path:
            self._mapping_model_arch_name_to_cls(modeling_path)

    @property
    def supported_models(self) -> Dict[str, Type[nn.Module]]:
        return self.model_arch_name_to_cls.keys()

    def get_model_cls_from_model_arch(self, model_arch: str) -> Type[nn.Module]:
        return self.model_arch_name_to_cls[model_arch]

    def _mapping_model_arch_name_to_cls(self, modeling_path: str):
        # 获取这个包下的所有模块
        package = importlib.import_module(modeling_path)
        # 遍历包中的所有模块
        for _, name, ispkg in pkgutil.walk_packages(package.__path__, modeling_path + "."):
            if not ispkg: # 如果模块不是包，则导入模块
                try:
                    module = importlib.import_module(name) # 导入模块
                except Exception as e:
                    logger.warning(f"Ignore import error when loading {name}. {e}")
                    continue
                if hasattr(module, "ModelClass"): # 如果模块中有 ModelClass 属性，则获取 ModelClass 的值
                    entry = module.ModelClass
                    if isinstance(entry, list):
                        for tmp in entry:
                            assert (
                                tmp.__name__ not in self.model_arch_name_to_cls
                            ), f"Duplicated model implementation for {tmp.__name__}"
                            self.model_arch_name_to_cls[tmp.__name__] = tmp
                    else:
                        assert (
                            entry.__name__ not in self.model_arch_name_to_cls
                        ), f"Duplicated model implementation for {entry.__name__}"
                        self.model_arch_name_to_cls[entry.__name__] = entry


@lru_cache # 缓存装饰器，确保_ModelRegistry 只被初始化一次，后续调用会直接返回缓存的实例
def get_registry():
    return _ModelRegistry(modeling_path=MODELING_PATH)
