import pytest
import torch
from transformers import GenerationConfig

from phagen import MDMGenerationConfig, PhaGenConfig, PhaGenModel


@pytest.mark.parametrize("config_source", ["explicit", "model_default", "transformers"])
def test_generation_overrides_are_limited_to_one_call(config_source):
    model = PhaGenModel(PhaGenConfig(
        dim=(16, 12, 8), depth=(1, 1, 1), max_seq_len=(2, 2, 4),
        heads=2, dim_head=4, attn_implementation="eager",
    )).eval()
    config_class = GenerationConfig if config_source == "transformers" else MDMGenerationConfig
    config = config_class(
        max_new_tokens=7, steps=1, alg="entropy", alg_temp=0.0,
        temperature=1.0, top_p=1.0, top_k=1,
        mask_token_id=6, pad_token_id=7, num_return_sequences=1, do_sample=True,
    )
    before = config.to_dict()
    if config_source == "model_default":
        model.generation_config = config
        kwargs = {}
    else:
        kwargs = {"generation_config": config}

    first = model.diffusion_generate(
        torch.tensor([[0]]), num_return_sequences=2, temperature=0.5, **kwargs,
    )
    assert first.shape == (2, 8)
    assert config.to_dict() == before

    second = model.diffusion_generate(torch.tensor([[0, 1, 2]]), **kwargs)
    assert second.shape == (1, 10)
    assert second[0, :3].tolist() == [0, 1, 2]
    assert config.to_dict() == before
