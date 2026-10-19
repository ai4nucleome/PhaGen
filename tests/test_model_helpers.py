"""Offline integration checks for the public model's inference helpers."""

import os
from pathlib import Path

import pytest
import torch
from phagen import MDMGenerationConfig, load_model, load_tokenizer


MODEL_DIR = os.environ.get("PHAGEN_MODEL_PATH")


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_DIR:
        pytest.skip("Set PHAGEN_MODEL_PATH to enable offline real-checkpoint tests.")
    assert Path(MODEL_DIR).is_dir(), "PHAGEN_MODEL_PATH must point to a local checkpoint."
    tokenizer = load_tokenizer(MODEL_DIR, local_files_only=True)
    model = load_model(MODEL_DIR, local_files_only=True)
    return model, tokenizer


def test_frozen_weight_forward_and_helper_shapes(bundle):
    model, tokenizer = bundle
    assert type(model).__name__ == "PhaGenModel"
    assert model.config.model_type == "phagen"
    assert model.config.architectures == ["PhaGenModel"]
    assert sum(parameter.numel() for parameter in model.parameters()) == 152421601
    inputs = tokenizer("ACGTACGTAC", return_tensors="pt", add_special_tokens=False)
    with torch.inference_mode():
        output = model(**inputs, is_causal=False)
    assert output.logits.shape == (1, 10, 9)
    assert torch.isfinite(output.logits).all()
    vector = model.embed(inputs["input_ids"], inputs["attention_mask"])
    assert vector.shape == (1, 964)
    assert vector.dtype == torch.float32
    assert not vector.requires_grad
    assert torch.isfinite(vector).all()


@pytest.mark.parametrize("mode", ["attend_pad", "mask_pad"])
def test_embedding_ignores_other_rows_and_extra_right_padding(bundle, mode):
    model, tokenizer = bundle
    sequences = ["ACGT" * 16 + "A", "ACGT" * 256, "ACGT" * 256 + "A"]
    references = []
    for sequence in sequences:
        ids = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"]
        references.append(model.embed(ids, pad_attention_mode=mode))
    expected = torch.cat(references)
    encoded = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=False)
    # Deliberately include a whole extra partial patch in the caller's batch.
    extra = 2048 - encoded["input_ids"].shape[1]
    ids = torch.nn.functional.pad(encoded["input_ids"], (0, extra), value=tokenizer.pad_token_id)
    mask = torch.nn.functional.pad(encoded["attention_mask"], (0, extra), value=0)
    actual = model.embed(ids, mask, pad_attention_mode=mode)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.embed(ids, pad_attention_mode=mode), expected, rtol=1e-5, atol=1e-6)
    order = torch.tensor([2, 0, 2, 1])
    reordered = model.embed(ids[order], mask[order], pad_attention_mode=mode)
    torch.testing.assert_close(reordered, expected[order], rtol=1e-5, atol=1e-6)


def test_embedding_input_contract(bundle):
    model, _ = bundle
    for ids, mask in [
        (torch.tensor([[7, 7]]), None),
        (torch.empty((1, 0), dtype=torch.long), None),
        (torch.tensor([1, 2]), None),
        (torch.tensor([[1.0, 2.0]]), None),
        (torch.tensor([[-1]]), None),
        (torch.tensor([[9]]), None),
        (torch.tensor([[1, 7, 2]]), None),
        (torch.tensor([[1, 2]]), torch.tensor([[0, 1]])),
        (torch.tensor([[1, 2]]), torch.tensor([[1, 0.5]])),
        (torch.tensor([[1, 7]]), torch.tensor([[1, 1]])),
        (torch.tensor([[1, 2]]), torch.tensor([[1]])),
    ]:
        with pytest.raises(ValueError):
            model.embed(ids, mask)
    with pytest.raises(ValueError, match="token canvas"):
        model.embed(torch.ones((1, 131073), dtype=torch.long))
    with pytest.raises(ValueError, match="pad_attention_mode"):
        model.embed(torch.tensor([[1]]), pad_attention_mode="unknown")
    # Explicitly masked tail content is discarded before embedding lookup.
    torch.testing.assert_close(
        model.embed(torch.tensor([[1, 999]]), torch.tensor([[1, 0]])),
        model.embed(torch.tensor([[1]])),
    )
    model.train()
    try:
        with pytest.raises(ValueError, match="eval"):
            model.embed(torch.tensor([[1]]))
    finally:
        model.eval()


def test_model_exposes_strict_contig_extraction(bundle):
    model, tokenizer = bundle
    ids = torch.tensor([0, 1, 3, 4, 2, 5, 0, 1, 6, 2, 5, 0, 4, 5, 0, 2])
    assert model.extract_contigs(ids, tokenizer) == ["ACGT", "G"]
    assert model.extract_contigs(ids, tokenizer, min_length=4) == ["ACGT"]
    assert model.extract_contigs(ids, tokenizer, min_length=2000) == []
    with pytest.raises(ValueError):
        model.extract_contigs(ids[None, :], tokenizer)


def test_short_diffusion_generation(bundle):
    model, tokenizer = bundle
    torch.manual_seed(42)
    prefix = torch.tensor([[tokenizer.bos_token_id]])
    config = MDMGenerationConfig(
        max_length=256, steps=32, alg="entropy", alg_temp=0.1,
        temperature=1.1, top_p=0.9, top_k=4,
        mask_token_id=tokenizer.mask_token_id, pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
    )
    result = model.diffusion_generate(
        input_ids=prefix, attention_mask=torch.ones_like(prefix), generation_config=config,
    )
    assert result.sequences.shape == (1, 256)
    assert result.sequences[0, 0].item() == tokenizer.bos_token_id
    assert result.sequences.min().item() >= 0
    assert result.sequences.max().item() < model.config.num_tokens
    assert not (result.sequences == tokenizer.mask_token_id).any()
    contigs = model.extract_contigs(result.sequences[0], tokenizer)
    assert all(contig and not (set(contig) - set("ACGT")) for contig in contigs)
