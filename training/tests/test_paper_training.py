"""CPU checks of the recorded paper implementation's training semantics."""

from unittest.mock import patch

import torch
import torch.nn.functional as F

from veomni.data.data_collator import DataCollatorWithPaddingMasking
from veomni.data.dataset import PackingMappingDataset
from veomni.models.transformers.phagen.configuration_phagen import PhaGenConfig
from veomni.models.transformers.phagen.generation_utils import MDMGenerationConfig
from veomni.models.transformers.phagen.modeling_phagen import PhaGenModel
from veomni.optim import build_optimizer


def make_model():
    torch.manual_seed(42)
    return PhaGenModel(PhaGenConfig(
        dim=(16, 12, 8), depth=(1, 1, 1), max_seq_len=(2, 2, 4),
        heads=2, dim_head=4, attn_implementation="eager",
        attn_dropout=0.0, ff_dropout=0.0,
    ))


def training_inputs():
    original = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 4, 3, 2, 1, 5, 7, 7, 7, 7]])
    masked = torch.zeros_like(original, dtype=torch.bool)
    masked[:, [0, 2, 5, 6, 8, 11, 13]] = True
    ids = original.masked_fill(masked, 6)
    labels = original.masked_fill(~masked, -100)
    return ids, labels


def test_packing_preserves_complete_sequences_and_attended_pad():
    sequences = [[0, 1, 2, 5], [0, 3, 5]]
    dataset = PackingMappingDataset(sequences, max_seq_len=16, pad_token_id=7, seed=42)
    assert len(dataset) == 1
    sample = dataset[0][0]
    ids = sample["input_ids"].tolist()
    assert ids[:7] in (sequences[0] + sequences[1], sequences[1] + sequences[0])
    assert ids[7:] == [7] * 9
    torch.testing.assert_close(sample["labels"], sample["input_ids"])
    assert sample["attention_mask"].tolist() == [1] * 16


def test_paper_masking_forces_boundaries_and_can_predict_pad():
    collator = DataCollatorWithPaddingMasking(mask_token_id=6, pad_token_id=7, bos_token_id=0, eos_token_id=5)
    original = torch.tensor([0, 1, 2, 5, 7, 7])

    def feature():
        return {"input_ids": original.clone(), "labels": original.clone(), "attention_mask": torch.ones_like(original)}

    with patch("torch.rand", return_value=torch.tensor([0.0])), patch("torch.rand_like", return_value=torch.ones(6)):
        sparse = collator([feature()])
    assert sparse["input_ids"].tolist() == [[6, 1, 2, 6, 7, 7]]
    assert sparse["labels"].tolist() == [[0, -100, -100, 5, -100, -100]]
    torch.testing.assert_close(sparse["mask_ratio"], torch.full((1, 6), 0.002))

    with patch("torch.rand", return_value=torch.tensor([1.0])), patch("torch.rand_like", return_value=torch.zeros(6)):
        dense = collator([feature()])
    assert dense["input_ids"].tolist() == [[6] * 6]
    assert dense["labels"].tolist() == [original.tolist()]
    torch.testing.assert_close(dense["mask_ratio"], torch.full((1, 6), 0.998))


def test_paper_objective_is_masked_ce_without_timestep_weighting():
    model = make_model().eval()
    ids, labels = training_inputs()
    with torch.no_grad():
        first = model(ids, attention_mask=torch.ones_like(ids), labels=labels, mask_ratio=torch.full_like(ids, 0.1, dtype=torch.float32))
        second = model(ids, attention_mask=torch.ones_like(ids), labels=labels, mask_ratio=torch.full_like(ids, 0.9, dtype=torch.float32))
    expected = F.cross_entropy(first.logits.reshape(-1, 9), labels.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(first.loss, expected)
    torch.testing.assert_close(second.loss, expected)


def test_paper_model_backward_and_original_adamw_step():
    model = make_model().train()
    ids, labels = training_inputs()
    optimizer = build_optimizer(model, lr=2e-5, weight_decay=0.1, optimizer_type="adamw")
    assert isinstance(optimizer, torch.optim.AdamW)
    output = model(ids, attention_mask=torch.ones_like(ids), labels=labels, mask_ratio=torch.full_like(ids, 0.5, dtype=torch.float32))
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)
    output.loss.backward()
    before = model.to_logits.weight.detach().clone()
    gradient = model.to_logits.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(model.to_logits.weight.detach(), before)


def test_generation_overrides_do_not_change_shared_config():
    model = make_model().eval()
    config = MDMGenerationConfig(
        max_new_tokens=7, steps=1, alg="entropy", alg_temp=0.0,
        temperature=1.0, top_p=1.0, top_k=1,
        mask_token_id=6, pad_token_id=7, num_return_sequences=1,
    )
    model.generation_config = config
    before = config.to_dict()
    first = model.diffusion_generate(torch.tensor([[0]]), num_return_sequences=2)
    assert first.shape == (2, 8)
    assert config.to_dict() == before
    second = model.diffusion_generate(torch.tensor([[0, 1, 2]]))
    assert second.shape == (1, 10)
    assert config.to_dict() == before
