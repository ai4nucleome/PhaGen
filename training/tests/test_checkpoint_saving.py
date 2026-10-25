from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tasks.train_torch_dllm import save_training_checkpoint


def write_local_checkpoint(root, state, global_steps):
    path = Path(root) / f"global_step_{global_steps}"
    path.mkdir(parents=True)
    torch.save({
        "model": state["model"].state_dict(),
        "optimizer": state["optimizer"].state_dict(),
        "extra_state": state["extra_state"],
    }, path / "state.pt")


@pytest.mark.parametrize("previous_step,final_step", [(None, 120), (200, 250)])
def test_final_checkpoint_contains_current_weights_and_training_state(tmp_path, previous_step, final_step):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    dataloader = SimpleNamespace(state_dict=lambda: {"next_record": 7})
    meter = SimpleNamespace(state_dict=lambda: {"consume_tokens": 32})
    checkpointer = SimpleNamespace(save=Mock(side_effect=write_local_checkpoint))
    kwargs = dict(
        checkpointer=checkpointer, checkpoint_root=str(tmp_path), model=model,
        optimizer=optimizer, lr_scheduler=scheduler, train_dataloader=dataloader,
        environ_meter=meter,
    )
    previous_path = None
    if previous_step is not None:
        previous_path = save_training_checkpoint(global_step=previous_step, **kwargs)
    previous_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    rng_state = torch.get_rng_state().clone()

    final_path = save_training_checkpoint(
        global_step=final_step, last_saved_path=previous_path, **kwargs,
    )
    saved = torch.load(Path(final_path) / "state.pt", weights_only=True)
    assert final_path == str(tmp_path / f"global_step_{final_step}")
    for name, value in model.state_dict().items():
        torch.testing.assert_close(saved["model"][name], value)
        assert not torch.equal(saved["model"][name], previous_weights[name])
    assert saved["optimizer"]["state"]
    assert saved["optimizer"]["param_groups"][0]["lr"] == optimizer.param_groups[0]["lr"]
    assert saved["extra_state"]["global_step"] == final_step
    assert saved["extra_state"]["lr_scheduler"] == scheduler.state_dict()
    assert saved["extra_state"]["train_dataloader"] == {"next_record": 7}
    assert saved["extra_state"]["environ_meter"] == {"consume_tokens": 32}
    torch.testing.assert_close(saved["extra_state"]["torch_rng_state"], rng_state)
    if previous_path is not None:
        previous = torch.load(Path(previous_path) / "state.pt", weights_only=True)
        for name, value in previous_weights.items():
            torch.testing.assert_close(previous["model"][name], value)


def test_final_checkpoint_keeps_an_already_saved_current_step(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    checkpointer = SimpleNamespace(save=Mock(side_effect=write_local_checkpoint))
    kwargs = dict(
        checkpointer=checkpointer, checkpoint_root=str(tmp_path), model=model,
        optimizer=optimizer, lr_scheduler=scheduler,
        train_dataloader=SimpleNamespace(state_dict=lambda: {}),
        environ_meter=SimpleNamespace(state_dict=lambda: {}), global_step=200,
    )
    saved_path = save_training_checkpoint(**kwargs)
    assert save_training_checkpoint(last_saved_path=saved_path, **kwargs) == saved_path
    assert checkpointer.save.call_count == 1
