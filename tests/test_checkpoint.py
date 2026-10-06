import torch

from rlhf.utils.checkpoint import latest_checkpoint, load_checkpoint, restore, save_checkpoint
from rlhf.utils.seed import seed_everything


def make_model_and_opt():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, opt


def train_step(model, opt):
    loss = model(torch.randn(8, 4)).pow(2).mean()  # uses the RNG, like real training
    opt.zero_grad()
    loss.backward()
    opt.step()


def test_resume_is_exact(tmp_path):
    """Train 3 steps, checkpoint, train 2 more. A resumed run must reproduce those 2 steps exactly."""
    seed_everything(0)
    model, opt = make_model_and_opt()
    for _ in range(3):
        train_step(model, opt)
    save_checkpoint(tmp_path, step=3, model=model, optimizer=opt, config={}, phase="test")
    for _ in range(2):
        train_step(model, opt)
    expected = model.weight.detach().clone()

    seed_everything(123)  # different seed: restore() must override it
    model2, opt2 = make_model_and_opt()
    step = restore(load_checkpoint(latest_checkpoint(tmp_path)), model2, opt2)
    assert step == 3
    for _ in range(2):
        train_step(model2, opt2)
    assert torch.equal(model2.weight, expected)


def test_keeps_last_three_and_milestones(tmp_path):
    model, opt = make_model_and_opt()
    for step in range(1, 6):
        save_checkpoint(tmp_path, step=step, model=model, optimizer=opt, config={}, phase="test")
    save_checkpoint(
        tmp_path, step=5, model=model, optimizer=opt, config={}, phase="test", name="iter_0005.pt"
    )
    names = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert names == ["ckpt_00000003.pt", "ckpt_00000004.pt", "ckpt_00000005.pt", "iter_0005.pt"]
    assert latest_checkpoint(tmp_path).name == "ckpt_00000005.pt"
    assert not list(tmp_path.glob("*.tmp"))  # atomic save leaves no temp files behind


def test_compiled_model_saves_clean_keys(tmp_path):
    model, opt = make_model_and_opt()

    class Wrapper(torch.nn.Module):  # stands in for torch.compile's wrapper
        def __init__(self, inner):
            super().__init__()
            self._orig_mod = inner

    save_checkpoint(tmp_path, step=1, model=Wrapper(model), optimizer=opt, config={}, phase="test")
    keys = load_checkpoint(latest_checkpoint(tmp_path))["model"].keys()
    assert set(keys) == {"weight", "bias"}


def test_no_checkpoint_returns_none(tmp_path):
    assert latest_checkpoint(tmp_path) is None
