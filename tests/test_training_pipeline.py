"""
Phase 3 validation suite for the training pipeline.

Covers the Part-10 checklist: config, LoRA, loss, checkpoint/resume, dataset,
gradient flow, and memory plumbing.

Most tests use a deliberately tiny TryOnModel (hidden 128, 1 block per stack) so the
whole suite runs in seconds on CPU. The architecture is identical to the real model —
only the widths and depths shrink — so every structural property under test
(injection points, gradient flow, checkpoint round-trip) is the same one that holds at
full scale. Tests needing the real 972 M checkpoint are marked `slow`.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from fashn_vton.tryon_mmdit import TryOnModel
from fashn_vton.training import memory as mem
from fashn_vton.training.config import LORA_PRESETS, Config, LoRAConfig
from fashn_vton.training import lora as L
from fashn_vton.training import losses


H, W = 48, 36  # both divisible by patch_size 12 -> 4x3 grid


def tiny_model(**kw) -> TryOnModel:
    """Structurally faithful miniature: sum(axes_dim)=128 must equal hidden//heads."""
    args = dict(
        input_shape=(H, W), hidden_size=128, n_heads=1,
        double_blocks_depth=1, single_blocks_depth=1, patch_mixer_depth=1,
        mlp_ratio=2, axes_dim=(16, 56, 56),
    )
    args.update(kw)
    return TryOnModel(**args)


def dummy_batch(bs: int = 1, device="cpu", dtype=torch.float32):
    return {
        "person": torch.randn(bs, 3, H, W, device=device, dtype=dtype),
        "ca_image": torch.randn(bs, 3, H, W, device=device, dtype=dtype),
        "garment": torch.randn(bs, 3, H, W, device=device, dtype=dtype),
        "person_pose": torch.randn(bs, 1, H, W, device=device, dtype=dtype),
        "garment_pose": torch.randn(bs, 1, H, W, device=device, dtype=dtype),
        "category": torch.full((bs,), 3, dtype=torch.long, device=device),
    }


def model_kwargs(bs: int = 1):
    b = dummy_batch(bs)
    return dict(ca_images=b["ca_image"], garment_images=b["garment"],
                person_poses=b["person_pose"], garment_poses=b["garment_pose"],
                garment_categories=b["category"])


# ---------------------------------------------------------------------- config
class TestConfig:
    def test_defaults_validate(self):
        Config().validate()

    def test_roundtrip(self, tmp_path):
        c = Config()
        c.lora.rank = 24
        c.optim.lr = 3e-5
        p = tmp_path / "c.yaml"
        c.save(p)
        assert Config.load(p).to_dict() == c.to_dict()

    def test_nested_types_survive_load(self, tmp_path):
        c = Config()
        p = tmp_path / "c.yaml"
        c.save(p)
        loaded = Config.load(p)
        assert isinstance(loaded.lora, type(c.lora))
        assert isinstance(loaded.memory, type(c.memory))

    def test_overrides_coerce_types(self):
        c = Config().apply_overrides(
            ["optim.lr=5e-5", "train.epochs=7", "memory.gradient_checkpointing=false",
             "data.max_train_samples=none", "checkpoint.resume=none"]
        )
        assert c.optim.lr == 5e-5 and isinstance(c.optim.lr, float)
        assert c.train.epochs == 7 and isinstance(c.train.epochs, int)
        assert c.memory.gradient_checkpointing is False
        assert c.data.max_train_samples is None
        # string fields keep the literal sentinel rather than becoming None
        assert c.checkpoint.resume == "none"

    @pytest.mark.parametrize("bad", [
        ["data.height=100"], ["lora.rank=0"], ["train.cond_dropout_prob=2"],
        ["memory.precision=int4"], ["train.timestep_sampling=magic"], ["lora.preset=nope"],
    ])
    def test_validation_rejects(self, bad):
        with pytest.raises(ValueError):
            Config().apply_overrides(bad)

    def test_unknown_key_rejected(self):
        with pytest.raises(ValueError):
            Config.from_dict({"data": {"not_a_field": 1}})

    def test_effective_batch(self):
        c = Config()
        c.train.batch_size, c.train.grad_accum_steps = 2, 8
        assert c.effective_batch_size == 16

    def test_numeric_list_override_keeps_numeric_type(self):
        """Regression: betas came back as strings and broke AdamW construction."""
        c = Config().apply_overrides(["optim.betas=0.9,0.95"])
        assert c.optim.betas == [0.9, 0.95]
        assert all(isinstance(x, float) for x in c.optim.betas)
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], betas=tuple(c.optim.betas))

    def test_string_list_override_stays_strings(self):
        c = Config().apply_overrides(["lora.target_modules=qkv,proj"])
        assert c.lora.target_modules == ["qkv", "proj"]
        assert c.lora.resolve_targets() == ["qkv", "proj"]


# ------------------------------------------------------------------------ LoRA
class TestLoRA:
    def test_injection_finds_modules(self):
        m, info = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        assert info["n_modules"] > 0
        assert info["n_lora_params"] > 0
        assert any(isinstance(x, L.LoRALinear) for x in m.modules())

    @pytest.mark.parametrize("preset", list(LORA_PRESETS))
    def test_all_presets_match_something(self, preset):
        _, info = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset=preset))
        assert info["n_modules"] > 0

    def test_preset_ordering_is_nested(self):
        counts = {}
        for p in ["attention", "attention_mlp", "attention_mlp_mod"]:
            _, i = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset=p))
            counts[p] = i["n_modules"]
        assert counts["attention"] < counts["attention_mlp"] < counts["attention_mlp_mod"]

    def test_init_is_identity(self):
        """
        B is zero-initialised, so injection must not change the model's output at all.

        Injection is done on the *same* instance so the base weights are provably
        identical — reloading a base state dict into an already-injected model would
        silently no-op, because injection renames keys (`qkv.weight` ->
        `qkv.base.weight`). See test_injection_renames_state_dict_keys.
        """
        m = tiny_model().eval()
        x, t, kw = torch.randn(1, 3, H, W), torch.full((1,), 0.5), model_kwargs()
        with torch.no_grad():
            before = m(x, t, **kw)["x"]

        L.inject_lora(m, LoRAConfig(rank=4, preset="attention_mlp"))
        m.eval()
        with torch.no_grad():
            after = m(x, t, **kw)["x"]
        assert torch.allclose(before, after, atol=1e-6)

    def test_injection_renames_state_dict_keys(self):
        """
        Guards the ordering requirement in engine.build_model.

        Wrapping inserts `.base.` into every adapted key, so a base checkpoint can only
        be loaded BEFORE injection. Loading afterwards with strict=True raises, and with
        strict=False silently leaves the base at random init — the worst failure mode.
        """
        m = tiny_model()
        before = set(m.state_dict())
        L.inject_lora(m, LoRAConfig(rank=4, preset="attention_mlp"))
        after = set(m.state_dict())
        renamed = before - after
        assert renamed, "expected adapted keys to be renamed"
        assert any(k.replace(".weight", ".base.weight") in after for k in renamed)
        with pytest.raises(RuntimeError):
            m.load_state_dict({k: v for k, v in tiny_model().state_dict().items()}, strict=True)

    def test_nonzero_b_changes_output(self):
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        m.eval()
        x, t, kw = torch.randn(1, 3, H, W), torch.full((1,), 0.5), model_kwargs()
        with torch.no_grad():
            a = m(x, t, **kw)["x"]
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.add_(0.05)
            b = m(x, t, **kw)["x"]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_only_lora_trainable(self):
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        n = L.mark_only_lora_trainable(m)
        assert n > 0
        for name, p in m.named_parameters():
            assert p.requires_grad == ("lora_" in name), name
        s = mem.count_parameters(m)
        assert s["trainable"] == n and s["trainable"] < s["total"]

    def test_state_dict_roundtrip(self):
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_" in n:
                    p.normal_()
        sd = L.lora_state_dict(m)
        m2, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        L.load_lora_state_dict(m2, sd, strict=True)
        for k, v in L.lora_state_dict(m2).items():
            assert torch.allclose(v, sd[k])

    def test_rank_mismatch_rejected(self):
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        sd = L.lora_state_dict(m)
        m2, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=8, preset="attention_mlp"))
        with pytest.raises(RuntimeError):
            L.load_lora_state_dict(m2, sd, strict=True)

    def test_merge_restores_plain_linear(self):
        m, info = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        m.eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.02)
        x, t, kw = torch.randn(1, 3, H, W), torch.full((1,), 0.5), model_kwargs()
        with torch.no_grad():
            before = m(x, t, **kw)["x"]
        n = L.merge_lora_into_base(m)
        assert n == info["n_modules"]
        assert not any(isinstance(x_, L.LoRALinear) for x_ in m.modules())
        with torch.no_grad():
            after = m(x, t, **kw)["x"]
        # merged weights must reproduce the adapted output
        assert torch.allclose(before, after, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_inject_into_gpu_model_keeps_devices_consistent(self):
        """
        Regression for the Phase 4 eval-path bug.

        Training injects while the model is still on CPU, then moves it — so a missing
        device= went unnoticed. The evaluation path injects into an already-on-GPU model,
        where CPU-created adapters raise "Expected all tensors to be on the same device"
        at forward and at merge time.
        """
        m = tiny_model().to("cuda")
        m, _ = L.inject_lora(m, LoRAConfig(rank=4, preset="attention_mlp"))
        for n, p in m.named_parameters():
            if "lora_" in n:
                assert p.device.type == "cuda", f"{n} landed on {p.device}"

        kw = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in model_kwargs().items()}
        x, t = torch.randn(1, 3, H, W, device="cuda"), torch.full((1,), 0.5, device="cuda")
        m.eval()
        with torch.no_grad():
            m(x, t, **kw)                    # forward must not raise
        assert L.merge_lora_into_base(m) > 0  # merge must not raise

    def test_no_match_raises(self):
        with pytest.raises(RuntimeError):
            L.inject_lora(tiny_model(), LoRAConfig(rank=4, target_modules=["does_not_exist"]))


# ------------------------------------------------------------------------ loss
class TestLoss:
    def test_flow_batch_endpoints(self):
        cfg = Config()
        x1 = torch.randn(4, 3, H, W)
        xt, t, v = losses.build_flow_batch(x1, cfg)
        assert xt.shape == x1.shape and v.shape == x1.shape and t.shape == (4,)
        assert ((t > 0) & (t < 1)).all()

    def test_interpolant_is_consistent(self):
        """xt = (1-t)x0 + t x1 and v = x1 - x0 must imply x1 = xt + (1-t)v."""
        cfg = Config()
        torch.manual_seed(0)
        x1 = torch.randn(3, 3, H, W)
        g = torch.Generator().manual_seed(0)
        xt, t, v = losses.build_flow_batch(x1, cfg, g)
        t_ = t.view(-1, 1, 1, 1)
        assert torch.allclose(xt + (1 - t_) * v, x1, atol=1e-5)

    @pytest.mark.parametrize("mode", ["uniform", "shifted", "logit_normal"])
    def test_timestep_modes(self, mode):
        cfg = Config()
        cfg.train.timestep_sampling = mode
        t = losses.sample_timesteps(256, cfg, torch.device("cpu"))
        assert t.shape == (256,) and ((t > 0) & (t < 1)).all()

    def test_shifted_biases_toward_high_noise(self):
        """The `shifted` schedule should concentrate mass at low t (high noise)."""
        cfg = Config()
        cfg.train.timestep_sampling = "uniform"
        u = losses.sample_timesteps(20000, cfg, torch.device("cpu"))
        cfg.train.timestep_sampling = "shifted"
        s = losses.sample_timesteps(20000, cfg, torch.device("cpu"))
        assert s.mean() < u.mean()

    def test_cond_dropout_mask(self):
        d = torch.device("cpu")
        assert losses.make_cond_dropout_mask(64, 0.0, d).all()
        m = losses.make_cond_dropout_mask(20000, 0.5, d)
        assert 0.45 < m.float().mean() < 0.55

    def test_loss_is_zero_for_perfect_prediction(self):
        cfg = Config()
        v = torch.randn(2, 3, H, W)
        t = torch.rand(2)
        assert losses.flow_matching_loss(v, v, t, cfg).item() == pytest.approx(0.0, abs=1e-8)

    def test_loss_positive_and_finite(self):
        cfg = Config()
        a, b, t = torch.randn(2, 3, H, W), torch.randn(2, 3, H, W), torch.rand(2)
        l = losses.flow_matching_loss(a, b, t, cfg)
        assert l.item() > 0 and torch.isfinite(l)

    def test_compute_loss_end_to_end(self):
        cfg = Config()
        m = tiny_model()
        out = losses.compute_loss(m, dummy_batch(2), cfg)
        assert torch.isfinite(out["loss"]) and out["loss"].item() > 0


# -------------------------------------------------------------- gradient flow
class TestGradients:
    def test_lora_params_receive_gradients(self):
        cfg = Config()
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        L.mark_only_lora_trainable(m)
        m.train()
        losses.compute_loss(m, dummy_batch(1), cfg)["loss"].backward()

        got = [n for n, p in m.named_parameters() if "lora_" in n and p.grad is not None
               and p.grad.abs().sum() > 0]
        # lora_A gets gradient only once B is non-zero; B always does at step 0.
        assert any("lora_B" in n for n in got), "no gradient reached lora_B"

    def test_frozen_params_get_no_gradients(self):
        cfg = Config()
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        L.mark_only_lora_trainable(m)
        m.train()
        losses.compute_loss(m, dummy_batch(1), cfg)["loss"].backward()
        for n, p in m.named_parameters():
            if "lora_" not in n:
                assert p.grad is None, f"frozen param {n} got a gradient"

    def test_optimizer_updates_lora_only(self):
        cfg = Config()
        cfg.optim.optimizer = "adamw"
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        L.mark_only_lora_trainable(m)
        m.train()
        from fashn_vton.training.engine import build_optimizer

        opt = build_optimizer(cfg, m)
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        for _ in range(3):
            losses.compute_loss(m, dummy_batch(1), cfg)["loss"].backward()
            opt.step()
            opt.zero_grad()

        changed = [n for n, p in m.named_parameters() if not torch.equal(p, before[n])]
        assert changed, "optimizer changed nothing"
        assert all("lora_" in n for n in changed), f"base weights moved: {changed[:3]}"

    def test_gradient_checkpointing_matches_plain(self):
        """Checkpointing must not change the gradients it recomputes."""
        cfg = Config()
        torch.manual_seed(0)
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention_mlp"))
        L.mark_only_lora_trainable(m)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_B" in n:
                    p.normal_(std=0.02)
        m.train()
        batch = dummy_batch(1)

        def grads(use_ckpt):
            m.set_gradient_checkpointing(use_ckpt)
            m.zero_grad(set_to_none=True)
            g = torch.Generator().manual_seed(7)
            losses.compute_loss(m, batch, cfg, g)["loss"].backward()
            return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}

        a, b = grads(False), grads(True)
        assert set(a) == set(b) and a
        for k in a:
            assert torch.allclose(a[k], b[k], atol=1e-5), f"gradient mismatch for {k}"


# ------------------------------------------------------------------ scheduler
class TestScheduler:
    def _lrs(self, cfg, total=100):
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=4, preset="attention"))
        L.mark_only_lora_trainable(m)
        from fashn_vton.training.engine import build_optimizer, build_scheduler

        opt = build_optimizer(cfg, m)
        sch = build_scheduler(cfg, opt, total)
        out = []
        for _ in range(total):
            out.append(sch.get_last_lr()[0])
            opt.step()
            sch.step()
        return out

    def test_warmup_then_decay(self):
        cfg = Config()
        cfg.optim.optimizer = "adamw"
        cfg.optim.warmup_steps, cfg.optim.scheduler = 10, "cosine"
        lrs = self._lrs(cfg)
        assert lrs[0] < lrs[5] < lrs[10]          # warming up
        assert lrs[-1] < lrs[10]                   # then decaying
        assert lrs[10] == pytest.approx(cfg.optim.lr, rel=1e-6)

    def test_constant_schedule_flat_after_warmup(self):
        cfg = Config()
        cfg.optim.optimizer = "adamw"
        cfg.optim.warmup_steps, cfg.optim.scheduler = 5, "constant"
        lrs = self._lrs(cfg, 50)
        assert len(set(round(x, 12) for x in lrs[6:])) == 1

    def test_lr_never_negative(self):
        for kind in ["cosine", "linear", "constant"]:
            cfg = Config()
            cfg.optim.optimizer = "adamw"
            cfg.optim.scheduler = kind
            assert all(x >= 0 for x in self._lrs(cfg, 60))


# ----------------------------------------------------------------- checkpoint
class TestCheckpoint:
    def _setup(self, tmp_path, rank=4):
        cfg = Config()
        cfg.optim.optimizer = "adamw"
        cfg.checkpoint.dir = str(tmp_path)
        cfg.lora.rank, cfg.lora.preset = rank, "attention_mlp"
        m, _ = L.inject_lora(tiny_model(), LoRAConfig(rank=rank, preset="attention_mlp"))
        L.mark_only_lora_trainable(m)
        from fashn_vton.training.engine import build_optimizer, build_scheduler

        opt = build_optimizer(cfg, m)
        sch = build_scheduler(cfg, opt, 100)
        return cfg, m, opt, sch

    def test_save_and_resume_restores_weights(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        with torch.no_grad():
            for n, p in m.named_parameters():
                if "lora_" in n:
                    p.normal_()
        ck = CheckpointManager(cfg, tmp_path)
        path = ck.save(m, opt, sch, epoch=2, step=50, metrics={"val_loss": 0.3})
        assert path.exists()

        cfg2, m2, opt2, sch2 = self._setup(tmp_path)
        state = CheckpointManager(cfg2, tmp_path).load(path, m2, opt2, sch2)
        assert state["epoch"] == 2 and state["step"] == 50
        a, b = L.lora_state_dict(m), L.lora_state_dict(m2)
        for k in a:
            assert torch.allclose(a[k], b[k]), f"{k} not restored"

    def test_checkpoint_excludes_base_weights(self, tmp_path):
        """A checkpoint must hold adapters only — not the 1.94 GB base."""
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        p = CheckpointManager(cfg, tmp_path).save(m, opt, sch, epoch=0, step=1)
        raw = torch.load(p, map_location="cpu", weights_only=False)
        assert all("lora_" in k for k in raw["lora"])
        assert "double_blocks.0.img_attn.qkv.base.weight" not in raw["lora"]

    def test_checkpoint_carries_config_and_rng(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        p = CheckpointManager(cfg, tmp_path).save(m, opt, sch, epoch=1, step=10)
        raw = torch.load(p, map_location="cpu", weights_only=False)
        assert raw["config"]["lora"]["rank"] == cfg.lora.rank
        assert "rng" in raw and "torch" in raw["rng"]
        assert raw["optimizer"] is not None and raw["scheduler"] is not None

    def test_best_tracking(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        ck = CheckpointManager(cfg, tmp_path)
        assert ck.update_best({"val_loss": 1.0}) is True
        assert ck.update_best({"val_loss": 1.5}) is False
        assert ck.update_best({"val_loss": 0.5}) is True
        assert ck.best_value == 0.5

    def test_resume_auto_finds_latest(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        cfg.checkpoint.resume = "auto"
        ck = CheckpointManager(cfg, tmp_path)
        assert ck.resolve_resume() is None          # nothing saved yet
        ck.save(m, opt, sch, epoch=0, step=1)
        assert ck.resolve_resume() is not None

    def test_tagged_save_updates_latest(self, tmp_path):
        """
        Regression: a run whose save_every_steps never fires used to leave no
        `latest.pt`, so `resume: auto` silently restarted from scratch.
        """
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        ck = CheckpointManager(cfg, tmp_path)
        ck.save(m, opt, sch, epoch=1, step=2, tag="epoch0001.pt")
        assert (tmp_path / "latest.pt").exists()
        cfg.checkpoint.resume = "auto"
        assert ck.resolve_resume() == tmp_path / "latest.pt"

    def test_resume_none_disables(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        ck = CheckpointManager(cfg, tmp_path)
        ck.save(m, opt, sch, epoch=0, step=1)
        cfg.checkpoint.resume = "none"
        assert ck.resolve_resume() is None

    def test_inspect(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path)
        p = CheckpointManager(cfg, tmp_path).save(m, opt, sch, epoch=3, step=99)
        info = CheckpointManager.inspect(p)
        assert info["epoch"] == 3 and info["step"] == 99 and info["n_lora_params"] > 0

    def test_rank_mismatch_on_resume_is_loud(self, tmp_path):
        from fashn_vton.training.checkpoint import CheckpointManager

        cfg, m, opt, sch = self._setup(tmp_path, rank=4)
        p = CheckpointManager(cfg, tmp_path).save(m, opt, sch, epoch=0, step=1)
        cfg2, m2, opt2, sch2 = self._setup(tmp_path, rank=8)
        with pytest.raises(RuntimeError):
            CheckpointManager(cfg2, tmp_path).load(p, m2, opt2, sch2)


# --------------------------------------------------------------------- memory
class TestMemory:
    def test_resolve_dtype(self):
        assert mem.resolve_dtype("bf16") is torch.bfloat16
        assert mem.resolve_dtype("fp16") is torch.float16
        assert mem.resolve_dtype("fp32") is torch.float32

    def test_alloc_conf_is_set(self):
        import os

        mem.configure_allocator(True)
        assert "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

    def test_estimate_is_ordered(self):
        e8 = mem.estimate_training_memory(11_000_000, 960_000_000, torch.bfloat16, "adamw8bit")
        e32 = mem.estimate_training_memory(11_000_000, 960_000_000, torch.bfloat16, "adamw")
        assert e8["static_total_gb"] < e32["static_total_gb"]
        assert e8["frozen_weights_gb"] == pytest.approx(1.92, abs=0.05)

    def test_checkpointing_toggle_requires_support(self):
        m = tiny_model()
        assert mem.enable_gradient_checkpointing(m) >= 2
        assert m.gradient_checkpointing is True
        with pytest.raises(AttributeError):
            mem.enable_gradient_checkpointing(torch.nn.Linear(2, 2))

    def test_is_oom_detection(self):
        assert mem.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
        assert not mem.is_oom(RuntimeError("shape mismatch"))

    def test_oom_safe_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("CUDA out of memory")
            return "ok"

        assert mem.oom_safe(flaky, max_retries=3) == "ok"
        assert calls["n"] == 3

    def test_oom_safe_reraises_non_oom(self):
        with pytest.raises(ValueError):
            mem.oom_safe(lambda: (_ for _ in ()).throw(ValueError("nope")))


# -------------------------------------------------------------------- dataset
class TestDataset:
    def _make_cache(self, root: Path, ids, h=H, w=W):
        from fashn_vton.training.data.preprocess import SUBDIRS

        cache = root / f"{h}x{w}"
        for s in SUBDIRS:
            (cache / s).mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for sid in ids:
            Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(cache / "person" / f"{sid}.jpg")
            Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(cache / "garment" / f"{sid}.jpg")
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(cache / "person_pose" / f"{sid}.png")
            Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(cache / "garment_pose" / f"{sid}.png")
            Image.fromarray(np.random.randint(0, 18, (h, w), dtype=np.uint8)).save(cache / "parse" / f"{sid}.png")
        return cache

    def _cfg(self, tmp_path, ids):
        import pandas as pd

        self._make_cache(tmp_path / "cache", ids)
        csv = tmp_path / "train.csv"
        pd.DataFrame({"id": ids, "source": ["x"] * len(ids)}).to_csv(csv, index=False)
        cfg = Config()
        cfg.data.cache_dir = str(tmp_path / "cache")
        cfg.data.height, cfg.data.width = H, W
        cfg.data.train_csv = cfg.data.val_csv = str(csv)
        cfg.data.num_workers = 0
        cfg.data.persistent_workers = False
        cfg.train.batch_size = 2
        return cfg

    def test_dataset_shapes_and_range(self, tmp_path):
        from fashn_vton.training.data.dataset import SareeVTONDataset

        ids = [f"{i:07d}" for i in range(4)]
        cfg = self._cfg(tmp_path, ids)
        ds = SareeVTONDataset(cfg, cfg.data.train_csv, "train")
        assert len(ds) == 4
        s = ds[0]
        assert s["person"].shape == (3, H, W)
        assert s["person_pose"].shape == (1, H, W)
        assert s["ca_image"].shape == (3, H, W)
        assert s["category"].item() == 3
        for k in ["person", "garment", "ca_image", "person_pose", "garment_pose"]:
            assert s[k].min() >= -1.0001 and s[k].max() <= 1.0001, k

    def test_agnostic_masking_changes_ca_image(self, tmp_path):
        from fashn_vton.training.data.dataset import SareeVTONDataset

        ids = [f"{i:07d}" for i in range(2)]
        cfg = self._cfg(tmp_path, ids)
        cfg.data.use_agnostic_mask = False
        off = SareeVTONDataset(cfg, cfg.data.train_csv, "train")[0]
        assert torch.allclose(off["ca_image"], off["person"])

        cfg.data.use_agnostic_mask = True
        on = SareeVTONDataset(cfg, cfg.data.train_csv, "train")[0]
        assert not torch.allclose(on["ca_image"], on["person"])

    def test_dataloader_batches(self, tmp_path):
        from fashn_vton.training.data.dataset import build_dataloader

        cfg = self._cfg(tmp_path, [f"{i:07d}" for i in range(6)])
        dl = build_dataloader(cfg, "train")
        b = next(iter(dl))
        assert b["person"].shape == (2, 3, H, W)
        assert b["category"].shape == (2,)
        assert isinstance(b["id"], list) and len(b["id"]) == 2

    def test_missing_cache_entries_excluded(self, tmp_path):
        import pandas as pd

        from fashn_vton.training.data.dataset import SareeVTONDataset

        ids = [f"{i:07d}" for i in range(3)]
        cfg = self._cfg(tmp_path, ids)
        csv = tmp_path / "with_ghost.csv"
        pd.DataFrame({"id": ids + ["9999999"]}).to_csv(csv, index=False)
        ds = SareeVTONDataset(cfg, csv, "train")
        assert len(ds) == 3      # the uncached id is dropped, not fatal

    def test_empty_dataset_raises(self, tmp_path):
        import pandas as pd

        from fashn_vton.training.data.dataset import SareeVTONDataset

        cfg = self._cfg(tmp_path, [f"{i:07d}" for i in range(2)])
        csv = tmp_path / "none.csv"
        pd.DataFrame({"id": ["8888888"]}).to_csv(csv, index=False)
        with pytest.raises(RuntimeError):
            SareeVTONDataset(cfg, csv, "train")

    def test_corrupt_sample_is_substituted(self, tmp_path):
        from fashn_vton.training.data.dataset import SareeVTONDataset

        ids = [f"{i:07d}" for i in range(4)]
        cfg = self._cfg(tmp_path, ids)
        ds = SareeVTONDataset(cfg, cfg.data.train_csv, "train")
        # corrupt one cached image on disk
        (Path(cfg.data.cache_dir) / f"{H}x{W}" / "person" / f"{ids[0]}.jpg").write_bytes(b"not a jpeg")
        s = ds[0]                       # must fall back rather than raise
        assert s["person"].shape == (3, H, W)
        assert ds.n_failures >= 1

    def test_determinism(self, tmp_path):
        from fashn_vton.training.data.dataset import SareeVTONDataset

        cfg = self._cfg(tmp_path, [f"{i:07d}" for i in range(3)])
        ds = SareeVTONDataset(cfg, cfg.data.train_csv, "train")
        assert torch.equal(ds[1]["person"], ds[1]["person"])
        assert torch.equal(ds[1]["ca_image"], ds[1]["ca_image"])


# ------------------------------------------------------------------- cleaning
class TestCleaning:
    def test_label_ids_come_from_the_package(self):
        """
        Regression for CONTEXT.md §26.2.

        Phase 2/3 hardcoded face/hair as 11/12 from a stale comment in
        scripts/debug_masks.py; in the installed parser those are glasses/arms. Ids must
        be resolved from LABELS_TO_IDS, never written as literals.
        """
        from fashn_human_parser import LABELS_TO_IDS

        from fashn_vton.training.data import clean as C

        assert C.PERSON_IDS == tuple(
            LABELS_TO_IDS[k] for k in ("face", "hair", "arms", "hands", "legs", "feet", "torso")
        )
        assert C.TOP_ID == LABELS_TO_IDS["top"]
        assert C.DRESS_ID == LABELS_TO_IDS["dress"]
        # the specific mix-up that caused the bug
        assert LABELS_TO_IDS["face"] != 11 and LABELS_TO_IDS["hair"] != 12

    @pytest.mark.parametrize("row,expected", [
        # templated blouse render: only `top`, modest coverage
        (dict(top_frac=0.218, dress_frac=0.0, skirt_frac=0.0, pants_frac=0.0), True),
        (dict(top_frac=0.205, dress_frac=0.0, skirt_frac=0.0, pants_frac=0.0), True),
        (dict(top_frac=0.283, dress_frac=0.0, skirt_frac=0.0, pants_frac=0.0), True),
        # real saree flat-lays
        (dict(top_frac=0.632, dress_frac=0.0, skirt_frac=0.0, pants_frac=0.0), False),   # large top
        (dict(top_frac=0.040, dress_frac=0.0, skirt_frac=0.254, pants_frac=0.0), False),  # skirt
        (dict(top_frac=0.302, dress_frac=0.067, skirt_frac=0.006, pants_frac=0.0), False),  # dress
        (dict(top_frac=0.0, dress_frac=0.732, skirt_frac=0.0, pants_frac=0.0), False),
        # plain fabric swatch: almost no garment structure
        (dict(top_frac=0.02, dress_frac=0.0, skirt_frac=0.0, pants_frac=0.0), False),
    ])
    def test_blouse_render_detector(self, row, expected):
        from fashn_vton.training.data.clean import is_blouse_render

        assert is_blouse_render(row) is expected

    def test_blouse_filter_configurable(self):
        c = Config()
        assert c.cleaning.reject_blouse_renders is True
        c.apply_overrides(["cleaning.reject_blouse_renders=false"])
        assert c.cleaning.reject_blouse_renders is False


# ----------------------------------------------------------------- evaluation
class TestEvaluationRegistry:
    def test_registry_has_pixel_and_saree_metrics(self):
        from fashn_vton.evaluation import available

        a = available()
        for k in ["ssim", "psnr", "lpips", "masked_ssim", "saree_structure"]:
            assert k in a

    def test_build_and_compute(self):
        from fashn_vton.evaluation import build

        pred = np.random.randint(0, 255, (32, 24, 3), dtype=np.uint8)
        m = build(["ssim", "psnr"])[0]
        assert m.compute(pred, pred.copy(), None) == pytest.approx(1.0, abs=1e-6)

    def test_unknown_metric_raises(self):
        from fashn_vton.evaluation import build

        with pytest.raises(KeyError):
            build(["not_a_metric"])

    def test_unimplemented_saree_metrics_return_nan(self):
        from fashn_vton.evaluation import build

        pred = np.zeros((16, 16, 3), dtype=np.uint8)
        for name in ["saree_structure", "pallu_presence"]:
            assert np.isnan(build([name])[0].compute(pred, pred, None))

    def test_masked_metric_needs_mask(self):
        from fashn_vton.evaluation import build

        pred = np.random.randint(0, 255, (32, 24, 3), dtype=np.uint8)
        gt = np.random.randint(0, 255, (32, 24, 3), dtype=np.uint8)
        m = build(["masked_ssim"])[0]
        assert np.isnan(m.compute(pred, gt, None))
        mask = np.zeros((32, 24), dtype=bool)
        mask[8:24, 6:18] = True
        assert not np.isnan(m.compute(pred, gt, {"mask": mask}))
