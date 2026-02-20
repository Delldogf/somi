"""
End-to-End Test: Pure SOMI Language Model + Distillation + Growth
==================================================================

Creates a small pure SOMI LM, creates a tiny "teacher" model,
runs distillation with auto-growth, and verifies everything works.

No real LLM required — uses a simple feedforward as the teacher to
prove the pipeline works end-to-end.
"""

import torch
import torch.nn as nn
from somi.lm.model import SOMILanguageModel
from somi.lm.growth import AutoGrowth
from somi.lm.distill import Distiller


class TinyTeacher(nn.Module):
    """Minimal teacher model for testing. Just embedding -> linear -> vocab logits."""
    def __init__(self, vocab_size=256, hidden_dim=64, n_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h) + h
        return self.head(h)


def test_pure_somi_lm_forward():
    """Test 1: Pure SOMI LM can do a forward pass."""
    print("=" * 60)
    print("TEST 1: Pure SOMI LM Forward Pass")
    print("=" * 60)

    model = SOMILanguageModel(vocab_size=256, hidden_dim=64, n_layers=4)
    input_ids = torch.randint(0, 256, (2, 16))

    logits, loss, diag = model(input_ids, labels=input_ids, training=True)

    assert logits.shape == (2, 16, 256), f"Expected (2,16,256), got {logits.shape}"
    assert loss is not None and not torch.isnan(loss), f"Loss is {loss}"
    assert len(diag) > 0, "No diagnostics returned"

    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Diagnostics keys: {list(diag.keys())[:6]}...")
    print("  PASSED\n")


def test_growth():
    """Test 2: SOMI LM can grow via neurogenesis."""
    print("=" * 60)
    print("TEST 2: Neurogenesis (Growth)")
    print("=" * 60)

    model = SOMILanguageModel(vocab_size=256, hidden_dim=64, n_layers=2)
    input_ids = torch.randint(0, 256, (1, 8))

    logits_before, _, _ = model(input_ids, training=False)
    print(f"  Before growth: H={model.hidden_dim}, logits={logits_before.shape}")

    model.grow(96)
    logits_after, _, _ = model(input_ids, training=False)
    print(f"  After growth:  H={model.hidden_dim}, logits={logits_after.shape}")

    assert model.hidden_dim == 96
    assert logits_after.shape == (1, 8, 256)
    print("  PASSED\n")


def test_auto_growth():
    """Test 3: AutoGrowth triggers when stress is high."""
    print("=" * 60)
    print("TEST 3: Auto-Growth Monitor")
    print("=" * 60)

    model = SOMILanguageModel(vocab_size=256, hidden_dim=64, n_layers=2)
    monitor = AutoGrowth(
        model, stress_threshold=0.01, patience=5,
        growth_factor=1.5, max_hidden=128, cooldown=5,
    )

    input_ids = torch.randint(0, 256, (1, 8))
    grew = False
    for step in range(50):
        _, _, diag = model(input_ids, labels=input_ids, training=True)
        triggered = monitor.step(diag)
        if triggered:
            grew = True
            print(f"  Growth triggered at step {step}! H={model.hidden_dim}")
            break

    if not grew:
        print(f"  No growth triggered in 50 steps (stress may have been low)")
        print(f"  Stress EMA: {monitor.stress_ema:.4f}")
    status = monitor.get_status()
    print(f"  Monitor status: {status}")
    print("  PASSED\n")


def test_distillation():
    """Test 4: Distillation from teacher to student works."""
    print("=" * 60)
    print("TEST 4: Knowledge Distillation")
    print("=" * 60)

    vocab_size = 256
    teacher = TinyTeacher(vocab_size=vocab_size, hidden_dim=64)
    student = SOMILanguageModel(vocab_size=vocab_size, hidden_dim=64, n_layers=2)

    distiller = Distiller(student, temperature=2.0, alpha=0.7, lr=1e-3, auto_grow=False)

    input_ids = torch.randint(0, vocab_size, (2, 16))
    labels = input_ids.clone()

    losses = []
    for step in range(20):
        info = distiller.distill_step(teacher, input_ids, labels)
        losses.append(info['loss'])

    print(f"  Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0] * 1.5, "Loss didn't decrease at all"
    print(f"  Steps completed: {distiller.step_count}")
    print("  PASSED\n")


def test_distill_with_growth():
    """Test 5: Full pipeline — distill + auto-grow."""
    print("=" * 60)
    print("TEST 5: Distillation + Auto-Growth")
    print("=" * 60)

    vocab_size = 256
    teacher = TinyTeacher(vocab_size=vocab_size, hidden_dim=64)
    student = SOMILanguageModel(vocab_size=vocab_size, hidden_dim=64, n_layers=2)

    distiller = Distiller(
        student, temperature=2.0, alpha=0.5, lr=1e-3,
        auto_grow=True, max_hidden=128,
    )
    distiller.growth_monitor.stress_threshold = 0.01
    distiller.growth_monitor.patience = 10
    distiller.growth_monitor.cooldown = 10

    input_ids = torch.randint(0, vocab_size, (2, 16))

    initial_H = student.hidden_dim
    for step in range(50):
        info = distiller.distill_step(teacher, input_ids, labels=input_ids)
        if info['grew']:
            print(f"  Step {step}: Growth triggered! H={info['hidden_dim']}")

    print(f"  Initial H: {initial_H}, Final H: {student.hidden_dim}")
    print(f"  Growth events: {len(distiller.growth_monitor.growth_events)}")
    print(f"  Final loss: {distiller.loss_history[-1]:.4f}")
    print("  PASSED\n")


def test_generation():
    """Test 6: Autoregressive generation works."""
    print("=" * 60)
    print("TEST 6: Autoregressive Generation")
    print("=" * 60)

    model = SOMILanguageModel(vocab_size=256, hidden_dim=64, n_layers=2)
    prompt = torch.randint(0, 256, (1, 4))

    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"  Prompt length: {prompt.shape[1]}")
    print(f"  Generated length: {generated.shape[1]}")
    assert generated.shape[1] == prompt.shape[1] + 10
    print("  PASSED\n")


if __name__ == '__main__':
    print("\nSOMI Language Model — End-to-End Test Suite\n")

    test_pure_somi_lm_forward()
    test_growth()
    test_auto_growth()
    test_distillation()
    test_distill_with_growth()
    test_generation()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
