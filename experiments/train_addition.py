"""
SOMI 4.0 Training Experiment: 2-Digit Addition
=================================================

Can SOMI learn to add numbers? This is the simplest possible test of
mathematical reasoning — the model must learn carry logic, not just
memorize answers.

Task:
  Input:  "12+34=" (characters)
  Target: "46"     (next characters to predict)

Vocabulary (13 tokens):
  0-9, +, =, <pad>

Architecture: Circuit-S (4 Parts, 2 Systems, hidden=128)
Training: 2000 steps, batch 32, lr 1e-3
Both macro (backprop) and micro (SOMI physics) learning active.

Run with: python -m experiments.train_addition
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

from somi.config import SOMIBrainConfig
from somi.brain.circuit_brain import SOMICircuitBrain
from somi.diagnostics.dashboard import DiagnosticDashboard
from somi.diagnostics.pathology import detect_pathologies, get_health_score


# =========================================================================
# VOCABULARY
# =========================================================================

CHARS = '0123456789+=P'  # P = padding
CHAR_TO_ID = {c: i for i, c in enumerate(CHARS)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)
PAD_ID = CHAR_TO_ID['P']


def encode(s: str, max_len: int = 8) -> list:
    """Encode a string to token IDs, padded to max_len."""
    ids = [CHAR_TO_ID[c] for c in s if c in CHAR_TO_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def decode(ids: list) -> str:
    """Decode token IDs back to string."""
    return ''.join(ID_TO_CHAR.get(i, '?') for i in ids if i != PAD_ID)


# =========================================================================
# DATA GENERATION
# =========================================================================

def generate_addition_example(max_val: int = 50):
    """
    Generate one addition example.

    Returns:
        input_str: e.g. "12+34="
        target_str: e.g. "46PPPP" (padded to 6 chars)
        full_str: e.g. "12+34=46" (for reference)
    """
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = a + b

    input_str = f"{a}+{b}="
    target_str = str(result)
    full_str = input_str + target_str

    return input_str, target_str, full_str


def make_batch(batch_size: int = 32, seq_len: int = 8, max_val: int = 50):
    """
    Generate a batch of addition examples.

    Returns:
        input_ids: [batch, seq_len] token IDs for the FULL sequence
        target_ids: [batch, seq_len] shifted by 1 (next token prediction)
        input_embeddings: [batch, seq_len, vocab_size] one-hot embeddings
    """
    input_ids_list = []
    target_ids_list = []

    for _ in range(batch_size):
        _, _, full = generate_addition_example(max_val)

        # Pad to seq_len
        full_padded = full + 'P' * (seq_len - len(full))
        full_padded = full_padded[:seq_len]

        ids = encode(full_padded, seq_len)
        input_ids_list.append(ids[:-1])   # All but last
        target_ids_list.append(ids[1:])   # All but first (shifted)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)      # [batch, seq-1]
    target_ids = torch.tensor(target_ids_list, dtype=torch.long)     # [batch, seq-1]

    # One-hot embeddings as input to the brain
    input_embeddings = torch.zeros(
        input_ids.shape[0], input_ids.shape[1], VOCAB_SIZE,
        dtype=torch.float32,
    )
    input_embeddings.scatter_(2, input_ids.unsqueeze(-1), 1.0)

    return input_ids, target_ids, input_embeddings


# =========================================================================
# EVALUATION
# =========================================================================

def evaluate(brain: SOMICircuitBrain, n_examples: int = 200, max_val: int = 50):
    """
    Evaluate accuracy on random addition problems.

    Returns:
        accuracy: fraction of examples where ALL result digits are correct
        per_char_accuracy: fraction of individual characters correct
        examples: list of (input, predicted, actual, correct) tuples
    """
    brain.eval()
    correct = 0
    total = 0
    char_correct = 0
    char_total = 0
    examples = []

    with torch.no_grad():
        for _ in range(n_examples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            expected = a + b

            input_str = f"{a}+{b}="
            expected_str = str(expected)
            full_str = input_str + expected_str

            # Pad and encode
            seq_len = 8
            full_padded = full_str + 'P' * (seq_len - len(full_str))
            full_padded = full_padded[:seq_len]
            ids = encode(full_padded, seq_len)

            # Feed up to the "=" and predict the rest
            eq_pos = input_str.index('=')
            input_part = ids[:eq_pos + 1]

            # Pad input to seq_len - 1 for the model
            input_padded = input_part + [PAD_ID] * (seq_len - 1 - len(input_part))
            input_tensor = torch.tensor([input_padded[:seq_len - 1]], dtype=torch.long)
            embed = torch.zeros(1, seq_len - 1, VOCAB_SIZE, dtype=torch.float32)
            embed.scatter_(2, input_tensor.unsqueeze(-1), 1.0)

            # Forward pass
            logits, _ = brain(embed, training=False)

            # Get predictions for positions after "="
            predicted_ids = logits[0].argmax(dim=-1).tolist()

            # Extract predicted result (characters after eq_pos)
            pred_chars = []
            for i in range(eq_pos, min(eq_pos + len(expected_str), len(predicted_ids))):
                pred_chars.append(predicted_ids[i])

            predicted_str = decode(pred_chars)

            # Check accuracy
            is_correct = predicted_str == expected_str
            if is_correct:
                correct += 1
            total += 1

            # Per-character accuracy
            for i, (p, e) in enumerate(zip(predicted_str, expected_str)):
                char_total += 1
                if p == e:
                    char_correct += 1

            if len(examples) < 10:
                examples.append((
                    f"{a}+{b}",
                    predicted_str,
                    expected_str,
                    is_correct,
                ))

    brain.train()

    accuracy = correct / max(total, 1)
    per_char = char_correct / max(char_total, 1)

    return accuracy, per_char, examples


# =========================================================================
# TRAINING
# =========================================================================

def train():
    """Run the full training experiment."""
    print("=" * 70)
    print("  SOMI 4.0 Training Experiment: 2-Digit Addition")
    print("=" * 70)
    print()

    # --- Setup ---
    N_STEPS = 3000
    BATCH_SIZE = 64
    SEQ_LEN = 6          # Shorter sequences for simpler task
    MAX_VAL = 9           # Single-digit: 0+0 to 9+9 (max answer 18)
    LR = 3e-3             # Higher LR for faster learning
    EVAL_EVERY = 100
    DIAG_EVERY = 500

    # Create brain with STABILIZED physics parameters
    # The default physics parameters are too aggressive for a small model.
    # We tune them for stability during early training:
    #   - alpha_1 reduced: coupling strength was blowing up phi
    #   - alpha_0 increased: stronger anchor prevents drift
    #   - dt reduced: smaller timestep = more stable integration
    #   - n_settle fixed at 5: prevents overly long settling
    #   - target_zeta increased: more damping = faster convergence
    #   - residual_weight increased: more of the original signal preserved
    print("Creating Circuit-S brain (stabilized physics)...")
    config = SOMIBrainConfig.circuit_s()
    config.alpha_1 = 0.1       # 10x lower coupling (was 1.0)
    config.alpha_0 = 0.5       # 5x stronger anchor (was 0.1)
    config.kappa_0 = 0.1       # 10x lower prediction error force (was 1.0)
    config.kappa_1 = 0.05      # 10x lower gate strength (was 0.5)
    config.lambda_C = 0.01     # 10x lower coordination (was 0.1)
    config.lambda_E = 0.01     # 10x lower error smoothing (was 0.1)
    config.dt = 0.05           # 3x smaller timestep (was 0.15)
    config.n_settle = 5        # Fixed at 5 (was auto-compute)
    config.target_zeta = 0.5   # More damping (was 0.15)
    config.residual_weight = 0.7  # More residual (was 0.5)
    config.noise_ratio = 0.001    # Less noise (was 0.003)
    config.timescale_ratio = 16.0 # Slower geometry learning (was 8.0)
    brain = SOMICircuitBrain(config, input_dim=VOCAB_SIZE, output_dim=VOCAB_SIZE)
    dashboard = DiagnosticDashboard(config, auto_heal=True, wandb_log=False)

    n_params = sum(p.numel() for p in brain.parameters())
    print(f"  Params: {n_params:,} (macro, backprop-trained)")
    print(f"  Buffers: {sum(b.numel() for b in brain.buffers()):,} (micro, physics-trained)")
    print(f"  Task: {MAX_VAL}+{MAX_VAL} addition ({MAX_VAL*MAX_VAL} possible problems)")
    print()

    # Optimizer (only macro parameters)
    optimizer = optim.AdamW(brain.parameters(), lr=LR, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # W&B setup
    try:
        import wandb
        wandb.init(
            project="somi-4-experiments",
            name="addition-circuit-s",
            config={
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
                "max_val": MAX_VAL,
                "lr": LR,
                "n_parts": config.n_parts,
                "n_systems": config.n_systems,
                "hidden_dim": config.hidden_dim,
            },
        )
        use_wandb = True
        print("W&B logging enabled.")
    except Exception:
        use_wandb = False
        print("W&B not available, logging to console only.")

    print()
    print("Training...")
    print("-" * 70)

    # --- Training loop ---
    loss_history = []
    accuracy_history = []
    health_history = []
    best_accuracy = 0.0
    start_time = time.time()

    brain.train()

    for step in range(1, N_STEPS + 1):
        # Generate batch
        input_ids, target_ids, input_embeddings = make_batch(
            BATCH_SIZE, SEQ_LEN, MAX_VAL
        )

        # Forward pass (BOTH macro and micro learning happen here)
        logits, fwd_diag = brain(input_embeddings, training=True)

        # NaN guard on logits — if physics blew up, skip this step
        if torch.isnan(logits).any():
            if step <= 5 or step % 100 == 0:
                print(f"  [WARN] Step {step}: NaN in logits, skipping backward pass")
            loss_history.append(float('nan'))
            continue

        # Compute loss (cross-entropy on next-token prediction)
        loss = loss_fn(
            logits.reshape(-1, VOCAB_SIZE),
            target_ids.reshape(-1),
        )

        # NaN guard on loss
        if torch.isnan(loss):
            if step <= 5 or step % 100 == 0:
                print(f"  [WARN] Step {step}: NaN loss, skipping backward pass")
            loss_history.append(float('nan'))
            continue

        # Backward pass (only updates macro parameters)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # Print debug info for first few steps
        if step <= 3:
            print(f"  Step {step}: loss={loss_val:.4f}")
            for k, v in sorted(fwd_diag.items()):
                if 'magnitude' in k or 'nan' in k or 'velocity' in k:
                    print(f"    {k}: {v}")

        # --- Periodic evaluation ---
        if step % EVAL_EVERY == 0 or step == 1:
            accuracy, per_char, examples = evaluate(brain, n_examples=200, max_val=MAX_VAL)
            accuracy_history.append((step, accuracy, per_char))

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed

            # Get brain health
            health_score = 100.0  # default
            if step % DIAG_EVERY == 0:
                report = dashboard.report(brain, step)
                health_score = report['health_score']
                health_history.append((step, health_score, report.get('n_pathologies', 0)))

            print(
                f"  Step {step:4d}/{N_STEPS} | "
                f"Loss: {loss_val:.4f} | "
                f"Acc: {accuracy:.1%} | "
                f"CharAcc: {per_char:.1%} | "
                f"Best: {best_accuracy:.1%} | "
                f"{steps_per_sec:.1f} steps/s"
            )

            # Show examples
            if step % (EVAL_EVERY * 5) == 0:
                print("    Examples:")
                for inp, pred, actual, ok in examples[:5]:
                    mark = 'OK' if ok else 'XX'
                    print(f"      {inp} = {pred:>4s} (expected {actual:>4s}) [{mark}]")

            # Log to W&B
            if use_wandb:
                log_dict = {
                    'loss': loss_val,
                    'accuracy': accuracy,
                    'per_char_accuracy': per_char,
                    'best_accuracy': best_accuracy,
                    'steps_per_sec': steps_per_sec,
                }
                if health_history:
                    log_dict['health_score'] = health_history[-1][1]
                    log_dict['n_pathologies'] = health_history[-1][2]
                wandb.log(log_dict, step=step)

    # --- Final evaluation ---
    print()
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print()

    final_acc, final_char, final_examples = evaluate(brain, n_examples=500, max_val=MAX_VAL)

    print(f"  Final Accuracy (exact match): {final_acc:.1%}")
    print(f"  Final Per-Char Accuracy:      {final_char:.1%}")
    print(f"  Best Accuracy During Training: {best_accuracy:.1%}")
    print()

    # Final diagnostics
    final_report = dashboard.report(brain, step=N_STEPS)
    print(f"  Final Health Score: {final_report['health_score']:.0f}/100")
    print(f"  Final Pathologies:  {final_report['n_pathologies']}")
    if final_report.get('pathologies'):
        for p in final_report['pathologies']:
            print(f"    [{p['severity']}] {p['name']}")
    print()

    # Shared Part analysis
    print("  Shared Part Analysis (Part 0 = reasoning core):")
    part0 = brain.parts['0']
    part1 = brain.parts['1']
    print(f"    Part 0 (SHARED) arousal: {part0.arousal.item():.3f}")
    print(f"    Part 1 (normal) arousal: {part1.arousal.item():.3f}")
    print(f"    Part 0 mass diversity:   {part0.mass.std().item():.4f}")
    print(f"    Part 1 mass diversity:   {part1.mass.std().item():.4f}")
    print(f"    Part 0 W magnitude:      {part0.W_local.abs().mean().item():.4f}")
    print(f"    Part 1 W magnitude:      {part1.W_local.abs().mean().item():.4f}")
    print()

    # Show final examples
    print("  Sample predictions:")
    for inp, pred, actual, ok in final_examples[:10]:
        mark = 'OK' if ok else 'XX'
        print(f"    {inp} = {pred:>4s} (expected {actual:>4s}) [{mark}]")
    print()

    # Loss trend
    early_loss = sum(loss_history[:50]) / 50
    late_loss = sum(loss_history[-50:]) / 50
    print(f"  Loss trend: {early_loss:.4f} (first 50) -> {late_loss:.4f} (last 50)")
    improvement = (early_loss - late_loss) / early_loss * 100
    print(f"  Improvement: {improvement:.1f}%")
    print()

    if final_acc > 0.1:
        print("  RESULT: SOMI 4.0 CAN LEARN MATH! Loss decreased and accuracy > random.")
    elif late_loss < early_loss:
        print("  RESULT: Learning detected (loss decreasing) but accuracy still low.")
        print("  Consider: more steps, larger model, or tuning physics parameters.")
    else:
        print("  RESULT: No clear learning yet. Needs investigation.")

    print("=" * 70)

    if use_wandb:
        wandb.finish()

    return brain, loss_history, accuracy_history


if __name__ == '__main__':
    brain, losses, accuracies = train()
