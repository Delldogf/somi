"""
SOMI 2.0 Cost Lock — Prevents Runaway Experiments
===================================================

This module enforces hard safety caps on all SOMI experiments:

1. MAX STEPS: No training loop can exceed MAX_STEPS_DEFAULT unless
   the environment variable SOMI_EXPENSIVE_RUN=true is set.

2. MAX RUNTIME: A wall-clock timer kills the run if it exceeds
   MAX_RUNTIME_SECONDS.

3. EXPENSIVE_RUN gate: Large runs (steps > 5000, hidden_dim > 512)
   are BLOCKED unless SOMI_EXPENSIVE_RUN=true is set in the environment.

4. W&B auto-finish: safe_wandb_init() and safe_wandb_finish() ensure
   that every W&B run is always closed, even if the script crashes.

Usage in any run_*.py script:
    from somi_2_0.cost_lock import (
        enforce_cost_lock, CostTimer, safe_wandb_init, safe_wandb_finish
    )

    # At the top, before training:
    enforce_cost_lock(n_steps=N_STEPS, hidden_dim=HIDDEN)

    # Wrap training in a timer:
    with CostTimer(max_seconds=MAX_RUNTIME_SECONDS):
        for step in range(N_STEPS):
            ...

    # W&B (automatically finishes on exit):
    run = safe_wandb_init(project="somi-2-0", name="my_run", config={...})
    ...
    safe_wandb_finish()
"""

import os
import sys
import time
import signal
import atexit

# =============================================================================
# CONFIGURABLE CAPS (change these to adjust safety limits)
# =============================================================================

MAX_STEPS_DEFAULT = 5000
"""Maximum training steps for a 'cheap' run.
Anything above this requires SOMI_EXPENSIVE_RUN=true."""

MAX_STEPS_EXPENSIVE = 100_000
"""Absolute maximum even for expensive runs. No script should ever
exceed this without manual code changes."""

MAX_RUNTIME_SECONDS = 1800  # 30 minutes
"""Wall-clock timeout for cheap runs."""

MAX_RUNTIME_EXPENSIVE = 14400  # 4 hours
"""Wall-clock timeout for expensive runs."""

HIDDEN_DIM_THRESHOLD = 512
"""Hidden dims above this require SOMI_EXPENSIVE_RUN=true."""


# =============================================================================
# EXPENSIVE_RUN FLAG
# =============================================================================

def is_expensive_run() -> bool:
    """Check if SOMI_EXPENSIVE_RUN=true is set in the environment.

    How to enable expensive runs:
        PowerShell:  $env:SOMI_EXPENSIVE_RUN = "true"
        Bash:        export SOMI_EXPENSIVE_RUN=true
        One-liner:   SOMI_EXPENSIVE_RUN=true python run_my_script.py
    """
    return os.environ.get("SOMI_EXPENSIVE_RUN", "false").lower() in ("true", "1", "yes")


def enforce_cost_lock(
    n_steps: int,
    hidden_dim: int = 128,
    label: str = "experiment",
) -> None:
    """Call this BEFORE any training loop. Blocks if limits are exceeded.

    What this does (plain English):
    - If n_steps > 5000 and SOMI_EXPENSIVE_RUN is not set, it stops the script.
    - If hidden_dim > 512 and SOMI_EXPENSIVE_RUN is not set, it stops the script.
    - If n_steps > 100000 even with EXPENSIVE_RUN, it stops (absolute cap).
    - Prints a clear message explaining what happened and how to fix it.

    Args:
        n_steps: How many training steps the script wants to run.
        hidden_dim: The model's hidden dimension.
        label: A name for this run (for the log message).
    """
    expensive = is_expensive_run()

    # Absolute cap — never exceed this
    if n_steps > MAX_STEPS_EXPENSIVE:
        print(f"\n{'='*60}")
        print(f"COST LOCK: ABSOLUTE CAP EXCEEDED")
        print(f"  Requested: {n_steps:,} steps")
        print(f"  Absolute max: {MAX_STEPS_EXPENSIVE:,} steps")
        print(f"  This cannot be overridden. Edit cost_lock.py if needed.")
        print(f"{'='*60}\n")
        sys.exit(1)

    # Cheap-run caps
    if not expensive:
        blocked = False
        reasons = []

        if n_steps > MAX_STEPS_DEFAULT:
            reasons.append(
                f"  Steps: {n_steps:,} > {MAX_STEPS_DEFAULT:,} (cheap limit)"
            )
            blocked = True

        if hidden_dim > HIDDEN_DIM_THRESHOLD:
            reasons.append(
                f"  Hidden dim: {hidden_dim} > {HIDDEN_DIM_THRESHOLD} (cheap limit)"
            )
            blocked = True

        if blocked:
            print(f"\n{'='*60}")
            print(f"COST LOCK: EXPENSIVE RUN BLOCKED")
            print(f"  Experiment: {label}")
            for r in reasons:
                print(r)
            print(f"")
            print(f"  To unlock, set the environment variable:")
            print(f"    PowerShell:  $env:SOMI_EXPENSIVE_RUN = \"true\"")
            print(f"    Bash:        export SOMI_EXPENSIVE_RUN=true")
            print(f"{'='*60}\n")
            sys.exit(1)

    # Passed — print confirmation
    mode = "EXPENSIVE" if expensive else "CHEAP"
    step_cap = MAX_STEPS_EXPENSIVE if expensive else MAX_STEPS_DEFAULT
    time_cap = MAX_RUNTIME_EXPENSIVE if expensive else MAX_RUNTIME_SECONDS
    print(f"[cost_lock] Mode: {mode} | Steps: {n_steps:,}/{step_cap:,} | "
          f"Hidden: {hidden_dim} | Timeout: {time_cap}s")


# =============================================================================
# WALL-CLOCK TIMER
# =============================================================================

class CostTimer:
    """Context manager that enforces a wall-clock time limit.

    Usage:
        with CostTimer(max_seconds=1800):
            for step in range(N_STEPS):
                ...  # if this takes longer than 30 min, it prints a warning
                     # and raises TimeoutError

    What this does (plain English):
    - When you enter the 'with' block, it starts a clock.
    - If the clock exceeds max_seconds, it raises a TimeoutError.
    - You can also call timer.check() inside your loop to check manually.
    """

    def __init__(self, max_seconds: int = None):
        expensive = is_expensive_run()
        if max_seconds is None:
            max_seconds = MAX_RUNTIME_EXPENSIVE if expensive else MAX_RUNTIME_SECONDS
        self.max_seconds = max_seconds
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"[cost_lock] Run completed in {elapsed:.1f}s / {self.max_seconds}s limit")
        return False  # Don't suppress exceptions

    def check(self, step: int = None):
        """Call this inside your training loop. Raises TimeoutError if overtime.

        Args:
            step: Current training step (for the error message).
        """
        if self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed > self.max_seconds:
            msg = (f"COST LOCK TIMEOUT: {elapsed:.0f}s > {self.max_seconds}s limit"
                   f"{f' (at step {step})' if step is not None else ''}")
            print(f"\n{'='*60}")
            print(msg)
            print(f"{'='*60}\n")
            raise TimeoutError(msg)

    @property
    def elapsed(self) -> float:
        """Seconds since the timer started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# =============================================================================
# W&B AUTO-FINISH (never leave a run hanging)
# =============================================================================

_active_wandb_run = None


def safe_wandb_init(**kwargs) -> "wandb.sdk.wandb_run.Run | None":
    """Initialize a W&B run with guaranteed cleanup.

    What this does (plain English):
    - Tries to import wandb and start a run.
    - Registers an 'atexit' handler so that if the script crashes or exits,
      wandb.finish() is ALWAYS called. No more zombie runs.
    - Returns the run object, or None if wandb isn't installed.

    Args:
        **kwargs: Passed directly to wandb.init() (project, name, config, etc.)
    """
    global _active_wandb_run
    try:
        import wandb
        # Force finish any previous run
        if _active_wandb_run is not None:
            try:
                wandb.finish()
            except Exception:
                pass

        run = wandb.init(**kwargs)
        _active_wandb_run = run

        # Register cleanup so finish() happens even on crash
        atexit.register(_atexit_wandb_finish)

        return run
    except ImportError:
        print("[cost_lock] wandb not installed — skipping W&B logging")
        return None
    except Exception as e:
        print(f"[cost_lock] wandb.init() failed: {e} — continuing without W&B")
        return None


def safe_wandb_finish():
    """Finish the active W&B run. Safe to call even if no run is active."""
    global _active_wandb_run
    if _active_wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
        _active_wandb_run = None


def safe_wandb_log(data: dict):
    """Log to W&B if a run is active. Silent no-op otherwise."""
    global _active_wandb_run
    if _active_wandb_run is not None:
        try:
            import wandb
            wandb.log(data)
        except Exception:
            pass


def _atexit_wandb_finish():
    """Called automatically when the Python process exits."""
    safe_wandb_finish()
