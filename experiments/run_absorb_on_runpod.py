"""
Run SOMI Absorb-All on RunPod
==============================

Spins up a B200 GPU pod, clones the somi repo, installs deps,
downloads all models in parallel, absorbs them into one SOMI brain,
then downloads results.

Usage:
    pip install runpod
    python experiments/run_absorb_on_runpod.py

Requires: runpod package, SSH key at ~/.ssh/id_ed25519
Results downloaded to: experiments/runpod_absorb_results/
"""

import runpod
import subprocess
import time
import os
import sys

# ============================================================
# Configuration
# ============================================================

RUNPOD_API_KEY = os.environ.get(
    "RUNPOD_API_KEY",
    "rpa_AK6YBHG3UBFGT7KCR2ZTJ7KPP822ZVUXTK6KQCHX1przg4"
)

WANDB_KEY = "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"

HF_TOKEN = os.environ.get("HF_TOKEN", "")

GITHUB_REPO = "https://github.com/Delldogf/somi.git"

POD_NAME = "somi-absorb-all"

GPU_TYPES = [
    "NVIDIA B200",
    "NVIDIA H200",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_GB = 200
DISK_GB = 50

SSH_KEY = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "runpod_absorb_results")


# ============================================================
# Helpers
# ============================================================

def run_cmd(cmd, timeout=300, check=True):
    """Run a local shell command."""
    print(f"  $ {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and check:
            print(f"  STDERR: {r.stderr[:500]}")
        return r
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None


def ssh_cmd(host, port, cmd, timeout=600):
    """Run a command on the pod via SSH."""
    ssh = (
        f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 '
        f'-i "{SSH_KEY}" -p {port} root@{host} "{cmd}"'
    )
    return run_cmd(ssh, timeout=timeout)


def ssh_cmd_stream(host, port, cmd, timeout=3600):
    """Run a command on the pod and stream output live."""
    ssh = (
        f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 '
        f'-i "{SSH_KEY}" -p {port} root@{host} "{cmd}"'
    )
    print(f"  $ {cmd[:80]}...")
    proc = subprocess.Popen(
        ssh, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [pod] {line}", end='')
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  TIMEOUT after {timeout}s")
    return proc


def scp_download_dir(host, port, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    cmd = (
        f'scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-i "{SSH_KEY}" -P {port} root@{host}:{remote_dir} "{local_dir}"'
    )
    return run_cmd(cmd, timeout=300, check=False)


def wait_for_ssh(host, port, max_wait=300):
    print(f"\n  Waiting for SSH on {host}:{port}...")
    start = time.time()
    while time.time() - start < max_wait:
        r = ssh_cmd(host, port, "echo OK", timeout=15)
        if r and r.returncode == 0 and "OK" in (r.stdout or ""):
            print(f"  SSH ready ({time.time()-start:.0f}s)")
            return True
        time.sleep(10)
    print(f"  SSH not ready after {max_wait}s")
    return False


# ============================================================
# Main
# ============================================================

def main():
    pod_id = None
    start_time = time.time()

    print("=" * 70)
    print("  SOMI: ABSORB ALL MODELS ON RUNPOD")
    print(f"  GPU preference: {GPU_TYPES[0]}")
    print(f"  Volume: {VOLUME_GB}GB, Disk: {DISK_GB}GB")
    print("=" * 70)
    print()

    runpod.api_key = RUNPOD_API_KEY

    # ---- Create or reuse pod ----
    print("  Checking for existing pod...")
    pods = runpod.get_pods()
    existing = [p for p in pods if p.get("name") == POD_NAME]

    if existing:
        pod = existing[0]
        pod_id = pod["id"]
        status = pod.get("desiredStatus", "unknown")
        print(f"  Found existing pod: {pod_id} (status: {status})")
        if status != "RUNNING":
            print(f"  Starting pod...")
            runpod.resume_pod(pod_id)
            time.sleep(15)
    else:
        print(f"  Creating new pod '{POD_NAME}'...")
        pod = None
        for gpu_type in GPU_TYPES:
            print(f"    Trying GPU: {gpu_type}...")
            try:
                pod = runpod.create_pod(
                    name=POD_NAME,
                    image_name=IMAGE,
                    gpu_type_id=gpu_type,
                    gpu_count=1,
                    volume_in_gb=VOLUME_GB,
                    container_disk_in_gb=DISK_GB,
                    ports="22/tcp",
                    cloud_type="COMMUNITY",
                )
                print(f"    Got {gpu_type}!")
                break
            except Exception as e:
                print(f"    {gpu_type} unavailable: {e}")
                continue

        if pod is None:
            print("  ERROR: No GPUs available. Try again later.")
            sys.exit(1)

        pod_id = pod["id"]
        print(f"  Created pod: {pod_id}")
        time.sleep(30)

    # ---- Wait for pod ----
    print("\n  Waiting for pod to start...")
    runtime = None
    for attempt in range(20):
        info = runpod.get_pod(pod_id)
        runtime = info.get("runtime")
        if runtime and runtime.get("ports"):
            break
        print(f"  Pod starting... ({attempt+1}/20)")
        time.sleep(10)
    else:
        print("  ERROR: Pod did not start in time.")
        print(f"  Check: https://www.runpod.io/console/pods/{pod_id}")
        sys.exit(1)

    # ---- Get SSH info ----
    ports = runtime.get("ports", [])
    ssh_host = ssh_port = None
    for p in ports:
        if p.get("privatePort") == 22:
            ssh_host = p.get("ip")
            ssh_port = p.get("publicPort")
            break

    if not ssh_host or not ssh_port:
        print(f"  ERROR: No SSH port found. Ports: {ports}")
        sys.exit(1)

    print(f"  SSH: {ssh_host}:{ssh_port}")

    if not wait_for_ssh(ssh_host, ssh_port):
        print("  ERROR: SSH not available.")
        sys.exit(1)

    # ---- Check GPU ----
    r = ssh_cmd(ssh_host, ssh_port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if r and r.returncode == 0:
        print(f"  GPU: {r.stdout.strip()}")

    # ---- Clone repo ----
    print("\n  Cloning somi repo...")
    ssh_cmd(ssh_host, ssh_port,
            f"rm -rf /workspace/somi && git clone {GITHUB_REPO} /workspace/somi",
            timeout=120)

    # ---- Install deps ----
    print("\n  Installing dependencies...")
    ssh_cmd(ssh_host, ssh_port,
            "cd /workspace/somi && pip install -e . && pip install transformers wandb huggingface_hub",
            timeout=300)

    # ---- Set env vars ----
    print("\n  Setting environment variables...")
    env_exports = (
        f"export HF_HOME=/workspace/.cache/huggingface "
        f"&& export WANDB_API_KEY={WANDB_KEY} "
        f"&& export WANDB_SILENT=true "
        f"&& export PYTHONUNBUFFERED=1"
    )
    if HF_TOKEN:
        env_exports += f" && export HF_TOKEN={HF_TOKEN}"

    # ---- Run absorption ----
    print("\n" + "=" * 70)
    print("  STARTING ABSORPTION")
    print("=" * 70)

    run_command = (
        f"{env_exports} && cd /workspace/somi "
        f"&& python -m experiments.absorb_all_models 2>&1 | tee /workspace/absorb_log.txt"
    )

    proc = ssh_cmd_stream(ssh_host, ssh_port, run_command, timeout=3600)

    if proc.returncode != 0:
        print(f"\n  WARNING: Script exited with code {proc.returncode}")

    # ---- Download results ----
    print("\n  Downloading results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scp_download_dir(ssh_host, ssh_port, "/workspace/somi_final/", RESULTS_DIR)
    scp_download_dir(ssh_host, ssh_port, "/workspace/somi_checkpoints/", RESULTS_DIR)

    # Download log
    log_cmd = (
        f'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-i "{SSH_KEY}" -P {ssh_port} root@{ssh_host}:/workspace/absorb_log.txt '
        f'"{os.path.join(RESULTS_DIR, "absorb_log.txt")}"'
    )
    run_cmd(log_cmd, timeout=60, check=False)

    results_cmd = (
        f'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-i "{SSH_KEY}" -P {ssh_port} root@{ssh_host}:/workspace/absorb_results.json '
        f'"{os.path.join(RESULTS_DIR, "absorb_results.json")}"'
    )
    run_cmd(results_cmd, timeout=60, check=False)

    total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  DONE in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Pod is still running: https://www.runpod.io/console/pods/{pod_id}")
    print(f"  STOP the pod when done to save money!")
    print("=" * 70)


if __name__ == '__main__':
    main()
