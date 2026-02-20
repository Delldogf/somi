"""
Launch Lossless Spectral Absorption on RunPod
===============================================

Creates (or reuses) a GPU pod, clones the repo, and runs the
lossless_absorb_experiment.py script. Downloads results when done
and terminates the pod.
"""

import os
import sys
import time
import subprocess
import runpod

RUNPOD_API_KEY = os.environ.get(
    "RUNPOD_API_KEY",
    "rpa_AK6YBHG3UBFGT7KCR2ZTJ7KPP822ZVUXTK6KQCHX1przg4"
)
WANDB_KEY = "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"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GITHUB_REPO = "https://github.com/Delldogf/somi.git"

POD_NAME = "somi-lossless-absorb"

GPU_TYPES = [
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
    "NVIDIA L40S",
    "NVIDIA RTX A6000",
]

IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_GB = 200
DISK_GB = 50

SSH_KEY = os.environ.get("SSH_KEY", os.path.expanduser("~/.ssh/id_ed25519"))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lossless_results")


def run_cmd(cmd, timeout=300, check=True):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and check:
            print(f"  STDERR: {r.stderr[:500]}")
        return r
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None


def ssh_cmd(host, port, cmd, timeout=600):
    ssh = (
        f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 '
        f'-i "{SSH_KEY}" -p {port} root@{host} "{cmd}"'
    )
    return run_cmd(ssh, timeout=timeout)


def ssh_cmd_stream(host, port, cmd, timeout=7200):
    ssh = (
        f'ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 '
        f'-i "{SSH_KEY}" -p {port} root@{host} "{cmd}"'
    )
    print(f"  Running: {cmd[:80]}...")
    proc = subprocess.Popen(
        ssh, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding='utf-8', errors='replace',
    )
    try:
        for line in iter(proc.stdout.readline, ''):
            print(f"  [pod] {line}", end='')
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  TIMEOUT after {timeout}s")
    return proc


def wait_for_ssh(host, port, max_wait=300):
    print(f"  Waiting for SSH on {host}:{port}...")
    start = time.time()
    while time.time() - start < max_wait:
        r = ssh_cmd(host, port, "echo OK", timeout=15)
        if r and r.returncode == 0 and "OK" in (r.stdout or ""):
            print(f"  SSH ready ({time.time()-start:.0f}s)")
            return True
        time.sleep(10)
    print(f"  SSH not ready after {max_wait}s")
    return False


def main():
    pod_id = None
    start_time = time.time()

    print("=" * 70)
    print("  SOMI: LOSSLESS SPECTRAL ABSORPTION ON RUNPOD")
    print("=" * 70)

    runpod.api_key = RUNPOD_API_KEY

    # Check for existing pod
    pods = runpod.get_pods()
    existing = [p for p in pods if p.get("name") == POD_NAME]

    if existing:
        pod = existing[0]
        pod_id = pod["id"]
        status = pod.get("desiredStatus", "unknown")
        print(f"  Found pod: {pod_id} (status: {status})")
        if status != "RUNNING":
            runpod.resume_pod(pod_id)
            time.sleep(15)
    else:
        print(f"  Creating pod '{POD_NAME}'...")
        pod = None
        for gpu_type in GPU_TYPES:
            print(f"    Trying: {gpu_type}...")
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
                print(f"    Unavailable: {e}")
                continue

        if pod is None:
            print("  No GPUs available.")
            sys.exit(1)

        pod_id = pod["id"]
        print(f"  Created: {pod_id}")
        time.sleep(30)

    # Wait for pod
    runtime = None
    for attempt in range(20):
        info = runpod.get_pod(pod_id)
        runtime = info.get("runtime")
        if runtime and runtime.get("ports"):
            break
        print(f"  Starting... ({attempt+1}/20)")
        time.sleep(10)
    else:
        print("  Pod didn't start.")
        sys.exit(1)

    # Get SSH info
    ssh_host = ssh_port = None
    for p in runtime.get("ports", []):
        if p.get("privatePort") == 22:
            ssh_host = p.get("ip")
            ssh_port = p.get("publicPort")
            break

    if not ssh_host:
        print("  No SSH port found.")
        sys.exit(1)

    print(f"  SSH: {ssh_host}:{ssh_port}")
    if not wait_for_ssh(ssh_host, ssh_port):
        sys.exit(1)

    # GPU info
    r = ssh_cmd(ssh_host, ssh_port, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if r and r.returncode == 0:
        print(f"  GPU: {r.stdout.strip()}")

    # Clone and install
    print("\n  Cloning repo...")
    ssh_cmd(ssh_host, ssh_port,
            f"rm -rf /workspace/somi && git clone {GITHUB_REPO} /workspace/somi",
            timeout=120)

    print("  Installing dependencies...")
    ssh_cmd(ssh_host, ssh_port,
            "cd /workspace/somi && pip install -e . && pip install transformers huggingface_hub accelerate",
            timeout=300)

    # Set env vars and run
    env_exports = (
        f"export HF_HOME=/workspace/.cache/huggingface "
        f"&& export WANDB_API_KEY={WANDB_KEY} "
        f"&& export WANDB_SILENT=true "
        f"&& export PYTHONUNBUFFERED=1"
    )
    if HF_TOKEN:
        env_exports += f" && export HF_TOKEN={HF_TOKEN}"

    print("\n" + "=" * 70)
    print("  STARTING LOSSLESS ABSORPTION")
    print("=" * 70)

    run_command = (
        f"{env_exports} && cd /workspace/somi "
        f"&& python -u experiments/lossless_absorb_experiment.py 2>&1 | tee /workspace/lossless_log.txt"
    )
    proc = ssh_cmd_stream(ssh_host, ssh_port, run_command, timeout=7200)

    # Download results
    print("\n  Downloading results...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for remote_file in [
        "/workspace/lossless_log.txt",
        "/workspace/lossless_absorb_results.json",
    ]:
        local_file = os.path.join(RESULTS_DIR, os.path.basename(remote_file))
        scp = (
            f'scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
            f'-i "{SSH_KEY}" -P {ssh_port} root@{ssh_host}:{remote_file} "{local_file}"'
        )
        run_cmd(scp, timeout=60, check=False)

    # Download checkpoint
    scp_ckpt = (
        f'scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
        f'-i "{SSH_KEY}" -P {ssh_port} root@{ssh_host}:/workspace/somi_lossless/ '
        f'"{os.path.join(RESULTS_DIR, "checkpoint")}"'
    )
    run_cmd(scp_ckpt, timeout=120, check=False)

    # Terminate pod
    print("\n  Terminating pod to save budget...")
    try:
        runpod.terminate_pod(pod_id)
        print(f"  Pod {pod_id} terminated.")
    except Exception as e:
        print(f"  Could not terminate: {e}")
        print(f"  MANUALLY STOP: https://www.runpod.io/console/pods/{pod_id}")

    total = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  DONE in {total:.0f}s ({total/60:.1f} min)")
    print(f"  Results: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
