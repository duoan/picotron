"""General Modal runner for picotron: run *any* command (tests, training, benchmarks) on cloud GPUs.

The image bundles the full picotron + DeepEP toolchain (CUDA 13, torch cu130, Triton, NCCL >= 2.30.4,
DeepEP built from source), so it works for plain training as well as the Hopper-only DeepEP path.

    pip install modal && modal setup

    # Default: run the EP integration test suite on 2x H100
    modal run modal_run.py

    # Any command (test or train), in /workspace:
    modal run modal_run.py --command "torchrun --nproc_per_node 2 tests/test_expert_parallel.py"
    modal run modal_run.py --command "python create_config.py --out_dir conf --ep 2 && \
                                       torchrun --nproc_per_node 2 train.py --config conf/config.json"

    # Pick GPU / timeout via env vars (the launcher is re-imported per run):
    MODAL_GPU=A100-80GB:2 modal run modal_run.py --command "pytest -q"
    MODAL_GPU=H100:4 MODAL_TIMEOUT=21600 modal run modal_run.py --command "torchrun --nproc_per_node 4 train.py ..."

GPU spec: "H100", "H100:2", "A100-80GB", "L40S", etc. (see modal.com/pricing). H100 = ~$3.95/GPU/hr.
Persistence: anything written under /outputs is saved to the 'picotron-outputs' Modal Volume
(`modal volume get picotron-outputs <path> .` to download) — point train checkpoints there.
"""

import os

import modal

# CUDA *devel* base (has nvcc) so the deep_ep C++/CUDA extension can build. Matches our local toolchain
# (CUDA 13 + torch cu130 + nccl-cu13), which is what DeepEP (NCCL Gin backend) expects.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu22.04", add_python="3.11")
    # libibverbs-dev/rdma-core: DeepEP's legacy internode kernels include <infiniband/mlx5dv.h>.
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libibverbs-dev", "rdma-core")
    .pip_install("torch", "triton", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("numpy", "setuptools", "wheel", "ninja")
    .pip_install("nvidia-nccl-cu13>=2.30.4")  # DeepEP needs NCCL Gin (>= 2.30.4)
    .run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepEP /opt/DeepEP",
        # Force g++ as host compiler: torch otherwise picks up a bogus clang++ (v0.0.0) and aborts.
        "cd /opt/DeepEP && CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='9.0' python setup.py install",
    )
    .env({"PYTHONPATH": "/workspace"})
    .add_local_dir(
        ".",
        remote_path="/workspace",
        ignore=["**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/*.log", "**/*.pyc", "**/outputs/**"],
    )
)

# Runtime-configurable (env vars, since the file is re-imported on every `modal run`).
GPU = os.environ.get("MODAL_GPU", "H100:2")
TIMEOUT = int(os.environ.get("MODAL_TIMEOUT", str(3600 * 2)))

app = modal.App("picotron-modal", image=image)
outputs = modal.Volume.from_name("picotron-outputs", create_if_missing=True)

# Default job: the EP integration suite (Hopper). `;` so every test runs even if one fails.
DEFAULT_COMMAND = (
    'echo "### megakernel" && python tests/test_megakernel.py ; '
    'echo "### fp8 dispatch" && torchrun --nproc_per_node 2 tests/test_fp8_dispatch.py ; '
    'echo "### deepep vs torch" && torchrun --nproc_per_node 2 tests/test_deepep_backend.py ; '
    'echo "### comm-volume reducers" && '
    "torchrun --nproc_per_node 2 tests/bench_ep_compress.py --hidden 4096 --latent 1024 --tokens 8192 ; "
    'echo "### overlap" && '
    "torchrun --nproc_per_node 2 tests/bench_ep_overlap.py --hidden 4096 --tokens 8192 --tiles 2 4"
)


@app.function(gpu=GPU, timeout=TIMEOUT, volumes={"/outputs": outputs})
def run_remote(command: str) -> int:
    """Run an arbitrary shell command in /workspace and return its exit code."""
    import subprocess

    import torch

    print("=" * 70)
    print(f"torch {torch.__version__} | {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)} | requested GPU: {GPU}")
    subprocess.run(["python", "-c", "import deep_ep; print('deep_ep:', deep_ep.__version__)"], check=False)
    print("=" * 70, flush=True)

    rc = subprocess.run(["bash", "-lc", command], cwd="/workspace").returncode
    outputs.commit()  # persist anything written under /outputs
    print(f"\n[modal_run] command exited with code {rc}", flush=True)
    return rc


@app.local_entrypoint()
def main(command: str = DEFAULT_COMMAND):
    code = run_remote.remote(command)
    print(f"Done. Remote exit code: {code}")
