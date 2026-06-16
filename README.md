# picotron
In the spirit of [NanoGPT](https://github.com/karpathy/nanoGPT), we created Picotron: The minimalist & most-hackable repository for pre-training Llama-like models with [4D Parallelism](https://arxiv.org/abs/2407.21783) (Data, Tensor, Pipeline, Context parallel). It is designed with simplicity and **educational** purposes in mind, making it an excellent tool for learning and experimentation.

![](assets/banière.png)
- The code itself is simple and readable: `train.py`, `model.py` and `[data|tensor|pipeline|context]_parallel.py` are all under **300** lines of code.

- Performance is not the best but still under active development. We observed 38% MFU on a LLaMA-2-7B model using 64 H100 GPUs and nearly 50% MFU on the SmolLM-1.7B model with 8 H100 GPUs. Benchmarks will come soon
- Compared to [Nanotron](https://github.com/huggingface/nanotron/tree/main), Picotron is primarily for educational purposes, helping people quickly get familiar with all the techniques in distributed training

# Extensions on top of the HuggingFace version

This repo is forked from the official HuggingFace picotron (baseline = upstream's last commit `59714b1`) and extends its 4D parallelism (DP/TP/PP/CP) as follows. Every parallelism module ships a from-scratch teaching `README`, companion slides (`teaching_slides.md` plus rendered `html/pdf/pptx`, see each module's `render_slides.sh`), SVG figures, and gradient-validated tests/benchmarks that run on CPU/gloo without a GPU cluster.

## 1. A 5th parallelism axis: Expert Parallelism (MoE)

New module `picotron/expert_parallel/`, extending the grid from 4D to `DP × PP × CP × EP × TP`. Non-expert weights are replicated and sync grads over the `cp_dp` group; expert weights are sharded across EP and not all-reduced.

- **`MoELayer`** (`expert_parallel.py`): replicated router picks top-k experts, two all-to-alls (dispatch / combine) + local expert compute + router-probability weighting.
- **Compute/comm overlap**:
  - tiled MoE (`tiled_moe.py`, MegaScale-style token-tiled pipeline, overlaps both forward and backward, enabled by `ep_num_tiles>1`).
  - shared-expert overlap (DeepSeek-style shared expert on a side stream, overlapping dispatch/combine, `ep_overlap=true`).
- **Comm-volume reduction**: FP8 (E4M3) dispatch (`ep_fp8_dispatch`) and LatentMoE (`moe_latent_dim`, route/compute in a low-dim latent space).
- **Optional DeepEP backend** (`deepep_backend.py`, `ep_backend="deepep"`): Hopper SM90+ CUDA dispatch/combine kernels, auto-fallback to naive when unavailable.
- **Training at scale**: fixed-capacity / static-memory dispatch (`capacity_moe.py`, capacity-capped + dropless multi-round) and router load balancing (aux loss + DeepSeek-V3 loss-free bias).
- **megakernel** (`megakernel.py`): Triton fused expert FFN (forward) + explicit `dgrad`/`wgrad` grouped GEMM (backward).
- Tests/benchmarks: `tests/test_expert_parallel.py`, `test_capacity_moe.py`, `test_load_balance.py`, `test_fp8_dispatch.py`, `test_deepep_backend.py`, `test_megakernel.py`, `bench_ep_*`, `bench_capacity_moe.*`, `ablation_moe.py`.

## 2. New Pipeline Parallelism (PP) schedules

`picotron/pipeline_parallel/pp_schedules.py` adds two modern schedules on top of the existing AFAB / 1F1B to cut the pipeline bubble:

- **Zero-Bubble** (`train_step_pipeline_zb`).
- **Interleaved 1F1B** (`train_step_pipeline_interleaved`).
- Tests/benchmarks: `tests/test_pipeline_parallel.py`, `tests/bench_pp_schedules.py`.

## 3. Three bit-exact Tensor Parallelism (TP) enhancements

`picotron/tensor_parallel/`, layered on top of the Megatron column/row split:

- **Async communication overlap**: a linear that overlaps comm with matmul.
- **Sequence Parallelism**: new `picotron/sequence_parallel/`, reusing the TP process group, with SP communication primitives + fused all-gather linear.
- **Vocab-parallel cross-entropy**: `vocab_parallel_cross_entropy`, avoiding a gather of the full logits.
- Tests/benchmarks: `tests/test_tp_sequence_parallel.py`, `tests/test_tp_vocab_ce.py`, `tests/bench_tp.py`.

## 4. New Context Parallelism (CP) algorithm: Ulysses

`picotron/context_parallel/` adds a second bit-exact algorithm alongside the existing ring attention:

- **DeepSpeed-Ulysses** (`ulysses.py`): two all-to-alls switch to a head-parallel layout and run one dense attention over the full sequence. Shares the same `cp_group` as ring; toggle via `CP_ATTENTION=ring|ulysses`.
- **Ring flash attention** (`RingFlashAttentionFunc`, `FLASH_ATTEN=1`): uses flash's low-level fwd/bwd with online-softmax merge; falls back to a readable pure-Python reference when `FLASH_ATTEN=0`.
- Tests/benchmarks: `tests/test_cp_ulysses.py`, `tests/test_cp_ring_flash.py`, `tests/bench_cp.py`.

## 5. Engineering & project infrastructure

- **Dependency management migrated to [uv](https://docs.astral.sh/uv/)**: `pyproject.toml` + `uv.lock` replace `requirements.txt` / `setup.py`.
- **Adopted [Ruff](https://docs.astral.sh/ruff/)** for linting / import sorting / formatting (incl. VS Code config).
- **Pervasive type annotations**: introduced `jaxtyping` and refactored `create_config.py`, `extract_metrics.py`, `train.py`, `checkpoint.py`, `bucket.py`, etc. for readability and type safety.
- **`modal_run.py`**: one-click training / benchmarking on [Modal](https://modal.com/).
- Added MoE config knobs (`create_config.py` / `template/base_config.json`) wired through to `train.py` and `model.py`.

> See each module's `picotron/<module>/README.md` for the deeper explanations.

# Tutorial videos

- A step by step tutorial on how to build Picotron distributed training framework form scratch:
    - [Picotron tutorial (playlist)](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S) 🎬
    - [Picotron tutorial (codebase)](https://github.com/huggingface/picotron_tutorial) 👷🏻‍♂️

# Install

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Requires Python 3.10 or 3.11 (`torch==2.1.0` has no wheels for 3.12+).

```sh
uv sync
```

To use the environment, either prefix commands with `uv run` (e.g. `uv run torchrun ...`)
or activate the virtualenv with `source .venv/bin/activate`.

# Lint & format

Uses [Ruff](https://docs.astral.sh/ruff/) for linting, import sorting, and formatting.
With the [Ruff VS Code extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff),
`.vscode/settings.json` enables format-on-save and auto-fix.

```sh
uv run ruff check .          # lint + type-import rules
uv run ruff format .         # auto-format
uv run ruff check . && uv run ruff format --check .  # CI: no changes allowed
```

# Quick start
- Get a HF token [here](https://huggingface.co/settings/tokens) to download models from HuggingFace

- GPU
    ```sh
    # To create a config file in json format under tmp by default
    python create_config.py --out_dir tmp --exp_name llama-1B --dp 8 --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 15  --grad_acc_steps 32 --mbs 4 --seq_len 1024 --hf_token <HF_TOKEN>

    # Locally
    torchrun --nproc_per_node 8 train.py --config tmp/llama-1B/config.json 

    # 3D Parallelism
    python create_config.py --out_dir tmp --dp 4 --tp 2 --pp 2 --pp_engine 1f1b --exp_name llama-7B --model_name meta-llama/Llama-2-7b-hf  --grad_acc_steps 32 --mbs 4 --seq_len 1024 --hf_token <HF_TOKEN>

    # Slurm
    python submit_slurm_jobs.py --inp_dir tmp/llama-7B --qos high --hf_token <HF_TOKEN>
    ```

-  CPU (expect it to be slow)
    ```sh
    # 3D Parallelism on CPU
    python create_config.py --out_dir tmp --exp_name llama-1B-cpu --dp 2 --tp 2 --pp 2 --pp_engine 1f1b --model_name HuggingFaceTB/SmolLM-1.7B --num_hidden_layers 5  --grad_acc_steps 2 --mbs 4 --seq_len 128 --hf_token <HF_TOKEN> --use_cpu

    # Locally
    torchrun --nproc_per_node 8 train.py --config tmp/llama-1B-cpu/config.json
    ```

# Citation
If you use Picotron, please cite it as:

```bibtex
@misc{zhao2025picotron,
  author = {Haojun Zhao and Ferdinand Mom},
  title = {Picotron: Distributed training framework for education and research experimentation},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/picotron}}
}
```

# Acknowledgements

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [FairScale](https://github.com/facebookresearch/fairscale)
- [LitGPT](https://github.com/Lightning-AI/lit-gpt)
