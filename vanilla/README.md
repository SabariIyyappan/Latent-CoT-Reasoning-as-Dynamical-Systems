# Vanilla CoT Paradigm

This folder contains everything paradigm-specific to **Vanilla CoT** CODI
and COCONUT runs on GSM8K. Both methods use checkpoints released by the
original authors on HuggingFace; no bootstrap step is required.

## Checkpoints

| Method | HuggingFace repo | Paper citation |
|---|---|---|
| CODI    | [`ModalityDance/latent-tts-codi`](https://huggingface.co/ModalityDance/latent-tts-codi) | Shen et al., 2025 (arXiv 2502.21074) |
| COCONUT | [`ModalityDance/latent-tts-coconut`](https://huggingface.co/ModalityDance/latent-tts-coconut) | Hao et al., 2024 (arXiv 2412.06769) |

CODI additionally loads a self-distillation projection module (`prj.pt`)
at inference time. The projection is bundled inside the HF release and
is loaded automatically by `analysis.wrappers.ModelWrapper`.

See [`checkpoints.md`](./checkpoints.md) for the exact expected layout
under `checkpoints/`.

## Configs

| File | Used by | Output |
|---|---|---|
| [`configs/inference_codi_gsm8k.yaml`](./configs/inference_codi_gsm8k.yaml)       | `runner.py --paradigm vanilla --method codi`    | `results/vanilla_codi/<run>/` |
| [`configs/inference_coconut_gsm8k.yaml`](./configs/inference_coconut_gsm8k.yaml) | `runner.py --paradigm vanilla --method coconut` | `results/vanilla_coconut/<run>/` |

Both configs target the **full GSM8K split** (N = 8,792 = train + test
combined) with stratified sampling across the 7 concept buckets, matching
§4.1 of the paper.

## How to run

### 1. Download checkpoints

```python
from huggingface_hub import snapshot_download
snapshot_download("ModalityDance/latent-tts-codi",    local_dir="checkpoints/codi")
snapshot_download("ModalityDance/latent-tts-coconut", local_dir="checkpoints/coconut")
```

### 2. Run inference

```bash
python runner.py --paradigm vanilla --method codi
python runner.py --paradigm vanilla --method coconut
```

Each command writes the latent-state HDF5 + every analysis output for
that cell. Approximate runtime on a single RTX 5070 Ti: ~30 minutes
per cell.

### 3. Quick smoke test (5 samples)

```bash
python runner.py --paradigm vanilla --method codi --n_samples 5
```

Finishes in ~1 minute. Useful for verifying the environment.

## Where the analysis flows next

After inference produces
`results/vanilla_<method>/<run>/latent_states/all_states.h5`, the full
analysis pipeline is invoked automatically by `runner.py` (you do not
need to call it separately).

For replay from a cached HDF5 (no GPU needed):

```bash
python scripts/replay/analyze_from_cache.py results/vanilla_codi/<run>/
```

This regenerates every figure that vanilla CoT contributes to the
paper (Figs 2-7, A.2, A.3) without re-running inference.
