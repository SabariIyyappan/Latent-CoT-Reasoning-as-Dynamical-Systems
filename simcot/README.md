# Sim-CoT Paradigm

This folder contains everything paradigm-specific to **Sim-CoT** CODI
and COCONUT runs on GSM8K. Sim-CoT models require a **one-time
bootstrap** that translates the upstream HuggingFace release into the
checkpoint layout `runner.py` expects.

## Checkpoints

| Method | HuggingFace repo | Paper citation |
|---|---|---|
| CODI    | [`internlm/SIM_COT-GPT2-CODI`](https://huggingface.co/internlm/SIM_COT-GPT2-CODI)       | Wei et al., 2025 (arXiv 2509.20317) |
| COCONUT | [`internlm/SIM_COT-GPT2-Coconut`](https://huggingface.co/internlm/SIM_COT-GPT2-Coconut) | Wei et al., 2025 |

These releases ship in upstream's training format (CODI as PEFT-LoRA on
a stock GPT-2; COCONUT as a single `checkpoint_28` torch dump). The
bootstrap scripts below translate them into HuggingFace-format
checkpoints that load through `analysis.wrappers.ModelWrapper` via the
same `CODIGPT2` / `COCONUTGPT2` classes vanilla CoT uses.

See [`checkpoints.md`](./checkpoints.md) for what the bootstrap
produces.

## How to run

```bash
python runner.py --paradigm simcot --method codi
python runner.py --paradigm simcot --method coconut
```

That's the whole flow: when `checkpoints/simcot_<method>_gpt2/` is
missing, `runner.py` invokes the bootstrap automatically before
loading the model. Each command then writes the latent-state HDF5 +
every analysis output for that cell (~30 minutes per cell on a single
A100).

### The bootstrap (runs automatically; manual invocation optional)

```bash
python simcot/bootstrap_codi.py    --out checkpoints/simcot_codi_gpt2/
python simcot/bootstrap_coconut.py --out checkpoints/simcot_coconut_gpt2/
```

Each bootstrap:

1. `snapshot_download("internlm/SIM_COT-GPT2-{CODI,Coconut}")`
2. Strips upstream's training-only auxiliary decoder tensors
3. Registers the three latent special tokens (`<|start-latent|>`,
   `<|latent|>`, `<|end-latent|>`) on the tokenizer at IDs 50257–50259
4. Saves a HuggingFace-format checkpoint under `--out`

Approximate runtime: ~2 minutes per bootstrap (download + translation).

### Smoke test

```bash
python runner.py --paradigm simcot --method codi --n_samples 5 --device cpu
```

## Why bootstrap instead of running upstream's inference?

The cam-ready trunk supports one Sim-CoT inference path: bootstrap-translated
checkpoint loaded through our `LatentGenerationMixin`. This is **Path A**
in the development history. Two alternative paths existed:

- **Path B** — re-implemented upstream's inference loop in our codebase
- **Path C** — invoked upstream's `test.py` directly via `importlib` from
  a sibling `../SIM-CoT/` clone

We validated that all three paths produce identical latent states on
GSM8K and standardised on Path A. Path B and Path C scripts are not in
the trunk but remain reachable on the `camera-ready-pre-cleanup` tag for
full provenance.

## Where the analysis flows next

After Sim-CoT inference produces
`results/simcot_<method>/<run>/latent_states/all_states.h5`, the full
analysis pipeline runs automatically. Same downstream entry as vanilla:

```bash
python scripts/replay/analyze_from_cache.py results/simcot_codi/<run>/
```

## Configs

| File | Used by | Output |
|---|---|---|
| [`configs/inference_codi_gsm8k.yaml`](./configs/inference_codi_gsm8k.yaml)       | `runner.py --paradigm simcot --method codi`    | `results/simcot_codi/<run>/` |
| [`configs/inference_coconut_gsm8k.yaml`](./configs/inference_coconut_gsm8k.yaml) | `runner.py --paradigm simcot --method coconut` | `results/simcot_coconut/<run>/` |

Both configs target the **full GSM8K split** (N = 8,792 = train + test
combined) with stratified sampling across the 7 concept buckets.
