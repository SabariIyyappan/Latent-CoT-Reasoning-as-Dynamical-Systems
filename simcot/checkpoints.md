# Sim-CoT Checkpoints

Sim-CoT checkpoints require a **one-time bootstrap step** that downloads
the upstream HuggingFace release and translates it into the layout
`runner.py` expects.

## Expected layout after bootstrap

```
checkpoints/simcot_codi_gpt2/
├── config.json                          # patched: projector=True, latent_id=50259, etc.
├── generation_config.json
├── model.safetensors                    # base_causallm.* tensors only; auxiliary decoder stripped
├── prj.pt                               # CODI projection module (registered separately)
├── tokenizer_config.json
├── tokenizer.json                       # with 50257-50259 latent tokens added
├── special_tokens_map.json
└── added_tokens.json
```

```
checkpoints/simcot_coconut_gpt2/
├── config.json
├── generation_config.json
├── model.safetensors                    # base_causallm.* tensors only
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
└── added_tokens.json
```

(COCONUT has no `prj.pt` — the latent loop lives entirely in
`LatentGenerationMixin`.)

## Running the bootstrap

```bash
# CODI
python simcot/bootstrap_codi.py --out checkpoints/simcot_codi_gpt2/

# COCONUT
python simcot/bootstrap_coconut.py --out checkpoints/simcot_coconut_gpt2/
```

Each bootstrap performs the following operations:

1. **Download** — `huggingface_hub.snapshot_download(<upstream repo>)`
   - CODI: `internlm/SIM_COT-GPT2-CODI`
   - COCONUT: `internlm/SIM_COT-GPT2-Coconut`
2. **Strip auxiliary decoder** — upstream's training-time decoder
   (`expainable_llm.*` keys for COCONUT; LoRA + projector + decoder for
   CODI) is dropped because it is not needed at inference time
3. **Re-emit in HF format** — only `base_causallm.*` tensors (plus
   `prj.pt` for CODI) are kept; written to `--out`
4. **Register latent tokens** — adds `<|start-latent|>`, `<|latent|>`,
   `<|end-latent|>` to the tokenizer at IDs 50257–50259 (matching what
   the vanilla GSM8K checkpoints expect)

## Verifying the bootstrap output

```bash
python -c "
from analysis import ModelWrapper
for method in ['codi', 'coconut']:
    w = ModelWrapper(method=method, checkpoint=f'checkpoints/simcot_{method}_gpt2', device='cpu')
    print(f'simcot_{method:8s} loaded ok')
"
```

## Source

| HF repo | Backbone | Trained by | Citation |
|---|---|---|---|
| `internlm/SIM_COT-GPT2-CODI`    | GPT-2-small (124 M) | InternLM team (Wei et al., 2025) on GSM8K | Wei et al., arXiv 2509.20317 |
| `internlm/SIM_COT-GPT2-Coconut` | GPT-2-small (124 M) | InternLM team on GSM8K | Wei et al., arXiv 2509.20317 |

We **do not train** the Sim-CoT variants. They are released by the
Sim-CoT paper authors and we use them at inference time only, per
paper §4.2.
