# Vanilla Checkpoints

Expected layout after a successful pull from HuggingFace:

```
checkpoints/codi/
├── config.json
├── generation_config.json
├── model.safetensors                    # or pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
├── added_tokens.json
├── prj.pt                               # CODI-only projection module
└── (vocab.json, merges.txt — BPE tokenizer files)
```

```
checkpoints/coconut/
├── config.json
├── generation_config.json
├── model.safetensors
├── tokenizer_config.json
├── tokenizer.json
├── special_tokens_map.json
├── added_tokens.json
└── (vocab.json, merges.txt)
```

## Verifying the download

```bash
python -c "
from analysis import ModelWrapper
for method in ['codi', 'coconut']:
    w = ModelWrapper(method=method, checkpoint=f'checkpoints/{method}', device='cpu')
    print(f'{method:8s} loaded ok')
"
```

## Source

Both checkpoints are the **community-released GPT-2-small variants** from
the original COCONUT and CODI papers, re-uploaded under
`ModalityDance/latent-tts-*`. Per the paper §4.2 they are taken as-is
and used at inference time only — we do **not** finetune them.

| HF repo | Backbone | Trained on | Citation |
|---|---|---|---|
| `ModalityDance/latent-tts-codi`    | GPT-2-small (124 M, vocab 50 260) | GSM8K (Cobbe et al., 2021) | Shen et al., 2025 |
| `ModalityDance/latent-tts-coconut` | GPT-2-small (124 M, vocab 50 260) | GSM8K | Hao et al., 2024 |
