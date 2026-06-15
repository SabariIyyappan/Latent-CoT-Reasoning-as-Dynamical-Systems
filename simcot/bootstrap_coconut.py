"""
Bootstrap the SIM-CoT GPT-2 Coconut HuggingFace release into a checkpoint
directory that runner.py can load through the existing COCONUTGPT2 +
LatentGenerationMixin path.

Background
----------
The InternLM HuggingFace release at internlm/SIM_COT-GPT2-Coconut ships a
single torch.save state-dict (``checkpoint_28``) plus an empty config.json,
no tokenizer files, and no transformers metadata. The state-dict carries
three top-level prefixes:

  base_causallm.*   149 keys, the actual generation model (vocab 50260)
  expainable_llm.*  149 keys, the SIM-CoT auxiliary supervised decoder
                    (vocab 50257; training-only, removed at inference)
  embedding.*       1 key, byte-identical duplicate of
                    base_causallm.transformer.wte.weight

This script keeps base_causallm.* (renamed to *), drops the auxiliary
decoder and the redundant embedding tensor, and writes the result as a
transformers-compatible checkpoint that pairs with a tokenizer carrying
three latent special tokens at IDs 50257-50259 to match the three extra
embedding rows. Without that registration, our LatentGenerationMixin
cannot resolve <|latent|> at generation time.

The conversation around the release format and weight naming sits in the
exp-simcot-gsm8k branch's design notes; the upstream code lives at
https://github.com/InternLM/SIM-CoT (paper: arXiv 2509.20317).

Usage
-----
    python scripts/finetune/bootstrap_simcot_coconut.py \\
        --out checkpoints/simcot_coconut_gpt2 \\
        --base-tokenizer openai-community/gpt2

The base-tokenizer flag picks the tokenizer to clone before adding the
three latent tokens. SIM-CoT trained on top of stock GPT-2's BPE so the
default points there.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.models.coconut import COCONUTGPT2, COCONUTGPT2Config

# Canonical SIM-CoT-Coconut release on HuggingFace
_HF_REPO = "internlm/SIM_COT-GPT2-Coconut"
_HF_FILE = "checkpoint_28"

# Latent token convention shared with the rest of the codebase.
# SIM-CoT trained 3 extra embedding rows beyond stock GPT-2's 50257; we
# register our names at the matching 50257-50259 slots so the trained
# weights line up with our LatentGenerationMixin's token resolution.
_LATENT_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    """Return a new dict with `prefix.` removed from every matching key."""
    out = {}
    pdot = prefix + "."
    for k, v in state_dict.items():
        if k.startswith(pdot):
            out[k[len(pdot):]] = v
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output checkpoint directory, e.g. checkpoints/simcot_coconut_gpt2",
    )
    parser.add_argument(
        "--base-tokenizer", default="openai-community/gpt2",
        help="HuggingFace id of the base tokenizer to clone before adding latent tokens.",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Save dtype. Default fp32 matches the release. Use bfloat16 to halve disk size.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # ─── 1. Download release weights ────────────────────────────────────────
    print(f"[1/5] Downloading {_HF_FILE} from {_HF_REPO} ...")
    ckpt_path = hf_hub_download(_HF_REPO, _HF_FILE)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"      loaded state dict with {len(sd)} keys")

    # ─── 2. Strip auxiliary decoder + redundant embedding ───────────────────
    base_sd = _strip_prefix(sd, "base_causallm")
    print(f"[2/5] Kept base_causallm.*: {len(base_sd)} keys "
          f"(dropped expainable_llm.* and embedding.* duplicate)")

    # Sanity: confirm vocab from the actual weights
    wte_shape = base_sd.get("transformer.wte.weight").shape
    vocab_size = int(wte_shape[0])
    hidden_size = int(wte_shape[1])
    print(f"      detected vocab_size={vocab_size}, hidden_size={hidden_size}")

    # ─── 3. Tokenizer + latent IDs ──────────────────────────────────────────
    print(f"[3/5] Cloning tokenizer from {args.base_tokenizer} and registering latent tokens ...")
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if tok.pad_token is None:
        # Match our existing GPT-2 CODI/COCONUT convention: pad token aliased
        # to eos so the collator works out of the box.
        tok.pad_token = tok.eos_token
    missing = [t for t in _LATENT_TOKENS if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        tok.add_tokens(missing, special_tokens=True)
        print(f"      registered {len(missing)} new tokens: {missing}")
    if len(tok) != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab size after registration ({len(tok)}) differs from "
            f"checkpoint embedding rows ({vocab_size}). "
            f"SIM-CoT trained {vocab_size - 50257} extra rows beyond stock GPT-2; "
            f"only {len(tok) - 50257} new tokens added here."
        )
    latent_id = int(tok.convert_tokens_to_ids("<|latent|>"))
    start_id = int(tok.convert_tokens_to_ids("<|start-latent|>"))
    end_id = int(tok.convert_tokens_to_ids("<|end-latent|>"))
    print(f"      latent ids: latent={latent_id} start={start_id} end={end_id}")

    # ─── 4. Build a COCONUTGPT2 config with the right latent metadata ───────
    print(f"[4/5] Building COCONUTGPT2 + loading state dict ...")
    cfg = COCONUTGPT2Config.from_pretrained(args.base_tokenizer)
    cfg.vocab_size = vocab_size
    cfg.latent_id = latent_id
    cfg.latent_start_id = start_id
    cfg.latent_end_id = end_id
    cfg.pad_token_id = int(tok.pad_token_id)

    model = COCONUTGPT2(cfg)
    # The model was constructed with vocab=50257 (from GPT-2 base config) by
    # default; resize embeddings to 50260 BEFORE loading so wte/lm_head
    # shapes match the loaded weights.
    if model.get_input_embeddings().weight.shape[0] != vocab_size:
        model.resize_token_embeddings(vocab_size)

    # Cast loaded weights to the requested dtype
    base_sd_cast = {k: v.to(save_dtype) for k, v in base_sd.items()}
    missing_keys, unexpected = model.load_state_dict(base_sd_cast, strict=False)
    print(f"      load: {len(missing_keys)} missing, {len(unexpected)} unexpected")
    if unexpected:
        # GPT2LMHeadModel ties wte = lm_head and discards the duplicate
        # automatically; only flag truly unexpected keys.
        true_unexp = [k for k in unexpected if not k.endswith("lm_head.weight")]
        if true_unexp:
            print(f"      unexpected (truly): {true_unexp[:10]}")
    if missing_keys:
        # GPT-2 keeps lm_head.weight tied to wte; missing lm_head is benign.
        true_miss = [k for k in missing_keys if not k.endswith("lm_head.weight")]
        if true_miss:
            print(f"      missing (truly): {true_miss[:10]}")
    model.to(save_dtype)
    model.eval()

    # ─── 5. Save checkpoint ─────────────────────────────────────────────────
    print(f"[5/5] Saving to {out_dir} ...")
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    # The from_pretrained may not persist our extra config keys if the saved
    # config rewrite drops them. Patch config.json to be safe.
    config_path = out_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        conf = json.load(f)
    conf["latent_id"] = latent_id
    conf["latent_start_id"] = start_id
    conf["latent_end_id"] = end_id
    conf["pad_token_id"] = int(tok.pad_token_id)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)
    print(f"      patched {config_path.name}")

    # ── Round-trip sanity check ────────────────────────────────────────────
    rtok = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    for name, expect in (("<|latent|>", latent_id), ("<|start-latent|>", start_id), ("<|end-latent|>", end_id)):
        got = rtok.convert_tokens_to_ids(name)
        assert got == expect, f"reload: {name} -> {got}, expected {expect}"
    print(f"      round-trip OK — checkpoint at {out_dir} is ready for inference.")


if __name__ == "__main__":
    main()
