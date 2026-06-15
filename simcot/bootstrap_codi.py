"""
Bootstrap the SIM-CoT GPT-2 CODI HuggingFace release into a checkpoint
directory that runner.py can load through the existing CODIGPT2 +
LatentGenerationMixin path.

Background
----------
The InternLM HuggingFace release at internlm/SIM_COT-GPT2-CODI ships a
PEFT-LoRA-wrapped safetensors plus an empty config.json and a tokenizer
that registers only [PAD] (no latent tokens). The state-dict carries
three top-level prefixes:

  codi.base_model.model.*   PEFT-wrapped GPT-2 with LoRA adapters on
                            attention and MLP projections (vocab 50260)
  prj.{1,3,ln}.*            CODI projector module
                            (Linear -> GELU -> Linear -> LayerNorm)
  decoder.*                 SIM-CoT auxiliary supervised decoder
                            (training-only, removed at inference)

Plan
----
1. Download the safetensors release.
2. Reconstitute a PEFT-wrapped GPT-2 by initialising stock GPT-2,
   resizing the embedding to 50260 to match the trained extra rows,
   wrapping it in LoRA with the published recipe (r=128, alpha=32,
   target_modules=[c_attn, c_proj, c_fc]), and loading the
   codi.base_model.model.* tensors into the wrapper.
3. Merge the LoRA adapter into the base weights so downstream code does
   not need PEFT to load the checkpoint.
4. Build a CODIGPT2 with the merged base weights and a projector module
   whose layers map cleanly from SIM-CoT's named keys to ours
   (prj.1 -> projector.1, prj.3 -> projector.3, prj.ln -> projector.4).
5. Register three latent special tokens at IDs 50257-50259 to match
   the trained extra embedding rows; drop the auxiliary decoder.
6. Save as a transformers checkpoint plus a tokenizer that downstream
   code can load through analysis/wrappers.py without modification.

Usage
-----
    python scripts/finetune/bootstrap_simcot_codi.py \\
        --out checkpoints/simcot_codi_gpt2 \\
        --base-model openai-community/gpt2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safetensors_load
from transformers import AutoTokenizer, GPT2LMHeadModel

from src.models.codi import CODIGPT2, CODIGPT2Config

_HF_REPO = "internlm/SIM_COT-GPT2-CODI"
_HF_FILE = "model-00001-of-00001.safetensors"

# Latent token convention shared with the rest of the codebase. SIM-CoT
# trained 3 extra rows beyond stock GPT-2 (vocab 50260 = 50257 + 3); we
# register the canonical names at 50257-50259 so the trained embeddings
# line up with our LatentGenerationMixin's resolution.
_LATENT_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>"]

# LoRA recipe from the SIM-CoT paper / GitHub README for the GPT-2 CODI run.
_LORA_RECIPE = dict(
    r=128,
    lora_alpha=32,
    target_modules=["c_attn", "c_proj", "c_fc"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)


def _select_keys(sd: dict, prefix: str) -> dict:
    """Return a new dict keeping only keys that start with `prefix.`,
    with the prefix stripped."""
    pdot = prefix + "."
    return {k[len(pdot):]: v for k, v in sd.items() if k.startswith(pdot)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output checkpoint directory, e.g. checkpoints/simcot_codi_gpt2",
    )
    parser.add_argument(
        "--base-model", default="openai-community/gpt2",
        help="HuggingFace base GPT-2 to load before applying SIM-CoT's LoRA.",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"),
        default="float32",
        help="Save dtype. Default fp32 keeps full precision; bf16 halves disk size.",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # ─── 1. Download release weights ────────────────────────────────────────
    print(f"[1/7] Downloading {_HF_FILE} from {_HF_REPO} ...")
    sf_path = hf_hub_download(_HF_REPO, _HF_FILE)
    full_sd = safetensors_load(sf_path)
    print(f"      loaded safetensors with {len(full_sd)} tensor keys")

    # ─── 2. Tokenizer + latent IDs ──────────────────────────────────────────
    print(f"[2/7] Cloning tokenizer from {args.base_model} and registering latent tokens ...")
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    missing = [t for t in _LATENT_TOKENS if tok.convert_tokens_to_ids(t) == tok.unk_token_id]
    if missing:
        tok.add_tokens(missing, special_tokens=True)
        print(f"      registered {len(missing)} latent tokens: {missing}")

    # Confirm vocab_size in the released weights
    wte = full_sd.get("codi.base_model.model.transformer.wte.weight")
    if wte is None:
        raise RuntimeError("Release missing codi.base_model.model.transformer.wte.weight")
    vocab_size = int(wte.shape[0])
    hidden_size = int(wte.shape[1])
    print(f"      detected vocab_size={vocab_size}, hidden_size={hidden_size}")
    if len(tok) != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab ({len(tok)}) != release vocab ({vocab_size}). "
            f"SIM-CoT shipped {vocab_size - 50257} extra rows; we registered "
            f"{len(tok) - 50257}. Adjust _LATENT_TOKENS if the release changes."
        )
    latent_id = int(tok.convert_tokens_to_ids("<|latent|>"))
    start_id = int(tok.convert_tokens_to_ids("<|start-latent|>"))
    end_id = int(tok.convert_tokens_to_ids("<|end-latent|>"))
    print(f"      latent ids: latent={latent_id} start={start_id} end={end_id}")

    # ─── 3. Reconstitute PEFT-wrapped GPT-2 + load codi.* weights ───────────
    print(f"[3/7] Building stock GPT-2 + LoRA wrapper to receive PEFT weights ...")
    base = GPT2LMHeadModel.from_pretrained(args.base_model)
    if base.get_input_embeddings().weight.shape[0] != vocab_size:
        base.resize_token_embeddings(vocab_size)

    from peft import LoraConfig, get_peft_model, TaskType
    peft_cfg = LoraConfig(
        r=_LORA_RECIPE["r"],
        lora_alpha=_LORA_RECIPE["lora_alpha"],
        target_modules=list(_LORA_RECIPE["target_modules"]),
        lora_dropout=_LORA_RECIPE["lora_dropout"],
        bias=_LORA_RECIPE["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base, peft_cfg)

    # Extract just the codi.* tensors, mapping codi -> the PEFT model
    codi_sd = _select_keys(full_sd, "codi")
    print(f"      filtering to codi.*: {len(codi_sd)} keys")
    # Cast to fp32 for safe loading; we will restore dtype after merge.
    codi_sd_f32 = {k: v.to(torch.float32) for k, v in codi_sd.items()}
    miss, unexp = peft_model.load_state_dict(codi_sd_f32, strict=False)
    # Tied lm_head shows up as missing; that is benign.
    benign_missing = [k for k in miss if k.endswith("lm_head.weight")]
    real_missing = [k for k in miss if k not in benign_missing]
    print(f"      PEFT load: {len(real_missing)} real missing, {len(unexp)} unexpected")
    if real_missing[:5]:
        print(f"      sample missing: {real_missing[:5]}")
    if unexp[:5]:
        print(f"      sample unexpected: {unexp[:5]}")

    # ─── 4. Merge LoRA into base weights ────────────────────────────────────
    print(f"[4/7] Merging LoRA adapter into base GPT-2 weights ...")
    merged = peft_model.merge_and_unload()
    merged.eval()
    print(f"      merged. trainable params after merge: "
          f"{sum(p.numel() for p in merged.parameters() if p.requires_grad):,}")

    # ─── 5. Build CODIGPT2 + copy merged base + projector ───────────────────
    print(f"[5/7] Building CODIGPT2 with projector enabled ...")
    cfg = CODIGPT2Config.from_pretrained(args.base_model)
    cfg.vocab_size = vocab_size
    cfg.latent_id = latent_id
    cfg.latent_start_id = start_id
    cfg.latent_end_id = end_id
    cfg.pad_token_id = int(tok.pad_token_id)
    cfg.projector = True
    cfg.projector_hidden_size = hidden_size
    cfg.projector_dropout = 0.0

    codi = CODIGPT2(cfg)
    if codi.get_input_embeddings().weight.shape[0] != vocab_size:
        codi.resize_token_embeddings(vocab_size)

    # Copy merged GPT-2 base into CODIGPT2 (both are GPT2LMHeadModel
    # subclasses with identical transformer + lm_head structure).
    merged_sd = merged.state_dict()
    miss, unexp = codi.load_state_dict(merged_sd, strict=False)
    real_missing = [k for k in miss if not k.startswith("projector.") and not k.endswith("lm_head.weight")]
    print(f"      copied merged base -> CODIGPT2 "
          f"({len(real_missing)} real missing, {len(unexp)} unexpected)")

    # ─── 6. Wire SIM-CoT projector into our projector module ────────────────
    # Mapping (SIM-CoT name -> our positional name):
    #   prj.1.{weight,bias}  -> projector.1.{weight,bias}    (first Linear)
    #   prj.3.{weight,bias}  -> projector.3.{weight,bias}    (second Linear)
    #   prj.ln.{weight,bias} -> projector.4.{weight,bias}    (LayerNorm)
    print(f"[6/7] Wiring SIM-CoT prj.* into our projector module ...")
    rename = {
        "prj.1.weight":  "projector.1.weight",
        "prj.1.bias":    "projector.1.bias",
        "prj.3.weight":  "projector.3.weight",
        "prj.3.bias":    "projector.3.bias",
        "prj.ln.weight": "projector.4.weight",
        "prj.ln.bias":   "projector.4.bias",
    }
    prj_sd = {}
    for src, dst in rename.items():
        if src not in full_sd:
            raise RuntimeError(
                f"Expected projector tensor {src} not present in release."
            )
        prj_sd[dst] = full_sd[src].to(torch.float32)
    miss, unexp = codi.load_state_dict(prj_sd, strict=False)
    # All non-projector keys will show up as missing here — that is by
    # design (we only loaded the projector). Filter to true unexpected.
    if unexp:
        print(f"      WARN: projector load reported unexpected keys: {unexp}")
    print(f"      projector wired ({len(prj_sd)} tensors).")

    # Cast to save dtype
    codi.to(save_dtype)
    codi.eval()

    # ─── 7. Save checkpoint + patch config.json ─────────────────────────────
    print(f"[7/7] Saving to {out_dir} ...")
    codi.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    config_path = out_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        conf = json.load(f)
    conf["latent_id"] = latent_id
    conf["latent_start_id"] = start_id
    conf["latent_end_id"] = end_id
    conf["projector"] = True
    conf["projector_hidden_size"] = hidden_size
    conf["projector_dropout"] = 0.0
    conf["pad_token_id"] = int(tok.pad_token_id)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2)
    print(f"      patched {config_path.name}")

    # Round-trip sanity check
    rtok = AutoTokenizer.from_pretrained(str(out_dir), use_fast=True)
    for name, expect in (("<|latent|>", latent_id), ("<|start-latent|>", start_id), ("<|end-latent|>", end_id)):
        got = rtok.convert_tokens_to_ids(name)
        assert got == expect, f"reload: {name} -> {got}, expected {expect}"
    print(f"      round-trip OK — checkpoint at {out_dir} is ready for inference.")


if __name__ == "__main__":
    main()
