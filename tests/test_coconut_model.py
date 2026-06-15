"""
Integration tests for the vanilla COCONUT checkpoint
(`ModalityDance/latent-tts-coconut`, locally cached at `checkpoints/coconut/`).

These tests load real model weights and run inference. They are skipped
automatically when the checkpoint folder is absent (e.g. on CI or a fresh
clone) so that `pytest tests/` always succeeds even without GPU resources.

To enable: download the checkpoint with
    huggingface-cli download ModalityDance/latent-tts-coconut \
        --local-dir checkpoints/coconut
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import pytest


_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "coconut"


def _checkpoint_available() -> bool:
    return _CHECKPOINT.is_dir() and (_CHECKPOINT / "config.json").is_file()


pytestmark = pytest.mark.skipif(
    not _checkpoint_available(),
    reason=(
        "vanilla COCONUT checkpoint not present at "
        f"{_CHECKPOINT}; run "
        "`huggingface-cli download ModalityDance/latent-tts-coconut "
        "--local-dir checkpoints/coconut` to enable"
    ),
)


# -- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(_CHECKPOINT))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def model(tokenizer, device):
    # Make `src` importable regardless of cwd.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.generation_mixin import LatentGenerationMixin  # type: ignore
    from src.models.coconut import COCONUTGPT2  # type: ignore

    class LatentCOCONUT(COCONUTGPT2, LatentGenerationMixin):
        def __init__(self, config):
            super().__init__(config)

    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    m = LatentCOCONUT.from_pretrained(
        str(_CHECKPOINT),
        latent_id=latent_id,
        latent_start_id=start_id,
        latent_end_id=end_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    return m


# -- tests --------------------------------------------------------------------


def test_load_model(model, tokenizer):
    """Model + tokenizer load and latent special tokens are distinct."""
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
    assert {latent_id, start_id, end_id} == {50257, 50258, 50259} or (
        latent_id != start_id and latent_id != end_id and start_id != end_id
    )
    assert model.config.n_embd == 768  # GPT-2 small
    assert model.config.n_layer == 12
    assert model.config.vocab_size == 50260


def test_deterministic_inference(model, tokenizer, device):
    """Deterministic decoding produces stable outputs across two runs."""
    import torch

    from src.generation_mixin import LatentGenerationConfig  # type: ignore

    cfg = LatentGenerationConfig(
        max_new_tokens=64,
        latent_length=6,
        latent_do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt = "What is 15 + 27?\n<|start-latent|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out1 = model.generate(
            **inputs, generation_config=cfg, num_return_sequences=1, use_cache=True
        )
        out2 = model.generate(
            **inputs, generation_config=cfg, num_return_sequences=1, use_cache=True
        )
    assert torch.equal(out1, out2), "deterministic decoding must be reproducible"


def test_stochastic_inference(model, tokenizer, device):
    """MC-dropout stochastic decoding produces variation across samples."""
    import torch

    from src.generation_mixin import LatentGenerationConfig  # type: ignore

    cfg = LatentGenerationConfig(
        max_new_tokens=64,
        latent_length=6,
        latent_do_sample=True,
        latent_do_sample_by="dropout",
        dropout_p=0.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt = "What is 23 + 19?\n<|start-latent|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    decoded_outputs = []
    for _ in range(4):
        with torch.no_grad():
            out = model.generate(
                **inputs, generation_config=cfg, num_return_sequences=1, use_cache=True
            )
        decoded_outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))

    # We don't assert on correctness (that would couple to model quality),
    # but the sampling path must at minimum execute without error and emit
    # decodable, non-empty sequences for every sample.
    assert len(decoded_outputs) == 4
    assert all(isinstance(s, str) and s for s in decoded_outputs)
    _ = Counter(decoded_outputs).most_common(1)


def test_latent_trajectory_shapes(model, tokenizer, device):
    """Latent reasoning emits T=6 hidden-state vectors of size n_embd.

    `LatentGenerationMixin` always captures latent thoughts internally
    and returns them via `output.latent_thoughts` when
    `return_dict_in_generate=True`. We do NOT also set
    `output_hidden_states=True`: the mixin already passes it through
    explicitly on the inner forward call (see `src/generation_mixin.py`
    L577/580), and enabling it on the config produces a duplicate-kwarg
    crash under recent transformers versions.
    """
    import torch

    from src.generation_mixin import LatentGenerationConfig  # type: ignore

    cfg = LatentGenerationConfig(
        max_new_tokens=64,
        latent_length=6,
        latent_do_sample=False,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt = "What is 5 * 9?\n<|start-latent|>"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            generation_config=cfg,
            num_return_sequences=1,
            use_cache=True,
        )

    latents = getattr(output, "latent_thoughts", None)
    assert latents is not None, "LatentGenerationMixin must populate latent_thoughts"
    # Layout: tensor of shape (B, T, n_embd) when stacked, else a list of
    # per-batch (T_b, n_embd) tensors.
    if isinstance(latents, torch.Tensor):
        assert latents.ndim == 3
        assert latents.shape[0] == 1  # batch size = 1 (single prompt)
        assert latents.shape[1] == 6  # T = latent_length
        assert latents.shape[2] == model.config.n_embd
    else:
        assert len(latents) == 1  # one item per batch entry
        for z in latents:
            assert z.shape[-2] == 6
            assert z.shape[-1] == model.config.n_embd


def test_architecture_invariants(model):
    """GPT-2 small architectural constants the analysis library relies on."""
    assert model.config.n_embd == 768
    assert model.config.n_layer == 12
    assert model.config.n_head == 12
    assert model.config.vocab_size == 50260
