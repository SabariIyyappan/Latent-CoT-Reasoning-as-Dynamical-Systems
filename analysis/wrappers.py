"""
Wrapper classes for the COCONUT and CODI latent reasoning models.

Combines the base model classes with LatentGenerationMixin to enable
latent thought extraction during inference.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Literal

import copy

import torch
from transformers import AutoTokenizer


# src/ at project root contains the LatentTTS source code
# (upstream: https://github.com/YRYangang/LatentTTS, see src/UPSTREAM.md)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.coconut import COCONUTGPT2
from src.models.codi import CODIGPT2
from src.generation_mixin import (
    LatentGenerationMixin,
    LatentGenerationConfig,
    LatentGenerateDecoderOnlyOutput,
)


class LatentCoconutGPT2(COCONUTGPT2, LatentGenerationMixin):
    """COCONUT model with latent generation capabilities."""

    def __init__(self, config):
        super().__init__(config)


class LatentCodiGPT2(CODIGPT2, LatentGenerationMixin):
    """CODI model with latent generation capabilities."""

    def __init__(self, config):
        super().__init__(config)


class PromptTooLongError(RuntimeError):
    """Raised when a tokenised prompt alone fills the model's position table."""


# Cam-ready scope: GSM8K x {CODI, COCONUT} x {Vanilla, Sim-CoT}. Both
# vanilla and Sim-CoT checkpoints share architectures (CODI uses prj.pt,
# COCONUT does not), so the registry keys remain method-based and the
# paradigm only swaps the checkpoint path.
MODEL_REGISTRY = {
    "coconut": {
        "class": LatentCoconutGPT2,
        "arch": "gpt2",
        "new_line_after_input": True,
        "answer_split": "#",
        "prompt_template": "{question}",  # handled by format_input below
        "use_start_latent": True,
        "use_max_latent_length": False,
    },
    "codi": {
        "class": LatentCodiGPT2,
        "arch": "gpt2",
        "new_line_after_input": False,
        "answer_split": None,
        "prompt_template": "{question}",
        "use_start_latent": True,
        "use_max_latent_length": False,
    },
}


class ModelWrapper:
    """
    Unified wrapper for loading and running latent reasoning models.

    Handles model loading, tokenizer setup, and inference with
    latent thought extraction.
    """

    def __init__(
        self,
        method: str,
        checkpoint: str,
        device: str = "cpu",
        latent_length: int = 6,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        max_latent_length: Optional[int] = None,
    ):
        if method not in MODEL_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Choose from: {list(MODEL_REGISTRY.keys())}")

        self.method = method
        self.device = device
        self.latent_length = latent_length
        self.max_latent_length = max_latent_length or latent_length
        self.registry = MODEL_REGISTRY[method]

        # Patch tokenizer_config.json if extra_special_tokens is a list.
        # Older checkpoints saved it as a list; transformers >=4.40 expects a dict
        # and raises AttributeError: 'list' object has no attribute 'keys'.
        import json as _json
        _tok_cfg_path = Path(checkpoint) / "tokenizer_config.json"
        if _tok_cfg_path.exists():
            with open(_tok_cfg_path) as _f:
                _tok_cfg = _json.load(_f)
            if isinstance(_tok_cfg.get("extra_special_tokens"), list):
                _tok_cfg["extra_special_tokens"] = {}
                with open(_tok_cfg_path, "w") as _f:
                    _json.dump(_tok_cfg, _f, indent=2)
                print("  Patched tokenizer_config.json: extra_special_tokens list → dict")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Resolve special token IDs
        self.latent_id = self.tokenizer.convert_tokens_to_ids("<|latent|>")
        if self.registry["use_start_latent"]:
            self.start_id = self.tokenizer.convert_tokens_to_ids("<|start-latent|>")
            self.end_id = self.tokenizer.convert_tokens_to_ids("<|end-latent|>")
        else:
            # No explicit start token; "###" doubles as the end marker
            self.start_id = -1
            self.end_id = self.tokenizer.convert_tokens_to_ids("###")

        # Build generation config
        if self.registry["use_max_latent_length"]:
            # Dynamic latent length
            gen_kwargs = dict(
                latent_length=None,
                max_latent_length=self.max_latent_length,
            )
        else:
            # GPT-2 based models use fixed latent length
            gen_kwargs = dict(latent_length=latent_length)

        self.generation_config = LatentGenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            return_dict_in_generate=True,
            **gen_kwargs,
        )

        # Load model - GPT-2 models use attn/embd_pdrop; Llama doesn't
        model_kwargs = dict(
            latent_id=self.latent_id,
            latent_start_id=self.start_id,
            latent_end_id=self.end_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.registry["arch"] == "gpt2":
            model_kwargs["attn_pdrop"] = 0.0
            model_kwargs["embd_pdrop"] = 0.0

        model_cls = self.registry["class"]
        self.model = model_cls.from_pretrained(checkpoint, **model_kwargs)
        self.model.to(device)
        self.model.eval()

        # Load CODI projection module if present (required for codi_math_distilled).
        # prj.pt is saved separately because config.projector=False at checkpoint time;
        # we build the projector here, load its weights, and enable it in config.
        prj_path = Path(checkpoint) / "prj.pt"
        if prj_path.exists():
            import torch.nn as nn
            hidden = self.model.config.hidden_size
            projector = nn.Sequential(
                nn.Dropout(0.0),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            _sd = torch.load(str(prj_path), map_location=device, weights_only=True)
            # prj.pt was saved with LayerNorm named "ln"; Sequential uses index "4"
            _sd = {"4." + k[3:] if k.startswith("ln.") else k: v for k, v in _sd.items()}
            projector.load_state_dict(_sd)
            projector.to(device).eval()
            self.model.projector = projector
            self.model.config.projector = True
            print(f"Loaded projector from {prj_path}")

        # Prevent double-advance of cache_position.
        #
        # transformers >= 4.44 increments cache_position inside
        # _update_model_kwargs_for_generation; LatentGenerationMixin._sample
        # (src/generation_mixin.py line 592) also increments it right after that
        # call.  The double-advance makes cache_position race one step ahead of
        # inputs_embeds.shape[1], so inputs_embeds[:, cache_position] in
        # prepare_inputs_for_generation fires a CUDA IndexKernel OOB on the very
        # first decode step for every sample.
        #
        # Fix: override _update_model_kwargs_for_generation on the model instance
        # to save/restore cache_position, leaving _sample's own increment as the
        # sole advance.  The rest of the method (past_key_values, attention_mask,
        # etc.) runs unchanged.
        import types as _types
        from transformers import GenerationMixin as _GM
        _base_upd = _GM._update_model_kwargs_for_generation

        def _patched_upd(model_self, outputs, model_kwargs,
                         is_encoder_decoder=False, num_new_tokens=1):
            saved_cp = model_kwargs.get("cache_position")
            model_kwargs = _base_upd(
                model_self, outputs, model_kwargs, is_encoder_decoder, num_new_tokens
            )
            if saved_cp is not None:
                model_kwargs["cache_position"] = saved_cp
            return model_kwargs

        self.model._update_model_kwargs_for_generation = _types.MethodType(
            _patched_upd, self.model
        )

    def format_input(self, question: str) -> str:
        """Format question with model-specific prompt template."""
        template = self.registry["prompt_template"]
        formatted = template.format(question=question)
        if self.registry["use_start_latent"]:
            postfix = "\n<|start-latent|>" if self.registry["new_line_after_input"] else "<|start-latent|>"
            formatted = formatted + postfix
        return formatted

    @torch.inference_mode()
    def run_inference(self, question: str) -> LatentGenerateDecoderOnlyOutput:
        """
        Run inference on a single question and return the full output
        including latent_thoughts.
        """
        formatted = self.format_input(question)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)

        # Guard against position-embedding overflow (GPT-2: n_positions=1024).
        # With KV cache + cache_position, each generated token at step k uses
        # position n_input+k.  If n_input+max_new_tokens > n_positions the GPU
        # raises an index-out-of-bounds assertion and crashes the process.
        # Latent thoughts are extracted from the first latent_length steps, so
        # capping max_new_tokens for long inputs does not affect the analysis.
        max_pos = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", None)
            or 1024
        )
        n_input = inputs["input_ids"].shape[1]
        # Need at least latent_length + 1 new positions (forced latent tokens +
        # end-latent) so _sample can collect exactly latent_length latent indices.
        # Skip the sample if the remaining capacity is too small.
        min_new = self.max_latent_length + 1
        if n_input + min_new > max_pos:
            raise PromptTooLongError(
                f"Input ({n_input} tokens) leaves only {max_pos - n_input} positions; "
                f"need at least {min_new} for the latent phase (table size {max_pos})"
            )

        gen_config = copy.copy(self.generation_config)
        if gen_config.max_new_tokens is not None and n_input + gen_config.max_new_tokens > max_pos:
            gen_config.max_new_tokens = max_pos - n_input

        output = self.model.generate(
            **inputs,
            generation_config=gen_config,
            num_return_sequences=1,
            use_cache=True,
        )
        return output

    def extract_latent_thoughts(self, output: LatentGenerateDecoderOnlyOutput) -> torch.Tensor:
        """
        Extract the latent thought tensor from model output.

        Returns:
            Tensor of shape [num_latent_steps, hidden_size]
        """
        thoughts = output.latent_thoughts
        if thoughts is None:
            raise ValueError("Model output does not contain latent_thoughts. Check generation config.")
        # If batched, take first sample
        if thoughts.dim() == 3:
            thoughts = thoughts[0]
        return thoughts.cpu()

    def decode_output(self, output: LatentGenerateDecoderOnlyOutput) -> str:
        """Decode the generated sequence to text."""
        return self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from decoded text.

        Uses the same strategy as _parse_latex_number in data_prep.py so that
        predicted and expected answers are compared on equal terms.  After
        splitting on the answer delimiter we try (in order):
          1. bare float  — handles plain integers / decimals
          2. eval_latex_expr  — handles \\frac, \\sqrt, \\pi, etc.
          3. last-number regex — fallback for anything else
        """
        import re
        from analysis.data_prep import eval_latex_expr
        if self.registry["answer_split"]:
            ans = text.split(self.registry["answer_split"])[-1].replace(",", "").strip()
        else:
            # No delimiter: pull the last number from the whole output
            numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
            return float(numbers[-1]) if numbers else None
        return eval_latex_expr(ans)
