"""Capture one fixed next-token logit vector with Transformers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
DEFAULT_PROMPT = "The sky appears blue because"


def capture_logits(
    model_name: str,
    prompt: str,
    output_dir: Path,
    dtype_name: str = "float32",
    revision: str | None = DEFAULT_REVISION,
) -> None:
    dtype = getattr(torch, dtype_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        device_map="auto",
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1].to(torch.float32).cpu().contiguous()

    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / "logits_fp32.pt"
    torch.save(logits, logits_path)
    metadata = {
        "model": model_name,
        "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "prompt": prompt,
        "model_dtype": dtype_name,
        "logits_dtype": "float32",
        "logits_shape": list(logits.shape),
        "input_ids": inputs["input_ids"][0].cpu().tolist(),
        "logits_file": str(logits_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("data/model_logits/smollm2-135m-fp32"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    capture_logits(args.model, args.prompt, args.output_dir, args.dtype, args.revision)


if __name__ == "__main__":
    main()