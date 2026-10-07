"""Pilot LoRA fine-tune of the pinned Qwen3-Embedding-0.6B on corpus-only pairs.

Small on purpose (runs under guard.py on the shared Studio, MPS, <8 GB): LoRA r=16
on all attention/MLP projections, CachedMultipleNegativesRankingLoss (large effective
batch at small memory), queries get the production instruction prefix, documents
none — the same asymmetric contract the service uses. The LoRA is merged into the
base weights and saved as a plain SentenceTransformer directory so the audit encoders
can load it like any other model.
"""

import argparse
import json
from pathlib import Path

from embed_utils import AUDIT_ROOT, DEFAULT_TASK, TEXT_MODELS, device


def main() -> None:
    """Train, merge, save."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", default=str(AUDIT_ROOT / "pilot_text_v1" / "pairs.jsonl"))
    parser.add_argument("--out", default=str(AUDIT_ROOT / "pilot_text_v1" / "model_lora_merged"))
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--mini-batch", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: use only N pairs")
    args = parser.parse_args()

    spec = TEXT_MODELS["qwen3-0.6b"]
    # bf16 weights + gradient checkpointing keep LoRA training inside the ~7 GB MPS budget on the shared box.
    model = SentenceTransformer(spec["id"], revision=spec["revision"], device=device(),
                                model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = args.max_seq
    # Wrap explicitly as a PeftModel so the adapter can be merged afterwards (ST 5 keeps the HF model in `.model`; `.auto_model` is a read-only alias).
    model[0].model = get_peft_model(model[0].model, LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model[0].model.print_trainable_parameters()
    model[0].model.enable_input_require_grads()
    model[0].model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    ds = Dataset.from_dict({"anchor": [r["anchor"] for r in rows], "positive": [r["positive"] for r in rows]})
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(Path(args.out).parent / "trainer_out"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={"anchor": f"Instruct: {DEFAULT_TASK}\nQuery: "},
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        seed=13,
    )
    trainer = SentenceTransformerTrainer(model=model, args=targs, train_dataset=ds, loss=loss)
    trainer.train()
    # merge LoRA into the base weights -> a plain ST model directory
    model[0].model = model[0].model.merge_and_unload()
    model.save(args.out)
    (Path(args.out) / "pilot_meta.json").write_text(json.dumps({**vars(args), "base": spec}, indent=1))
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
