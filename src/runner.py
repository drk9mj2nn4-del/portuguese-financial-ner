"""Runner for individual BERTimbau NER experiments and benchmark mode."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataset import (
    ChunkRecord,
    DocumentRecord,
    aggregate_word_predictions,
    chunk_document_word_bounded,
    load_test_dataset,
    load_train_and_val_datasets,
)
from metrics import compute_strict_entity_metrics

try:
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
        set_seed,
    )
    from trainer import CustomWeightedLossTrainer, InformativeEarlyStoppingCallback
    HAS_TORCH_HF = True
except ImportError:
    HAS_TORCH_HF = False
    torch = None  # type: ignore


def check_mps_precision(precision: str, torch_module: Any = None) -> None:
    """Validate device precision compatibility. Reject bf16 on MPS with clear message."""
    t_mod = torch_module if torch_module is not None else torch
    if precision.lower() == "bf16" and t_mod is not None:
        is_mps = False
        try:
            if hasattr(t_mod.backends, "mps") and t_mod.backends.mps.is_available():
                is_mps = True
        except Exception:
            pass
        if is_mps:
            raise ValueError(
                "Bfloat16 (bf16) is not supported on Apple Silicon MPS devices. "
                "Execution on MPS must use --precision fp32."
            )


def find_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Find the most recent checkpoint folder in output_dir by step index."""
    if not output_dir.exists():
        return None
    checkpoints = [
        p for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and p.name.split("-")[-1].isdigit()
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


class ChunkDataset:
    """PyTorch-compatible Dataset for ChunkRecords with transportable chunk IDs."""

    def __init__(self, chunks: List[ChunkRecord]):
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        chunk = self.chunks[idx]
        return {
            "input_ids": list(chunk.subtoken_ids),
            "attention_mask": list(chunk.attention_mask),
            "labels": list(chunk.subtoken_labels),
            "chunk_id": chunk.chunk_id,
            "chunk_idx": idx,
        }


class IdentifiableChunkCollator:
    """Data collator that pads sequences while preserving chunk identity for order verification."""

    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chunk_ids = [f["chunk_id"] for f in features]
        chunk_indices = [f["chunk_idx"] for f in features]

        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)
            batch_labels.append(f["labels"] + [-100] * pad_len)

        if torch is not None:
            tensor_ids = torch.tensor(batch_input_ids, dtype=torch.long)
            tensor_mask = torch.tensor(batch_attention_mask, dtype=torch.long)
            tensor_labels = torch.tensor(batch_labels, dtype=torch.long)
        else:
            tensor_ids = batch_input_ids
            tensor_mask = batch_attention_mask
            tensor_labels = batch_labels

        return {
            "input_ids": tensor_ids,
            "attention_mask": tensor_mask,
            "labels": tensor_labels,
            "chunk_id": chunk_ids,
            "chunk_idx": chunk_indices,
        }


def run_experiment(
    clean_dir: Path,
    output_root: Path,
    run_id: str,
    model_name_or_path: str = "neuralmind/bert-base-portuguese-cased",
    condition: str = "baseline",
    seed: int = 42,
    o_class_weight: float = 1.0,
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    max_epochs: int = 15,
    early_stopping_patience: int = 3,
    gradient_accumulation_steps: int = 2,
    max_length: int = 512,
    overlap_subtokens: int = 128,
    precision: str = "fp32",
    benchmark_mode: bool = False,
    local_files_only: bool = False,
    save_total_limit: int = 2,
) -> Dict[str, Any]:
    """Execute or resume a single NER experiment or run benchmark mode."""
    if not HAS_TORCH_HF:
        raise RuntimeError(
            "PyTorch and Hugging Face Transformers are required to run experiments. "
            "Please install dependencies from requirements.txt."
        )

    # Device and precision compatibility check
    check_mps_precision(precision)

    is_mps = False
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            is_mps = True
    except Exception:
        pass

    if is_mps:
        device_name = "Apple Silicon MPS (GPU)"
    elif torch.cuda.is_available():
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device_name = "CPU"

    pin_memory = False if is_mps else True

    t_start_total = time.time()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "run_summary.json"

    # 2. Check for completed run (Safe Resumption)
    if not benchmark_mode and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "completed":
                pred_test_file = run_dir / "predictions_test.jsonl"
                test_metrics_file = run_dir / "test_metrics.json"
                if pred_test_file.exists() and test_metrics_file.exists():
                    print(f"[{run_id}] [IGNORADA] Run já concluída anteriormente (Test F1={summary.get('test_micro_f1', 0.0):.4f}). Skipping.")
                    return summary
        except Exception:
            pass  # If corrupted, proceed with resume/re-run

    # 3. Benchmark mode adjustments
    if benchmark_mode:
        max_epochs = 1
        condition = "baseline"
        o_class_weight = 1.0
        learning_rate = 2e-5
        batch_size = 16
        gradient_accumulation_steps = 2

    # Set seeds
    set_seed(seed)

    # 4. Data preparation & timing
    # In benchmark mode, test.jsonl is NEVER loaded or opened (Requirement 5)
    print(f"\n[{run_id}] [1/6] Preparação dos dados:")
    t_prep_start = time.time()
    fit_docs, val_docs, labels = load_train_and_val_datasets(clean_dir)
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for i, label in enumerate(labels)}
    o_index = labels.index("O")

    print(f"[{run_id}] [2/6] Carregamento do tokenizer:")
    print(f"  - Modelo/Origem: {model_name_or_path} (local_files_only={local_files_only})")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_prefix_space=True,
        local_files_only=local_files_only,
    )
    cls_token_id = tokenizer.cls_token_id or 101
    sep_token_id = tokenizer.sep_token_id or 102
    pad_token_id = tokenizer.pad_token_id or 0

    def tokenize_word_fn(word: str) -> List[int]:
        return tokenizer.encode(word, add_special_tokens=False)

    # Chunk fit (training) documents
    fit_chunks: List[ChunkRecord] = []
    for doc in fit_docs:
        fit_chunks.extend(
            chunk_document_word_bounded(
                doc=doc,
                tokenize_word_fn=tokenize_word_fn,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                label_to_id=label_to_id,
                max_length=max_length,
                overlap_subtokens=overlap_subtokens,
                is_train=True,
            )
        )

    # Chunk validation documents
    val_chunks: List[ChunkRecord] = []
    for doc in val_docs:
        val_chunks.extend(
            chunk_document_word_bounded(
                doc=doc,
                tokenize_word_fn=tokenize_word_fn,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                label_to_id=label_to_id,
                max_length=max_length,
                overlap_subtokens=overlap_subtokens,
                is_train=False,
            )
        )

    print(f"  - Fit: {len(fit_docs)} documentos -> {len(fit_chunks)} chunks (palavra ativa única)")
    print(f"  - Validação: {len(val_docs)} documentos -> {len(val_chunks)} chunks")
    print(f"  - Etiquetas ({len(labels)}): {', '.join(labels[:5])}... ('O' no índice {o_index})")
    print(f"  - Janelas deslizantes: max_length={max_length}, overlap_subtokens={overlap_subtokens}")

    fit_dataset = ChunkDataset(fit_chunks)
    val_dataset = ChunkDataset(val_chunks)
    data_collator = IdentifiableChunkCollator(pad_token_id=pad_token_id)
    t_prep_end = time.time()
    prep_time_seconds = t_prep_end - t_prep_start
    print(f"  - Concluído em {prep_time_seconds:.2f}s")

    # 5. Model and class weights setup
    print(f"[{run_id}] [3/6] Carregamento do modelo:")
    print(f"  - Dispositivo selecionado: {device_name}")
    print(f"  - Precisão: {precision}")
    print(f"  - Hiperparâmetros: lr={learning_rate}, batch_size={batch_size}, grad_accum={gradient_accumulation_steps}, max_epochs={max_epochs}, patience={early_stopping_patience}")
    print(f"  - Ponderação da classe 'O': peso={o_class_weight} (restantes 8 classes com peso 1.0)")

    class_weights = torch.ones(len(labels), dtype=torch.float32)
    class_weights[o_index] = float(o_class_weight)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )

    use_bf16 = (precision.lower() == "bf16")

    # TrainingArguments using eval_strategy and dataloader_pin_memory
    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        eval_strategy="no" if benchmark_mode else "epoch",
        save_strategy="no" if benchmark_mode else "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=max_epochs,
        weight_decay=0.01,
        seed=seed,
        metric_for_best_model="strict_entity_exact_match_micro_f1",
        greater_is_better=True,
        load_best_model_at_end=False if benchmark_mode else True,
        save_total_limit=save_total_limit,
        logging_steps=10,
        report_to="none",
        bf16=use_bf16,
        fp16=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=pin_memory,
    )

    callbacks = []
    if not benchmark_mode and early_stopping_patience > 0:
        callbacks.append(InformativeEarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = CustomWeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=fit_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        class_weights=class_weights,
        eval_doc_records=val_docs,
        eval_chunk_records=val_chunks,
        id_to_label=id_to_label,
        callbacks=callbacks,
    )

    # 6. Check for existing checkpoint to resume
    resume_checkpoint = None
    if not benchmark_mode:
        resume_checkpoint = find_latest_checkpoint(run_dir / "checkpoints")
        if resume_checkpoint:
            print(f"  - [RETOMA] Checkpoint encontrado: {resume_checkpoint}. A retomar treino.")
        else:
            print(f"  - [INÍCIO] Nenhum checkpoint prévio encontrado. A iniciar treino de raiz.")

    # 7. Train model
    print(f"[{run_id}] [4/6] Treino:")
    print(f"  - A iniciar treino (máx {max_epochs} épocas, early stopping patience={early_stopping_patience})...")
    t_train_start = time.time()
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    t_train_end = time.time()
    train_time_seconds = t_train_end - t_train_start
    print(f"  - Treino concluído em {train_time_seconds:.2f}s")

    # 8. Evaluate on validation set
    print(f"[{run_id}] [5/6] Avaliação final:")
    t_val_start = time.time()
    val_eval_results = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    t_val_end = time.time()
    val_time_seconds = t_val_end - t_val_start

    # Benchmark mode exits early without test.jsonl access or heavy checkpoints (Requirement 7)
    if benchmark_mode:
        benchmark_results = {
            "mode": "benchmark",
            "run_id": run_id,
            "model": model_name_or_path,
            "condition": condition,
            "seed": seed,
            "fit_documents": len(fit_docs),
            "val_documents": len(val_docs),
            "epochs": 1,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "precision": precision,
            "validation_metrics": val_eval_results,
            "timings": {
                "prep_time_seconds": round(prep_time_seconds, 4),
                "train_time_seconds": round(train_time_seconds, 4),
                "val_time_seconds": round(val_time_seconds, 4),
                "total_time_seconds": round(time.time() - t_start_total, 4),
            },
        }
        (run_dir / "benchmark_results.json").write_text(
            json.dumps(benchmark_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[{run_id}] Benchmark completed in {benchmark_results['timings']['total_time_seconds']}s")
        return benchmark_results

    # 9. Extract best checkpoint metadata accurately (Requirement 6)
    best_metric, best_checkpoint, best_epoch = trainer.get_best_run_metadata()

    # 10. Reconstruct validation predictions for file export
    print(f"  - A reconstruir predições palavra a palavra da validação ({len(val_docs)} docs)...")
    val_loader = trainer.get_eval_dataloader(val_dataset)
    val_chunk_logits: List[List[List[float]]] = []
    with torch.no_grad():
        for step, inputs in enumerate(val_loader):
            inputs.pop("chunk_id", None)
            inputs.pop("chunk_idx", None)
            inputs = trainer._prepare_inputs(inputs)
            outputs = trainer.model(**inputs)
            logits = outputs.logits.detach().cpu().to(torch.float32).numpy()
            for i in range(logits.shape[0]):
                seq_len = int(inputs["attention_mask"][i].sum().item())
                val_chunk_logits.append(logits[i, :seq_len].tolist())

    val_chunks_by_doc: Dict[str, List[Tuple[ChunkRecord, List[List[float]]]]] = {}
    for chunk, c_logits in zip(val_chunks, val_chunk_logits):
        val_chunks_by_doc.setdefault(chunk.doc_id, []).append((chunk, c_logits))

    val_pred_records: List[Dict[str, Any]] = []
    val_gold_records: List[Dict[str, Any]] = []
    for doc in val_docs:
        val_gold_records.append({"doc_id": doc.doc_id, "tokens": doc.tokens, "ner_tags": doc.ner_tags})
        pair_list = val_chunks_by_doc.get(doc.doc_id, [])
        pair_list.sort(key=lambda item: item[0].chunk_index)
        c_records = [p[0] for p in pair_list]
        c_logits_list = [p[1] for p in pair_list]
        preds = aggregate_word_predictions(doc, c_records, c_logits_list, id_to_label)
        val_pred_records.append({"doc_id": doc.doc_id, "tokens": doc.tokens, "pred_tags": preds})

    val_metrics = compute_strict_entity_metrics(val_gold_records, val_pred_records)
    with (run_dir / "predictions_val.jsonl").open("w", encoding="utf-8") as f:
        for rec in val_pred_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (run_dir / "val_metrics.json").write_text(
        json.dumps(val_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 11. Load and evaluate on held-out manual Test Set (Requirement 5: loaded only now)
    t_test_start = time.time()
    test_docs, _ = load_test_dataset(clean_dir)
    print(f"  - A avaliar conjunto de teste ({len(test_docs)} documentos)...")
    test_chunks: List[ChunkRecord] = []
    for doc in test_docs:
        test_chunks.extend(
            chunk_document_word_bounded(
                doc=doc,
                tokenize_word_fn=tokenize_word_fn,
                cls_token_id=cls_token_id,
                sep_token_id=sep_token_id,
                label_to_id=label_to_id,
                max_length=max_length,
                overlap_subtokens=overlap_subtokens,
                is_train=False,
            )
        )

    test_dataset = ChunkDataset(test_chunks)
    test_loader = trainer.get_eval_dataloader(test_dataset)
    test_chunk_logits: List[List[List[float]]] = []
    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            inputs.pop("chunk_id", None)
            inputs.pop("chunk_idx", None)
            inputs = trainer._prepare_inputs(inputs)
            outputs = trainer.model(**inputs)
            logits = outputs.logits.detach().cpu().to(torch.float32).numpy()
            for i in range(logits.shape[0]):
                seq_len = int(inputs["attention_mask"][i].sum().item())
                test_chunk_logits.append(logits[i, :seq_len].tolist())

    test_chunks_by_doc: Dict[str, List[Tuple[ChunkRecord, List[List[float]]]]] = {}
    for chunk, c_logits in zip(test_chunks, test_chunk_logits):
        test_chunks_by_doc.setdefault(chunk.doc_id, []).append((chunk, c_logits))

    test_pred_records: List[Dict[str, Any]] = []
    test_gold_records: List[Dict[str, Any]] = []
    for doc in test_docs:
        test_gold_records.append({"doc_id": doc.doc_id, "tokens": doc.tokens, "ner_tags": doc.ner_tags})
        pair_list = test_chunks_by_doc.get(doc.doc_id, [])
        pair_list.sort(key=lambda item: item[0].chunk_index)
        c_records = [p[0] for p in pair_list]
        c_logits_list = [p[1] for p in pair_list]
        preds = aggregate_word_predictions(doc, c_records, c_logits_list, id_to_label)
        test_pred_records.append({"doc_id": doc.doc_id, "tokens": doc.tokens, "pred_tags": preds})

    test_metrics = compute_strict_entity_metrics(test_gold_records, test_pred_records)
    t_test_end = time.time()
    test_time_seconds = t_test_end - t_test_start

    with (run_dir / "predictions_test.jsonl").open("w", encoding="utf-8") as f:
        for rec in test_pred_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (run_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 12. Save run summary with accurate best_metric, best_checkpoint, and best_epoch
    print(f"[{run_id}] [6/6] Gravação dos resultados:")
    print(f"  - [CHECKPOINT SELECIONADO] {best_checkpoint} (Época {best_epoch})")
    print(f"  - Validação Strict Micro F1: {val_metrics['entity_exact_match_micro']['f1']:.4f}")
    print(f"  - Teste Strict Micro F1:     {test_metrics['entity_exact_match_micro']['f1']:.4f}")

    # 12. Save run summary with accurate best_metric, best_checkpoint, and best_epoch
    summary = {
        "status": "completed",
        "run_id": run_id,
        "model": model_name_or_path,
        "condition": condition,
        "seed": seed,
        "o_class_weight": o_class_weight,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "precision": precision,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "save_total_limit": save_total_limit,
        "best_metric": best_metric,
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
        "validation_micro_f1": val_metrics["entity_exact_match_micro"]["f1"],
        "test_micro_f1": test_metrics["entity_exact_match_micro"]["f1"],
        "timings": {
            "prep_seconds": round(prep_time_seconds, 2),
            "train_seconds": round(train_time_seconds, 2),
            "val_seconds": round(val_time_seconds, 2),
            "test_seconds": round(test_time_seconds, 2),
            "total_seconds": round(time.time() - t_start_total, 2),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[{run_id}] Run completed. Best epoch={best_epoch}, Val F1={best_metric:.4f}, Test F1={summary['test_micro_f1']:.4f}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for BERTimbau NER experiments.")
    parser.add_argument("--clean-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent.parent / "outputs")
    parser.add_argument("--run-id", type=str, default="bertimbau_baseline_s42")
    parser.add_argument("--model-name-or-path", "--model-name", type=str, default="neuralmind/bert-base-portuguese-cased", dest="model_name_or_path")
    parser.add_argument("--condition", type=str, default="baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--o-class-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to retain")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--local-files-only", action="store_true", help="Force loading weights from local cache/filesystem only")
    parser.add_argument("--benchmark", action="store_true", help="Run 1-epoch benchmark on baseline without test eval")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        clean_dir=args.clean_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        model_name_or_path=args.model_name_or_path,
        condition=args.condition,
        seed=args.seed,
        o_class_weight=args.o_class_weight,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        gradient_accumulation_steps=args.grad_accum,
        precision=args.precision,
        benchmark_mode=args.benchmark,
        local_files_only=args.local_files_only,
        save_total_limit=args.save_total_limit,
    )


if __name__ == "__main__":
    main()
