"""Custom Trainer with configurable class weights, chunk order validation, and strict F1 evaluation."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import SequentialSampler
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import EarlyStoppingCallback
    HAS_TORCH_HF = True
except ImportError:
    HAS_TORCH_HF = False
    Trainer = object  # type: ignore
    TrainingArguments = object  # type: ignore
    EarlyStoppingCallback = object  # type: ignore
    torch = None  # type: ignore
    nn = None  # type: ignore
    SequentialSampler = None  # type: ignore

from dataset import ChunkRecord, DocumentRecord, aggregate_word_predictions
from metrics import compute_strict_entity_metrics


class CustomWeightedLossTrainer(Trainer if HAS_TORCH_HF else object):
    """Hugging Face Trainer subclass that:
    1. Injects class-weighted CrossEntropyLoss targeting the majority 'O' class (index 0).
    2. Enforces strictly sequential single-process evaluation and validates chunk ordering.
    3. Overrides evaluate() to reconstruct word-level predictions per document and calculate
       strict entity-level exact-match micro F1 for model checkpoint selection.
    4. Accurately tracks best_metric, best_checkpoint, and best_epoch.
    """

    def __init__(
        self,
        *args,
        class_weights: Optional[Any] = None,
        eval_doc_records: Optional[List[DocumentRecord]] = None,
        eval_chunk_records: Optional[List[ChunkRecord]] = None,
        id_to_label: Optional[Dict[int, str]] = None,
        **kwargs,
    ):
        if not HAS_TORCH_HF:
            raise RuntimeError(
                "PyTorch and Hugging Face Transformers are required to instantiate CustomWeightedLossTrainer."
            )
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.eval_doc_records = eval_doc_records or []
        self.eval_chunk_records = eval_chunk_records or []
        self.id_to_label = id_to_label or {}

        # Accurate tracking of best checkpoint metadata
        self.best_eval_micro_f1: Optional[float] = None
        self.best_eval_epoch: Optional[float] = None
        self.best_eval_step: Optional[int] = None
        self.best_eval_checkpoint: Optional[str] = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute CrossEntropyLoss with configurable class weights and ignore_index=-100."""
        # Explicitly pop chunk identification metadata before forwarding to model
        # ensuring chunk_id and chunk_idx never reach the model signature
        chunk_ids = inputs.pop("chunk_id", None)
        chunk_indices = inputs.pop("chunk_idx", None)

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        active_loss = inputs.get("attention_mask", None) is not None
        if active_loss:
            active_logits = logits.view(-1, model.config.num_labels)
            active_labels = labels.view(-1)
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def get_eval_dataloader(self, eval_dataset=None):
        """Force strictly sequential, deterministic single-process evaluation dataloader."""
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")

        return torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            sampler=SequentialSampler(eval_dataset),
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Run evaluation by aggregating chunk logits into document-level word predictions.

        Guarantees:
        - Validates chunk IDs and strictly sequential order. Any discrepancy raises an immediate error.
        - Reconstructs word-level predictions per document.
        - Calculates strict entity-level exact-match micro F1 matching evaluate_ner_predictions.py.
        - Accurately tracks best_eval_epoch and best_eval_micro_f1.
        """
        model = self._wrap_model(self.model, training=False, dataloader=None)
        model.eval()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        chunk_logits_all: List[List[List[float]]] = []
        expected_next_pos = 0

        with torch.no_grad():
            for step, inputs in enumerate(eval_dataloader):
                # Extract chunk identifiers for order and integrity validation
                batch_chunk_ids = inputs.pop("chunk_id", None)
                batch_chunk_indices = inputs.pop("chunk_idx", None)

                inputs = self._prepare_inputs(inputs)
                outputs = model(**inputs)
                logits = outputs.logits.detach().cpu().to(torch.float32).numpy()

                batch_size = logits.shape[0]
                for i in range(batch_size):
                    # Validate chunk sequence order if chunk identifiers are transported
                    if batch_chunk_ids is not None:
                        c_id = batch_chunk_ids[i]
                        c_idx = batch_chunk_indices[i] if batch_chunk_indices is not None else None
                        expected_chunk = self.eval_chunk_records[expected_next_pos]
                        if c_id != expected_chunk.chunk_id or (c_idx is not None and c_idx != expected_next_pos):
                            raise ValueError(
                                f"DataLoader chunk order divergence! At position {expected_next_pos}, "
                                f"expected chunk '{expected_chunk.chunk_id}' (idx={expected_next_pos}), "
                                f"but received '{c_id}' (idx={c_idx}). Shuffled evaluation is forbidden."
                            )

                    seq_len = int(inputs["attention_mask"][i].sum().item())
                    expected_len = len(self.eval_chunk_records[expected_next_pos].subtoken_ids)
                    if seq_len != expected_len:
                        raise ValueError(
                            f"Chunk subtoken length mismatch at position {expected_next_pos} "
                            f"({self.eval_chunk_records[expected_next_pos].chunk_id}): "
                            f"expected {expected_len}, got {seq_len}."
                        )

                    chunk_logits_all.append(logits[i, :seq_len].tolist())
                    expected_next_pos += 1

        if expected_next_pos != len(self.eval_chunk_records):
            raise ValueError(
                f"Incomplete evaluation! Processed {expected_next_pos} chunks out of "
                f"{len(self.eval_chunk_records)} expected chunks."
            )

        # Group chunk logits by doc_id
        chunks_by_doc: Dict[str, List[Tuple[ChunkRecord, List[List[float]]]]] = {}
        for chunk, c_logits in zip(self.eval_chunk_records, chunk_logits_all):
            chunks_by_doc.setdefault(chunk.doc_id, []).append((chunk, c_logits))

        # Reconstruct word-level predictions for each document
        pred_records: List[Dict[str, Any]] = []
        gold_records: List[Dict[str, Any]] = []

        for doc in self.eval_doc_records:
            gold_records.append({
                "doc_id": doc.doc_id,
                "tokens": doc.tokens,
                "ner_tags": doc.ner_tags,
            })
            pair_list = chunks_by_doc.get(doc.doc_id, [])
            pair_list.sort(key=lambda item: item[0].chunk_index)
            c_records = [p[0] for p in pair_list]
            c_logits_list = [p[1] for p in pair_list]

            pred_tags = aggregate_word_predictions(
                doc=doc,
                chunks=c_records,
                chunk_logits_list=c_logits_list,
                id_to_label=self.id_to_label,
            )
            pred_records.append({
                "doc_id": doc.doc_id,
                "tokens": doc.tokens,
                "pred_tags": pred_tags,
            })

        metrics = compute_strict_entity_metrics(gold_records, pred_records)
        micro = metrics["entity_exact_match_micro"]
        micro_f1 = float(micro["f1"])

        current_epoch = float(self.state.epoch) if self.state.epoch is not None else 0.0
        current_step = int(self.state.global_step)

        # Track best epoch and metric accurately. A tie is not a new best,
        # matching the strict comparison used by Hugging Face Trainer.
        is_new_best = self.best_eval_micro_f1 is None or micro_f1 > self.best_eval_micro_f1
        if is_new_best:
            self.best_eval_micro_f1 = micro_f1
            self.best_eval_epoch = current_epoch
            self.best_eval_step = current_step

        output_metrics = {
            f"{metric_key_prefix}_loss": 0.0,
            f"{metric_key_prefix}_strict_entity_exact_match_micro_f1": micro_f1,
            f"{metric_key_prefix}_precision": float(micro["precision"]),
            f"{metric_key_prefix}_recall": float(micro["recall"]),
            f"{metric_key_prefix}_macro_f1": float(metrics["entity_exact_match_macro_f1"]),
            f"{metric_key_prefix}_token_accuracy": float(metrics["token_accuracy"]),
        }

        best_so_far = self.best_eval_micro_f1 if self.best_eval_micro_f1 is not None else micro_f1
        best_marker = " ★ NOVO MELHOR CHECKPOINT!" if is_new_best else ""
        print(
            f"\n[AVALIAÇÃO ÉPOCA {current_epoch:.1f} | Passo {current_step}] "
            f"F1 Estrito: {micro_f1:.4f}{best_marker} | "
            f"Precisão: {float(micro['precision']):.4f} | Recall: {float(micro['recall']):.4f} | "
            f"Melhor F1 até ao momento: {best_so_far:.4f}"
        )

        self.log(output_metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output_metrics)
        return output_metrics

    def get_best_run_metadata(self) -> Tuple[Optional[float], Optional[str], Optional[float]]:
        """Return (best_metric, best_checkpoint, best_epoch).

        Extracts the exact epoch where best_metric occurred from log_history or tracked state,
        avoiding using final state.epoch as a surrogate.
        """
        best_metric = self.state.best_metric
        best_checkpoint = self.state.best_model_checkpoint

        # Search log_history for the entry matching best_metric
        best_epoch = self.best_eval_epoch
        if best_metric is not None:
            metric_key = "eval_strict_entity_exact_match_micro_f1"
            for entry in self.state.log_history:
                if metric_key in entry:
                    if abs(entry[metric_key] - best_metric) < 1e-7:
                        best_epoch = entry.get("epoch", best_epoch)
                        break

        return best_metric, best_checkpoint, best_epoch


class InformativeEarlyStoppingCallback(EarlyStoppingCallback if HAS_TORCH_HF else object):
    """EarlyStoppingCallback with explicit terminal message when patience triggers."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        super().on_evaluate(args, state, control, metrics, **kwargs)
        if getattr(control, "should_training_stop", False):
            print(
                f"\n[EARLY STOPPING] Critério de paragem antecipada atingido na época {state.epoch:.1f} "
                f"(passo {state.global_step}) após {self.early_stopping_patience} avaliações consecutivas sem melhoria."
            )
