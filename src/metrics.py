"""Strict entity-level exact matching evaluation conforming to evaluate_ner_predictions.py."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


def entities(tags: List[str]) -> Set[Tuple[str, int, int]]:
    """Convert IOB2 tags to (type, start, end-exclusive) entities.

    An I-tag without a compatible preceding B/I-tag starts a new entity. This
    follows the standard CoNLL-style repair and keeps evaluation deterministic.
    """
    result: Set[Tuple[str, int, int]] = set()
    active_type: str | None = None
    active_start: int | None = None

    def close(end: int) -> None:
        nonlocal active_type, active_start
        if active_type is not None and active_start is not None:
            result.add((active_type, active_start, end))
        active_type = None
        active_start = None

    for index, tag in enumerate(tags):
        if tag == "O":
            close(index)
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid IOB2 tag: {tag!r}")
        prefix, entity_type = tag.split("-", 1)
        if prefix not in {"B", "I"} or not entity_type:
            raise ValueError(f"Invalid IOB2 tag: {tag!r}")
        if prefix == "B" or active_type != entity_type:
            close(index)
            active_type = entity_type
            active_start = index
    close(len(tags))
    return result


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> Dict[str, float | int]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_strict_entity_metrics(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute strict entity-level exact matching metrics across documents."""
    gold_by_id = {record["doc_id"]: record for record in gold_records}
    pred_by_id = {record["doc_id"]: record for record in pred_records}

    if len(gold_by_id) != len(gold_records):
        raise ValueError("Duplicate doc_id in gold data")
    if len(pred_by_id) != len(pred_records):
        raise ValueError("Duplicate doc_id in predictions")

    missing = sorted(set(gold_by_id) - set(pred_by_id))
    unexpected = sorted(set(pred_by_id) - set(gold_by_id))
    if missing or unexpected:
        raise ValueError(f"Document mismatch: missing={missing}, unexpected={unexpected}")

    gold_entities: Set[Tuple[str, str, int, int]] = set()
    pred_entities: Set[Tuple[str, str, int, int]] = set()
    token_counts: Dict[str, Counter] = defaultdict(Counter)
    token_total = 0
    token_correct = 0

    for doc_id in sorted(gold_by_id):
        gold = gold_by_id[doc_id]
        pred = pred_by_id[doc_id]
        gold_tags = gold["ner_tags"]
        pred_tags = pred.get("pred_tags", pred.get("ner_tags"))

        if pred_tags is None:
            raise ValueError(f"{doc_id}: prediction has no pred_tags or ner_tags")
        if len(gold_tags) != len(pred_tags):
            raise ValueError(
                f"{doc_id}: tag length mismatch ({len(gold_tags)} vs {len(pred_tags)})"
            )
        if "tokens" in pred and pred["tokens"] != gold.get("tokens"):
            raise ValueError(f"{doc_id}: prediction tokens differ from gold tokens")

        gold_entities.update((doc_id, kind, start, end) for kind, start, end in entities(gold_tags))
        pred_entities.update((doc_id, kind, start, end) for kind, start, end in entities(pred_tags))

        for gold_tag, pred_tag in zip(gold_tags, pred_tags):
            token_total += 1
            token_correct += int(gold_tag == pred_tag)
            for label in {gold_tag, pred_tag}:
                token_counts[label]["tp"] += int(gold_tag == label and pred_tag == label)
                token_counts[label]["fp"] += int(gold_tag != label and pred_tag == label)
                token_counts[label]["fn"] += int(gold_tag == label and pred_tag != label)

    exact = gold_entities & pred_entities
    overall = prf(
        len(exact),
        len(pred_entities - gold_entities),
        len(gold_entities - pred_entities),
    )
    entity_types = sorted({item[1] for item in gold_entities | pred_entities})
    per_type = {}
    for entity_type in entity_types:
        gold_type = {item for item in gold_entities if item[1] == entity_type}
        pred_type = {item for item in pred_entities if item[1] == entity_type}
        per_type[entity_type] = prf(
            len(gold_type & pred_type),
            len(pred_type - gold_type),
            len(gold_type - pred_type),
        )
    macro_entity_f1 = safe_div(
        sum(float(metrics["f1"]) for metrics in per_type.values()), len(per_type)
    )

    token_per_label = {
        label: prf(counts["tp"], counts["fp"], counts["fn"])
        for label, counts in sorted(token_counts.items())
    }
    token_macro_f1 = safe_div(
        sum(float(metrics["f1"]) for metrics in token_per_label.values()),
        len(token_per_label),
    )

    return {
        "documents": len(gold_records),
        "primary_metric": "strict_entity_exact_match_micro_f1",
        "entity_exact_match_micro": overall,
        "entity_exact_match_macro_f1": macro_entity_f1,
        "entity_exact_match_per_type": per_type,
        "token_accuracy": safe_div(token_correct, token_total),
        "token_macro_f1": token_macro_f1,
        "token_per_label": token_per_label,
    }
