from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dataset import DocumentRecord, aggregate_word_predictions, chunk_document_word_bounded
from metrics import compute_strict_entity_metrics
from run_matrix import load_matrix


class CoreTests(unittest.TestCase):
    def test_exact_entity_metrics(self):
        gold = [{"doc_id": "d1", "tokens": ["A", "B", "C"], "ner_tags": ["B-Evento", "I-Evento", "O"]}]
        pred = [{"doc_id": "d1", "tokens": ["A", "B", "C"], "pred_tags": ["B-Evento", "I-Evento", "O"]}]
        result = compute_strict_entity_metrics(gold, pred)
        self.assertEqual(result["entity_exact_match_micro"]["f1"], 1.0)

    def test_word_bounded_chunk_ownership(self):
        doc = DocumentRecord("d1", "synthetic", "finance", [f"w{i}" for i in range(20)], ["O"] * 20)
        chunks = chunk_document_word_bounded(
            doc,
            tokenize_word_fn=lambda _: [10, 11],
            cls_token_id=1,
            sep_token_id=2,
            label_to_id={"O": 0},
            max_length=14,
            overlap_subtokens=4,
            is_train=True,
        )
        owned = []
        for chunk in chunks:
            self.assertLessEqual(len(chunk.subtoken_ids), 14)
            for word_index, position in chunk.word_to_subtoken_idx.items():
                if chunk.subtoken_labels[position] != -100:
                    owned.append(word_index)
        self.assertEqual(sorted(owned), list(range(20)))

    def test_word_prediction_reconstruction(self):
        doc = DocumentRecord("d1", "synthetic", "finance", ["Empresa", "subiu"], ["B-Participante", "B-Evento"])
        chunks = chunk_document_word_bounded(
            doc,
            tokenize_word_fn=lambda _: [10],
            cls_token_id=1,
            sep_token_id=2,
            label_to_id={"O": 0, "B-Participante": 1, "B-Evento": 2},
            is_train=False,
        )
        logits = [[[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0]]]
        labels = aggregate_word_predictions(doc, chunks, logits, {0: "O", 1: "B-Participante", 2: "B-Evento"})
        self.assertEqual(labels, ["B-Participante", "B-Evento"])

    def test_fixed15_matrix(self):
        rows = load_matrix(REPO_ROOT / "configs" / "experiment_matrix.csv")
        self.assertEqual(len(rows), 6)
        self.assertEqual({row["seed"] for row in rows}, {42, 123, 2025})
        self.assertEqual({row["condition"] for row in rows}, {"baseline", "weight_only"})
        self.assertEqual([row["o_class_weight"] for row in rows].count(0.05), 3)


if __name__ == "__main__":
    unittest.main()
