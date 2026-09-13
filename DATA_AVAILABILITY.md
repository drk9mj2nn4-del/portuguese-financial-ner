# Data availability and local schema

The corpus is not distributed with this release because the underlying news texts cannot be republished. The code therefore expects a user-supplied local directory and never downloads the dataset.

## Required files

- `train.jsonl`: 121 manually reviewed documents used as the development pool. The identifiers listed in `validation_ids.json` form the 13-document validation partition; the other 108 documents form the fitting partition.
- `test.jsonl`: 29 manually annotated held-out documents.
- `validation_ids.json`: JSON array containing the 13 validation `doc_id` values.
- `labels.json`: ordered list of the nine IOB2 labels. Index 0 must be `O`; see `configs/labels.json`.

Each JSONL line must contain:

```json
{
  "doc_id": "synthetic_001",
  "origin": "manual",
  "topic": "finance",
  "tokens": ["Empresa", "X", "subiu", "5", "%", "."],
  "ner_tags": ["B-Participante", "I-Participante", "B-Evento", "B-Quantificacao", "I-Quantificacao", "O"]
}
```

The example is synthetic and is not drawn from the corpus. `tokens` and `ner_tags` must have identical lengths, all identifiers must be unique, and the fitting, validation, and test identifiers must be disjoint.

For anonymous review, do not commit the private data directory, real validation identifiers, model predictions, or checkpoints.
