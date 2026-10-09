# Assisted annotation and class-weighted NER for Portuguese financial news

This repository accompanies the paper **"LLM-Assisted Corpus Construction and Recall-Oriented Class Weighting for Portuguese Financial NER"**, accepted at RECPAD 2026. It contains the BERTimbau fine-tuning and evaluation code, the fixed-15 experimental configuration, aggregate results, and the prompt templates used in the assisted-annotation workflow.

The corpus, document identifiers, model checkpoints, predictions, and execution logs are deliberately excluded. The source news articles cannot be redistributed. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the expected local data format.

## Corpus provenance and collection

[CORPUS_COLLECTION.md](CORPUS_COLLECTION.md) documents the Arquivo.pt search window, publisher filters, search terms, extraction and topic-selection rules, LLM-assisted curation, deduplication, and final partitions. It deliberately contains no article text, archived URLs, annotations, or real document identifiers. The exact private corpus cannot be reconstructed from this public repository.

## Experimental protocol

- Encoder: `neuralmind/bert-base-portuguese-cased` (BERTimbau Base).
- Task: IOB2 NER with `Participante`, `Evento`, `Tempo`, and `Quantificacao`.
- Conditions: ordinary cross-entropy (`w_O = 1.0`) and down-weighted outside class (`w_O = 0.05`; all entity-label weights remain 1.0).
- Seeds: 42, 123, and 2025.
- Optimizer settings: learning rate `2e-5`, batch size 16, gradient accumulation 2.
- Training: exactly 15 epochs; early stopping disabled.
- Long documents: word-bounded windows of at most 512 subtokens with a 128-subtoken overlap budget. Only the first subtoken of each word receives a label, and each word contributes to the training loss in exactly one window.
- Model selection: strict entity-level exact-match micro-F1 on the validation partition.
- Test use: the held-out test partition is loaded only after training and checkpoint selection.

The six rows used in the paper are in [`configs/experiment_matrix.csv`](configs/experiment_matrix.csv). Reported results are in [`results/`](results/).

## Installation

Python 3.9 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data preparation

Create a local `data/` directory with:

```text
data/
  train.jsonl
  test.jsonl
  labels.json
  validation_ids.json
```

The real corpus is not part of this repository. Copy [`configs/labels.json`](configs/labels.json) to `data/labels.json`, then provide the remaining private files according to [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md).

## Running the fixed-15 protocol

```bash
bash run_fixed15.sh
```

Optional environment variables:

```bash
DATASET_DIR=/path/to/private/data \
MODEL_NAME_OR_PATH=neuralmind/bert-base-portuguese-cased \
PRECISION=fp32 \
bash run_fixed15.sh
```

Use `LOCAL_FILES_ONLY=1` when the model is already cached and network access must be disabled. Additional arguments are forwarded to `src/run_matrix.py`.

Outputs are written to `outputs/`, which is ignored by Git. Completed runs are skipped, and interrupted runs resume from the latest retained checkpoint.

## Tests

The lightweight tests use only synthetic records and do not access the private corpus:

```bash
python3 -m unittest discover -s tests -v
```

## Prompt templates

The prompt templates in [`prompts/`](prompts/) document the preliminary model screening, the selected Portuguese sequential workflow for entity scouting and attribute enrichment, Qwen-family auditing and correction, and the final cross-family Mistral audit. Dataset-derived demonstrations have been replaced by placeholders to prevent redistribution of source text.

## Authors

- Pedro Herdeiro — Faculty of Sciences, University of Porto
- Pedro Campos — School of Economics and Management, University of Porto
- Evelin Freire Amorim — INESC TEC

## Licence

The code and documentation in this repository are released under the [MIT License](LICENSE). The corpus and underlying news articles are not covered by this licence and are not distributed.
