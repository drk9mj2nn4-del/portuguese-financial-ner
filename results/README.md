# Result files

- `per_seed_results.csv` contains the validation score used for checkpoint selection and the held-out test metrics for each of the six fixed-15 runs.
- `aggregate_results.csv` reports the arithmetic mean and sample standard deviation across the three seeds in each condition.
- `per_type_f1.csv` reports the mean strict entity-level exact-match F1 for each entity type across the three seeds.
- `llm_screening_results.csv` records the preliminary 15-document model/prompt screening used to choose the annotation model. Its pooled metrics sum TP, FP, and FN over the four types after type-consistent span matching at a 50% overlap threshold. `NA` denotes an unusable structured output rather than a measured zero.

All BERTimbau values are copied from the machine-readable output of the completed fixed-15 runs. Predictions, checkpoints, logs, corpus records, and document identifiers are not included.
