#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET_DIR="${DATASET_DIR:-${ROOT_DIR}/data}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-neuralmind/bert-base-portuguese-cased}"
PRECISION="${PRECISION:-fp32}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

required_files=(train.jsonl test.jsonl labels.json validation_ids.json)
for filename in "${required_files[@]}"; do
  if [[ ! -f "${DATASET_DIR}/${filename}" ]]; then
    echo "Missing required private-data file: ${DATASET_DIR}/${filename}" >&2
    echo "See DATA_AVAILABILITY.md for the expected schema." >&2
    exit 2
  fi
done

extra_args=()
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  extra_args+=(--local-files-only)
fi

echo "Running six fixed-15 BERTimbau experiments"
echo "Dataset: ${DATASET_DIR}"
echo "Model:   ${MODEL_NAME_OR_PATH}"
echo "Output:  ${OUTPUT_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/src/run_matrix.py" \
  --clean-dir "${DATASET_DIR}" \
  --output-root "${OUTPUT_DIR}" \
  --matrix-csv "${ROOT_DIR}/configs/experiment_matrix.csv" \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --conditions baseline,weight_only \
  --precision "${PRECISION}" \
  --max-epochs 15 \
  --patience 0 \
  --grad-accum 2 \
  --save-total-limit 2 \
  "${extra_args[@]}" \
  "$@"
