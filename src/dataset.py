"""Data loading, word-bounded sliding-window chunking, and alignment for NER."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class DocumentRecord:
    doc_id: str
    origin: str
    topic: str
    tokens: List[str]
    ner_tags: List[str]

    def __post_init__(self):
        if len(self.tokens) != len(self.ner_tags):
            raise ValueError(
                f"Document {self.doc_id}: tokens count ({len(self.tokens)}) does not "
                f"match ner_tags count ({len(self.ner_tags)})."
            )


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_index: int
    total_chunks: int
    start_word: int
    end_word: int
    own_start_word: int
    own_end_word: int
    subtoken_ids: List[int]
    attention_mask: List[int]
    subtoken_labels: List[int]  # -100 for ignored / special / continuation
    word_to_subtoken_idx: Dict[int, int]  # maps word index -> first subtoken index in chunk

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}::chunk_{self.chunk_index}"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at {path}:{line_num}: {e}") from e
    return records


def load_train_and_val_datasets(
    clean_dir: Path,
) -> Tuple[List[DocumentRecord], List[DocumentRecord], List[str]]:
    """Load train.jsonl (split into fit and val) and labels.json.

    Notice: test.jsonl is NEVER opened or accessed by this function.
    """
    clean_dir = clean_dir.resolve()
    train_path = clean_dir / "train.jsonl"
    val_ids_path = clean_dir / "validation_ids.json"
    labels_path = clean_dir / "labels.json"

    validation_ids = set(json.loads(val_ids_path.read_text(encoding="utf-8")))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    train_raw = read_jsonl(train_path)

    fit_docs: List[DocumentRecord] = []
    val_docs: List[DocumentRecord] = []

    for item in train_raw:
        doc = DocumentRecord(
            doc_id=item["doc_id"],
            origin=item.get("origin", ""),
            topic=item.get("topic", ""),
            tokens=item["tokens"],
            ner_tags=item["ner_tags"],
        )
        if doc.doc_id in validation_ids:
            val_docs.append(doc)
        else:
            fit_docs.append(doc)

    val_loaded_ids = {d.doc_id for d in val_docs}
    if val_loaded_ids != validation_ids:
        missing = validation_ids - val_loaded_ids
        raise ValueError(f"Validation IDs not found in train.jsonl: {missing}")

    return fit_docs, val_docs, labels


def load_test_dataset(clean_dir: Path) -> Tuple[List[DocumentRecord], List[str]]:
    """Load test.jsonl and labels.json independently from train data."""
    clean_dir = clean_dir.resolve()
    test_path = clean_dir / "test.jsonl"
    labels_path = clean_dir / "labels.json"

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    test_raw = read_jsonl(test_path)

    test_docs: List[DocumentRecord] = []
    for item in test_raw:
        test_docs.append(
            DocumentRecord(
                doc_id=item["doc_id"],
                origin=item.get("origin", ""),
                topic=item.get("topic", ""),
                tokens=item["tokens"],
                ner_tags=item["ner_tags"],
            )
        )

    return test_docs, labels


def load_raw_dataset(
    clean_dir: Path,
) -> Tuple[List[DocumentRecord], List[DocumentRecord], List[DocumentRecord], List[str]]:
    """Convenience loader that loads fit, val, and test together."""
    fit_docs, val_docs, labels = load_train_and_val_datasets(clean_dir)
    test_docs, _ = load_test_dataset(clean_dir)
    return fit_docs, val_docs, test_docs, labels


def chunk_document_word_bounded(
    doc: DocumentRecord,
    tokenize_word_fn: Callable[[str], List[int]],
    cls_token_id: int,
    sep_token_id: int,
    label_to_id: Dict[str, int],
    max_length: int = 512,
    overlap_subtokens: int = 128,
    is_train: bool = True,
) -> List[ChunkRecord]:
    """Slice document into overlapping chunks with strict word boundaries.

    Guarantees:
    1. Validates len(tokens) == len(ner_tags) and unknown labels raise ValueError.
    2. A word producing 0 subtokens raises ValueError.
    3. A word exceeding window capacity on its own raises ValueError.
    4. A chunk NEVER starts in the middle of a word's subtokens.
    5. Overlap calculation strictly respects the target overlap budget.
    6. In training, each word is assigned active training loss in exactly
       one chunk (where it is most centrally located). Context words from
       the overlap receive label -100 to prevent loss duplication and bias.
    7. For evaluation/inference, word_to_subtoken_idx tracks the first subtoken
       of every word present in the chunk so logits can be aggregated.
    """
    if len(doc.tokens) != len(doc.ner_tags):
        raise ValueError(
            f"Document {doc.doc_id}: tokens count ({len(doc.tokens)}) does not "
            f"match ner_tags count ({len(doc.ner_tags)})."
        )

    max_word_subtokens = max_length - 2  # reserve space for [CLS] and [SEP]
    words = doc.tokens
    tags = doc.ner_tags
    num_words = len(words)

    # Validate labels and pre-tokenize each word
    word_subtokens: List[List[int]] = []
    word_subtoken_lens: List[int] = []

    for w_idx, (w, tag) in enumerate(zip(words, tags)):
        if tag not in label_to_id:
            raise ValueError(
                f"Document {doc.doc_id}: unknown label {tag!r} at word index {w_idx} ('{w}'). "
                f"Valid labels are: {sorted(label_to_id.keys())}"
            )

        subtokens = tokenize_word_fn(w)
        if len(subtokens) == 0:
            raise ValueError(
                f"Document {doc.doc_id}: word at index {w_idx} ({w!r}) produced 0 subtokens."
            )
        if len(subtokens) > max_word_subtokens:
            raise ValueError(
                f"Document {doc.doc_id}: word at index {w_idx} ({w!r}) produced {len(subtokens)} "
                f"subtokens, exceeding maximum window capacity ({max_word_subtokens} subtokens)."
            )

        word_subtokens.append(subtokens)
        word_subtoken_lens.append(len(subtokens))

    # 1. Compute chunk spans [start_word, end_word)
    windows: List[Tuple[int, int]] = []
    start_word = 0

    while start_word < num_words:
        accumulated = 0
        end_word = start_word
        while end_word < num_words and (accumulated + word_subtoken_lens[end_word]) <= max_word_subtokens:
            accumulated += word_subtoken_lens[end_word]
            end_word += 1

        if end_word == start_word:
            # Word fits max_word_subtokens as validated above
            end_word = start_word + 1

        windows.append((start_word, end_word))
        if end_word >= num_words:
            break

        # Calculate next start_word strictly respecting word boundaries and target overlap budget
        target_overlap = min(overlap_subtokens, max_word_subtokens // 2)
        overlap_accum = 0
        next_start = end_word
        for w in range(end_word - 1, start_word, -1):
            if overlap_accum + word_subtoken_lens[w] <= target_overlap:
                overlap_accum += word_subtoken_lens[w]
                next_start = w
            else:
                break

        # If no word fits within target_overlap, use zero overlap (next_start = end_word).
        # Never open an exception for individual words larger than the budget.
        if next_start < end_word:
            if next_start <= start_word:
                next_start = start_word + 1
        else:
            next_start = end_word

        start_word = next_start

    num_chunks = len(windows)

    # 2. Compute ownership boundaries [own_start_word, own_end_word)
    ownership: List[Tuple[int, int]] = []
    for k in range(num_chunks):
        start_k, end_k = windows[k]
        if k == 0:
            own_start = 0
        else:
            prev_start, prev_end = windows[k - 1]
            own_start = (start_k + prev_end) // 2

        if k == num_chunks - 1:
            own_end = num_words
        else:
            next_start, next_end = windows[k + 1]
            own_end = (next_start + end_k) // 2

        ownership.append((own_start, own_end))

    # 3. Build ChunkRecord objects
    chunk_records: List[ChunkRecord] = []
    for k, ((start_w, end_w), (own_start_w, own_end_w)) in enumerate(zip(windows, ownership)):
        subtoken_ids: List[int] = [cls_token_id]
        subtoken_labels: List[int] = [-100]  # [CLS] is always ignored
        word_map: Dict[int, int] = {}

        for w_idx in range(start_w, end_w):
            subtokens = word_subtokens[w_idx]
            first_pos = len(subtoken_ids)
            word_map[w_idx] = first_pos

            is_owned = own_start_w <= w_idx < own_end_w
            word_label = tags[w_idx]
            label_id = label_to_id[word_label]

            for sub_i, s_id in enumerate(subtokens):
                subtoken_ids.append(s_id)
                if is_train:
                    if is_owned and sub_i == 0:
                        subtoken_labels.append(label_id)
                    else:
                        subtoken_labels.append(-100)
                else:
                    if sub_i == 0:
                        subtoken_labels.append(label_id)
                    else:
                        subtoken_labels.append(-100)

        subtoken_ids.append(sep_token_id)
        subtoken_labels.append(-100)  # [SEP] is always ignored
        attention_mask = [1] * len(subtoken_ids)

        chunk_records.append(
            ChunkRecord(
                doc_id=doc.doc_id,
                chunk_index=k,
                total_chunks=num_chunks,
                start_word=start_w,
                end_word=end_w,
                own_start_word=own_start_w,
                own_end_word=own_end_w,
                subtoken_ids=subtoken_ids,
                attention_mask=attention_mask,
                subtoken_labels=subtoken_labels,
                word_to_subtoken_idx=word_map,
            )
        )

    return chunk_records


def aggregate_word_predictions(
    doc: DocumentRecord,
    chunks: List[ChunkRecord],
    chunk_logits_list: List[List[List[float]]],
    id_to_label: Dict[int, str],
) -> List[str]:
    """Aggregate multi-chunk subtoken logits back to word-level predictions.

    For each word in the document, averages the logits of its first subtoken
    across all chunks where it appears (sliding window ensemble), then takes argmax.
    Lança erro explícito se alguma palavra ficar sem previsão.
    """
    num_words = len(doc.tokens)
    num_labels = len(id_to_label)

    accumulated_logits: List[List[float]] = [[0.0] * num_labels for _ in range(num_words)]
    counts: List[int] = [0] * num_words

    for chunk, chunk_logits in zip(chunks, chunk_logits_list):
        for w_idx, pos in chunk.word_to_subtoken_idx.items():
            if 0 <= w_idx < num_words and pos < len(chunk_logits):
                logits = chunk_logits[pos]
                for label_idx in range(min(num_labels, len(logits))):
                    accumulated_logits[w_idx][label_idx] += logits[label_idx]
                counts[w_idx] += 1

    predictions: List[str] = []
    for w_idx in range(num_words):
        if counts[w_idx] == 0:
            raise ValueError(
                f"Document {doc.doc_id}: word index {w_idx} ({doc.tokens[w_idx]!r}) "
                f"received 0 predictions across all {len(chunks)} chunks."
            )
        avg_logits = [val / counts[w_idx] for val in accumulated_logits[w_idx]]
        best_id = max(range(num_labels), key=lambda idx: avg_logits[idx])
        predictions.append(id_to_label[best_id])

    return predictions
