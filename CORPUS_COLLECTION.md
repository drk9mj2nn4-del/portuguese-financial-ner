# Corpus collection and curation (metadata only)

This document describes how the private corpus used in the RECPAD 2026 paper was assembled. It contains no article text, archived article URLs, document identifiers, annotations, or evaluation records. The description is intended to make the methodology inspectable; it does **not** enable reconstruction of the exact corpus without access to the restricted source material.

## Starting collection

The starting corpus comprised 97 European Portuguese news articles obtained through Arquivo.pt. A doctoral researcher in linguistics manually annotated them under a scheme designed by a senior linguist. The four topics represented in the final corpus are fuel, water, cars, and electricity. The original manual collection was not produced by the augmentation search described below; its complete search history and article list are not available in this release. Two exact duplicates were removed from these 97 articles, leaving 95 manual documents.

## Additional article retrieval

To increase coverage of the under-represented water, cars, and electricity topics, the historical augmentation script searched the Arquivo.pt `textsearch` API. Its fixed search parameters were:

- Archived-capture window: `20130101000000` through `20201231235959` (`from` and `to`). This is a capture-time filter, not a guarantee about article publication dates.
- Publisher-domain filter (`siteSearch`): `jornaldenegocios.pt`, `dinheirovivo.pt`, `expresso.sapo.pt`, `dn.pt`, `publico.pt`, or `observador.pt`.
- Up to 80 returned items per query and publisher (`maxItems=80` in the final collection loop).
- Search terms (`q`), grouped by intended topic:

| Topic | Search terms |
| --- | --- |
| Water | `preço da água Portugal`; `fatura da água`; `tarifas de água`; `saneamento e águas`; `gestão de recursos hídricos`; `abastecimento de água`; `faturação de água`; `seca barragens`; `água potável`; `EPAL abastecimento` |
| Cars | `mercado automóvel Portugal`; `venda de carros`; `imposto automóvel`; `veículos elétricos Portugal` |
| Electricity | `tarifa elétrica Portugal`; `fatura da luz`; `preço da eletricidade`; `mercado de energia elétrica` |

For each returned archived page, the script used `trafilatura` to extract article text and metadata. It kept candidate bodies of 150--1,500 words, rejected missing extractions, and avoided obvious repeats using the first 100 lower-cased, trimmed characters of the body. A local `qwen2.5:14b` model then classified the article's main topic with temperature 0; only articles classified as water, cars, or electricity were accepted for augmentation. Attempted retrievals were logged locally, but the log and archived URLs are not distributed. The script's per-topic targets (25 water, 20 cars, 15 electricity) were collection stopping criteria, **not** the final topic counts. The resulting working pool contained 63 candidate articles.

Arquivo.pt's index, archived-page availability, extraction library, and local topic-classifier output can change. Re-running these queries should therefore be understood as reproducing the *collection procedure*, not as a guarantee of obtaining the same 63 documents. The retrieval heuristic above was followed by the stricter corpus-level duplicate checks below.

## Assisted annotation and final selection

A 15-document model--prompt screening selected the locally run `qwen3:14b` annotation workflow. Its extraction prompts used three manually annotated articles as in-context examples. Model-generated spans were aligned to character offsets, audited by `qwen3.5:9b`, and corrected by `qwen3:14b` over nine rounds. This left 60 Qwen-approved candidates. A separate Mistral-Nemo-12B audit agreed on 57 of the 60, but its votes were diagnostic and did not determine inclusion: one of its three rejected articles was subsequently removed as an exact duplicate, while two remained in the training corpus.

The final deterministic pass removed out-of-schema spans, checked text/offset agreement, converted discontinuous and overlapping annotations to flat NER, and deduplicated by normalised full text. Five of the 60 Qwen-approved articles were exact duplicates, leaving 55 LLM-assisted articles. Together with the 95 retained manual articles, the final corpus has 150 documents. The LLM-assisted annotations were **not** exhaustively re-annotated by a human and should not be interpreted as gold-standard labels.

## Partitioning and availability

The 150 documents were divided into 108 for fitting (53 manual, 55 LLM-assisted), 13 manual-only for validation, and 29 manual-only for final testing. The three manual in-context examples are among the 53 fitting documents and were excluded from validation and test. Exact duplicates were removed before partitioning; near-duplicate groups identified using word 6-gram Jaccard similarity were kept within one partition.

The source articles and derived annotations cannot be redistributed. This repository releases the experimental code, prompt templates with corpus-derived demonstrations removed, aggregate results, and data-format documentation, but not the corpus, archived URLs, true document identifiers, split-ID file, predictions, or model checkpoints. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the expected private local format.
