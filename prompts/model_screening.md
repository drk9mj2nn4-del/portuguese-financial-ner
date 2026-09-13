# Preliminary LLM and prompt screening

The annotation teacher was selected through a 15-document local screening. Qwen3-14B, Gemma4-12B, Mistral-Nemo-12B, and Llama3.1-8B were tested with the same three-stage extraction design wherever valid structured output could be obtained. Three manual demonstrations were used; their text is not redistributed.

The selected Portuguese production templates are reproduced in [`annotation.md`](annotation.md). The common English entity-scouting variant used in the cross-model screening was:

```text
You are an Expert NLP Information Extraction Agent.
Your only goal is to find the text boundaries (spans) and the primary class based on the following schema:

Extract ONLY the following entity classes:
- Participante
- Evento
- Tempo
- Quantificacao

Strict instructions:
Step 1: Use <thinking>...</thinking> tags to reason about the entities in the text.
Step 2: After reasoning, the final response must be strictly a JSON array containing objects with the keys "span" and "type". Do not analyse attributes at this stage.

Examples:
{{FEW_SHOT_EXAMPLES}}

Input:
{{TEXT}}

Output:
```

The corresponding nominal and verbal stages preserved the schemas and JSON contracts shown in `annotation.md`, translated to English. Screening used temperature `0.0`, context window `8192`, and the same type-consistent 50%-overlap scoring procedure for all parseable outputs. Pooled precision, recall, and F1 sum TP, FP, and FN over the four entity types. The comparison values and structured-output outcomes are recorded in [`../results/llm_screening_results.csv`](../results/llm_screening_results.csv).
