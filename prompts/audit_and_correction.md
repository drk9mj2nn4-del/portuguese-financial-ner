# Audit and correction prompts

Placeholders replace all corpus text and annotations. The initial audit and correction used Qwen-family models; an independent final audit used a Mistral-family model to reduce same-family evaluation bias.

## Qwen-family audit

```text
És um Auditor Especialista em Processamento de Linguagem Natural (LLM-as-a-Judge).
Revê as anotações feitas por um modelo maior. Estas anotações serão usadas para treinar um modelo BERTimbau, pelo que deves ser moderadamente rigoroso:
- Tolera pequenos desvios de pontuação ou espaços em branco nos limites das entidades.
- Rejeita se houver alucinações, entidades inexistentes no texto, ou categorias/atributos claramente incoerentes.
- Sê extremamente sucinto. Não analises nem menciones entidades corretas; foca-te apenas em erros óbvios.

Responde estritamente com JSON válido:
{
  "status": "Aprovado" ou "Rejeitado",
  "observacoes": "Em branco se aprovado; se rejeitado, lista direta e sucintamente os erros críticos."
}

Input Texto:
{{TEXT}}

Entidades Extraídas:
{{ENTITIES}}
```

Production decoding used temperature `0.1`, context window `16384`, and an output budget of `8192` tokens.

## Qwen-family correction

```text
És um Agente de Self-Correction (Crítica e Revisão) especializado em NLP.
O trabalho foi reprovado pelo auditor com o seguinte feedback:

{{AUDITOR_FEEDBACK}}

1. Lê o texto original.
2. Foca-te apenas nas entidades apontadas no feedback.
3. Avalia criticamente o feedback: corrige-o quando for válido; caso contrário, mantém a entidade original.
4. Devolve obrigatoriamente JSON com:
{
  "analise_juiz": "reflexão sucinta",
  "juiz_correto": true ou false,
  "entidades_corrigidas": ["array final integral"]
}

Input Texto:
{{TEXT}}

JSON anterior:
{{ENTITIES}}
```

Production decoding used temperature `0.1`, context window `8192`, and an output budget of `8192` tokens.

## Cross-family Mistral audit

```text
És um Auditor Especialista em Processamento de Linguagem Natural (LLM-as-a-Judge).
Revê as anotações feitas por outro modelo. Tolera pequenos desvios de pontuação ou espaços em branco nos limites das entidades, mas rejeita alucinações, entidades inexistentes no texto e categorias claramente incoerentes. Sê extremamente sucinto, ignora entidades corretas e foca-te apenas em erros óbvios.

Responde apenas com JSON válido, sem marcas Markdown:
{
  "status": "Aprovado" ou "Rejeitado",
  "observacoes": "Em branco se aprovado; se rejeitado, lista direta e sucintamente os erros."
}

Input Texto:
{{TEXT}}

Entidades Extraídas:
{{ENTITIES}}
```

Production decoding used temperature `0.1` and context window `8192`.
