# Assisted-annotation prompts

These are the Portuguese prompt templates used by the sequential annotation workflow. The three in-context demonstrations used during production are not reproduced because they contain corpus text; insert them at `{{FEW_SHOT_EXAMPLES}}`.

## 1. Entity scouting

```text
És um Agente Especialista em Extração de Informação NLP.
O teu único objetivo é encontrar os limites de texto (spans) e a classe principal baseada no seguinte esquema:

Extrai APENAS as seguintes classes de entidades:
- Participante
- Evento
- Tempo
- Quantificacao

Instruções estritas:
Passo 1: Usa as tags <thinking>...</thinking> para raciocinar sobre as entidades no texto.
Passo 2: Após o raciocínio, a resposta final deve ser estritamente um array JSON contendo objetos com as chaves "span" e "type". Não analises atributos nesta fase.

Exemplos:
{{FEW_SHOT_EXAMPLES}}

Input:
{{TEXT}}

Output:
```

## 2. Nominal attributes

```text
És um Agente Especialista em Extração de Informação NLP focado em atributos nominais.
Recebeste um texto e um conjunto de entidades extraídas. Analisa cada entidade Participante e Quantificacao e adiciona os atributos corretos:

Participante:
- Nucleo_Lexical: "Nome" ou "Pronome"
- Individualizacao_Dominio: "Individual" ou "Coletivo"
- Participante_Tipo_Dominio: "Pessoa", "Organizacao", "Cargo", "Profissao", "Produto" ou "Ativo_Financeiro"
- Envolvimento: "1", "2" ou "3"

Quantificacao:
- Subtipo: "Moeda", "Percentagem", "Numero", "Data" ou "Variacao"
- Quantificacao_Preco_Unidade: "Valor_Exato", "Valor_Aproximado" ou "Intervalo"
- Envolvimento: "1", "2" ou "3"

Usa <thinking>...</thinking> para um raciocínio sucinto. A resposta final deve ser estritamente o array JSON atualizado; cada entidade deve manter o "span" original e conter um dicionário "attributes".

Exemplos:
{{FEW_SHOT_EXAMPLES}}

Input Texto:
{{TEXT}}

Input Entidades Base:
{{ENTITIES}}
```

## 3. Verbal attributes

```text
És um Agente Especialista em Extração de Informação NLP focado em atributos verbais (eventos).
Recebeste um texto e um conjunto de entidades Evento. Analisa cada evento e adiciona:

- Evento_Tipo: "Estado", "Transicao", "Processo", "Atividade" ou "Acao"
- Pos: "Verbo" ou "Nome"
- Tempo_Verbal: "Pres_Ind", "Pret_Perf_Ind", "Pret_Imper_Ind", "Fut_Ind", "Condicional" ou "Infinitivo"
- Movimento: "Ascendente", "Descendente", "Estavel" ou "Oscilacao"

Usa <thinking>...</thinking> para um raciocínio sucinto. A resposta final deve ser estritamente o array JSON atualizado; cada evento deve manter o "span" original e conter um dicionário "attributes".

Input Texto:
{{TEXT}}

Input Eventos Base:
{{EVENT_ENTITIES}}
```

Production decoding used temperature `0.1`, context window `8192`, and an output budget of `8192` tokens. The flat NER experiments released here use only entity spans and the four main types; the auxiliary attributes are not model targets.
