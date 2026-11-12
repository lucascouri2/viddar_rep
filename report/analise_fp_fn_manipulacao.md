# Falsos positivos e falsos negativos na detecção de manipulação: análise e caminhos de melhoria

*Projeto: benchmark de VLMs abertos no ViDDAR (Xiu et al., IEEE TVCG 2025). Novembro de 2026.*

Este documento faz duas coisas:
1. Analisa **por que** os VLMs abertos erram na detecção de manipulação de informação, separando
   falsos positivos (FP) de falsos negativos (FN) e atribuindo cada erro a um fator do ViDDAR.
2. Detalha dois caminhos de melhoria escolhidos para o benchmark: **(A) ajustes de prompt** e
   **(C) novos componentes no pipeline**.

Todos os números da Seção 2 foram calculados a partir das execuções já feitas (prompt `paper`,
decodificação greedy), em `results/manipulation/<vlm>/paper/predictions.csv`, sem nenhuma chamada nova
ao modelo.

---

## 1. Base da análise

### 1.1 Os dados permitem atribuir erros por fator

O ViDDAR avalia cada par (imagem real, imagem com RA) por três **fatores**. Cada fator é uma condição
que o conteúdo virtual precisa satisfazer para enganar o usuário, e recebe um rótulo binário:
**1 = a condição é atendida**, **0 = não é atendida**. O significado de cada valor, a partir das
perguntas 3–5 do prompt e da Fig. 4 do artigo, é este:

| Fator | Valor 1 (condição atendida) | Valor 0 (condição não atendida) |
|---|---|---|
| **A, alinhamento** (pergunta 3) | O conteúdo virtual está encostado ou preso ao objeto-chave, sem espaço visível, como se fizesse parte dele. Ex.: uma planta virtual apoiada sobre uma caixa de som. | O conteúdo está deslocado, flutuando ou afastado do objeto. Ex.: a mesma planta flutuando acima da caixa de som. O desalinhamento denuncia que o conteúdo é virtual. |
| **S, estilo** (pergunta 4) | A textura, a cor e a iluminação são realistas, e o conteúdo se mistura à cena como um objeto real. | O conteúdo tem aparência artificial (textura pobre, cores chapadas, sem sombra) e é facilmente reconhecido como virtual. |
| **I, informação** (pergunta 5) | A combinação faz o usuário acreditar que o objeto tem uma função ou informação que não tem, ou esconde uma que tem. Ex.: no artigo, uma planta sobre uma caixa de som inteligente pode enganar o usuário sobre o que é o objeto (Fig. 4a). | A combinação não sugere nada falso sobre o objeto. Ex.: uma xícara de café sobre a caixa de som, mesmo bem alinhada e realista, não altera o que o usuário entende sobre ela (Fig. 4d). |

A manipulação só acontece quando o conteúdo é, ao mesmo tempo, bem encaixado, realista e enganoso. Por
isso o ViDDAR usa **M = A ∧ S ∧ I** (Eq. 7 do artigo): M = 1 (ataque) só se os três fatores valem 1.
Basta um fator valer 0 para o par ser benigno (M = 0). Por exemplo, um conteúdo enganoso mas
desalinhado não engana, porque o usuário percebe que ele é virtual.

O VLM responde aos três fatores nas perguntas 3, 4 e 5 e dá a conclusão na pergunta 6. O dataset de
manipulação (114 pares) traz o **rótulo verdadeiro de cada fator**, além do rótulo M. Assim, cada erro
do VLM pode ser atribuído a um fator sem anotação nova.

Composição do dataset, agrupada pela combinação dos rótulos verdadeiros:

| A | S | I | O que acontece no par | Pares | M = A ∧ S ∧ I |
|---|---|---|---|---|---|
| 1 | 1 | 1 | alinhado, realista e enganoso | 44 | **1** (ataque) |
| 0 | 1 | 1 | realista e enganoso, mas **desalinhado** | 32 | 0 (benigno) |
| 1 | 0 | 1 | alinhado e enganoso, mas com **textura artificial** | 25 | 0 (benigno) |
| 1 | 1 | 0 | alinhado e realista, mas **não sugere nada falso** | 11 | 0 (benigno) |
| 0 | 0 | 1 | enganoso, mas **desalinhado e com textura artificial** | 2 | 0 (benigno) |
| | | | **Total** | **114** (44 ataques, 70 benignos) | |

Só a primeira linha tem M = 1, porque é a única em que os três fatores valem 1. As outras quatro linhas
são os quatro tipos de par benigno do dataset, cada um definido pelo fator (ou fatores) que vale 0,
marcado em negrito. Outras combinações, como (0, 1, 0), não ocorrem no dataset.

Somando por fator, os 70 pares benignos se dividem em 34 com A = 0 (32 + 2), 27 com S = 0 (25 + 2) e 11
com I = 0. Ou seja, os pares benignos foram construídos sobretudo com **desalinhamento** ou **textura
artificial**. Só 11 pares são benignos por não sugerirem nada falso (I = 0), e por isso as conclusões
sobre I terão intervalos largos.

No restante do documento, **"aceitar" um fator** significa que o VLM respondeu "sim" àquela pergunta, e
**"rejeitar"**, que respondeu "não". Aceitar um fator cujo rótulo é 0 pode levar a um falso positivo;
rejeitar um fator cujo rótulo é 1 leva a um falso negativo.

### 1.2 Definições

- **FP:** verdade M = 0, previsão M = 1. **FPR** = FP / 70.
- **FN:** verdade M = 1, previsão diferente de 1. **FNR** = FN / 44.
- Resposta não interpretável (`None`) conta como erro, conforme o princípio 6 do AGENTS.md.
- Intervalos de confiança de 95% de Wilson (`viddar.metrics.wilson_ci`).

### 1.3 Procedimento de atribuição

**Passo 1. Classificar cada erro**, nesta ordem:

| Categoria | Falso positivo (M = 0 → 1) | Falso negativo (M = 1 → 0) |
|---|---|---|
| **Ilegível** | algum fator ou a conclusão veio `None` | idem |
| **Erro de conclusão** | o E das respostas 3–5 dá 0, mas a pergunta 6 diz "sim" | o E das respostas 3–5 dá 1, mas a pergunta 6 diz "não" |
| **Erro de percepção** | coerente; aceitou ("sim") os fatores cuja verdade é 0 | coerente; rejeitou ("não") fatores cuja verdade é 1 |

No erro de percepção, registra-se o **conjunto** de fatores errados (só A, só S, A+S…).

**Passo 2. Métricas por fator:**
- P(sim | F = 0), a aceitação indevida, que gera FP;
- P(não | F = 1), a rejeição indevida, que gera FN.

Quando as duas taxas são parecidas, o modelo não discrimina aquele fator.

**Passo 3. Oráculo contrafactual.** Substitui-se a resposta do VLM pela verdade em um fator e
recalcula-se M = A ∧ S ∧ I a partir das respostas 3–5. A mudança na FPR e na FNR mede quanto do erro
aquele fator explica. É também o **teto de ganho** de qualquer intervenção nesse fator.

**Passo 4. Leitura qualitativa.** Para cada categoria relevante, leem-se as justificativas ("explain
why") de uma amostra de erros, para identificar o mecanismo.

---

## 2. Resultados

### 2.1 Taxas de erro da decisão

| Modelo | FPR | FNR | Acurácia |
|---|---|---|---|
| Qwen3-VL-2B | **97,1** [90–99] (68/70) | 4,5 [1–15] (2/44) | 38,6 |
| Qwen3-VL-4B | 4,3 [1–12] (3/70) | **93,2** [82–98] (41/44) | 61,4 |
| LLaVA-Next-8B (4 bits) | **90,0** [81–95] (63/70) | 15,9 [8–29] (7/44) | 38,6 |

A decisão recalculada pelo E das respostas 3–5 é idêntica à da pergunta 6 no Qwen3-VL-4B e no LLaVA.
No Qwen3-VL-2B ela se inverte: FPR 0% e FNR 100%.

Nenhum modelo opera num ponto útil. Dois erram quase sempre para o "sim", e o Qwen3-VL-4B, que parecia
o melhor pela FPR, **deixa passar 93% dos ataques**.

### 2.2 Respostas por fator

| Modelo | A: P(sim \| A=0) | A: P(não \| A=1) | S: P(sim \| S=0) | S: P(não \| S=1) | I: P(sim \| I=0) | I: P(não \| I=1) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 100 (34/34) | 2,5 (2/80) | 100 (27/27) | 2,3 (2/87) | 0 (0/11) | 100 (103/103) |
| Qwen3-VL-4B | 64,7 (22/34) | 26,2 (21/80) | 44,4 (12/27) | 24,1 (21/87) | 63,6 (7/11) | 66,0 (68/103) |
| LLaVA-8B | 97,1 (33/34) | 16,2 (13/80) | 88,9 (24/27) | 12,6 (11/87) | 100 (11/11) | 0 (0/103) |

### 2.3 Causas dos erros

Cada célula mostra o número de erros sobre o total de pares em que aquele erro poderia ocorrer. Assim,
os modelos ficam comparáveis entre si.

**Falsos positivos** (70 pares benignos):

| Causa | Pares em que pode ocorrer | Qwen3-VL-2B | Qwen3-VL-4B | LLaVA-8B |
|---|---|---|---|---|
| Ilegível | 70 | 0/70 | 0/70 | 0/70 |
| Erro de conclusão | 70 | **68/70** | 0/70 | 0/70 |
| Percepção: aceitou A nos pares desalinhados (só A = 0) | 32 | 0/32 | 2/32 | **31/32** |
| Percepção: aceitou S nos pares com textura artificial (só S = 0) | 25 | 0/25 | 1/25 | **22/25** |
| Percepção: aceitou I nos pares que não sugerem nada falso (só I = 0) | 11 | 0/11 | 0/11 | **8/11** |
| Percepção: aceitou A e S nos pares com A = 0 e S = 0 | 2 | 0/2 | 0/2 | 2/2 |
| **Total de FP** | **70** | **68/70 (97,1%)** | **3/70 (4,3%)** | **63/70 (90,0%)** |

**Falsos negativos** (44 ataques):

| Causa | Pares em que pode ocorrer | Qwen3-VL-2B | Qwen3-VL-4B | LLaVA-8B |
|---|---|---|---|---|
| Ilegível | 44 | 2/44 | 0/44 | 0/44 |
| Erro de conclusão | 44 | 0/44 | 0/44 | 0/44 |
| Percepção: rejeitou só I | 44 | 0/44 | **38/44** | 0/44 |
| Percepção: rejeitou A e S | 44 | 0/44 | 3/44 | **7/44** |
| Percepção: rejeitou só A ou só S | 44 | 0/44 | 0/44 | 0/44 |
| **Total de FN** | **44** | **2/44 (4,5%)** | **41/44 (93,2%)** | **7/44 (15,9%)** |

**Interpretação**

- **Os três modelos erram por mecanismos diferentes**, o que já indica que não existe uma correção
  única.
  - O Qwen3-VL-2B erra na **conclusão**.
  - O LLaVA erra na **percepção**, aceitando fatores que valem 0.
  - O Qwen3-VL-4B erra na **percepção**, rejeitando fatores verdadeiros.
- **Qwen3-VL-2B:** todos os FP são erros de conclusão. As respostas por fator chegam a estar "certas"
  para negar o ataque, porque o modelo sempre responde "não" a I, mas a pergunta 6 diz "sim" mesmo
  assim. Os poucos FN (2/44) não são acertos de raciocínio: são respostas ilegíveis. O baixo número de
  FN vem do mesmo "sim" automático que gera os FP.
- **LLaVA:** aceita quase todo fator que vale 0, qualquer que seja ele: 31/32 desalinhados, 22/25 com
  textura artificial e 8/11 que não sugerem nada falso. O problema não é um fator específico, mas um **viés geral
  de "sim"**. Como o alinhamento é o tipo de negativo mais comum no dataset, é ele que concentra a maior
  parte dos FP. Por isso o oráculo em A tem o maior efeito (Seção 2.4).
- **Qwen3-VL-4B:** quase não aceita fatores que valem 0 (3/70 FP), mas rejeita I em 38 dos 44 ataques. Os
  FN dele não vêm de alinhamento ou textura, e sim da **interpretação de I** (Seção 2.5). É um erro
  concentrado num único fator, o que o torna o candidato mais claro a ganhar com a variante A3
  (definição explícita de I).
- **Implicação para os caminhos:**
  - erros de conclusão (Qwen3-VL-2B): a decisão em código (A2) é a intervenção direta;
  - viés de "sim" na percepção (LLaVA): o prompt neutro (A1), as perguntas isoladas (A2) e, para o
    alinhamento, o componente geométrico (C2);
  - erro concentrado em I (Qwen3-VL-4B): a definição explícita de I (A3).

### 2.4 Oráculo contrafactual

FPR / FNR, em %, com M = A ∧ S ∧ I calculado a partir das respostas 3–5:

| Modelo | Sem oráculo | Oráculo em A | Oráculo em S | Oráculo em I |
|---|---|---|---|---|
| Qwen3-VL-2B | 0,0 / 100 | 0,0 / 100 | 0,0 / 100 | 84,3 / 4,5 |
| Qwen3-VL-4B | 4,3 / 93,2 | 1,4 / 93,2 | 2,9 / 93,2 | **47,1 / 6,8** |
| LLaVA-8B | 90,0 / 15,9 | **42,9** / 15,9 | 55,7 / 15,9 | 78,6 / 15,9 |

### 2.5 Interpretação por modelo

**Qwen3-VL-2B: respostas constantes e conclusão desconectada.**
- O modelo diz "sim" para A e S em praticamente todos os pares, e "não" para I em **todos**.
- Na pergunta 6, ele diz "sim" mesmo tendo respondido "não" na 5, o que contraria a própria instrução
  do prompt.
- Todos os 68 FP são erros de conclusão. Conferi nas respostas brutas: não é erro de parsing.
- Corrigir só a conclusão não resolve, porque as respostas por fator são constantes e não carregam
  informação: o modelo passaria a nunca detectar (FNR de 100%).
- O problema é de **percepção e de instrução** ao mesmo tempo.

**Qwen3-VL-4B: interpretação restritiva de I.**
- **38 dos 41 FN** vêm da rejeição de I.
- Ele identifica corretamente o conteúdo virtual e o objeto, mas entende "função falsa" como algo
  absurdo ou contraditório, e trata adições plausíveis como inofensivas:
  > *"No. The addition of a basket does not imply any false functionality… It is a plausible accessory
  > that enhances the scooter's utility."*
  >
  > *"No… The emergency exit sign is a standard, functional object… It simply adds a real-world safety
  > element that is semantically appropriate."*
- O ViDDAR define o contrário: o que torna o conteúdo enganoso é justamente ser **plausível**. O usuário
  passa a acreditar que a patinete tem uma cesta, ou que ali existe uma saída de emergência.
- O modelo raciocina sobre a definição errada.
- A rejeição de I não discrimina: P(sim | I = 1) = 34% é até menor que P(sim | I = 0) = 64%, embora esta
  última se baseie em só 11 pares.
- **A FPR baixa não é mérito.** Ela é consequência de rejeitar I. Com I correto (oráculo), a FPR sobe
  para 47%, porque o modelo também aceita 65% dos conteúdos desalinhados.
- Às vezes o modelo julga o alinhamento em relação a outro objeto:
  > *"The plant is positioned directly on the desk surface, just to the left of the speaker… There is
  > no visible gap."*

  O alinhamento deveria ser julgado em relação à caixa de som, não à mesa.

**LLaVA-8B: viés de "sim" na percepção.**
- Todos os FP são erros de percepção, principalmente em **A** (31 só em A) e **S** (22).
- I é aceito em 100% dos casos, sem nenhuma discriminação.
- O modelo **percebe** o desalinhamento, mas não usa isso no julgamento:
  > Resposta 1: *"a cup of coffee with a saucer, floating above the fan"*
  >
  > Resposta 3: *"Yes, the virtual content is accurately aligned to the fan without a significant gap."*
- O raciocínio sobre I é trivial:
  > *"may make users believe that the table has a coffee cup on it"*
- Com A correto (oráculo), a FPR cairia de 90% para 43%.
- Os 7 FN vêm da rejeição simultânea de A e S.

### 2.6 Síntese: três problemas e o que os ataca

| Problema | Onde aparece | Intervenção |
|---|---|---|
| **P1. Conclusão desconectada dos fatores** | Qwen3-VL-2B | A2 (decisão em código) |
| **P2. Viés de "sim" ou falta de discriminação em A e S** (efeito halo: um "sim" puxa os outros) | LLaVA; Qwen3-VL-4B em parte | A2 (perguntas isoladas), C1 (localizar o conteúdo), **C2 (alinhamento geométrico)** |
| **P3. Interpretação errada de I**: restritiva no Qwen3-VL-4B (gera FN), trivial no LLaVA (gera FP) | Qwen3-VL-4B, LLaVA | A1 (prompt neutro), **A3 (definição de I explícita)** |

O alinhamento (A) é o fator que mais gera FP e é o mais "geométrico" dos três. Por isso o caminho C
começa por ele: na obstrução, a decisão geométrica do ViDDAR é justamente o que protege contra os erros
do VLM.

---

## 3. Caminho A: ajustes de prompt

### 3.1 Regras gerais

- O prompt `paper` não muda: é a transcrição literal do artigo. Cada variante ganha uma chave nova em
  `prompts.MANIPULATION` e é exposta na opção `--prompt` de `scripts/run_manipulation.py`.
- A decisão `pred` (última ocorrência de "yes"/"no") continua sendo a do artigo. Decisões alternativas
  entram como colunas novas.
- A chave do cache já inclui o prompt, então variantes novas não contaminam o cache.
- **Para evitar ajustar o prompt aos dados de teste:**
  - escrever as variantes a partir das definições do artigo (§3.2), e não a partir dos erros
    observados nos pares;
  - congelar o texto antes da execução completa;
  - reportar todas as variantes executadas.
  - Se for preciso iterar, separar um subconjunto fixo de desenvolvimento de cerca de 20 pares,
    estratificado por (A, S, I), e reportar os resultados só nos demais.

### 3.2 Variantes

**A1. Prompt neutro** (`neutral`). Mesma estrutura de 6 perguntas, para o parsing continuar igual,
sem as duas instruções que puxam para o "sim":
- a afirmação de que I_a contém conteúdo virtual passa a ser condicional ("may contain virtual
  content");
- sai a instrução da pergunta 5 *"Think creatively, only say no if the combination has no specific
  semantic relation"*.

Essa variante mede quanto da FPR do LLaVA vem do prompt e não da percepção.

**A2. Prompt decomposto** (`decomposed`). Uma pergunta por chamada, com a decisão em código:
1. **Chamada 1:** pergunta qual é o conteúdo virtual e qual é o objeto-chave, cada um em até 4
   palavras.
2. **Chamadas 2 a 4:** uma por fator (A, S, I), cada uma com as duas imagens e citando o conteúdo e o
   objeto da chamada 1. A resposta deve **começar** por "yes" ou "no" e é lida com `first_yes_no`.
3. **Decisão:** `pred_decomposed = A ∧ S ∧ I`, calculada em código.

Isso elimina o erro de conclusão (P1) por construção e reduz o efeito halo (P2). O custo é de 4
chamadas por par.

**A3. Definição explícita de I** (`definition`). A pergunta 5 passa a trazer a definição do artigo: há
manipulação quando um usuário que acredita que o conteúdo é real passa a ter uma informação falsa sobre
o objeto. Isso inclui adições **plausíveis**, porque a plausibilidade é o que faz o usuário acreditar.
O exemplo do prompt deve ser genérico e não estar no dataset, para não vazar o gabarito. Esta variante
ataca P3: a rejeição de I no Qwen3-VL-4B e a aceitação trivial no LLaVA.

**A2 + A3.** Prompt decomposto com a definição de I na chamada do fator I. É a combinação candidata a
melhor ponto de operação.

### 3.3 Passos de implementação

1. Em `viddar/prompts.py`: adicionar `MANIPULATION["neutral"]` e `MANIPULATION["definition"]`, e um
   dicionário novo `MANIPULATION_DECOMPOSED` com os textos das 4 chamadas, com e sem a definição de I.
2. Em `viddar/manipulation.py`:
   - criar `run_decomposed()`, que grava as respostas brutas de cada chamada (`resp_objects`,
     `resp_A`, `resp_S`, `resp_I`), os fatores e `pred_decomposed`;
   - manter as mesmas colunas `pred_A`, `pred_S`, `pred_I`, para a análise da Seção 2 rodar sem
     mudanças;
   - ajustar `summarize()` para reportar a métrica sobre `pred_decomposed` quando existir.
3. Em `viddar/experiments.py`: adicionar um campo `mode: str = "joint"` em `ManipulationConfig`
   (valores `"joint"` ou `"decomposed"`), gravado em `metrics.json`, e a saída em
   `results/manipulation/<vlm>/<prompt>_<mode>/`.
4. Em `scripts/run_manipulation.py`: ampliar as opções de `--prompt` e adicionar `--mode`, com os
   mesmos nomes da dataclass.
5. Implementar a atribuição da Seção 1.3 como função reutilizável (por exemplo,
   `manipulation.attribute_errors(df)` e `manipulation.oracle_rates(df)`), com testes em `tests/`, e
   expor o resultado em `scripts/summarize.py`.
6. Fazer o smoke test com `--limit 3 --out results/_smoke` e ler as respostas brutas: sem `None` nos
   fatores, sem cópia de exemplos do prompt e com a resposta começando por "yes" ou "no" no modo
   decomposto.
7. Rodar as 4 variantes (A1, A2, A3, A2 + A3) nos 3 modelos atuais e nos modelos novos do benchmark.
8. Avaliar cada variante:
   - FPR e FNR com intervalo de confiança;
   - **teste de McNemar** contra o prompt `paper` nos mesmos pares, que é pareado e tem mais poder
     que o teste de Fisher aqui;
   - a atribuição da Seção 2 refeita, para ver qual problema (P1, P2, P3) cada variante resolveu.
9. Registrar as variantes e o motivo de cada uma na tabela da seção 4 do REPRODUCAO.md.

**Custo:** 114 pares por modelo e variante (456 chamadas no modo decomposto). Cada execução deve levar
da ordem de minutos a uma hora por modelo nesta GPU.

---

## 4. Caminho C: novos componentes no pipeline de manipulação

A ideia é tirar do VLM o que a geometria resolve, como o ViDDAR já faz na obstrução.

**Pré-condição verificada:** os 114 pares de manipulação são alinhados pixel a pixel. O conteúdo
virtual ocupa uma mediana de cerca de 1,5% dos pixels, e a diferença entre as imagens é limpa (os pixels com
diferença maior que 0 praticamente coincidem com os de diferença maior que 30). Portanto, a máscara do
conteúdo virtual **m_c** pode ser obtida com `viddar.data.derive_content_mask`, o mesmo código da
obstrução.

**Escopo atual (sem anotação manual):** seguimos com o **C2**. O C1 fica registrado como opção futura;
dele, só o passo 1 (gerar m_c) entra no plano, porque é pré-requisito do C2.

### 4.1 C1. Mostrar ao VLM onde está o conteúdo virtual (fora do escopo atual)

**Hipótese:** parte dos erros vem de o VLM não saber exatamente o que é virtual. Isso inclui tomar um
objeto real pelo virtual, que foi observado no Draft 2, ou julgar o alinhamento em relação ao objeto
errado.

**Passos:**
1. Gerar m_c para os 114 pares e salvar em `data/content_masks/manipulation/`. Montar uma folha de
   contato para conferir as máscaras e registrar no manifesto uma coluna `content_mask_path`.
2. **Gabarito do conteúdo** (cerca de 1 h): anotar o nome do conteúdo virtual e do objeto-chave de cada
   par numa tabela auxiliar em `data/annotations/`. O dataset reutiliza poucos conteúdos (planta, xícara,
   cesta, placa de saída, relógio, livro…).
   - Isso permite medir automaticamente se as respostas 1 e 2 do VLM estão certas.
   - É uma nova dimensão da atribuição: separa "não identificou o conteúdo ou o objeto" de "identificou,
     mas julgou errado".
3. Criar uma variante de entrada com uma **terceira imagem**: o recorte de I_a em volta de m_c, com
   margem. O prompt (`content_hint`) diz que a terceira imagem mostra o conteúdo virtual.
   - Uma alternativa seria desenhar o contorno em I_a, mas isso altera a imagem em que o VLM julga a
     textura (S). Fica como variante secundária.
4. Adicionar o campo `content_hint: bool` em `ManipulationConfig` e no script, gravado em
   `metrics.json`. A imagem extra muda o hash das imagens, então o cache não precisa ser invalidado.
5. Avaliar como no caminho A (FPR, FNR, McNemar e atribuição) e medir também a acurácia das respostas
   1 e 2 com e sem a dica.

**Custo:** baixo, uma execução por modelo. Combina com qualquer variante de prompt do caminho A.

### 4.2 C2. Alinhamento geométrico (A_geo) no lugar da resposta 3

**Hipótese:** o desalinhamento dos negativos do dataset é geométrico (conteúdo flutuando acima do
objeto ou afastado dele), e uma medida em pixels o detecta melhor que o VLM. O teto já está
quantificado: no LLaVA, acertar A levaria a FPR de 90% para 43% (oráculo em A).

**Passos:**
1. **Objeto-chave.**
   - Extrair o nome do objeto da resposta 2 já gravada (coluna `key_object`), com um extrator de
     sintagma para frases do tipo "The key object … is the black fan". Isso exige um teste novo em
     `tests/`.
   - No modo decomposto (A2), usar a resposta curta da chamada 1, que dispensa o extrator. É mais um
     motivo para fazer o caminho A antes do C2.
2. **Máscara do objeto, m_k:** Grounding DINO + SAM sobre **I_r**, onde o objeto não está coberto
   pelo conteúdo, usando `GroundedSegmenter.detect_and_segment` (top-1), como na obstrução.
   - Seguir o princípio 4: primeiro todas as chamadas ao VLM, depois `close()`, depois o segmentador.
   - Quando já houver respostas gravadas, **não é preciso chamar o VLM de novo**: o estágio
     geométrico roda sozinho, em minutos.
3. **Medidas geométricas entre m_c e m_k**, todas gravadas como valores contínuos (princípio 2):
   - `gap`: distância mínima entre os pixels de m_c e de m_k (transformada de distância), normalizada
     pela altura do conteúdo;
   - `contact`: fração da borda de m_c a até *k* px de m_k;
   - `overlap`: |m_c ∩ dilatação(m_k)| / |m_c|.
4. **Validação da medida, antes de qualquer limiar:** AUC de cada medida contra o rótulo A (80
   positivos e 34 negativos), com intervalo por bootstrap. Se nenhuma medida separar bem os casos
   (AUC baixa), o componente não se justifica, e isso também é um resultado.
5. **Decisão:** `A_geo = gap_norm ≤ τ` (ou a medida de maior AUC). O limiar τ é escolhido por
   **validação cruzada estratificada** (por exemplo, 5 partições repetidas), nunca no conjunto
   inteiro.
6. **Decisão híbrida:** `pred_hybrid = A_geo ∧ S_vlm ∧ I_vlm`, com S e I vindos do VLM (prompt
   `paper` ou a melhor variante do caminho A). Fica como coluna nova; `pred` continua sendo a do
   artigo.
7. **Atribuição dos erros do componente:** separar a falha de detecção (nenhuma caixa) dos demais
   erros. Sem gabarito do objeto-chave, "objeto errado" e "medida errada com objeto certo" não podem ser
   separados automaticamente; isso fica declarado como limitação.
8. **Implementação:**
   - funções em `viddar/manipulation.py` (ou num módulo novo `viddar/alignment.py`), com testes das
     medidas em máscaras sintéticas;
   - campos `alignment: str = "vlm"` (`"vlm"` ou `"geometric"`) e `alignment_tau` em
     `ManipulationConfig`;
   - registro no REPRODUCAO.md.

**Limitações:** a medida é um substituto 2D de um alinhamento 3D. Um objeto à frente de outro pode se
sobrepor na imagem e ainda assim "flutuar" em profundidade. Uma extensão possível, fora do prazo atual,
é usar profundidade monocular. O fator S continua com o VLM: realismo de textura não tem um substituto
geométrico confiável no prazo.

---

## 5. Ordem de execução sugerida

| Etapa | Conteúdo | GPU | Por quê nessa ordem |
|---|---|---|---|
| 1 | Atribuição (Seção 1.3) implementada e testada | nenhuma | base para avaliar tudo o que vem depois |
| 2 | Caminho A: variantes A1, A2, A3 e A2 + A3 nos 3 modelos atuais | poucas horas | atacam P1, P2 e P3, incluindo a FNR do Qwen3-VL-4B |
| 3 | Caminho A nos modelos novos do benchmark | algumas horas | comparação entre VLMs com cada prompt |
| 4 | C2: m_c (passo 1 de C1), A_geo e decisão híbrida com o S e o I da melhor variante de A | minutos (só Grounding DINO + SAM) | reaproveita as respostas gravadas e ataca a maior fonte de FP |
| 5 | Repetições da melhor configuração para estimar a variância | algumas horas | resultado final |

**Critério de sucesso:** reduzir a FPR **sem** aumentar a FNR, ou vice-versa no caso do Qwen3-VL-4B,
com diferença significativa no teste de McNemar. Um ganho só em acurácia, trocando FP por FN, não conta
como melhoria.

**Cuidados estatísticos:**
- com 114 pares, diferenças de poucos pontos não são significativas;
- os resultados sobre I se apoiam em só 11 negativos;
- todo limiar é escolhido por validação cruzada;
- todas as variantes executadas são reportadas.
