# Benchmark de VLMs abertos na detecção de manipulação de informação: resultados

*Projeto: benchmark de VLMs abertos no ViDDAR (Xiu et al., IEEE TVCG 2025). Execução completa de
16–17/11/2026.*

Este relatório apresenta os resultados do benchmark de detecção de manipulação de informação com cinco
VLMs abertos. O plano e as hipóteses estão em [analise_fp_fn_manipulacao.md](analise_fp_fn_manipulacao.md).
O objetivo é **caracterizar o problema**: medir como cada modelo erra e verificar se os erros vêm das
causas levantadas naquele documento. Não se propõe aqui uma solução.

Os números vêm de `results/manipulation/<modelo>/<variante>/` e podem ser recalculados sem chamar os
modelos pelo notebook [notebooks/benchmark_manipulacao.ipynb](../notebooks/benchmark_manipulacao.ipynb),
que traz também as tabelas completas e os exemplos.

---

## Resumo

- **Com o prompt do artigo, nenhum VLM aberto supera o detector trivial**, que nunca dispara e acerta
  61,4% dos pares.
  - Três modelos disparam em 90 a 100% dos pares benignos (acurácia perto de 38%).
  - Um quase nunca dispara (61,4%) e o outro fica em 55%.
- **Os modelos erram por motivos diferentes:**
  - conclusão que contradiz as próprias respostas: Qwen3-VL-2B e 8B;
  - leitura restritiva do fator I: Qwen3-VL-4B;
  - aceitação de alinhamento e estilo ruins: LLaVA-Next e Gemma 4.
- **Dois modelos não percebem os fatores:** no Qwen3-VL-2B e no LLaVA-Next, as respostas não dependem da
  imagem. Nenhum prompt testado mudou isso.
- **Nenhum modelo distingue o fator I** (deturpação da informação), que é o que define o ataque.
- **Só 3 das 25 configurações superam o detector trivial com significância.** A melhor é o
  Qwen3-VL-8B com perguntas separadas e definição de I: 75,4% de acurácia, precisão 67,4%, recall
  70,5%. O GPT-4o do artigo tem 82,5%.
- **O alinhamento geométrico (C2) mede o fator A melhor que os VLMs,** mas quase nunca melhora a
  decisão: o gargalo está nos fatores S e I.

---

## 1. Configuração

### 1.1 Tarefa e dados

- **Entrada:** pares (imagem crua I_r, imagem aumentada I_a) do dataset de manipulação do ViDDAR.
- **Tamanho:** 114 pares, sendo 44 ataques (M = 1) e 70 benignos (M = 0).
- **Rótulos:** cada par tem o rótulo de três fatores.
  - **A (alinhamento):** o conteúdo virtual está preso ao objeto real, sem separação.
  - **S (estilo):** a textura é realista e se mistura à cena.
  - **I (informação):** a combinação faz o usuário atribuir ao objeto uma função ou informação falsa.
- **Regra de ataque:** M = A ∧ S ∧ I (Eq. 7 do artigo).

Os fatores são desbalanceados:

| Fator | pares com o fator falso (= 0) | pares com o fator verdadeiro (= 1) |
|---|---|---|
| A | 34 | 80 |
| S | 27 | 87 |
| I | 11 | 103 |

### 1.2 Modelos

| Preset | Modelo | Precisão numérica |
|---|---|---|
| `qwen3-vl-2b` | Qwen/Qwen3-VL-2B-Instruct | bf16 |
| `qwen3-vl-4b` | Qwen/Qwen3-VL-4B-Instruct | bf16 |
| `qwen3-vl-8b-4bit` | Qwen/Qwen3-VL-8B-Instruct | 4 bits (NF4) |
| `llava-next-8b-4bit` | llava-hf/llama3-llava-next-8b-hf (o LLaVA do artigo) | 4 bits (NF4) |
| `gemma4-12b-4bit` | google/gemma-4-12B-it | 4 bits (NF4) |

- **Comparações que o conjunto permite:** os três Qwen mostram o efeito do tamanho dentro de uma
  família; LLaVA e Gemma trazem outras arquiteturas.
- **Configuração comum:** GPU RTX 3060 de 12 GB, decodificação greedy (temperatura 0), imagens limitadas
  a 1 Mpx, até 768 tokens de resposta.

### 1.3 Variantes de prompt (caminho A)

Cada variante testa uma hipótese sobre a origem dos erros.

| Variante | O que muda em relação ao prompt do artigo | Hipótese testada |
|---|---|---|
| `paper` | nada: é o prompt do artigo (6 perguntas, uma chamada; a decisão é a pergunta 6) | referência |
| `neutral` | "augmented image, **which may contain** virtual content" e sem "Think creatively" | H1: o prompt induz um viés de "sim" |
| `definition` | a pergunta 5 começa com a definição de I do §3.2 do artigo | H3: o modelo lê I de forma restritiva |
| `paper_decomposed` | 4 chamadas por par (objetos, depois A, S e I separados); a decisão A ∧ S ∧ I é calculada em código | H2: a conclusão se desconecta das respostas e cada resposta contamina as outras |
| `definition_decomposed` | igual a `paper_decomposed`, com a definição de I na chamada do fator I | H2 e H3 juntas |

### 1.4 Caminho C2: alinhamento geométrico

O fator A é medido em pixels, sem consultar o VLM de novo:

1. **Conteúdo virtual:** máscara obtida pela diferença de pixels entre I_a e I_r.
2. **Objeto-chave:** o nome que o VLM deu ao objeto vai para o Grounding DINO, e a caixa encontrada é
   segmentada pelo SAM.
3. **Medida:** distância normalizada entre o conteúdo e o objeto.
4. **A geométrico:** o par é considerado alinhado quando a distância fica abaixo de um limiar, escolhido
   por validação cruzada em 5 partições.
5. **Decisão híbrida:** A_geo ∧ S_VLM ∧ I_VLM.

O C2 foi rodado no prompt `paper` e na melhor variante de cada modelo.

### 1.5 Métricas e testes

- **Métricas do artigo (Tabela 2):** acurácia, precisão e recall. Somam-se a elas FPR (benignos
  marcados como ataque / 70) e FNR (ataques perdidos / 44).
- **Acurácia balanceada** = (recall + especificidade) / 2. Qualquer detector constante tem 50%; ela serve
  para comparar detectores que disparam muito com os que quase não disparam.
- **J de Youden por fator** = 1 − P(erro | fator = 0) − P(erro | fator = 1). Mede se a resposta do modelo
  sobre o fator depende da imagem: 0 = não depende; 1 = perfeita. Ao contrário da acurácia por fator, não
  premia o viés para o rótulo majoritário.
- **Teste de McNemar exato** nos mesmos 114 pares, com p < 0,05, para três comparações: variante ×
  `paper`, execução × detector trivial, e modelo × modelo.
- **Respostas ilegíveis** contam como erro. Foram no máximo 2 por execução.

---

## 2. Resultado geral

### 2.1 Todas as execuções

A coluna "vs. `paper`" mostra a diferença de acurácia em pontos percentuais e o p do teste de McNemar.
A coluna "vs. trivial" compara a execução com o detector que nunca dispara (61,4%). Os valores estão em
%.

| Modelo | Variante | Acurácia | Precisão | Recall | FPR | FNR | Acur. balanceada | vs. `paper` | vs. trivial |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | paper | 38,6 | 38,2 | 95,5 | 97,1 | 4,5 | 49,2 | referência | **abaixo** |
| | neutral | 48,2 | 42,7 | 93,2 | 78,6 | 6,8 | 56,6 | **+9,6 (p = 0,019)** | n.s. |
| | definition | 39,5 | 38,3 | 93,2 | 94,3 | 6,8 | 49,4 | +0,9 (n.s.) | **abaixo** |
| | paper_decomposed | 37,7 | 37,4 | 90,9 | 95,7 | 9,1 | 47,6 | −0,9 (n.s.) | **abaixo** |
| | definition_decomposed | 37,7 | 37,4 | 90,9 | 95,7 | 9,1 | 47,6 | −0,9 (n.s.) | **abaixo** |
| Qwen3-VL-4B | paper | 61,4 | 50,0 | 6,8 | 4,3 | 93,2 | 51,3 | referência | n.s. |
| | neutral | 61,4 | – | 0,0 | 0,0 | 100,0 | 50,0 | 0,0 (n.s.) | n.s. |
| | definition | 52,6 | 40,7 | 50,0 | 45,7 | 50,0 | 52,1 | −8,8 (n.s.) | n.s. |
| | paper_decomposed | 73,7 | 93,8 | 34,1 | 1,4 | 65,9 | 66,3 | **+12,3 (p = 0,003)** | **acima** |
| | definition_decomposed | 73,7 | 93,8 | 34,1 | 1,4 | 65,9 | 66,3 | **+12,3 (p = 0,003)** | **acima** |
| Qwen3-VL-8B | paper | 37,7 | 38,1 | 97,7 | 100,0 | 2,3 | 48,9 | referência | **abaixo** |
| | neutral | 60,5 | 49,4 | 90,9 | 58,6 | 9,1 | 66,2 | **+22,8 (p < 0,001)** | n.s. |
| | definition | 58,8 | 48,2 | 90,9 | 61,4 | 9,1 | 64,7 | **+21,1 (p < 0,001)** | n.s. |
| | paper_decomposed | 69,3 | 61,5 | 54,5 | 21,4 | 45,5 | 66,6 | **+31,6 (p < 0,001)** | n.s. |
| | definition_decomposed | **75,4** | 67,4 | 70,5 | 21,4 | 29,5 | **74,5** | **+37,7 (p < 0,001)** | **acima** |
| LLaVA-Next-8B | paper | 38,6 | 37,0 | 84,1 | 90,0 | 15,9 | 47,0 | referência | **abaixo** |
| | neutral | 53,5 | 42,6 | 59,1 | 50,0 | 40,9 | 54,5 | **+14,9 (p = 0,014)** | n.s. |
| | definition | 41,2 | 37,6 | 79,5 | 82,9 | 20,5 | 48,3 | +2,6 (n.s.) | **abaixo** |
| | paper_decomposed | 63,2 | 55,0 | 25,0 | 12,9 | 75,0 | 56,1 | **+24,6 (p = 0,003)** | n.s. |
| | definition_decomposed | 63,2 | 55,0 | 25,0 | 12,9 | 75,0 | 56,1 | **+24,6 (p = 0,003)** | n.s. |
| Gemma 4 12B | paper | 55,3 | 46,2 | 95,5 | 70,0 | 4,5 | 62,7 | referência | n.s. |
| | neutral | 59,6 | 48,2 | 61,4 | 41,4 | 38,6 | 60,0 | +4,4 (n.s.) | n.s. |
| | definition | 63,2 | 51,3 | 88,6 | 52,9 | 11,4 | 67,9 | +7,9 (n.s.) | n.s. |
| | paper_decomposed | 61,4 | – | 0,0 | 0,0 | 100,0 | 50,0 | +6,1 (n.s.) | n.s. |
| | definition_decomposed | 61,4 | – | 0,0 | 0,0 | 100,0 | 50,0 | +6,1 (n.s.) | n.s. |

Referência do artigo (Tabela 2, prompt `paper`):

| Modelo | Acurácia | Precisão | Recall |
|---|---|---|---|
| GPT-4o | 82,5% | 74,0% | 84,1% |
| LLaVA-Next-8B | 43,0% | 39,8% | 95,4% |

**O que a tabela mostra:**

- **Com o prompt do artigo, nenhum modelo passa do detector trivial.** 2B, 8B e LLaVA ficam abaixo dele
  com significância. O LLaVA (38,6%) reproduz de perto o artigo (43,0%), com o mesmo padrão de recall
  alto e precisão baixa.
- **Quase todo ganho de acurácia troca FP por FN.** O LLaVA decomposto, por exemplo, sobe para 63,2%, mas
  o recall cai de 84% para 25%.
- **A exceção é o Qwen3-VL-8B `definition_decomposed`:** a FPR cai de 100% para 21,4% mantendo recall de
  70,5%. É a única execução com precisão e recall acima de 2/3.
- **Só 3 das 25 execuções superam o detector trivial com significância:** as duas decompostas do 4B (15
  ataques detectados contra 1 alarme falso) e o 8B `definition_decomposed` (31 contra 15).
- **7 execuções ficam significativamente abaixo do trivial.** Nas outras 15 não há diferença
  significativa. O Gemma `definition`, por exemplo, detecta 39 ataques, mas dá 37 alarmes falsos.

### 2.2 Comparação entre modelos

O teste de McNemar foi aplicado entre as melhores variantes de cada modelo. Como a melhor variante foi
escolhida olhando os dados, os números são otimistas para todos.

| Comparação | Acurácias | Só o 1º acerta / só o 2º acerta | p | Resultado |
|---|---|---|---|---|
| Qwen-8B `definition_decomposed` × Qwen-4B `paper_decomposed` | 75,4 × 73,7 | 19 / 17 | 0,87 | empate |
| Qwen-8B × Gemma `definition` | 75,4 × 63,2 | 33 / 19 | 0,07 | empate (n.s.) |
| Qwen-8B × LLaVA `paper_decomposed` | 75,4 × 63,2 | 28 / 14 | 0,044 | Qwen-8B melhor |
| Qwen-4B × LLaVA | 73,7 × 63,2 | 20 / 8 | 0,036 | Qwen-4B melhor |
| Qwen-4B × Gemma | 73,7 × 63,2 | 39 / 27 | 0,18 | empate (n.s.) |
| LLaVA × Gemma | 63,2 × 63,2 | 34 / 34 | 1,00 | empate |
| Qwen-2B `neutral` × demais | 48,2 × 63,2–75,4 | – | 0,002–0,071 | o 2B perde para o 4B, o 8B e o Gemma |

**Leitura:**

- **O Qwen-8B e o Qwen-4B empatam em acurácia, mas por caminhos diferentes.**
  - O 8B é um detector equilibrado: 31 ataques detectados, 15 alarmes falsos, acurácia balanceada de
    74,5%.
  - O 4B quase não dispara: 15 ataques, 1 alarme falso, acurácia balanceada de 66,3%.
- **Nenhum modelo aberto alcança o GPT-4o (82,5%).** Sem as respostas do GPT-4o par a par, não há teste
  pareado. O IC95 do 8B, [66,8; 82,4], só encosta nesse valor.

---

## 3. Por que os modelos erram

### 3.1 Causa dominante dos erros

Cada erro foi atribuído a uma causa, pelo procedimento da Seção 1.3 de
[analise_fp_fn_manipulacao.md](analise_fp_fn_manipulacao.md):

- **conclusão:** a pergunta 6 contradiz as respostas 3–5;
- **percepção:** o modelo errou um fator específico.

| Modelo | Causa dominante no `paper` | No modo decomposto |
|---|---|---|
| Qwen3-VL-2B | conclusão: 68 dos 70 erros | os erros viram percepção; aceita A em 100% dos pares |
| Qwen3-VL-4B | rejeita I: 38 dos 44 erros, todos FN | a rejeição migra para S (23–24 FN) |
| Qwen3-VL-8B | conclusão: 44 dos 71 erros; depois S (11) e A (10) | A, S e I espalhados; com a definição, os FN por I somem (7 → 0) |
| LLaVA-Next-8B | aceita A (31 FP) e S (22 FP) | passa a rejeitar A nos ataques (28 FN) |
| Gemma 4 12B | aceita A (22 FP) e S (18 FP) | rejeita S em todos os pares: 44 FN, nenhum FP |

Não há uma causa comum, e por isso nenhuma variante funciona para todos os modelos.

### 3.2 Os modelos percebem os fatores?

J de Youden por fator. O J de I usa só 11 pares com I = 0 e é pouco preciso.

| Modelo | Variante | J A | J S | J I |
|---|---|---|---|---|
| Qwen3-VL-2B | paper | −0,03 | −0,02 | 0,00 |
| | paper_decomposed | −0,03 | 0,07 | 0,00 |
| Qwen3-VL-4B | paper | 0,09 | 0,31 | −0,30 |
| | paper_decomposed | 0,25 | 0,22 | −0,01 |
| Qwen3-VL-8B | paper | **0,48** | 0,25 | −0,01 |
| | paper_decomposed | 0,32 | **0,50** | −0,09 |
| | definition_decomposed | 0,32 | 0,50 | 0,00 |
| LLaVA-Next-8B | paper | −0,13 | −0,02 | 0,00 |
| | paper_decomposed | 0,05 | −0,06 | 0,00 |
| Gemma 4 12B | paper | 0,20 | 0,16 | **0,25** |
| | paper_decomposed | 0,35 | 0,00 | 0,21 |

(Tabela completa, com as 25 execuções, na seção 5.4 do notebook.)

- **Qwen3-VL-2B e LLaVA-Next não percebem nenhum fator.**
  - O J fica entre −0,13 e +0,09 em todos os fatores e variantes: as respostas não dependem da imagem.
  - As variantes só deslocam o viés. Por isso seus "ganhos" trocam FP por FN.
  - O LLaVA é o modelo usado no artigo, e isso é coerente com o desempenho baixo dele na Tabela 2.
- **Na família Qwen, o tamanho melhora a percepção de A e S.** O J de A vai de −0,03 (2B) a 0,09 (4B) e
  0,48 (8B).
- **Nenhum modelo distingue I.**
  - O J de I fica entre −0,30 e +0,25 nas 25 execuções.
  - Só o Gemma tem valores positivos em todas as variantes (0,07 a 0,25), com intervalos largos.
  - Como I = 1 em 103 dos 114 pares, dizer sempre "sim" a I é a estratégia que mais acerta. As variantes
    que melhoram o resultado fazem exatamente isso, e na prática o detector se reduz a A ∧ S.

### 3.3 A etapa de conclusão

No modo conjunto, a decisão do artigo é a pergunta 6: *"If you answered 'yes' in all questions 3, 4,
and 5, you must say 'yes.'"*. A tabela compara essa decisão com o E das respostas 3–5 (A ∧ S ∧ I),
calculado em código, nos modelos em que as duas divergem.

| Modelo | Variante | Q6 coerente com Q3–5 | Acurácia: Q6 → A∧S∧I | Pares a favor / contra | p |
|---|---|---|---|---|---|
| Qwen3-VL-2B | paper | 1,8% | 38,6 → 61,4 | 68 / 42 | 0,017 |
| Qwen3-VL-8B | paper | 60,5% | 37,7 → 75,4 | 44 / 1 | < 0,001 |
| Qwen3-VL-8B | neutral | 69,0% | 60,5 → **80,7** | 29 / 6 | < 0,001 |
| Qwen3-VL-8B | definition | 85,1% | 58,8 → 73,7 | 17 / 0 | < 0,001 |
| LLaVA-Next-8B | definition | 26,3% | 41,2 → 55,3 | 50 / 34 | 0,10 |
| 4B, Gemma, LLaVA (`paper` e `neutral`) | – | 88 a 100% | sem mudança relevante | – | – |

- **O Qwen3-VL-8B ignora as próprias respostas na conclusão.**
  - No `paper`, a pergunta 6 é "Yes." em 113 dos 114 pares, mesmo depois de "No" nas perguntas 3 ou 4.
    Exemplo: *"3. No. The virtual plant is positioned floating in the air directly above the hard drive,
    with a clear gap [...] 6. Yes."*
  - A percepção do 8B é razoável; quem derruba a acurácia é a conclusão.
- **No 2B, a conclusão também é incoerente, mas as respostas não salvam o resultado.** Como ele diz "não"
  a I em todos os pares, A ∧ S ∧ I dá 61,4%, que é o detector trivial.
- **O 8B com `neutral` e a regra A ∧ S ∧ I chega a 80,7%.** É a única execução do modo conjunto acima do
  trivial (p = 0,002) e a mais próxima do GPT-4o.
  - É um resultado **exploratório**: a melhor de 15 combinações (variante × regra), escolhida depois de
    ver os dados.
  - A decisão do artigo (`pred`) continua sendo a métrica principal.

---

## 4. Hipóteses

### H1. O prompt induz um viés de "sim" (variante `neutral`)

**Confirmada nos modelos que disparam demais.** A FPR caiu em todos os modelos com viés de "sim":

| Modelo | FPR: `paper` → `neutral` | Acurácia | Observação |
|---|---|---|---|
| Qwen3-VL-2B | 97% → 79% | sobe, p = 0,019 | |
| Qwen3-VL-8B | 100% → 59% | sobe, p < 0,001 | parte do ganho vem de menos erros de conclusão (44 → 29) |
| LLaVA-Next-8B | 90% → 50% | sobe, p = 0,014 | |
| Gemma 4 12B | 70% → 41% | sem ganho significativo | a FNR sobe na mesma medida (4,5% → 38,6%) |
| Qwen3-VL-4B | 4,3% → 0% | sem mudança | já quase não disparava; passa a nunca disparar |

Nos modelos que não percebem os fatores, a queda de FP vira aumento de FN.

### H2. A conclusão se desconecta das respostas e as respostas se contaminam (modo decomposto)

**Confirmada em parte.**

- **O que funcionou:** decompor elimina os erros de conclusão e dá os maiores ganhos do benchmark: 4B
  +12,3 p.p.; 8B +31,6 e +37,7; LLaVA +24,6.
- **O que aparece no lugar:** cada fator perguntado isoladamente tem o próprio viés, que varia por
  modelo.
  - O 4B e o Gemma passam a rejeitar S. O Gemma descreve o conteúdo como *"a 2D sticker placed over the
    image"* em todos os pares e vira um detector que nunca dispara.
  - O LLaVA passa a rejeitar A nos ataques.
  - O 2B não muda.

### H3. O modelo lê I de forma restritiva (variantes `definition`)

**Não confirmada como uma leitura melhor de I.**

- **No modo conjunto, os efeitos vão em direções opostas:**
  - o 4B passa a disparar mais em todos os fatores (−8,8 p.p., n.s.);
  - o LLaVA desconecta a conclusão;
  - o Gemma ganha +7,9 p.p. (n.s.);
  - no 8B, o ganho vem da conclusão (erros de conclusão 44 → 17), não de I.
- **No 8B decomposto, a definição tem efeito significativo:** ganha 6,1 p.p. sobre `paper_decomposed`
  (7 pares a favor, 0 contra, p = 0,016), e os FN por I somem (7 → 0).
  - Mas o J de I não sobe (de −0,09 para 0,00): o modelo passa a aceitar I em todos os pares.
  - O ganho vem de I = 1 ser majoritário no dataset, não de uma interpretação melhor.
- **No 2B, no 4B e no LLaVA, as duas decompostas são idênticas,** porque esses modelos já aceitavam I em
  todos os pares no modo decomposto.

### C2. O VLM julga mal o alinhamento

**Confirmada no nível do fator, não no nível da decisão.**

Comparação com dois baselines:

- **pergunta 6:** a decisão do artigo;
- **A ∧ S ∧ I do próprio VLM:** como o híbrido só troca A, é esta a comparação que isola o efeito
  geométrico.

| Execução | Acerto em A: geometria × VLM | AUC da distância | Acurácia: Q6 → híbrido | Acurácia: A∧S∧I → híbrido | p (vs A∧S∧I) |
|---|---|---|---|---|---|
| Qwen-2B paper | 0,79 × 0,70 | 0,79 | 38,6 → 61,4 | 61,4 → 61,4 | 1,00 |
| Qwen-2B neutral | 0,79 × 0,70 | 0,80 | 48,2 → 60,5 | 60,5 → 60,5 | 1,00 |
| Qwen-4B paper | 0,76 × 0,62 | 0,72 | 61,4 → 61,4 | 61,4 → 61,4 | 1,00 |
| Qwen-4B paper_decomposed | 0,76 × 0,63 | 0,72 | 73,7 → 71,1 | 73,7 → 71,1 | 0,25 |
| Qwen-8B paper | 0,80 × 0,75 | 0,76 | 37,7 → 78,1 | 75,4 → 78,1 | 0,58 |
| Qwen-8B definition_decomposed | 0,75 × 0,74 | 0,75 | 75,4 → 75,4 | 75,4 → 75,4 | 1,00 |
| LLaVA paper | 0,65 × 0,60 | 0,72 | 38,6 → 52,6 | 38,6 → **52,6** | **0,036** |
| LLaVA paper_decomposed | 0,59 × 0,41 | 0,65 | 63,2 → 62,3 | 63,2 → 62,3 | 1,00 |
| Gemma paper | 0,73 × 0,74 | 0,70 | 55,3 → 59,6 | 54,4 → 59,6 | 0,36 |
| Gemma definition | 0,75 × 0,69 | 0,72 | 63,2 → 65,8 | 61,4 → 65,8 | 0,46 |

- **A geometria julga A melhor.** Acerta mais em 9 de 10 execuções (a exceção é o Gemma `paper`,
  empatado), com AUC de 0,65 a 0,80.
- **Isolando o efeito geométrico, só o LLaVA `paper` melhora com significância,** e por troca: a FPR cai
  de 90% para 41%, a FNR sobe de 16% para 57%.
- **Os saltos do 2B (38,6 → 61,4) e do 8B (37,7 → 78,1) no `paper` não vêm da geometria.** Vêm de trocar
  a pergunta 6 pela regra A ∧ S ∧ I.
- **Por que um A melhor não ajuda: a decisão é um E.**
  - Onde S ou I já bloqueiam o ataque, A não importa: I no 2B e no 4B, S no 4B decomposto.
  - Onde o VLM aceita tudo, a geometria corta FP, mas também ataques, porque a medida erra de 20 a 40%
    dos pares.

---

## 5. Conclusão

- **O detector de manipulação do ViDDAR não funciona com os VLMs abertos testados** usando o prompt do
  artigo: todos ficam no nível do detector trivial ou abaixo dele.
- **Os falsos positivos não têm uma causa única.** Vêm de três fontes, com remédios diferentes:
  1. **viés de "sim" do prompt:** só se corrige nos modelos que percebem os fatores;
  2. **conclusão desconectada das respostas:** dá para corrigir tirando a decisão do VLM e calculando-a em
     código;
  3. **percepção fraca de A e S:** a geometria corrige A só em parte, e S continua dependendo do VLM.
- **Os únicos casos em que os FP caíram sem custar quase todos os ataques foram no Qwen3-VL-8B.** Ali se
  juntaram duas condições:
  - um modelo com percepção mínima dos fatores (J > 0 em A e S);
  - a decisão tirada da etapa de conclusão do VLM.
- **Escala ajuda:** dentro da família Qwen, o modelo maior percebe melhor A e S, mas isso só aparece na
  acurácia quando a conclusão sai do VLM.
- **O fator I segue sem ser medido de forma confiável**, tanto pelos modelos quanto pelo dataset, que tem
  só 11 pares benignos em I. É uma limitação central, porque I é o que distingue um ataque de um
  conteúdo apenas bem encaixado.

---

## 6. Limitações

- **Tamanho da amostra:** 114 pares (44 ataques), com só 11 benignos com I = 0 e 27 com S = 0. Diferenças
  de poucos pontos não são significativas.
- **Escolhas feitas depois de ver os dados:** a melhor variante de cada modelo e a regra A ∧ S ∧ I foram
  escolhidas olhando os resultados, sem conjunto de teste separado. Com 25 execuções e vários testes,
  algum p < 0,05 pode ser acaso.
- **Quantização:** Qwen3-VL-8B, LLaVA-Next e Gemma 4 rodam em 4 bits (NF4), o que pode mudar as
  respostas em relação aos modelos completos.
- **Uma execução por configuração:** decodificação greedy, sem medir a variância entre amostras.
- **Comparação com o GPT-4o:** usa os números publicados no artigo. Não temos as respostas dele par a
  par.
- **Dependências do C2:** usa o nome do objeto dado pelo VLM e a diferença de pixels, que só funciona
  com pares alinhados pixel a pixel.

---

## 7. Reprodução

Há duas formas de reproduzir:

- **Pelo notebook:** abra [notebooks/benchmark_manipulacao.ipynb](../notebooks/benchmark_manipulacao.ipynb)
  com o kernel "Python (ViDDAR .venv)" e execute todas as células. Ele roda só o que falta, usando o
  cache de respostas, e recalcula todas as tabelas.
- **Pela linha de comando,** uma execução por vez:

```bash
.venv/Scripts/python scripts/run_manipulation.py --vlm qwen3-vl-8b-4bit --prompt definition --mode decomposed
.venv/Scripts/python scripts/run_alignment.py --run-dir results/manipulation/qwen3-vl-8b-4bit/definition_decomposed
```

**Custo das execuções novas (RTX 3060, segundos por par):**

| Modelo | `paper` | `neutral` | `definition` | `paper_decomposed` | `definition_decomposed` |
|---|---|---|---|---|---|
| Qwen3-VL-8B (4 bits) | 19,5 | 18,5 | 19,1 | 38,9 | 13,5 |
| Gemma 4 12B (4 bits) | 20,6 | 19,5 | 22,7 | 36,9 | 16,8 |

- **Por que `definition_decomposed` é mais rápida:** reaproveita do cache as chamadas iguais às de
  `paper_decomposed` e só refaz a do fator I.
- **C2:** cerca de 1,1 s por par.
