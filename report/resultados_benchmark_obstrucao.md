# Benchmark de VLMs abertos na detecção de obstrução: resultados

*Projeto: benchmark de VLMs abertos no ViDDAR (Xiu et al., IEEE TVCG 2025). Execução completa de
17/11/2026.*

Este relatório apresenta os resultados da detecção de **ataques de obstrução** do ViDDAR com cinco VLMs
abertos, os mesmos do benchmark de manipulação
([resultados_benchmark_manipulacao.md](resultados_benchmark_manipulacao.md)). O pipeline é o replicado
do artigo, sem variantes de prompt. O que muda é só o VLM.

O objetivo é medir **como a escolha do VLM afeta o detector** e **de onde vêm os falsos positivos**.

Os números vêm de `results/obstruction/<modelo>/<método>/` e podem ser recalculados sem chamar os modelos
pelo notebook [notebooks/benchmark_obstrucao.ipynb](../notebooks/benchmark_obstrucao.ipynb), que traz
também as tabelas completas e os exemplos.

---

## Resumo

- **O pipeline do ViDDAR é robusto à troca do VLM:** todos os modelos ficam entre 84,6% e 91,8% de
  acurácia.
  - Os três melhores (Qwen3-VL-4B, Qwen3-VL-8B e Gemma 4 12B) empatam entre si e chegam ao nível do
    GPT-4o do artigo (92,2%).
  - O 4B e o 8B empatam também com o limite superior do pipeline, que recebe o nome verdadeiro do
    objeto.
- **Separar reconhecimento (VLM) e decisão (geometria) é o que faz o detector funcionar.** Quando o VLM
  decide sozinho (end-to-end), a acurácia cai para 45% a 82%, e cada modelo tem um viés próprio.
- **Os falsos positivos são raros (FPR de 1,9% a 6,8%) e todos de reconhecimento.** Aparecem em cenas
  com mais de um objeto relevante, quando o VLM escolhe outro objeto que o conteúdo virtual cobre.
- **Os falsos negativos são o erro dominante (FNR de 13,8% a 24,8%).** A maioria tem cobertura logo
  abaixo do limiar α, e o pipeline erra esses casos mesmo com o nome correto do objeto.
- **O prompt do objeto-chave importa.** Sem a orientação de segurança, quatro dos cinco modelos perdem de
  5 a 10 pontos.

---

## 1. Configuração

### 1.1 Tarefa e dados

- **Entrada:** pares (imagem crua I_r, imagem aumentada I_a).
- **Definição de obstrução:** o conteúdo virtual cobre um objeto real importante, o *objeto-chave*
  (placa, extintor, tomada, faca...).
- **Tamanho:** 306 amostras de 23 classes de objeto, sendo 145 obstruídas (rótulo 1) e 161 não
  obstruídas (rótulo 0). As classes têm de 3 a 29 amostras.

### 1.2 Pipeline do ViDDAR e linhas de base (Tabela 1 do artigo)

O pipeline do ViDDAR (§4.1 e §5.1 do artigo, com os prompts literais):

1. o VLM recebe I_r e nomeia o objeto-chave;
2. o Grounding DINO localiza esse nome em I_r e o SAM recorta a máscara do objeto, m_k;
3. a máscara do conteúdo virtual, m_c, vem da diferença de pixels entre I_r e I_a;
4. há obstrução se |m_k ∩ m_c| / |m_k| ≥ α, com α = 0,25 (Eq. 4).

O VLM participa só do passo 1. Os métodos comparados:

| Método | O que faz | Usa VLM |
|---|---|---|
| `viddar` | o pipeline acima | sim, só no passo 1 |
| `end_to_end` | o VLM nomeia o objeto e decide sozinho se ele está obstruído (vê I_r e I_a) | sim, decide tudo |
| `underdetailed` | como o ViDDAR, com prompt mínimo: sem papel de especialista, sem menção a segurança, sem exemplos | sim |
| `greedy` | o VLM lista vários objetos; há obstrução se qualquer um passar de α | sim |
| `prior` | o pipeline recebe o **nome verdadeiro** do objeto: limite superior | não |
| `saliency`, `canny` | linhas de base clássicas de visão computacional | não |

### 1.3 Modelos

Os mesmos do benchmark de manipulação: Qwen3-VL-2B e 4B (bf16), Qwen3-VL-8B, LLaVA-Next-8B (o modelo do
artigo) e Gemma 4 12B, os três em 4 bits (NF4).

A execução usou uma GPU RTX 3060 de 12 GB, com decodificação greedy (temperatura 0).

### 1.4 Métricas

- **Reconhecimento do objeto-chave:** dois critérios.
  - **Léxico:** a frase do VLM nomeia a classe certa. É permissivo: "black tv stand" conta como TV.
  - **Com máscara (IoU ≥ 0,5):** a máscara recortada a partir da frase cobre o objeto verdadeiro. É o
    critério usado nas conclusões.
- **mIoU:** IoU médio entre a máscara prevista do objeto e a verdadeira.
- **Detecção:** acurácia (acertos / 306), precisão, recall, FPR (não obstruídas marcadas / 161) e FNR
  (obstruídas perdidas / 145).
- **Comparações:** teste de McNemar exato nas mesmas 306 amostras; uma diferença só conta com p < 0,05.
- **Atribuição de erros:** cada erro do ViDDAR é atribuído à etapa que falhou:
  - **nada detectado:** o Grounding DINO não encontrou o nome;
  - **objeto errado:** IoU < 0,5 com o objeto verdadeiro;
  - **objeto certo, razão do lado errado de α.**

  Para cada erro, verifica-se também se o `prior` erra a mesma amostra. Se o `prior` acerta, o erro é do
  VLM; se também erra, o problema está depois do VLM.

---

## 2. Resultado geral

### 2.1 Tabela 1 com os cinco modelos

Valores em %.

| Modelo | Método | Objeto-chave (léxico) | Objeto-chave (IoU ≥ 0,5) | mIoU | Acurácia | Precisão | Recall | FPR | FNR |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-2B | viddar | 65,7 | 72,5 | 66,2 | 84,6 | 90,8 | 75,2 | 6,8 | 24,8 |
| Qwen3-VL-4B | viddar | 83,0 | **89,9** | **80,6** | 91,5 | 95,4 | **86,2** | 3,7 | **13,8** |
| Qwen3-VL-8B | viddar | 80,4 | 85,0 | 77,1 | **91,8** | **97,6** | 84,8 | **1,9** | 15,2 |
| LLaVA-Next-8B | viddar | 70,3 | 76,1 | 68,8 | 88,2 | 91,6 | 82,8 | 6,8 | 17,2 |
| Gemma 4 12B | viddar | 80,7 | 88,6 | **80,6** | 90,5 | 97,5 | 82,1 | **1,9** | 17,9 |
| Qwen3-VL-2B | end_to_end | 65,7 | – | – | 45,1 | 45,6 | 81,4 | 87,6 | 18,6 |
| Qwen3-VL-4B | end_to_end | 83,0 | – | – | 72,9 | 73,5 | 66,9 | 21,7 | 33,1 |
| Qwen3-VL-8B | end_to_end | 80,4 | – | – | 73,5 | 67,6 | 84,8 | 36,6 | 15,2 |
| LLaVA-Next-8B | end_to_end | 70,3 | – | – | 51,3 | 42,9 | 8,3 | 9,9 | 91,7 |
| Gemma 4 12B | end_to_end | 80,7 | – | – | **81,7** | 80,7 | 80,7 | 17,4 | 19,3 |
| Qwen3-VL-2B | underdetailed | 72,5 | 70,6 | 63,8 | 82,7 | 87,1 | 74,5 | 9,9 | 25,5 |
| Qwen3-VL-4B | underdetailed | 75,5 | 76,1 | 69,0 | 85,6 | 89,8 | 78,6 | 8,1 | 21,4 |
| Qwen3-VL-8B | underdetailed | 68,3 | 73,2 | 66,2 | 82,7 | 85,9 | 75,9 | 11,2 | 24,1 |
| LLaVA-Next-8B | underdetailed | 68,6 | 69,3 | 62,6 | 82,7 | 87,7 | 73,8 | 9,3 | 26,2 |
| Gemma 4 12B | underdetailed | 57,5 | 62,1 | 56,4 | 80,1 | 83,3 | 72,4 | 13,0 | 27,6 |
| Qwen3-VL-2B | greedy | 79,4 | 81,4 | 74,4 | 88,9 | 94,4 | 81,4 | 4,3 | 18,6 |
| Qwen3-VL-4B | greedy | 84,6 | 89,5 | 80,4 | 90,5 | 95,3 | 84,1 | 3,7 | 15,9 |
| Qwen3-VL-8B | greedy | 81,7 | 86,6 | 78,4 | 92,2 | 98,4 | 84,8 | 1,2 | 15,2 |
| LLaVA-Next-8B | greedy | 68,0 | 77,8 | 70,2 | 87,3 | 92,1 | 80,0 | 6,2 | 20,0 |
| Gemma 4 12B | greedy | 79,7 | 87,9 | 79,9 | 90,5 | 97,5 | 82,1 | 1,9 | 17,9 |
| – | prior | – | 94,4 | 85,1 | 93,8 | 99,2 | 87,6 | 0,6 | 12,4 |
| – | saliency | – | – | – | 52,3 | 49,7 | 53,1 | 48,4 | 46,9 |
| – | canny | – | – | – | 49,7 | 47,1 | 49,7 | 50,3 | 50,3 |

No `greedy`, o mIoU usa a união das máscaras de todos os objetos listados.

Referência do artigo (Tabela 1, acurácia):

| Método | GPT-4o | LLaVA-Next-8B |
|---|---|---|
| viddar | 92,2% | 89,2% |
| end_to_end | 81,0% | 55,2% |
| underdetailed | 86,9% | 81,4% |
| greedy | 88,9% | 85,6% |
| prior / saliency / Canny | 93,1% / 51,6% / 52,0% | |

**O que a tabela mostra:**

- **A replicação confere com o artigo.**

  | Método | Nosso resultado | Artigo |
  |---|---|---|
  | LLaVA, viddar | 88,2% | 89,2% |
  | LLaVA, end_to_end | 51,3% | 55,2% |
  | prior | 93,8% | 93,1% |
  | saliência | 52,3% | 51,6% |
  | Canny | 49,7% | 52,0% |

- **Com o ViDDAR, 4B, 8B e Gemma ficam a 0,4–1,7 ponto do GPT-4o do artigo.**
- **O detector erra mais por omissão do que por alarme falso:** precisão de 90,8% a 97,6% contra recall
  de 75,2% a 86,2%.

### 2.2 ViDDAR × end-to-end

| Modelo | ViDDAR | End-to-end | Diferença | Só ViDDAR acerta / só end-to-end acerta | p |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 84,6 | 45,1 | +39,5 | 152 / 31 | < 0,001 |
| Qwen3-VL-4B | 91,5 | 72,9 | +18,6 | 74 / 17 | < 0,001 |
| Qwen3-VL-8B | 91,8 | 73,5 | +18,3 | 68 / 12 | < 0,001 |
| LLaVA-Next-8B | 88,2 | 51,3 | +36,9 | 126 / 13 | < 0,001 |
| Gemma 4 12B | 90,5 | 81,7 | +8,8 | 43 / 16 | 0,001 |

- **A tese central do artigo se mantém nos cinco modelos.** A vantagem do ViDDAR é maior nos modelos mais
  fracos.
- **No end-to-end, cada modelo tem o próprio viés:**
  - o 2B diz "obstruído" em 259 de 306 amostras (FPR de 87,6%);
  - o LLaVA diz "obstruído" em só 28 (recall de 8,3%);
  - o 8B tende ao "sim" (FPR de 36,6%), e o 4B ao "não" (FNR de 33,1%).
- **O Gemma é o único equilibrado:** diz "sim" em 145 amostras, exatamente o número de obstruídas. Seu
  end-to-end (81,7%) iguala o do GPT-4o no artigo (81,0%).

### 2.3 Comparação entre modelos

O teste de McNemar foi aplicado entre os modelos, nas mesmas 306 amostras.

**Com o ViDDAR:**

| Comparação | Acurácias | Só o 1º acerta / só o 2º acerta | p | Resultado |
|---|---|---|---|---|
| 4B × 8B | 91,5 × 91,8 | 6 / 7 | 1,00 | empate |
| 4B × Gemma | 91,5 × 90,5 | 11 / 8 | 0,65 | empate |
| 8B × Gemma | 91,8 × 90,5 | 8 / 4 | 0,39 | empate |
| 4B × `prior` | 91,5 × 93,8 | 2 / 9 | 0,065 | empate (n.s.) |
| 8B × `prior` | 91,8 × 93,8 | 3 / 9 | 0,15 | empate (n.s.) |
| Gemma × `prior` | 90,5 × 93,8 | 2 / 12 | 0,013 | `prior` melhor |
| LLaVA × `prior` | 88,2 × 93,8 | 4 / 21 | 0,001 | `prior` melhor |
| LLaVA × 4B / 8B / Gemma | 88,2 × 90,5–91,8 | – | 0,052–0,28 | empate (n.s.) |
| 2B × 4B / 8B / Gemma / `prior` | 84,6 × 90,5–93,8 | – | ≤ 0,001 | o 2B perde |

- **4B, 8B e Gemma empatam.** O 4B e o 8B não se distinguem do `prior`: com eles, o reconhecimento do
  objeto-chave deixa de ser o gargalo do pipeline.
- **Dentro da família Qwen, o tamanho ajuda do 2B para o 4B (+6,9 p.p.), mas não do 4B para o 8B.** O 8B
  reconhece um pouco menos objetos que o 4B (85,0% × 89,9%), mas erra menos nos pares não obstruídos (3 FP
  contra 6).
- **No end-to-end, o Gemma supera todos (p ≤ 0,004).** O 4B e o 8B empatam (p = 0,91); o LLaVA e o 2B
  empatam no fim da tabela (p = 0,24).

### 2.4 Prompt do objeto-chave: `greedy` e `underdetailed`

| Modelo | ViDDAR | Greedy | p | Underdetailed | p |
|---|---|---|---|---|---|
| Qwen3-VL-2B | 84,6 | **88,9** | 0,007 | 82,7 | 0,46 |
| Qwen3-VL-4B | 91,5 | 90,5 | 0,38 | 85,6 | 0,001 |
| Qwen3-VL-8B | 91,8 | 92,2 | 1,00 | 82,7 | < 0,001 |
| LLaVA-Next-8B | 88,2 | 87,3 | 0,58 | 82,7 | 0,012 |
| Gemma 4 12B | 90,5 | 90,5 | 1,00 | 80,1 | < 0,001 |

- **O `greedy` empata com o ViDDAR em quase todos os modelos.** Só o 2B melhora: listar vários objetos
  compensa a má escolha de um único.
- **O `underdetailed` piora 4 dos 5 modelos, de 5,5 a 10,4 pontos,** como no artigo (GPT-4o: 86,9% ×
  92,2%).
- **Sem a orientação de segurança, os modelos nomeiam objetos salientes, mas irrelevantes.** "door" é a
  resposta mais frequente do Gemma (29 vezes) e do 2B (27). O reconhecimento do Gemma cai de 88,6% para
  62,1%.
- **O papel e os exemplos do prompt do artigo dizem *que tipo* de objeto importa,** e é isso que define o
  objeto-chave.

---

## 3. De onde vêm os erros do ViDDAR

### 3.1 Erros por etapa do pipeline

Cada célula traz o número de erros. Entre parênteses, quantas dessas amostras o `prior` também erra.

| Erro | Etapa | 2B | 4B | 8B | LLaVA | Gemma |
|---|---|---|---|---|---|---|
| FP (de 161) | objeto errado | 11 (0) | 6 (0) | 3 (0) | 11 (0) | 3 (0) |
| FN (de 145) | nada detectado | 3 (0) | 0 | 0 | 0 | 0 |
| FN (de 145) | objeto errado | 17 (2) | 5 (3) | 7 (2) | 15 (6) | 12 (4) |
| FN (de 145) | objeto certo, razão abaixo de α | 16 (14) | 15 (14) | 15 (14) | 10 (9) | 14 (13) |

O `prior` tem 1 FP e 18 FN, dos quais 16 são de limiar.

### 3.2 Falsos positivos: escolha de outro objeto em cenas ambíguas

**Todos os FP do ViDDAR são de reconhecimento,** e o `prior` acerta todas essas amostras. Em cada caso, o
VLM nomeia um objeto diferente do anotado, e o conteúdo virtual cobre esse outro objeto. Os casos se
concentram em poucas classes:

| Classe | Modelos com FP | O que o VLM nomeou |
|---|---|---|
| *warning sign* | 4 de 5 (até 4 dos 8 não obstruídos) | "black door with sign" (a porta onde está a placa), "yellow safety bollard", "yellow pole" |
| *boiling pot* | 4 de 5 | "glowing red burner", "hot stove burner", "black stove", "red heating element" |
| *electrical outlet* | 2B e Gemma | "black power adapter", "hand sanitizer bottle", "red box" |

- **Fora dessas três classes, 4B, 8B e Gemma têm só 2 FP no total.**
- **No 2B, 5 dos 11 FP vêm de "red box",** cópia do exemplo do prompt ("such as 'red box'"). É o único
  modelo que copia: "red box" é a resposta mais frequente dele (36 vezes), e essa classe não existe no
  dataset.

**A leitura para o estudo de falsos positivos:** os FP da obstrução não vêm de erro de percepção visual
nem da geometria. Vêm de **qual objeto importa**. Em cenas com mais de um objeto relevante (a porta e a
placa, a panela e o queimador), o VLM escolhe um candidato razoável diferente do anotado, e esse candidato
está coberto pelo conteúdo. São exatamente os cenários ambíguos em que o rótulo depende de uma escolha
que o prompt não especifica.

### 3.3 Falsos negativos: cobertura logo abaixo de α

- **Limiar (10 a 16 por modelo):** o objeto certo foi recortado, mas a cobertura ficou abaixo de 25%.
  - A razão desses casos fica entre 0,06 e 0,249, com mediana de cerca de 0,21.
  - O `prior` erra 9 a 14 deles, e 7 se repetem nos cinco modelos (6 também no `prior`).
  - É um limite do limiar, da máscara m_c ou do rótulo, não do VLM.
- **Objeto errado (5 a 17 por modelo):**
  - A maioria é erro do VLM. Exemplos: "orange spray bottle" em vez da TV; "plugged-in hair dryer" em
    vez da tomada.
  - Alguns são do detector. Nas amostras de *traffic light*, todos os modelos dizem "red traffic light",
    mas o semáforo segmentado não é o anotado (IoU < 0,1, inclusive no `prior`).

### 3.4 Sensibilidade ao limiar α

Acurácia do ViDDAR (%) para alguns valores de α. A curva completa está no notebook.

| α | 2B | 4B | 8B | LLaVA | Gemma | `prior` |
|---|---|---|---|---|---|---|
| 0,10 | 84,0 | 88,6 | 88,6 | 85,3 | 87,6 | 91,2 |
| 0,15 | 85,9 | 91,8 | 91,5 | 87,9 | 90,5 | 94,1 |
| **0,20** | **86,3** | **93,5** | **93,5** | **89,2** | **92,5** | **95,4** |
| 0,25 (artigo) | 84,6 | 91,5 | 91,8 | 88,2 | 90,5 | 93,8 |
| 0,30 | 82,0 | 87,9 | 87,6 | 83,7 | 86,9 | 89,9 |
| 0,40 | 77,5 | 82,0 | 83,0 | 79,1 | 81,0 | 83,0 |

- **Todas as curvas têm o pico em α = 0,20, inclusive a do `prior`.** O α do artigo fica um pouco acima do
  ótimo, coerente com os FN de limiar da Seção 3.3.
- **Explicação provável:** as nossas máscaras m_c, derivadas por diferença de pixels, são um pouco menores
  que as dos autores. O dataset publica só 32 máscaras, sem mapeamento.
- **Mantivemos α = 0,25,** porque escolher o limiar no próprio conjunto de teste seria otimista.
- **A ordem dos modelos e a distância até o `prior` se mantêm em todos os α.**

---

## 4. Conclusão

1. **Com o pipeline do ViDDAR, a escolha do VLM importa pouco a partir de 4B parâmetros.**
   - Qwen3-VL-4B, Qwen3-VL-8B e Gemma 4 12B empatam entre si e chegam ao nível do GPT-4o do artigo.
   - O 4B e o 8B empatam também com o limite superior do pipeline.
   - Modelos menores (2B) ou mais antigos (LLaVA-Next) perdem de 2 a 7 pontos, por reconhecerem pior o
     objeto-chave.
2. **O que torna o detector robusto é a divisão de trabalho.**
   - O VLM só nomeia um objeto, e a geometria decide. Quando o VLM decide sozinho, aparecem os mesmos
     vieses de "sim" e "não" vistos na manipulação.
   - Na manipulação, onde o VLM decide sozinho, nenhum dos cinco modelos supera o detector trivial com o
     prompt do artigo. Na obstrução, onde só nomeia, chegam a ~91%.
   - Os melhores resultados da manipulação também vieram de tirar a decisão do VLM.
3. **Os falsos positivos da obstrução vêm da ambiguidade de qual objeto importa.**
   - São raros (1,9% a 6,8%) e todos de reconhecimento.
   - Concentram-se em cenas com mais de um objeto relevante: porta e placa, panela e queimador.
   - É um caso concreto do cenário ambíguo que o estudo de FP investiga.
4. **Os falsos negativos são o erro mais comum e, na maioria, não dependem do VLM.** São coberturas logo
   abaixo do limiar, em que o pipeline erra mesmo com o nome correto do objeto.
5. **O prompt do objeto-chave é parte essencial do método.** Sem a orientação sobre segurança, os modelos
   escolhem objetos salientes, mas irrelevantes, e perdem até 10 pontos.

---

## 5. Limitações

- **Tamanho das classes:** 306 amostras em 23 classes, algumas com só 3 a 8 amostras. As taxas por classe
  servem para localizar casos, não para comparar modelos.
- **Máscaras derivadas:** as máscaras m_c vêm da diferença de pixels, e não das máscaras dos autores. Isso
  pode explicar o α ótimo diferente.
- **Atribuição de "objeto errado":** mistura erros do VLM e do detector (Grounding DINO + SAM). A
  comparação com o `prior` separa parte deles.
- **Critério léxico permissivo:** o reconhecimento léxico superestima o acerto; as conclusões usam o
  critério com máscara.
- **Quantização:** Qwen3-VL-8B, LLaVA-Next e Gemma 4 rodam em 4 bits (NF4).
- **Uma execução por configuração:** decodificação greedy, sem medir a variância entre amostras.
- **Comparação com o GPT-4o:** usa os números publicados no artigo, sem teste pareado.

---

## 6. Reprodução

Há duas formas de reproduzir:

- **Pelo notebook:** abra [notebooks/benchmark_obstrucao.ipynb](../notebooks/benchmark_obstrucao.ipynb)
  com o kernel "Python (ViDDAR .venv)" e execute todas as células. Ele roda só o que falta, usando o cache
  de respostas.
- **Pela linha de comando:**

```bash
.venv/Scripts/python scripts/run_obstruction.py --methods viddar end_to_end underdetailed greedy --vlm qwen3-vl-8b-4bit
```

**Custo das execuções novas (RTX 3060, segundos por amostra):**

| Modelo | VLM: viddar | VLM: end_to_end | VLM: underdetailed | VLM: greedy | Grounding DINO + SAM (por método) |
|---|---|---|---|---|---|
| Qwen3-VL-8B (4 bits) | 2,0 | 2,7 | 1,5 | 1,7 | 0,8 |
| Gemma 4 12B (4 bits) | 1,4 | 1,3 | 0,9 | 1,3 | 0,8 |

Cada modelo leva de 40 min (Gemma) a 1 h (Qwen3-VL-8B) para os quatro métodos.
