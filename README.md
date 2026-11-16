# ViDDAR — reprodução da avaliação com VLMs locais

Este repositório parte do repositório oficial do artigo
**ViDDAR: Vision Language Model-Based Task-Detrimental Content Detection for Augmented Reality**
(Yanming Xiu, Tim Scargill, Maria Gorlatova — IEEE VR 2025 / IEEE TVCG, vol. 31, n. 5, 2025) e
acrescenta uma **reimplementação completa do pipeline de avaliação do artigo**, rodando com modelos de
visão e linguagem (VLMs) **pequenos e locais** em uma GPU de 12 GB.

O repositório original publica o dataset e quatro notebooks de demonstração, mas não o código que gera
os resultados do artigo. Aqui esse código foi reconstruído a partir da descrição do artigo e dos
notebooks, validado contra os números publicados e preparado para ser reaproveitado em outros
datasets. O objetivo da pesquisa em andamento é investigar em quais cenários o detector produz
**falsos positivos**.

- Artigo: [IEEE Computer Society](https://www.computer.org/csdl/journal/tg/2025/05/10924575/251sSyPifN6) · [arXiv 2501.12553](https://arxiv.org/abs/2501.12553)
- Dataset original: [Google Drive](https://drive.google.com/drive/folders/1aZ-2n-_DjUDqTJuQUbqMWiohOCxcp2UO?usp=drive_link)
- Detalhes de cada decisão de implementação e análise completa dos resultados: [REPRODUCAO.md](REPRODUCAO.md)
- Recomendações para quem for desenvolver no projeto: [AGENTS.md](AGENTS.md)

## Sumário

1. [O que é o ViDDAR](#1-o-que-é-o-viddar)
2. [O que veio do repositório original e o que foi reconstruído](#2-o-que-veio-do-repositório-original-e-o-que-foi-reconstruído)
3. [Como a reimplementação funciona](#3-como-a-reimplementação-funciona)
4. [Resultados da reprodução](#4-resultados-da-reprodução)
5. [Instalação](#5-instalação)
6. [Uso](#6-uso)
7. [Usando outro dataset](#7-usando-outro-dataset)
8. [Estrutura do repositório](#8-estrutura-do-repositório)
9. [Observações sobre o dataset original](#9-observações-sobre-o-dataset-original)
10. [O que o artigo propõe e não está implementado](#10-o-que-o-artigo-propõe-e-não-está-implementado)
11. [Limitações desta reimplementação](#11-limitações-desta-reimplementação)
12. [Citação e créditos](#12-citação-e-créditos)

## 1. O que é o ViDDAR

Em realidade aumentada (RA), conteúdo virtual mal posicionado ou mal projetado pode prejudicar a
tarefa do usuário. O artigo define dois tipos de conteúdo prejudicial, a partir de um par de imagens:
a imagem crua da câmera, I_r, e a imagem aumentada que o usuário vê, I_a.

- **Ataque de obstrução:** o conteúdo virtual esconde um objeto real importante (o *objeto-chave*),
  por exemplo uma seta de navegação sobre uma placa de pare. Formalmente (Eq. 4 do artigo), há ataque
  se, para algum objeto-chave k_i com máscara m_k^i, a máscara do conteúdo virtual m_c cobre pelo
  menos uma fração α dele: |m_k^i ∩ m_c| ≥ α·|m_k^i|, com α = 0,25. Os objetos-chave não são
  predefinidos: dependem do contexto da cena.
- **Ataque de manipulação de informação:** o conteúdo virtual faz o usuário interpretar errado um
  objeto real, por exemplo uma planta virtual sobre uma caixa de som, que passa a parecer um vaso.
  O artigo modela isso com três fatores booleanos para cada par (objeto real, conteúdo virtual):
  **A** (alinhamento preciso), **S** (estilo visual semelhante ao mundo real) e **I** (a combinação
  distorce a função ou a informação do objeto). Há ataque se A ∧ S ∧ I (Eq. 7).

O sistema ViDDAR detecta os dois ataques com uma arquitetura dispositivo–borda–nuvem:

| Obstrução | Manipulação de informação |
|---|---|
| <img src="image/diagram_obstruction.png" height="210" alt="Arquitetura do ViDDAR para obstrução"/> | <img src="image/diagram_information_manipulation.png" height="210" alt="Arquitetura do ViDDAR para manipulação de informação"/> |

- **Obstrução:** um VLM na nuvem (GPT-4o ou LLaVA-Next-8b) recebe I_r e nomeia o objeto-chave. Na
  borda, o Grounding DINO localiza esse objeto, o SAM gera sua máscara, e ela é comparada, pixel a
  pixel, com a máscara do conteúdo virtual obtida da diferença entre I_r e I_a.
- **Manipulação de informação:** o VLM recebe I_r e I_a com um questionário de seis perguntas (qual
  é o conteúdo virtual, qual objeto real interage com ele, A, S, I e a conclusão). Um pós-processamento
  toma como decisão o "yes" ou "no" que aparece por último na resposta.

Resultados do artigo: 92,15% de acurácia na obstrução e 82,46% na manipulação de informação, ambos com
GPT-4o.

## 2. O que veio do repositório original e o que foi reconstruído

### Presente no repositório original (autores do artigo)

| Item | Conteúdo | Limitações encontradas |
|---|---|---|
| Dataset (Google Drive) | 306 pares para obstrução (23 classes de objeto-chave, máscaras dos objetos-chave) e 114 pares para manipulação de informação (10 combinações de conteúdo virtual e objeto real) | só 32 máscaras de conteúdo virtual, sem indicação de a qual par pertencem; ver [§9](#9-observações-sobre-o-dataset-original) |
| `obstruction_labels.csv`, `information_manipulation_labels.csv` | rótulos (objeto-chave e obstrução; A, S, I e M) | o segundo é um arquivo Excel com extensão `.csv` |
| [Related Material/Obstruction-Detection-Proposed.ipynb](Related%20Material/Obstruction-Detection-Proposed.ipynb) | demonstração do ViDDAR com GPT-4o + LangSAM | caminhos removidos; limiar sorteado entre 0,175 e 0,225 (o artigo diz 0,25); prompt diferente do artigo; não calcula reconhecimento do objeto-chave nem mIoU; reporta 94,11% (o artigo, 92,15%) |
| [Related Material/Information-Manipulation-Detection-Proposed.ipynb](Related%20Material/Information-Manipulation-Detection-Proposed.ipynb) | demonstração da detecção de manipulação com GPT-4o | caminhos e nome do arquivo de rótulos diferentes dos publicados; só calcula acurácia |
| [Related Material/*_likert_demo.ipynb](Related%20Material/) | questionários do estudo com usuários (validação dos rótulos) | sem as respostas dos participantes |
| `image/` | diagramas e exemplos usados no README original | – |

Não havia código para: os modelos LLaVA-Next-8b e outros VLMs, nenhum dos seis baselines da Tabela 1,
as métricas de reconhecimento do objeto-chave e de mIoU, precisão e recall, a geração das máscaras de
conteúdo virtual nem a consolidação das tabelas. O app Android e o servidor de borda, usados nas
medições de latência, também não foram publicados.

### Reconstruído neste repositório

Cada parte abaixo segue a metodologia descrita no artigo; as seções citadas são do artigo.

| Parte | Metodologia do ViDDAR (artigo) | Implementação aqui |
|---|---|---|
| Máscara do conteúdo virtual m_c | Sec. 3.1: "m_c pode ser extraída diretamente comparando I_r e I_a no nível de pixel" | [viddar/data.py](viddar/data.py): pixel pertence a m_c se algum canal RGB difere mais de 30, com limpeza morfológica. Gerada para os 306 pares; coincide com as 32 máscaras publicadas (IoU mediano 0,99) |
| Identificação do objeto-chave | Sec. 4.1 e 5.1.1: o VLM recebe I_r e um prompt com atribuição de papel, exemplos e formato ("You are an expert in observing the world…") | [viddar/prompts.py](viddar/prompts.py) com o prompt literal do artigo; o do notebook fica como variante |
| Detecção e segmentação | Sec. 4.1: Grounding DINO gera caixas a partir do nome do objeto; SAM ViT-B gera a máscara dentro da caixa | [viddar/grounding.py](viddar/grounding.py): Grounding DINO aberto (o 1.5 do artigo só existe via API fechada) + SAM ViT-B, com os limiares e a escolha da caixa de maior confiança do notebook original |
| Decisão de obstrução | Eq. 4 com α = 0,25 | [viddar/obstruction.py](viddar/obstruction.py): razão |m_k ∩ m_c| / |m_k| salva por amostra, decisão com α configurável e varredura de α |
| Baseline *Prior knowledge* | Sec. 5.1.1: o detector recebe o nome correto do objeto-chave (limite superior) | usa a classe do rótulo como texto do Grounding DINO |
| Baseline *End-to-end* | Sec. 5.1.1: o VLM identifica o objeto e depois decide sozinho, vendo I_r e I_a, se ele está obstruído | prompt literal do passo 2; decisão pelo último "yes"/"no" |
| Baseline *Underdetailed* | Sec. 5.1.1: mesmo pipeline com prompt mínimo, sem papel nem exemplos | prompt literal do artigo |
| Baseline *Greedy* | Sec. 5.1.1: o VLM lista todos os objetos importantes, detectados um a um | prompt do artigo com "Give any object…"; obstrução se qualquer objeto listado passar de α |
| Baselines *Saliency* e *Canny* | Sec. 5.1.1: obstrução se a média do mapa de saliência (Hou & Zhang) ou da densidade de bordas (Canny) sob m_c supera a média da imagem | resíduo espectral como no OpenCV; Canny com limiares 100/200 (não informados no artigo) |
| Métricas de obstrução | Sec. 5.1.2: reconhecimento do objeto-chave, mIoU da segmentação, acurácia da detecção | as três, mais precisão, recall, taxa de falsos positivos (FPR) e intervalos de confiança; o critério de reconhecimento não é publicado, então há dois: léxico (sinônimos da classe) e *grounded* (IoU ≥ 0,5 com a máscara verdadeira) |
| Detecção de manipulação | Sec. 4.2 e 5.2.1: VLM recebe I_r e I_a com o questionário de seis perguntas; decisão pelo "yes"/"no" mais próximo do fim | [viddar/manipulation.py](viddar/manipulation.py) com o prompt literal e a mesma regra; **além disso**, extrai as respostas às perguntas 3, 4 e 5 como previsões de A, S e I e recalcula A ∧ S ∧ I (Eq. 7) |
| Métricas de manipulação | Sec. 5.2.2: acurácia, precisão e recall | as três, mais FPR, acurácia por fator e coerência entre a conclusão e as respostas 3–5 |
| VLMs | GPT-4o (API OpenAI) e LLaVA-Next-8b (API Hugging Face) | [viddar/vlm.py](viddar/vlm.py): qualquer VLM do Hugging Face rodando localmente (Qwen3-VL-2B, Qwen3-VL-4B, LLaVA-Next-8b em 4-bit) e o GPT-4o via API, com cache das respostas |

A lista completa de decisões nos pontos em que o artigo não dá detalhes (resolução das imagens,
decodificação, o que fazer quando o objeto não é detectado etc.) está na seção 4 do
[REPRODUCAO.md](REPRODUCAO.md).

## 3. Como a reimplementação funciona

```
 manifesto CSV ──► estágio 1: VLM ──► estágio 2: Grounding DINO + SAM ──► decisão ──► métricas
 (pares I_r/I_a,    (objeto-chave ou     (só obstrução; a GPU é liberada     (Eq. 4 ou   (predictions.csv,
  rótulos, grupos)   questionário;        do VLM antes de carregar os          regra do    metrics.json,
                     respostas em cache)  modelos de visão)                    artigo)     metrics_by_group.csv)
```

- **Manifestos:** todo experimento lê um CSV com os caminhos das imagens e os rótulos. O script
  `prepare_viddar.py` gera os manifestos do dataset original. Para outro dataset, basta outro manifesto.
- **Estágios:** os métodos com VLM rodam primeiro. O VLM é descarregado da GPU e só então o Grounding
  DINO e o SAM são carregados, o que permite usar VLMs de até ~9 GB numa GPU de 12 GB.
- **Cache:** cada resposta do VLM é gravada em `results/cache/<modelo>.jsonl`, indexada pelo modelo,
  prompt, hash das imagens e parâmetros de geração. Mudar só o α ou recalcular métricas não consulta o
  modelo de novo, e a identificação do objeto-chave é compartilhada entre ViDDAR e End-to-end.
- **Saídas por execução:** `results/<tarefa>/<vlm>/<método>/` com `predictions.csv` (uma linha por
  amostra, incluindo a resposta bruta do VLM), `metrics.json` e `metrics_by_group.csv`.
  `scripts/summarize.py` junta tudo em `results/summary.md`, ao lado dos números do artigo.

## 4. Resultados da reprodução

Decodificação greedy, α = 0,25, imagens reduzidas para até 1 megapixel, RTX 3060 12 GB. O LLaVA-Next-8b
é o mesmo modelo usado no artigo, aqui quantizado em 4 bits.

**Obstrução (306 pares) — acurácia (%)**

| Método | Qwen3-VL-2B | Qwen3-VL-4B | LLaVA-Next-8b (4-bit) | Artigo: LLaVA-Next-8b | Artigo: GPT-4o |
|---|---|---|---|---|---|
| ViDDAR | 84,6 | **91,5** | 88,2 | 89,21 | 92,15 |
| End-to-end | 45,1 | 72,9 | 51,3 | 55,23 | 81,04 |
| Underdetailed | 82,7 | 85,6 | 82,7 | 81,37 | 86,92 |
| Greedy | 88,9 | 90,5 | 87,3 | 85,62 | 88,89 |
| Saliency map | 52,3 | | | | 51,63 |
| Canny edge | 49,7 | | | | 51,96 |
| Prior knowledge | 93,8 | | | | 93,14 |

**Manipulação de informação (114 pares)**

| Modelo | Acurácia | Precisão | Recall | FPR |
|---|---|---|---|---|
| Qwen3-VL-2B | 38,6 | 38,2 | 97,7 | 97,1 |
| Qwen3-VL-4B | 61,4 | 50,0 | 6,8 | 4,3 |
| LLaVA-Next-8b (4-bit) | 38,6 | 37,0 | 84,1 | 90,0 |
| Artigo: LLaVA-Next-8b | 42,98 | 39,80 | 95,35 | – |
| Artigo: GPT-4o | 82,46 | 74,00 | 84,09 | – |

- Com o mesmo VLM do artigo, todos os métodos ficam a menos de 4 pontos dos valores publicados, e os
  métodos sem VLM também coincidem. Isso valida a reimplementação.
- Um VLM de 4B rodando localmente praticamente iguala o ViDDAR com GPT-4o na obstrução.
- Na manipulação de informação, modelos com taxas de erro parecidas erram por motivos diferentes:
  o Qwen3-VL-2B contradiz as próprias respostas na conclusão, o LLaVA aceita quase toda combinação
  como enganosa e o Qwen3-VL-4B quase nunca aceita. A análise completa está na seção 7 do
  [REPRODUCAO.md](REPRODUCAO.md).

## 5. Instalação

Testado em Windows 10, Python 3.12, NVIDIA RTX 3060 12 GB (driver 560), CUDA 12.6.

**1. Dados.** As imagens não estão no repositório (são cerca de 1,8 GB); baixe-as do Drive dos autores.

1. Abra o [Google Drive do dataset](https://drive.google.com/drive/folders/1aZ-2n-_DjUDqTJuQUbqMWiohOCxcp2UO?usp=drive_link)
   e baixe a pasta inteira (menu ⋮ da pasta → **Fazer download**). O Drive entrega um ou mais
   arquivos `.zip`; se vierem vários (`...-001.zip`, `...-002.zip`), baixe todos.
2. Na raiz deste repositório, crie a pasta `ViDDAR dataset` (com espaço e "dataset" minúsculo).
3. Descompacte **todos** os `.zip` dentro de `ViDDAR dataset/`. Se vierem vários, extraia-os no mesmo
   lugar, aceitando mesclar as pastas.
4. Confira se a estrutura ficou assim (o zip já traz a pasta `ViDDAR Dataset`):

   ```
   ViDDAR dataset/
   └── ViDDAR Dataset/
       ├── obstruction_raw_img/            306 imagens
       ├── obstruction_ar_img/             306 imagens
       ├── obstruction_key_object_mask/    322 máscaras
       ├── obstruction_ar_content_mask/     32 máscaras
       ├── raw/                            115 imagens (manipulação, cruas)
       ├── confusion_ar_img/               114 imagens (manipulação, aumentadas)
       ├── obstruction_labels.csv
       └── confusion_labels.csv
   ```

   Para conferir as contagens no PowerShell, a partir da raiz do repositório:

   ```powershell
   Get-ChildItem "ViDDAR dataset\ViDDAR Dataset" -Directory | ForEach-Object { "{0,-32} {1}" -f $_.Name, (Get-ChildItem $_.FullName -File).Count }
   ```

   Se a extração criar um nível a mais (por exemplo `ViDDAR dataset/ViDDAR Dataset/ViDDAR Dataset/`),
   mova o conteúdo para cima até ficar como acima. Se preferir guardar o dataset em outro lugar, use
   `scripts/prepare_viddar.py --dataset <caminho>`.

A pasta `ViDDAR dataset/` está no `.gitignore` e nunca é enviada ao GitHub. Os arquivos gerados a
partir dela (manifestos e máscaras) são criados no passo de preparação (seção 6).

**2. Ambiente Python** (isolado; não use um ambiente conda base já cheio de pacotes):

```bash
python -m venv .venv
# Windows: .venv\Scripts\python   ·   Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
# opcional, para o preset llava-next-8b-4bit:
.venv/Scripts/python -m pip install bitsandbytes==0.50.2
# kernel do Jupyter para o notebook de experimentos:
.venv/Scripts/python -m ipykernel install --user --name viddar --display-name "Python (ViDDAR .venv)"
```

**3. Espaço em disco.** O ambiente ocupa ~5 GB. Os modelos são baixados na primeira execução para
`~/.cache/huggingface` (mude com a variável `HF_HOME`): Grounding DINO + SAM 1,3 GB, Qwen3-VL-2B 4,3 GB,
Qwen3-VL-4B 8,9 GB, LLaVA-Next-8b 16,7 GB.

**4. Verificação:**

```bash
.venv/Scripts/python -m pytest tests
```

Problemas conhecidos:

- `CERTIFICATE_VERIFY_FAILED` ao baixar modelos: acontece quando antivírus ou proxy re-assinam o
  tráfego HTTPS. O pacote `truststore`, já incluído, faz o Python usar os certificados do sistema.
  Se o erro for no próprio `pip`, use `pip install --use-feature=truststore ...`.
- O aviso sobre *symlinks* do Hugging Face no Windows é inofensivo; silencie com
  `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- No Windows, carregar modelos grandes pode fazer o `pagefile.sys` crescer vários GB no disco C:.

## 6. Uso

Há duas formas equivalentes de rodar os experimentos. As duas usam as mesmas funções
([viddar/experiments.py](viddar/experiments.py)), gravam nos mesmos lugares e compartilham o cache.

### Pelo notebook (recomendado para explorar)

Abra [notebooks/experimentos_viddar.ipynb](notebooks/experimentos_viddar.ipynb) no VS Code ou no
Jupyter, selecione o kernel **"Python (ViDDAR .venv)"** e rode as células em ordem. O notebook:

1. verifica o ambiente (GPU, espaço em disco);
2. tem uma célula de configuração (VLM, `LIMIT`, α, variante de prompt, manifestos);
3. prepara o dataset na primeira execução e mostra o relatório de validação;
4. roda a obstrução com todos os métodos e compara com a Tabela 1 do artigo;
5. mostra a acurácia em função de α, os erros por classe, as respostas mais frequentes do VLM e
   imagens dos falsos positivos e negativos (objeto-chave verdadeiro e conteúdo virtual sobrepostos);
6. roda a manipulação de informação, compara com a Tabela 2 e mostra qual fator (A, S ou I) provoca
   os erros, com exemplos e a resposta completa do VLM;
7. reabre execuções anteriores sem rodar nada de novo e junta todas as execuções lado a lado.

Com `LIMIT = 10` os resultados vão para `results/_smoke/`, sem sobrescrever as execuções completas.

O segundo notebook, [notebooks/benchmark_manipulacao.ipynb](notebooks/benchmark_manipulacao.ipynb),
implementa o estudo de falsos positivos e negativos da manipulação descrito em
[report/analise_fp_fn_manipulacao.md](report/analise_fp_fn_manipulacao.md):

- a atribuição de cada erro a um fator ou à conclusão do VLM;
- as variantes de prompt (`neutral`, `definition` e o modo decomposto), comparadas pelo teste de
  McNemar;
- o alinhamento geométrico (A_geo), que substitui o fator A do VLM por uma medida em pixels.

### Pela linha de comando (recomendado para execuções longas)

```bash
# 1) manifestos + máscaras de conteúdo virtual + relatório de validação do dataset
.venv/Scripts/python scripts/prepare_viddar.py

# 2) obstrução (Tabela 1). "all" = viddar end_to_end underdetailed greedy saliency canny prior
.venv/Scripts/python scripts/run_obstruction.py --methods all --vlm qwen3-vl-2b

# 3) manipulação de informação (Tabela 2)
.venv/Scripts/python scripts/run_manipulation.py --vlm qwen3-vl-2b

# 4) tabelas consolidadas -> results/summary.md
.venv/Scripts/python scripts/summarize.py
```

Opções principais:

| Opção | Efeito |
|---|---|
| `--vlm` | preset (`qwen3-vl-2b`, `qwen3-vl-4b`, `qwen2.5-vl-3b`, `llava-next-8b-4bit`, `gpt-4o`) ou qualquer id do Hugging Face compatível com `AutoModelForImageTextToText` |
| `--limit N` | avalia só as N primeiras amostras (teste rápido) |
| `--alpha` | limiar da Eq. 4 (padrão 0,25, como no artigo) |
| `--key-prompt notebook` / `--prompt notebook` | usa os prompts dos notebooks originais em vez dos do artigo |
| `--prompt neutral` / `--prompt definition` | variantes do prompt de manipulação para o estudo de falsos positivos (ver `viddar/prompts.py`) |
| `--mode decomposed` | manipulação com uma chamada por fator e M = A ∧ S ∧ I calculado em código |
| `--max-pixels` | resolução máxima enviada ao VLM (padrão 1.000.000 pixels por imagem) |
| `--temperature T --seed S` | amostragem estocástica, para medir a variação entre execuções |
| `--all-boxes` | usa a união de todas as caixas detectadas, em vez da de maior confiança |
| `--manifest` | avalia outro dataset (ver §7) |

Tempo aproximado na RTX 3060: obstrução ~1–2 s por imagem no estágio do VLM e ~1 s na segmentação;
manipulação ~13 s (Qwen3-VL-2B) a ~20 s (Qwen3-VL-4B, LLaVA-Next-8b) por par.

O preset `gpt-4o` usa a API da OpenAI e exige a variável `OPENAI_API_KEY`.

## 7. Usando outro dataset

Crie um manifesto CSV (caminhos relativos ao próprio CSV) e passe-o com `--manifest`.

| Tarefa | Colunas obrigatórias | Colunas opcionais |
|---|---|---|
| Obstrução | `id, raw_path, ar_path, label` | `key_object` (necessária para `prior` e para medir reconhecimento), `key_mask_path` (necessária para mIoU), `content_mask_path` (sem ela, m_c é derivada de I_r e I_a, que precisam estar alinhadas pixel a pixel), `group` |
| Manipulação | `id, raw_path, ar_path, label` | `A, S, I` (rótulos dos fatores), `group` |

A coluna `group` serve para marcar cenários (por exemplo `ambiguo` e `controle`): cada execução gera
`metrics_by_group.csv` com verdadeiros e falsos positivos, FPR e FNR por grupo. Na manipulação,
`predictions.csv` guarda `pred_A`, `pred_S`, `pred_I` e `pred_factor_rule`, o que permite atribuir
cada falso positivo a um fator ou a uma conclusão incoerente do VLM.

## 8. Estrutura do repositório

```
viddar/                 pacote Python da reimplementação
  data.py               manifestos, rótulos, máscara m_c, razão de obstrução (Eq. 4)
  prompts.py            prompts do artigo (literais) e variantes dos notebooks
  vlm.py                backends de VLM (transformers local, OpenAI), presets, cache de respostas
  grounding.py          Grounding DINO + SAM
  obstruction.py        ViDDAR e baselines da Tabela 1
  manipulation.py       detecção de manipulação (Tabela 2), extração de A, S, I e atribuição de erros
  alignment.py          alinhamento geométrico A_geo (m_c × objeto-chave segmentado)
  parsing.py            pós-processamento das respostas; critério de reconhecimento do objeto-chave
  metrics.py            métricas binárias com IC de Wilson; métricas por grupo
  experiments.py        execução de experimentos (usada pelos scripts e pelo notebook)
  reporting.py          consolidação das execuções e números de referência do artigo
  viz.py                gráficos e visualização de amostras para o notebook
notebooks/              experimentos_viddar.ipynb, benchmark_manipulacao.ipynb
scripts/                prepare_viddar.py, run_obstruction.py, run_manipulation.py, run_alignment.py, summarize.py
tests/                  testes de parsing, máscaras e métricas
data/manifests/         manifestos gerados e validation_report.json (versionados)
data/content_masks/     máscaras m_c derivadas (geradas, fora do git)
results/                saídas, logs e cache de respostas (fora do git)
Related Material/       notebooks originais dos autores (não modificados)
image/                  figuras do repositório original
REPRODUCAO.md           decisões de implementação, observações e análise completa dos resultados
AGENTS.md               recomendações de desenvolvimento
```

## 9. Observações sobre o dataset original

Verificadas pelo `prepare_viddar.py` (resultado em `data/manifests/validation_report.json`):

- `information_manipulation_labels.csv` (e `confusion_labels.csv` no Drive) é um arquivo **Excel** com
  extensão `.csv`. A coluna `confusion` é o rótulo M, e M = A ∧ S ∧ I vale para os 114 pares. A coluna
  extra `result` não é documentada e é ignorada.
- A pasta `obstruction_ar_content_mask` tem só 32 máscaras, reutilizadas entre pares, sem mapeamento
  para os 306 pares. Por isso as máscaras são derivadas das imagens.
- Com as máscaras verdadeiras dos objetos-chave, os rótulos de obstrução são melhor explicados por
  α ≈ 0,2 (98,4% de acerto) do que pelo α = 0,25 citado no artigo (94,1%).
- `raw/` tem 115 imagens para 114 pares, e há 16 máscaras de `laptop` sem par (classe fora do conjunto
  de 306).
- No README original, "style similarity (1 for high, 2 for low)" deve ser lido como 1/0: os valores no
  arquivo são 0 e 1.

## 10. O que o artigo propõe e não está implementado

Este repositório reproduz a **avaliação sobre o dataset** (Seções 5.1 e 5.2 do artigo). Os itens
abaixo são descritos ou propostos no artigo, mas não existem aqui, nem no código original publicado,
salvo indicação.

### 10.1 Sistema em tempo real (Seções 4 e 5.4)

| Item do artigo | O que o artigo descreve | Situação aqui |
|---|---|---|
| Dispositivo de RA | app Android (Unity 2022.3.28f1 + Google ARCore, Google Pixel 7 Pro) que captura I_r, sobrepõe o conteúdo virtual e envia I_r e I_a ao servidor de borda | não implementado; não publicado pelos autores |
| Mitigação da obstrução | ao detectar obstrução de um objeto-chave, o app reduz a opacidade do conteúdo virtual (Fig. 1d); o artigo também ilustra mover o conteúdo (Fig. 1c) | não implementado; aqui a saída é só a classificação |
| Aviso de manipulação | ao detectar manipulação de informação, o app mostra uma mensagem de alerta | não implementado |
| Servidor de borda | recebe I_r e I_a por HTTP, codifica em Base64 e roda detecção e segmentação a **cada quadro** | não implementado; o pipeline processa pares de imagens estáticas em lote |
| Controlador de prompts | decide quando consultar o VLM, para reduzir custo e latência; na versão do artigo o disparo é manual | não implementado; o VLM é consultado uma vez por par de imagens |
| Lista de objetos-chave | a borda mantém os nomes dos objetos-chave retornados pelo VLM e os reutiliza nos quadros seguintes, sem nova consulta | não implementado; cada par é avaliado de forma independente |
| Teste com fluxo de imagens em tempo real e latência (Tabela 3) | 20 medições por configuração, em rede de um salto (mesma rede do campus) e de seis saltos (rede residencial): 533 ms e 960 ms na obstrução; 9,62 s e 12,30 s na manipulação, com GPT-4o | não reproduzido; fora do escopo deste estudo |

### 10.2 Avaliação e dados

| Item do artigo | O que o artigo descreve | Situação aqui |
|---|---|---|
| Estudo com usuários (Seção 5.3, Fig. 11) | 20 participantes avaliam, em escala Likert de 5 pontos, 10 pares por tarefa para validar os rótulos; itens negativos são invertidos (6 − nota) | só os questionários originais em `Related Material/*_likert_demo.ipynb`; as respostas dos participantes e o código que gera a Fig. 11 não foram publicados |
| Coleta do dataset (Seção 5) | o mesmo app gera as cenas: formas geométricas aleatórias para obstrução e, para manipulação, conteúdo ancorado por rastreamento de imagem de um marcador de 15×15 cm exibido num iPad | não reproduzível; são usadas as imagens publicadas. O iPad do marcador aparece nas imagens de manipulação e já confundiu o LLaVA (§4) |
| Estatísticas do dataset (Fig. 8) | distribuição da área obstruída por rótulo e distribuição dos rótulos A, S, I e M | só os números, em `data/manifests/validation_report.json` (razão de obstrução com as máscaras verdadeiras, proporção de cada rótulo); os gráficos não foram refeitos |
| Detector Grounding DINO 1.5 | usado em todos os métodos com detecção | substituído pelo Grounding DINO aberto, pois o 1.5 só existe via API fechada; a segmentação ficou até melhor (mIoU 85,1% contra 78,8% no Prior knowledge) |
| Resultados com GPT-4o | linha principal das Tabelas 1 e 2 | o preset `gpt-4o` existe, mas não foi executado (exige chave de API e tem custo) |
| Critério de acerto do objeto-chave | métrica *key object recognition accuracy* da Tabela 1 | o artigo não diz como a resposta do VLM é comparada ao rótulo; aqui há dois critérios próprios (léxico e *grounded*), então os valores não são diretamente comparáveis |

### 10.3 Modelo formal × implementação

Estas diferenças já existem no próprio artigo e foram mantidas por fidelidade:

- **Vários objetos-chave:** as Eq. 1–4 admitem um conjunto K de n objetos-chave por cena, mas o prompt
  do ViDDAR pede um único objeto e o dataset tem um objeto-chave por imagem. Só o baseline Greedy trata
  vários objetos.
- **Vários objetos reais na manipulação:** a Eq. 7 considera todos os pares (r_i, c), mas o
  questionário pede ao VLM um único "objeto-chave" que interage com o conteúdo virtual.
- **Fatores A, S e I:** o modelo define a decisão como A ∧ S ∧ I, mas o sistema do artigo usa a
  conclusão livre do VLM (pergunta 6). Aqui as duas decisões são calculadas, e a diferença entre elas é
  justamente uma fonte de falsos positivos (§4).

### 10.4 Trabalhos futuros citados no artigo (Seções 4.1 e 6)

Nada disso foi feito pelos autores no artigo nem aqui:

- um escalonador automático de prompts, substituindo o disparo manual;
- VLMs leves o bastante para rodar na borda, incluindo ajuste fino de um VLM pequeno (NVILA) com dados
  de RA;
- modelagem mais refinada da manipulação de informação e módulos de detecção adicionais para ela;
- tratamento de conteúdo dinâmico com técnicas de pergunta e resposta sobre vídeo (os autores seguiram
  nessa direção no [AR-VIM](https://github.com/YM-Xiu/AR-VIM));
- implantação em óculos de RA: o artigo menciona uma demonstração de viabilidade no Meta Quest 3, que
  não foi publicada;
- estudo da aleatoriedade das respostas do VLM, que o artigo aponta como fonte de instabilidade. Aqui
  existe suporte (`--temperature` e `--seed`), mas nenhuma análise com várias execuções foi feita
  ainda.

## 11. Limitações desta reimplementação

- Os VLMs usados são pequenos (2B–8B) e rodam localmente; o LLaVA-Next-8b está quantizado em 4 bits,
  enquanto o artigo o usou pela API do Hugging Face.
- As imagens são reduzidas para até 1 megapixel antes de ir ao VLM; o artigo não informa a resolução
  usada com o LLaVA (com o GPT-4o, o notebook usava `detail: high`).
- Todos os resultados vêm de uma única execução com decodificação greedy; não há ainda estimativa de
  variância entre execuções.
- O limiar α = 0,25 segue o artigo, embora os rótulos sejam melhor explicados por α ≈ 0,2 (§9); a
  varredura de α é salva em cada `metrics.json`.
- Outras escolhas feitas onde o artigo é omisso estão na seção 4 do [REPRODUCAO.md](REPRODUCAO.md).

## 12. Citação e créditos

O dataset, a formulação dos ataques, a arquitetura do ViDDAR, os prompts e os notebooks originais são
dos autores do artigo. Se usar este material, cite:

```
@ARTICLE{xiu2025viddar,
  author={Xiu, Yanming and Scargill, Tim and Gorlatova, Maria},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  title={{ViDDAR}: Vision Language Model-Based Task-Detrimental Content Detection for Augmented Reality},
  year={2025},
  volume={31},
  number={5},
  pages={3194-3203},
  doi={10.1109/TVCG.2025.3549147}}
```

Os autores também publicaram um trabalho posterior sobre manipulação visual de informação com um
dataset em vídeo, o [AR-VIM](https://github.com/YM-Xiu/AR-VIM).

Contato dos autores originais: Yanming Xiu (yanming.xiu AT duke.edu), Tim Scargill
(timothyjames.scargill AT duke.edu), Maria Gorlatova (maria.gorlatova AT duke.edu). O trabalho original
foi financiado, entre outros, por NSF (CSR-2312760, CNS-2112562, IIS-2231975, CAREER IIS-2046072,
NAIAD 2332744), Cisco Research, Meta Research, DARPA (HR0011-24-1-0001) e Army Research Laboratory
(W911NF-23-2-0224).

Licença: MIT (ver [LICENSE](LICENSE)).
