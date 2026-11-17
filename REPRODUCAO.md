# Reprodução da avaliação do ViDDAR com VLM local

Reimplementação do pipeline de avaliação do artigo *ViDDAR: Vision Language Model-Based
Task-Detrimental Content Detection for Augmented Reality* (Xiu, Scargill, Gorlatova — IEEE TVCG 2025),
Seções 5.1 (obstrução, Tabela 1) e 5.2 (manipulação de informação, Tabela 2), usando VLMs pequenos
rodando localmente (testado em RTX 3060 12 GB). Latência, app Android e estudo com usuários estão
fora do escopo.

O repositório original só traz notebooks de demonstração (ViDDAR + GPT-4o). Todo o código em
`viddar/` e `scripts/` foi reconstruído a partir do artigo e desses notebooks.

## 1. Ambiente

```bash
python -m venv .venv                       # Python 3.12
.venv/Scripts/python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt
```

- Espaço em disco: ~6 GB para o ambiente, 4,3 GB para o Qwen3-VL-2B e 1,3 GB para Grounding DINO + SAM
  (cache em `%USERPROFILE%\.cache\huggingface`; mude com a variável `HF_HOME`).
- `truststore` faz o Python usar os certificados do Windows. Sem isso, os downloads do Hugging Face
  falham nesta máquina (`CERTIFICATE_VERIFY_FAILED`), porque algum software re-assina o tráfego HTTPS.
- Opcional: `set HF_HUB_DISABLE_SYMLINKS_WARNING=1` silencia um aviso inofensivo no Windows.

## 2. Execução

```bash
# 1) manifestos + máscaras de conteúdo virtual derivadas + relatório de validação do dataset
.venv/Scripts/python scripts/prepare_viddar.py

# 2) obstrução (Tabela 1). "all" = viddar end_to_end underdetailed greedy saliency canny prior
.venv/Scripts/python scripts/run_obstruction.py --methods all --vlm qwen3-vl-2b

# 3) manipulação de informação (Tabela 2)
.venv/Scripts/python scripts/run_manipulation.py --vlm qwen3-vl-2b

# 3b) estudo de falsos positivos (report/analise_fp_fn_manipulacao.md): variantes de prompt e
#     alinhamento geométrico sobre uma execução pronta (só Grounding DINO + SAM, sem VLM)
.venv/Scripts/python scripts/run_manipulation.py --vlm qwen3-vl-4b --prompt definition --mode decomposed
.venv/Scripts/python scripts/run_alignment.py --run-dir results/manipulation/qwen3-vl-4b/paper

# 4) tabelas consolidadas (com os números do artigo ao lado) -> results/summary.md
.venv/Scripts/python scripts/summarize.py

# testes
.venv/Scripts/python -m pytest tests
```

Opções úteis: `--limit N` (teste rápido), `--alpha` (limiar da Eq. 4), `--key-prompt notebook` /
`--prompt notebook` (prompts dos notebooks originais em vez dos do artigo), `--max-pixels` (resolução
enviada ao VLM), `--temperature T --seed S` (amostragem estocástica para medir variância),
`--all-boxes` (união de todas as caixas em vez da de maior confiança).

**VLMs:** `--vlm` aceita os presets de [viddar/vlm.py](viddar/vlm.py) (`qwen3-vl-2b` padrão,
`qwen3-vl-4b`, `qwen2.5-vl-3b`, `llava-next-8b-4bit` — o LLaVA do artigo, exige `bitsandbytes` —,
`qwen3-vl-8b-4bit` e `gemma4-12b-4bit` — extensão do benchmark, também em 4 bits —,
`gpt-4o` via API com `OPENAI_API_KEY`) ou qualquer id do Hugging Face compatível com
`AutoModelForImageTextToText`.

**Cache:** toda resposta do VLM fica em `results/cache/<modelo>.jsonl`, indexada por
modelo + prompt + hash das imagens + parâmetros de geração. A quantização não entra na chave: um
mesmo id do Hugging Face deve rodar sempre com a mesma quantização (os presets `*-4bit` garantem isso). Re-executar (por exemplo com outro α) não
consulta o modelo de novo; o passo de identificação do objeto-chave é compartilhado entre ViDDAR e
End-to-End.

## 3. Estrutura

| Arquivo | Conteúdo |
|---|---|
| [viddar/data.py](viddar/data.py) | manifestos, leitura dos rótulos, máscara m_c por diferença de pixels, razão de obstrução (Eq. 4) |
| [viddar/prompts.py](viddar/prompts.py) | prompts do artigo (literais) e variantes dos notebooks |
| [viddar/vlm.py](viddar/vlm.py) | backends (transformers local / OpenAI), presets e cache |
| [viddar/grounding.py](viddar/grounding.py) | Grounding DINO + SAM (equivalente ao `LangSAM` do notebook) |
| [viddar/obstruction.py](viddar/obstruction.py) | ViDDAR e baselines da Tabela 1 |
| [viddar/manipulation.py](viddar/manipulation.py) | detecção de manipulação (Tabela 2) + extração de A, S, I; modo decomposto; atribuição de FP/FN e oráculo por fator |
| [viddar/alignment.py](viddar/alignment.py) | alinhamento geométrico A_geo: medidas entre m_c e o objeto-chave, limiar por validação cruzada, decisão híbrida |
| [viddar/parsing.py](viddar/parsing.py) | pós-processamento das respostas; critério de acerto do objeto-chave |
| [viddar/metrics.py](viddar/metrics.py) | acurácia, precisão, recall, FPR/FNR com IC de Wilson; métricas por grupo |

Cada execução grava `predictions.csv` (uma linha por amostra, com a resposta bruta do VLM),
`metrics.json` e `metrics_by_group.csv` em `results/<tarefa>/<vlm>/<método>/`.

## 4. Decisões de implementação e desvios do artigo

| Item | Artigo | Aqui | Motivo |
|---|---|---|---|
| VLM | GPT-4o-2024-08-06, LLaVA-Next-8b (API Hugging Face) | Qwen3-VL-2B (padrão), Qwen3-VL-4B, LLaVA-Next-8b em 4-bit NF4 | objetivo do estudo: modelo pequeno local; o 8B só cabe em 12 GB quantizado |
| Ordem texto/imagens | texto, depois imagem crua e aumentada | igual para o Qwen; o template do LLaVA sempre põe as imagens antes do texto | a ordem crua → aumentada, citada no prompt, é mantida |
| Decodificação | não informado | greedy (temperatura 0) | determinismo; use `--temperature` para variância |
| Resolução das imagens | GPT "detail: high" | reduzidas para ≤ 1 MP por imagem | memória da GPU; ajustável |
| Detector | Grounding DINO 1.5 (API fechada) | `IDEA-Research/grounding-dino-base` | modelo aberto; o notebook original também usava o GDINO aberto via LangSAM |
| Segmentação | SAM ViT-B | `facebook/sam-vit-base` | igual |
| Limiares do detector | não informados | box 0,3 / texto 0,25, caixa de maior confiança | padrão do LangSAM + `largest_flag=True` do notebook |
| α (Eq. 4) | 0,25 | 0,25 (varredura de 0,05 a 0,5 salva em `metrics.json`) | o notebook usava U(0,175; 0,225) sorteado por imagem |
| Máscara m_c | "comparando I_r e I_a" | diferença > 30 em algum canal + abertura 3×3 + fechamento 7×7 | o dataset só publica 32 máscaras sem mapeamento; as derivadas coincidem com elas (IoU mediano 0,99). Os ~8% com IoU < 0,8 são configurações de conteúdo virtual para as quais nenhuma máscara foi publicada (verificado visualmente); podem ter pequenos buracos onde o objeto virtual tem a mesma cor do fundo |
| Sem detecção do objeto | não informado | razão = 0 → "não obstruído" (contado em `no_detection`) | comportamento de um sistema real (não alarma) |
| Greedy | "detecta individualmente" | obstrução se *qualquer* objeto listado tiver razão ≥ α; mIoU com a união das máscaras | interpretação mais direta do texto |
| Acerto do objeto-chave | critério não informado | (a) léxico: resposta contém um sinônimo da classe (`KEY_OBJECT_SYNONYMS`); (b) *grounded*: IoU(máscara prevista, máscara GT) ≥ 0,5 | (a) aproxima uma avaliação manual; (b) é objetivo. O léxico é permissivo ("tv stand" conta como TV) |
| Saliency | Hou & Zhang, S_o > S_r | resíduo espectral (64×64, como no OpenCV) | igual |
| Canny | C_o > C_r | Canny(100, 200) | limiares não informados |
| Decisão de manipulação | último "yes"/"no" da resposta | igual; **também** calcula A∧S∧I a partir das respostas 3–5 | mostra inconsistências do VLM (ver §6) |
| Retentativas | o notebook repetia a pergunta se a resposta contivesse "sorry", "however" etc. | não | com decodificação greedy a resposta seria a mesma; o filtro também enviesa o resultado |
| Variantes de prompt de manipulação (estudo de FP) | só o prompt do §5.2.1 | `neutral` (sem "created by adding some virtual content" e sem "Think creatively…"); `definition` (definição de I do §3.2 acrescentada à pergunta 5) | testar se os FP vêm do viés do prompt (A1) ou da interpretação de I (A3); o prompt `paper` continua sendo a referência |
| Modo decomposto | uma chamada com 6 perguntas; decisão = último "yes"/"no" | `--mode decomposed`: uma chamada para conteúdo e objeto-chave (≤ 4 palavras) e uma por fator; resposta começa por "yes"/"no"; `pred` = A ∧ S ∧ I em código (não há pergunta 6) | elimina o erro de conclusão do Qwen3-VL-2B e reduz o efeito halo (A2) |
| Taxas de erro no estudo de FP | – | FPR = FP / benignos; FNR = FN / ataques; decisão ilegível num ataque conta como FN | denominadores fixos, comparáveis entre execuções |
| Objeto-chave da manipulação | resposta livre à pergunta 2 | `extract_object_phrase`: "… is the black fan" → "black fan"; corta orações de lugar e alternativas ("tablet or a smartphone" → "tablet") | entrada do Grounding DINO no alinhamento geométrico |
| Ganho de uma variante | – | acurácia, precisão e recall (Tabela 2 do artigo) + FPR/FNR; "melhorou"/"piorou" só com McNemar p < 0,05 (`reporting.gain_table`) | mesma métrica do artigo, com teste pareado |
| VLMs do benchmark estendido | GPT-4o e LLaVA-Next-8b | Qwen3-VL 2B, 4B e 8B, LLaVA-Next-8B e Gemma 4 12B; os de 8B ou mais em 4 bits (NF4, bitsandbytes), com o mesmo protocolo e as mesmas variantes | modelos abertos que cabem em 12 GB e aceitam duas imagens por conversa; Llama 3.2 Vision ficou de fora por ser treinado com uma imagem por prompt |
| Resolução no Gemma 4 | – | o processador do Gemma 4 usa um orçamento fixo de 280 tokens por imagem, menor que o dos Qwen (≤ 1 MP); o preset converte os `pixel_values` para bf16 (`cast_inputs`), o que a torre de visão dele não faz sozinha | é o comportamento padrão do modelo; mudar o orçamento seria uma variante à parte |
| Extração do objeto-chave para o C2 | – | a mesma regra para todos os modelos; respostas com parênteses ("black speaker (a small …") chegam truncadas ao Grounding DINO, que ainda assim detecta o objeto | mudar a regra alteraria o C2 dos modelos já executados |
| Alinhamento geométrico (A_geo) | fator A julgado pelo VLM | m_c pela diferença de pixels (mesmo código da obstrução; os 114 pares são alinhados) e m_k = Grounding DINO + SAM no objeto citado pelo VLM, em I_r. Medida de decisão fixada antes: `gap_norm` = menor distância entre m_c e m_k ÷ maior lado da caixa de m_c; limiar por validação cruzada estratificada (5 partições, 20 repetições para a dispersão); sem detecção, vale o A do VLM | caminho C2 do estudo de FP; nenhum limiar é ajustado no conjunto inteiro |

## 5. Observações sobre o dataset (ver `data/manifests/validation_report.json`)

- `information_manipulation_labels.csv` / `confusion_labels.csv` são arquivos **Excel** com extensão
  `.csv`. A coluna `result` não é documentada e é ignorada; `confusion` é o rótulo M, e M = A∧S∧I
  vale para as 114 amostras.
- Há 115 imagens em `raw/` para 114 pares (a `115_raw.png` fica de fora) e 16 máscaras de `laptop`
  sem par (classe excluída, como no notebook, que usava `dataset-gt-no-laptop.csv`).
- Com as **máscaras GT** do objeto-chave, o maior valor da razão entre os negativos é 0,198 e o menor
  entre os positivos é 0,145. A acurácia "oráculo" é 98,4% com α = 0,2 e 94,1% com α = 0,25: os
  rótulos parecem ter sido feitos com fronteira em torno de 0,2, apesar de o artigo dizer 0,25.

## 6. Usando outro dataset (estudo de falsos positivos)

Basta criar um manifesto CSV (caminhos relativos ao próprio CSV) e passá-lo com `--manifest`:

- Obstrução: `id, raw_path, ar_path, label` + opcionais `key_object` (necessário para `prior` e
  para a métrica de reconhecimento), `key_mask_path` (necessário para mIoU), `content_mask_path`
  (senão é derivada de |I_r − I_a|, o que exige imagens alinhadas pixel a pixel), `group`.
- Manipulação: `id, raw_path, ar_path, label` + opcionais `A, S, I`, `group`.

Use a coluna `group` para marcar os cenários (por exemplo `ambiguo_tipo1`, `controle`). Cada
execução gera `metrics_by_group.csv` com TP/FP/TN/FN, FPR e FNR por grupo, e `metrics.json` traz
os intervalos de confiança de Wilson. Na manipulação, `predictions.csv` guarda `pred_A`, `pred_S`,
`pred_I` (respostas às perguntas 3–5) e `pred_factor_rule` (A∧S∧I), o que permite atribuir cada
falso positivo ao fator que o causou, ou a uma resposta final inconsistente com os próprios fatores
(métrica `final_answer_consistent_with_factors`).

## 7. Resultados (decodificação greedy, α = 0,25, ≤ 1 MP por imagem — 28/09/2026)

Modelos: Qwen3-VL-2B-Instruct (bf16), Qwen3-VL-4B-Instruct (bf16) e LLaVA-Next-8B
(`llava-hf/llama3-llava-next-8b-hf`, o modelo do artigo, em 4-bit NF4). Tabelas completas, com
precisão, recall, FPR e IC 95% de Wilson, em `results/summary.md`.

**Obstrução (306 pares) — acurácia (%)**

| Método | Qwen3-VL-2B | Qwen3-VL-4B | LLaVA-8B 4-bit | Artigo GPT-4o | Artigo LLaVA-8B |
|---|---|---|---|---|---|
| ViDDAR | 84,6 | **91,5** | 88,2 | 92,15 | 89,21 |
| End-to-End | 45,1 (FPR 87,6%) | 72,9 | 51,3 (recall 8,3%) | 81,04 | 55,23 |
| Underdetailed | 82,7 | 85,6 | 82,7 | 86,92 | 81,37 |
| Greedy | 88,9 | 90,5 | 87,3 | 88,89 | 85,62 |
| Saliency | 52,3 | | | 51,63 | |
| Canny | 49,7 | | | 51,96 | |
| Prior knowledge | 93,8 (mIoU 85,1) | | | 93,14 (mIoU 78,83) | |

- **A replicação confere:** com o mesmo VLM do artigo (LLaVA-8B), todos os métodos ficam a menos de
  4 pontos dos números publicados. As partes sem VLM também batem (Prior 93,8 vs 93,1;
  Saliency/Canny ≈ acaso), validando detector, segmentação, máscara m_c e Eq. 4.
- O Qwen3-VL-4B, rodando localmente, praticamente iguala o ViDDAR+GPT-4o (91,5 vs 92,15).
- A métrica de reconhecimento do objeto-chave não é comparável com a do artigo (critério não
  publicado): o LLaVA atinge acurácia de detecção igual à do artigo com 70% no critério léxico daqui,
  contra 86% no artigo.
- O End-to-End falha de formas opostas: o Qwen-2B diz "obstruído" quase sempre (FPR 88%) e o LLaVA
  quase nunca (recall 8%). Deixar a decisão quantitativa inteira para o VLM é instável.
- Qwen-2B: todos os 11 falsos positivos do ViDDAR vêm de objeto-chave errado; em 36/306 imagens ele
  respondeu **"red box"**, copiando o exemplo do próprio prompt.
- Varredura de α (ViDDAR, Qwen-2B): 0,2 → 86,3%; 0,25 → 84,6%; 0,3 → 82,0%. O pico em 0,2 é
  coerente com a fronteira dos rótulos (§5).

**Manipulação de informação (114 pares; 44 positivos)**

| Modelo | Acurácia | Precisão | Recall | FPR | Q6 coerente com Q3–5 | "yes" em I (GT: 90%) |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B | 38,6 | 38,2 | 97,7 | **97,1** | 1,8% | 0% |
| Qwen3-VL-4B | 61,4 | 50,0 | 6,8 | 4,3 | 100% | 37% |
| LLaVA-8B 4-bit | 38,6 | 37,0 | 84,1 | **90,0** | 100% | 100% |
| Artigo GPT-4o | 82,46 | 74,00 | 84,09 | – | | |
| Artigo LLaVA-8B | 42,98 | 39,80 | 95,35 | – | | |

- O LLaVA reproduz o resultado do artigo (38,6 / 37,0 / 84,1 contra 43,0 / 39,8 / 95,4).
- Os três modelos erram por **mecanismos diferentes**, visíveis graças à extração de A, S e I:
  - *Qwen-2B, falso positivo por incoerência lógica:* responde "no" ao fator I em todos os casos,
    mas "yes" na pergunta 6 (muitas vezes alegando que 3, 4 e 5 foram "yes"). Aplicando A∧S∧I às
    próprias respostas dele, o FPR cai de 97% para 0%.
  - *LLaVA, falso positivo por credulidade:* é coerente, mas diz "yes" ao fator I sempre e a A/S em
    88% dos casos, então quase toda combinação vira manipulação.
  - *Qwen-4B, falso negativo por ceticismo:* coerente, mas só aceita o fator I em 37% dos casos;
    acerta 61,4%, exatamente a taxa de responder sempre "não" (70/114).
- Exemplo de confusão com o mundo real: na amostra 1, o LLaVA tomou o **iPad de calibração** (que
  mostra um relógio) pelo conteúdo virtual.
- Há 1 resposta degenerada do Qwen-2B (loop de repetição), contada como erro.
