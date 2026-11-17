# AGENTS.md — recomendações de desenvolvimento

Guia para quem for desenvolver neste repositório, sejam pessoas ou agentes de código. Visão geral do
projeto no [README.md](README.md); decisões de implementação e resultados de referência no
[REPRODUCAO.md](REPRODUCAO.md).

## Contexto em uma frase

Reimplementação do pipeline de avaliação do ViDDAR (Xiu et al., IEEE TVCG 2025) com VLMs locais, usada
numa pesquisa de mestrado sobre **falsos positivos** do detector em cenários ambíguos. Toda mudança
deve preservar duas coisas: a **fidelidade ao artigo** e a **comparabilidade entre execuções**.

## Ambiente

- Windows 10, GPU RTX 3060 12 GB, Python 3.12 em `.venv/` isolado (PyTorch 2.14 + CUDA 12.6,
  transformers 5.17). Não use o Anaconda base da máquina: ele mistura pacotes compilados para NumPy 1 e
  2 e um TensorFlow quebrado no site-packages do usuário.
- Sempre rode com o Python do venv: `.venv/Scripts/python ...`, a partir da raiz do repositório.
- `viddar/__init__.py` define `USE_TF=0` e ativa o `truststore`. Não remova: sem ele os downloads do
  Hugging Face falham nesta máquina (`CERTIFICATE_VERIFY_FAILED`). Se o `pip` falhar com erro de
  certificado, use `pip install --use-feature=truststore ...`.
- Espaço no disco C: é limitado; os modelos ficam em `~/.cache/huggingface` e o `pagefile.sys` cresce
  ao carregar modelos grandes. Confira `df -h /c` antes de baixar um modelo novo.
- Versões fixadas em [requirements.txt](requirements.txt). Ao atualizar uma dependência, rode a
  checagem completa da seção "Validação" e registre a mudança, porque ela pode alterar as respostas
  dos modelos.

## Comandos

```bash
.venv/Scripts/python -m pytest tests                                   # testes (segundos, sem GPU)
.venv/Scripts/python scripts/prepare_viddar.py                         # manifestos + máscaras m_c
.venv/Scripts/python scripts/run_obstruction.py --methods viddar --vlm qwen3-vl-2b --limit 5
.venv/Scripts/python scripts/run_manipulation.py --vlm qwen3-vl-2b --limit 3
.venv/Scripts/python scripts/summarize.py                              # -> results/summary.md
```

Use `--limit` e `--out results/_smoke` para testes rápidos que não poluem os resultados reais; apague
`results/_smoke` depois. Execuções completas levam de 30 min a 2 h por modelo: rode em segundo plano,
com log em `results/logs/<tarefa>_<vlm>.log`.

## Notebook e scripts

- A lógica de execução fica em [viddar/experiments.py](viddar/experiments.py)
  (`ObstructionConfig`/`run_obstruction`, `ManipulationConfig`/`run_manipulation`, `load_run`).
  O estudo de falsos positivos da manipulação usa ainda `AlignmentConfig`/`run_alignment` e as
  funções de atribuição de `viddar/manipulation.py`. Os scripts em `scripts/` e os notebooks
  [notebooks/experimentos_viddar.ipynb](notebooks/experimentos_viddar.ipynb),
  [notebooks/benchmark_manipulacao.ipynb](notebooks/benchmark_manipulacao.ipynb) e
  [notebooks/benchmark_obstrucao.ipynb](notebooks/benchmark_obstrucao.ipynb) (atribuição de erros em
  `obstruction.error_causes`) são camadas finas sobre
  essas funções. Não duplique lógica de experimento no notebook: se algo for útil nos dois, coloque em
  `viddar/` (gráficos e visualizações em [viddar/viz.py](viddar/viz.py), consolidação de resultados em
  [viddar/reporting.py](viddar/reporting.py)).
- Os campos das dataclasses de configuração têm os mesmos nomes dos argumentos dos scripts e são
  gravados em `metrics.json`. Ao adicionar um parâmetro, inclua-o nos dois lugares.
- O notebook usa o kernel "Python (ViDDAR .venv)" (`ipykernel install`, ver README §5) e funciona
  aberto a partir da raiz ou de `notebooks/`.
- Versione o notebook **sem saídas** (limpe as saídas antes do commit): imagens embutidas deixam o
  arquivo com vários MB e geram diffs ilegíveis.
- Para testar o notebook inteiro sem tocar nos resultados completos, execute uma cópia com
  `LIMIT = 5` via `nbclient` (kernel `viddar`) e apague a cópia e `results/_smoke` em seguida.
- Gráficos: um único eixo, cores categóricas na ordem fixa de `viz.SERIES` (a cor segue a série,
  não a posição), rótulos diretos, legenda fora da área dos dados e uma tabela com os mesmos números
  ao lado de cada gráfico. Confira a figura renderizada antes de concluir uma mudança.

## O que não pode ser modificado

- **Material dos autores:** `ViDDAR dataset/`, `Related Material/`, `image/`,
  `obstruction_labels.csv`, `information_manipulation_labels.csv`, `LICENSE`. O código lê esses arquivos
  como estão, incluindo o `.csv` que é na verdade Excel (tratado em `read_label_table`).
- **Prompts `paper` em [viddar/prompts.py](viddar/prompts.py):** são transcrições literais do artigo.
  Para testar um prompt novo, crie uma variante com outra chave (`KEY_OBJECT["minha_variante"]`,
  `MANIPULATION["minha_variante"]`) e exponha-a nas opções `--key-prompt` / `--prompt` dos scripts.
- **Regra de decisão da manipulação** (`last_yes_no`): é a regra do artigo. Regras alternativas entram
  como colunas ou métricas adicionais, como já é feito com `pred_factor_rule`, nunca substituindo
  `pred`.

## Princípios de implementação

1. **Artigo primeiro.** Onde o artigo especifica algo, siga-o. Onde ele é omisso, escolha, deixe
   configurável se for razoável e registre a escolha e o motivo na tabela da seção 4 do REPRODUCAO.md.
   Onde o notebook original diverge do artigo, o padrão é o artigo e o notebook vira variante.
2. **Guarde o dado bruto, decida depois.** `predictions.csv` guarda a resposta bruta do VLM, a razão de
   obstrução e os escores. Limiar, regra de decisão e métricas devem poder ser recalculados sem consultar
   o modelo de novo.
3. **Independência de dataset.** Nada em `viddar/` pode conter caminhos ou nomes específicos do dataset
   original; tudo passa por manifestos (formato no README, seção 7). O que é específico do ViDDAR fica
   em `scripts/prepare_viddar.py`, e um dataset novo ganha seu próprio `prepare_<nome>.py`.
4. **Um modelo por vez na GPU.** Os métodos rodam em estágios (VLM → `close()` → Grounding DINO + SAM).
   Não carregue o VLM e o segmentador ao mesmo tempo; chame `close()` ao terminar um estágio.
5. **Reprodutibilidade.** O padrão é decodificação greedy (temperatura 0). Todo parâmetro que afeta o
   resultado deve aparecer em `config` dentro do `metrics.json` e, se afetar a saída do VLM, também na
   chave do cache.
6. **Métricas honestas.** Resposta não interpretável (`None`) conta como erro, nunca é descartada.
   Reporte sempre a taxa de falsos positivos (FPR) e os intervalos de confiança: com 114 amostras,
   diferenças de poucos pontos não são significativas.

## Cache de respostas do VLM (cuidado)

`results/cache/<modelo>.jsonl` é indexado por modelo, prompt, hash das imagens, `max_pixels`,
`max_new_tokens`, temperatura e seed (`CachedVLM.ask` em [viddar/vlm.py](viddar/vlm.py)).

- Mudar pós-processamento, parsing, métricas ou α **não** exige invalidar o cache.
- Mudar o que acontece **dentro** da geração (`HFVLM.generate`: pré-processamento das imagens, ordem
  das mensagens, template, decodificação) altera as respostas **sem** alterar a chave, e o cache
  passaria a misturar respostas antigas e novas. Nesse caso, inclua um novo campo na chave (por exemplo
  uma constante de versão do backend) ou apague o `.jsonl` do modelo afetado e rode de novo.
  Precedente: `clean_up_tokenization_spaces=False` entrou depois das execuções dos modelos Qwen; as
  respostas deles no cache são anteriores à mudança.
- Nunca edite o `.jsonl` à mão. Ele é append-only; em caso de chave repetida, vale a última linha.

## Como estender

**Novo VLM.** Qualquer modelo compatível com `AutoModelForImageTextToText` funciona direto com
`--vlm <id do Hugging Face>`. Para um nome curto, adicione um preset em `PRESETS` (use
`load_in_4bit: True` se não couber em 12 GB em bf16). Faça antes um smoke test com `--limit 3`, leia as
respostas brutas em `predictions.csv` e verifique se o `split_numbered_answers` separou as perguntas
(sem valores `None` em `pred_A`, `pred_S` e `pred_I`). Modelos com raciocínio (`<think>`) são tratados
por `_strip_thinking` e `enable_thinking=False`.

**Novo método de obstrução.** Adicione-o em [viddar/obstruction.py](viddar/obstruction.py) no conjunto
certo (`VLM_METHODS`, `GROUNDING_METHODS`, `CV_METHODS`) e em `ALL_METHODS`, trate a decisão em
`add_predictions` e grave o valor contínuo (razão, escore) para permitir varreduras de limiar.

**Novo dataset.** Escreva `scripts/prepare_<nome>.py`, que gera um manifesto em `data/manifests/` e um
relatório de validação (arquivos faltando, distribuição de rótulos). Preencha a coluna `group` com os
cenários do estudo, pois é ela que alimenta o `metrics_by_group.csv`. Se as imagens cruas e aumentadas
não forem alinhadas pixel a pixel (vídeo, compressão), forneça `content_mask_path`; a diferença de
pixels não funciona nesse caso.

**Nova métrica.** Implemente em [viddar/metrics.py](viddar/metrics.py) ou nos `summarize()` dos módulos
de tarefa, com teste em `tests/`, e exponha em `scripts/summarize.py`.

## Validação antes de reportar resultados

1. `pytest tests` passando. Toda mudança em `parsing.py`, `metrics.py` ou `data.py` ganha um teste.
2. Smoke test com `--limit` no método alterado, lendo algumas respostas brutas.
3. Referências que devem se manter depois de mudanças no pipeline (α = 0,25):
   - Prior knowledge ≈ 93,8% de acurácia, mIoU ≈ 85% (não usa VLM; valida detector, SAM, m_c e Eq. 4);
   - Saliency ≈ 52% e Canny ≈ 50%;
   - Qwen3-VL-2B: ViDDAR 84,6%, manipulação 38,6%; LLaVA-Next-8b 4-bit: ViDDAR 88,2%, manipulação 38,6%.

   Um desvio grande sem mudança intencional indica regressão.
4. `scripts/summarize.py` e atualização da seção de resultados do REPRODUCAO.md, com data, modelos e
   parâmetros.

## Pontos que já causaram confusão

- `information_manipulation_labels.csv` / `confusion_labels.csv` são Excel; o rótulo M é a coluna
  `confusion`, e `result` é ignorada.
- O dataset publica só 32 máscaras de conteúdo virtual, sem mapeamento; as máscaras usadas são
  derivadas em `data/content_masks/`.
- O critério léxico de reconhecimento do objeto-chave é permissivo ("black tv stand" conta como TV).
  Para conclusões, prefira o critério *grounded* (`key_object_grounded_accuracy`, IoU ≥ 0,5).
- O template do LLaVA-Next coloca as imagens antes do texto, ao contrário da ordem passada; a ordem
  imagem crua → aumentada é mantida.
- O VLM pode copiar exemplos do prompt (o Qwen3-VL-2B respondeu "red box" em 36/306 imagens).
  Verifique as respostas mais frequentes em `predictions.csv` ao testar prompts novos.
- Os scripts preenchem `sys.path` com a raiz do repositório; rode-os a partir da raiz.

## Estilo e convenções

- Código, comentários, docstrings e nomes em inglês; documentação (`README.md`, `REPRODUCAO.md`,
  `AGENTS.md`) em português.
- Siga o estilo existente: `from __future__ import annotations`, type hints, funções pequenas, docstring
  curta explicando o porquê e citando a seção ou equação do artigo quando ela é implementada.
- Scripts com `argparse`, docstring de módulo com exemplos de uso e saídas em
  `results/<tarefa>/<vlm>/<variante>/`.
- Git: `results/`, `data/content_masks/`, `ViDDAR dataset/` e `.venv/` ficam fora do repositório;
  `data/manifests/` é versionado. Não versione pesos de modelos nem o dataset. Trabalhe e faça commits
  direto na `main`, sem branches separadas.
