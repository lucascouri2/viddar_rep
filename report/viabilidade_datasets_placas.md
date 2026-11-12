# Viabilidade dos datasets de placas e retorno à proposta de benchmark

*Projeto: robustez de detectores de conteúdo prejudicial em RA (reimplementação do ViDDAR, Xiu et al.,
IEEE TVCG 2025). Novembro de 2026.*

## Resumo

A proposta do Draft 2 era testar se **placas de trânsito fisicamente degradadas** (desgaste, pichação,
adesivos, dano estrutural) aumentam a taxa de falsos positivos (FPR) dos detectores do ViDDAR. Para
isso, avaliamos o dataset recebido no projeto (vwise), o dataset de placas descrito no rascunho do
artigo do grupo (VSigns) e quatro datasets públicos de placas. A conclusão é a seguinte:

- **Nenhum dataset disponível tem placas degradadas em quantidade suficiente.** Nenhum deles tem um
  rótulo do estado físico da placa.
- **O VSigns é o único construído com esse foco, mas não está disponível para nós.** Mesmo que
  estivesse, precisaria de anotação adicional antes de servir ao experimento.
- **Os datasets públicos têm placas íntegras.** Eles serviriam como grupo de controle, mas não testam
  a hipótese sozinhos.

Por isso decidimos **não seguir com a análise de placas** e **voltar à proposta inicial**: um
benchmark de VLMs de código aberto no próprio ViDDAR, com foco em falsos positivos. Essa proposta usa
os dados já rotulados do ViDDAR e o pipeline já reconstruído e validado, e pode começar
imediatamente.

---

## 1. O que o experimento com placas exigiria

O ViDDAR avalia **pares** de imagens alinhadas pixel a pixel: a cena real (I_r) e a mesma cena com
conteúdo virtual de RA (I_a). Os datasets de placas trazem apenas imagens reais. O experimento
proposto no Draft 2 (§III) precisaria, portanto, gerar as imagens aumentadas sinteticamente em cinco
condições:

| Cond. | Conteúdo virtual | Obstrução esperada | Manipulação esperada |
|---|---|---|---|
| C0 | nenhum (I_a = I_r) | 0 | 0 |
| C1 | benigno, longe da placa | 0 | 0 |
| C2 | benigno, encostado na placa, cobrindo menos que α | 0 | 0 |
| C3 | cobre pelo menos α da placa | 1 | — |
| C4 | altera o significado da placa | — | 1 |

Essas condições seriam cruzadas com o estado da placa: íntegra ou degradada, e o tipo de degradação.
Com isso, cada dataset candidato precisaria oferecer:

1. **Placas fisicamente degradadas e placas íntegras**, com um rótulo de estado (ou ao menos
   material suficiente para anotarmos esse rótulo);
2. **Localização da placa** (caixa ou máscara), usada para obter a máscara m_k;
3. **Placas grandes o bastante** na imagem. Nas imagens do ViDDAR, o objeto-chave aparece em
   primeiro plano. Em quadros amplos de direção, o VLM nem sempre escolhe a placa: no piloto com o
   vwise, o Qwen3-VL-4B a nomeou em 61% dos quadros;
4. **Cerca de 140 pares sem ataque por grupo.** É o tamanho estimado para detectar um aumento de FPR
   de 5% para 15%, com significância de 5% e poder de 80%;
5. De preferência, **contexto latino-americano**, que motivava a proposta.

---

## 2. Datasets avaliados

### 2.1 vwise (dataset recebido no projeto)

Detalhado no Draft 2, Seção VI.

- **O que é:** 44 vídeos de direção, cortados em 521 segmentos de cerca de 10 s (86,6 min), com 5
  quadros por segmento (2.605 imagens).
- **Rótulos:** tipo de ambiente, em 6 classes. O rascunho do artigo do grupo confirma que esses
  rótulos foram gerados com o CLIP e revisados por anotadores. **Não há anotação de placas, de
  degradação nem de localidade.**
- **Presença de placas:** o Grounding DINO, com a consulta "traffic sign", encontra placas em
  26,3% dos segmentos com confiança ≥ 0,45 e em 8,1% com confiança ≥ 0,55. Esses números incluem
  detecções que não são placas (adesivos, logotipos, cones).
- **Qualidade das placas:** são pequenas (mediana de 0,6% da área do quadro), em geral pouco nítidas
  e sem degradação física evidente.
- **Localidade:** há vários exemplos gravados fora da América Latina, com placas de idiomas diversos.
- **Sobreposições:** pelo menos 41% dos segmentos já têm sobreposições 2D gravadas no vídeo
  (logotipos, legendas, marcas d'água).

**Veredito:** o vwise é um dataset de classificação de cenas, não de placas. O professor confirmou
que ele foi enviado por engano no lugar do dataset de placas correlacionado.

### 2.2 VSigns ("VWays"): o dataset de placas do grupo

Descrito no rascunho *VSigns and VWise: Challenges and Solutions in Creating Vision Datasets for Latin
American Vehicular Applications* (SIBGRAPI 2025) e na planilha *research gap - VWays.xlsx*.

- **O que é:** placas brasileiras e latino-americanas, com mais de 10.500 imagens e 11.800
  anotações. A planilha informa 7.871 imagens antes da aumentação. Suporta três tarefas: detecção,
  classificação e segmentação de instâncias.
- **Classes:** 18 subclasses em 4 macroclasses (Advertência, Direção, Proibição e Velocidade).
- **Origem das imagens:**
  - busca no Google Imagens e no YouTube, inclusive com os termos "placa danificada", "placa
    enferrujada" e "placa pichada";
  - Google Street View;
  - gravações com celular em Recife, sob chuva, oclusão por vegetação e pouca luz;
  - renderizações sintéticas 3D com dano, desbotamento e marcações.
- **Máscaras:** geradas com SAM a partir das caixas e validadas manualmente.

**Por que seria adequado:** é o único dataset que corresponde à hipótese. Tem degradação real em
contexto latino-americano, caixas e máscaras prontas e classes que permitem análise por tipo de
placa. As placas de velocidade também seriam alvos naturais para a condição C4 (trocar "40" por
"60").

**Por que não seguimos com ele:**

1. **Não está disponível.** Não é público (o artigo indica divulgação futura), e o professor ainda
   não o localizou. O cronograma não pode depender disso.
2. **Não tem rótulo do estado da placa.** O artigo não descreve uma anotação de "íntegra ×
   degradada" nem do tipo de degradação. Ela teria de ser feita por nós, a menos que os metadados de
   origem (o termo de busca de cada imagem) tenham sido guardados.
3. **Mistura imagens reais, sintéticas e aumentadas.**
   - As renderizações 3D teriam de ser separadas: uma placa renderizada pode ser tomada pelo VLM como
     conteúdo virtual, o que contamina justamente a medida de falsos positivos.
   - A aumentação de balanceamento (rotações e flips) teria de ser descartada: uma placa espelhada
     muda de significado ("proibido virar à direita" vira "à esquerda").
4. **Uso das imagens em publicação:** imagens coletadas do Google Imagens, do YouTube e do Street View
   têm restrições de licença a esclarecer.

### 2.3 Brazilian Vertical Traffic Signs and Lights Dataset (VTSLD)

*Recognition of Brazilian vertical traffic signs and lights from a car using Single Shot Multibox
Detector* (SBC, 2023).

- **O que é:**
  - 1.363 quadros de 800×450 extraídos de vídeos de direção no Brasil (há letreiros com nomes de vias
    de São Paulo);
  - 2.107 objetos anotados em 16 classes, das quais 3 são semáforos. Sem os semáforos, restam 1.542
    placas;
  - anotações em XML (formato VOC), com divisão treino/validação/teste.
- **Classes sem nome:** as classes aparecem só como códigos numéricos, sem arquivo de mapeamento. O
  mapeamento abaixo foi inferido pelos recortes:

  | Código | Classe |
  |---|---|
  | 000 | pare |
  | 001 | dê a preferência |
  | 003 | proibido virar à esquerda |
  | 004 | proibido virar à direita |
  | 007 | proibido estacionar |
  | 008 | estacionamento |
  | 009 | proibido parar e estacionar |
  | 023 | velocidade máxima |
  | 025 | lombada |
  | 028 | sentido obrigatório |
  | 035 | proibido caminhão |
  | 040 | faixa de ônibus |
  | 042 | bicicleta |
  | 051 / 052 / 053 | semáforo amarelo / vermelho / verde |

- **Aumentações:** os demais cerca de 54 mil arquivos são **aumentações fotométricas** de cada
  quadro: inversão de cores, solarização, troca de canais, ruído e contraste. Isso não é degradação
  física.
- **Tamanho das placas:** muito pequenas. O menor lado tem mediana de 34 px; só cerca de 200 placas
  têm 50 px ou mais, e 21 têm 80 px ou mais.
- **Estado das placas:** nas amostras de recortes examinadas, as placas estão praticamente todas
  íntegras.

**Veredito:** é brasileiro e real, mas não tem placas degradadas, e as placas são pequenas demais
para o VLM. As aumentações teriam de ser descartadas: uma imagem com cores invertidas é justamente o
tipo de alteração que o VLM pode ler como manipulação. O dataset serviria, no máximo, como um pequeno
grupo de controle de placas brasileiras íntegras.

### 2.4 BelgiumTS (BTSD)

Timofte et al., *Multi-view traffic sign detection, recognition, and 3D localisation* (MVA, 2011).

- **O que é:**
  - imagens de 1628×1236 (JPEG 2000) gravadas por uma van de mapeamento com 8 câmeras na Bélgica;
  - 13.480 anotações em 9.006 imagens, com 62 dos 210 tipos de placa definidos e 12 superclasses de
    forma e cor;
  - poses de câmera para localização 3D e imagens sem placas.
- **Tamanho das placas:** as maiores entre os datasets públicos. O menor lado tem mediana de 65 px, e
  cerca de 2.000 placas têm 120 px ou mais.
- **Placas repetidas:** muitas anotações são a mesma placa física, vista em quadros e câmeras
  diferentes. O número de placas distintas é bem menor.
- **Estado das placas:** íntegras e em boas condições de captura. Não encontramos pichação nem dano
  nas amostras examinadas.

**Veredito:** tem boa qualidade, mas não é latino-americano e não tem degradação. Serviria apenas como
controle de placas íntegras.

### 2.5 GTSDB (German Traffic Sign Detection Benchmark)

- **O que é:** 900 imagens de 1360×800 da Alemanha. A cópia baixada só tem o gabarito das 600 de
  treino: 852 placas em 506 imagens, 43 classes.
- **Tamanho das placas:** muito pequenas. O menor lado tem mediana de 36 px, e só 54 placas têm 80 px
  ou mais.
- **Estado das placas:** não há degradação.

**Veredito:** é o menos útil. Serviria no máximo como verificação de sanidade.

### 2.6 Datasets não disponíveis publicamente

- **Traffic Sign Recognition with Deep Learning: Vegetation Occlusion Detection in Brazilian
  Environments** (Sensors, 2023): seria o mais próximo do nosso interesse, porque a oclusão natural
  por vegetação é uma obstrução real que o VLM poderia atribuir ao conteúdo virtual. Não está
  disponível.
- **Construction of Brazilian Regulatory Traffic Sign Recognition Dataset** (CIARP, 2022): não está
  disponível.
- **DITS, Dataset of Italian Traffic Signs** (ACIVS, 2016): não está disponível e também não é
  latino-americano.

---

## 3. Comparação

| Dataset | Degradação física | América Latina | Tamanho das placas | Anotação de placas | Disponível | Papel possível |
|---|---|---|---|---|---|---|
| vwise | não identificada | parcial (com exemplos de fora) | muito pequenas | não | sim | nenhum |
| **VSigns** | **sim (é o foco)** | sim | variado | caixas, classes e máscaras | **não** | seria a base do estudo |
| VTSLD | praticamente nenhuma | sim (Brasil) | muito pequenas | caixas (códigos sem nome) | sim | controle pequeno |
| BelgiumTS | praticamente nenhuma | não | grandes | caixas e classes | sim | controle |
| GTSDB | não | não | muito pequenas | caixas (só treino) | sim | sanidade |
| Vegetation Occlusion | oclusão natural | sim (Brasil) | ? | sim | não | — |
| Brazilian Regulatory TSR | ? | sim (Brasil) | ? | sim | não | — |
| DITS | ? | não | ? | sim | não | — |

**Conclusão da análise:** os datasets disponíveis permitem montar o grupo de **placas íntegras**, mas
não o grupo de **placas degradadas**, que é o centro da hipótese. Sem esse grupo não há comparação
possível, e o experimento perde o sentido. A única fonte adequada (VSigns) está indisponível e, mesmo
quando chegar, exigiria:

- separar as imagens reais das sintéticas e aumentadas;
- anotar o estado e o tipo de degradação de cada placa;
- esclarecer a licença das imagens.

Gerar degradação sintética sobre placas íntegras também não resolveria: a hipótese trata de
degradação **real**. Uma degradação gerada por nós mediria o efeito do nosso gerador, e não o das
condições encontradas nas ruas.

---

## 4. Decisão: voltar ao benchmark de VLMs no ViDDAR

Retomamos a proposta inicial, anterior à sugestão do dataset de placas: um **benchmark de VLMs de
código aberto na detecção de conteúdo prejudicial em RA, com foco em falsos positivos**, usando os
dados do próprio ViDDAR.

### 4.1 Por que essa proposta é viável agora

- **Os dados estão prontos e rotulados:** 306 pares de obstrução e 114 de manipulação, com rótulos e
  máscaras dos autores.
- **O pipeline está reconstruído e validado.** Com o mesmo VLM do artigo, os resultados de
  obstrução ficam a menos de 4 pontos percentuais dos publicados. O baseline sem VLM (prior knowledge) reproduz o artigo
  (93,8% contra 93,14%), o que valida o detector, o segmentador e a regra de decisão.
- **Ele preserva o tema da pesquisa:** falsos positivos do detector em cenários ambíguos. Os
  resultados preliminares já mostram que esse é o ponto fraco dos modelos abertos.
- **Não depende de coleta, anotação ou licença de dados de terceiros.**

### 4.2 Resultados preliminares que motivam o benchmark

Obstrução (acurácia, %, α = 0,25):

| Método | Qwen3-VL-2B | Qwen3-VL-4B | LLaVA-Next-8B (4 bits) | Artigo: LLaVA-Next-8B | Artigo: GPT-4o |
|---|---|---|---|---|---|
| ViDDAR | 84,6 | 91,5 | 88,2 | 89,21 | 92,15 |
| End-to-end | 45,1 | 72,9 | 51,3 | 55,23 | 81,04 |
| Underdetailed | 82,7 | 85,6 | 82,7 | 81,37 | 86,92 |
| Greedy | 88,9 | 90,5 | 87,3 | 85,62 | 88,89 |

Manipulação de informação (%):

| Modelo | Acurácia | Precisão | Recall | FPR |
|---|---|---|---|---|
| Qwen3-VL-2B | 38,6 | 38,2 | 97,7 | 97,1 |
| Qwen3-VL-4B | 61,4 | 50,0 | 6,8 | 4,3 |
| LLaVA-Next-8B (4 bits) | 38,6 | 37,0 | 84,1 | 90,0 |
| GPT-4o (artigo; FPR derivada da matriz de confusão) | — | — | — | 18,6 |

Os modelos abertos falham de formas diferentes na manipulação:

- **Qwen3-VL-2B (incoerência):** a conclusão final contradiz as próprias respostas aos fatores.
- **LLaVA (credulidade):** aceita o fator "informação falsa" em todos os casos.
- **Qwen3-VL-4B (ceticismo):** quase nunca detecta ataque.

Na obstrução, a decisão geométrica do ViDDAR protege contra esses erros. Na manipulação e no
end-to-end, em que o VLM decide sozinho, a escolha do modelo é decisiva. Esse contraste é o objeto do
benchmark.

### 4.3 Escopo proposto

1. **Mais modelos.** Comparar VLMs abertos de famílias e tamanhos diferentes que caibam numa GPU de
   12 GB, com quantização em 4 bits quando necessário. Além dos três já avaliados, há candidatos como
   o Qwen2.5-VL-3B (já configurado) e o Qwen3-VL-8B em 4 bits. A lista final ainda será definida.
2. **Análise dos falsos positivos.** Medir FPR com intervalos de confiança e atribuir cada falso
   positivo da manipulação a um fator (alinhamento, estilo ou informação) ou a uma conclusão
   incoerente. Na obstrução, separar os erros de escolha do objeto-chave dos erros de
   localização/segmentação.
3. **Variância entre execuções.** Repetir com amostragem estocástica para estimar a estabilidade das
   métricas, um ponto que o próprio artigo aponta.
4. **Variantes de prompt (opcional, como mitigação).** Avaliar variantes como acréscimo, mantendo
   sempre o prompt original do artigo como referência.

### 4.4 Limitações

- O estudo deixa de tratar de placas degradadas e de condições realistas de rua.
- Com 114 pares de manipulação, diferenças de poucos pontos percentuais entre modelos não são
  estatisticamente significativas. Os intervalos de confiança serão sempre reportados.
- Usamos apenas modelos de código aberto. A comparação com o GPT-4o limita-se aos números publicados.

---

## 5. Se o VSigns for obtido no futuro

A metodologia de pares C0–C4 do Draft 2 continua válida, e o pipeline foi escrito para ser
independente do dataset: um conjunto novo entra por um script de preparação que gera o manifesto de
pares. Se o VSigns for disponibilizado com as imagens originais, a separação entre reais e sintéticas
e informação sobre o estado das placas, a análise de placas degradadas pode voltar como **extensão**
do benchmark, sem refazer o trabalho atual.
