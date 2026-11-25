<img src="assets/icon.svg" width="64" height="64" alt="">

# Verificação de impressão digital 1:1 com embeddings CNN

Projeto final da disciplina de Biometria (UTFPR Ponta Grossa), Dupla 3: impressão digital, base principal
**FVC2002 DB1**, segunda base **FVC2004 DB1**. Cada imagem vira um embedding de 256 dimensões gerado por uma CNN.
Duas amostras são comparadas por similaridade de cosseno contra um limiar escolhido na validação e congelado.

O artigo está em [`artigo/`](artigo/) (template IEEE de conferência).

## Dados

As duas bases têm o conjunto A (100 dedos) e o conjunto B (10 dedos), 8 impressões por dedo, 880 imagens cada. Os
dois conjuntos estão no material suplementar do capítulo 4 do *Handbook of Fingerprint Recognition* (3ª ed.);
o conjunto B desse material é idêntico, byte a byte, ao do site da competição. Como obter e organizar:
[data/README.md](data/README.md).

- **Treino, validação e teste (experimento A):** FVC2002 DB1, divisão por identidade com seed 42 em 60/20/30
  dedos (`protocols/fvc2002_db1/`). A validação escolhe a época e fixa o limiar.
- **Segunda base (experimento C, cross-dataset):** FVC2004 DB1, nunca usada no treino. Avaliada nos 30 dedos de
  teste da mesma divisão (`protocols/fvc2004_db1/`) e no protocolo
  oficial FVC do conjunto A (`protocols/fvc2004_db1_official/`, 2.800 genuínos e 4.950 impostores).

## Ficha do sistema

| Item | Valor |
|---|---|
| Modalidade | Impressão digital, verificação 1:1 |
| Datasets | FVC2002 DB1 (Identix TouchView II, 388x374, 500 dpi), conjuntos A e B, 110 dedos: treino, validação e teste. FVC2004 DB1 (CrossMatch V300, 640x480, 500 dpi), conjuntos A e B: só teste |
| CNN | ConvNeXt-Tiny (principal, registrada com o professor), comparada com ResNet-50 e EfficientNet-B0, torchvision |
| Origem dos pesos | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, `ResNet50_Weights.IMAGENET1K_V2` e `EfficientNet_B0_Weights.IMAGENET1K_V1`, pré-treino ImageNet-1k |
| Camadas congeladas | ConvNeXt-Tiny: `features.0` a `features.3` (stem, estágio 1, redução e estágio 2). ResNet-50: `conv1`, `bn1`, `layer1`, `layer2`. EfficientNet-B0: `features.0` a `features.3`. Pesos sem gradiente e módulos congelados em modo eval durante o treino |
| Camadas treinadas | ConvNeXt-Tiny: `features.4` a `features.7` (estágios 3 e 4) e a LayerNorm após o pooling; ResNet-50: `layer3`, `layer4`; EfficientNet-B0: `features.4` a `features.8`; todas: Dropout 0,2 + Linear -> 256 + BatchNorm1d e a cabeça ArcFace |
| Pré-processamento | Reamostragem para 500 dpi -> controle de qualidade (FTA se a área útil < 40 mm²) -> ROI por desvio-padrão em blocos de 16 px (Otsu, morfologia, maior componente, preenchimento) -> escala de cinza -> normalização de Hong (média 128, variância 2500) -> realce de cristas com 16 filtros de Gabor escolhidos pela orientação local -> fundo 128 -> janela fixa 320x320 no centroide da ROI, sem redimensionar cada imagem |
| Entrada | 1 canal, 320x320 px na resolução nativa de 500 dpi |
| Embedding | 256 dimensões, normalizado L2 |
| Perda e treino | ArcFace (m = 0,3, s = 30), batches P x K = 16 dedos x 4 impressões, AdamW (lr 3e-4, weight decay 5e-4), cosseno em 80 épocas, bfloat16 na GPU, augmentação por rotação de até 30°, translação de 10%, blur e brilho/contraste. Melhor época pelo EER de validação |
| Matching | Similaridade de cosseno; score >= limiar => genuíno |
| Regra do limiar | Limiar de EER nos pares de validação (20 dedos da FVC2002 DB1), congelado e reutilizado sem mudança nos testes das duas bases |
| Ajuste de hiperparâmetros | Nenhum nesta fase. A configuração (realce de Gabor, 80 épocas, lr 3e-4) veio de uma fase anterior, sem o conjunto A, em que o treino usava bases de outros sensores ([results/fase_anterior/](results/fase_anterior/)). Naquela fase o conjunto B das duas bases era o teste, e parte das escolhas foi informada por ele; dois desses dedos (107 e 109) caíram no teste atual, e `results/sensitivity_setb.csv` traz a EER sem eles |
| Nº de parâmetros | ConvNeXt-Tiny: 28,01 M, 26,78 M treináveis (112,1 MB de state_dict). ResNet-50: 24,03 M (96,4 MB). EfficientNet-B0: 4,34 M (17,6 MB) |
| Seeds | Divisão e pares: 42. Treinos: 2026, 1999 e 1234, média ± desvio |
| Hardware | AMD Ryzen 5 5600X (6 núcleos), 32 GB de RAM, NVIDIA GeForce RTX 4060 8 GB, Windows 11. Um treino leva cerca de 5,5 min (ConvNeXt-Tiny), 5 min (ResNet-50) e 3,4 a 4,8 min (EfficientNet-B0) |
| Python e bibliotecas | Python 3.13.13 (requer 3.12 ou 3.13), PyTorch 2.6.0+cu124, torchvision 0.21.0, NumPy 2.5.4, OpenCV 5.0.0, scikit-image 0.26.0, scikit-learn 1.9.1, pandas 3.1.1. Versões travadas em `uv.lock` e `requirements.txt` |

## Protocolo

- Divisão por identidade (dedo), nunca por imagem, com seed 42: 60 dedos de treino, 20 de validação e 30 de teste.
- Validação: 560 pares genuínos e 12.160 impostores (todos os pares entre dedos diferentes).
- Teste em cada base: 840 genuínos e 27.840 impostores entre os 30 dedos de teste. Protocolo oficial na FVC2004
  DB1-A: todas as impressões do mesmo dedo (2.800) e a primeira impressão de cada dedo contra a primeira dos
  outros (4.950).
- Os arquivos `test_pairs.csv` nomeiam amostras pelo nome original (`7_1.tif`). Para o experimento B,
  `scripts/import_scores.py` confere os pares antes de importar o `scores.csv` de outra equipe.
- FTA é a taxa por amostra; `metrics.json` traz também `pairs_dropped`. Na robustez nenhum par é descartado e o
  FTA aparece em coluna separada.

## Como reproduzir

```bash
# dependências (GPU com CUDA 12.4). Em CPU troque cu124 por cpu.
uv sync --extra cu124 --extra dev
uv run --extra cu124 pytest -q

# baixe os dados conforme data/README.md e então rode tudo (protocolos, pré-processamento, 12 treinos, referências, fusão,
# avaliação, robustez, tabelas e figuras do artigo):
bash scripts/run_pipeline.sh
# CPU com uma seed: EXTRA=cpu SEEDS=2026 bash scripts/run_pipeline.sh

# compilar o artigo
cd artigo && latexmk -pdf main.tex
```

Não use `uv run` sem `--extra`: ele sincroniza o ambiente sem o extra do PyTorch e desinstala o torch.
O script pula etapas cujo resultado já existe; `FORCE=1` refaz tudo. Qualquer chave de `config.yaml` pode ser
sobrescrita com `--set chave.subchave=valor`. Sem `uv`:
`pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`.

## Resultados

EER, média ± desvio sobre 3 seeds de treino, todos os pares. Tabelas completas em
[results/summary.md](results/summary.md), intervalos em [results/bootstrap_ci.csv](results/bootstrap_ci.csv).

| Execução | Validação | FVC2002 DB1, teste (A) | FVC2004 DB1, teste (C) | FVC2004 DB1-A, oficial (C) |
|---|---|---|---|---|
| **ConvNeXt-Tiny com realce de Gabor** | 3,84 ± 0,44% | 7,87 ± 0,29% | 8,32 ± 0,44% | 10,42 ± 0,17% |
| ResNet-50 com realce | 5,36 ± 0,36% | 9,54 ± 0,95% | 13,20 ± 0,64% | 14,78 ± 1,02% |
| EfficientNet-B0 com realce | 5,10 ± 0,76% | 7,92 ± 0,40% | 12,24 ± 0,39% | 14,08 ± 0,46% |
| ConvNeXt-Tiny sem realce (ablação) | 4,38 ± 0,54% | 6,89 ± 0,78% | 10,35 ± 0,72% | 11,29 ± 0,84% |
| ConvNeXt-Tiny da ImageNet, sem ajuste fino | 27,50% | 29,49% | 34,88% | 34,75% |
| SourceAFIS 3.18.1 (minúcias) | 2,61% | 2,71% | 9,31% | 12,18% |
| Fusão ConvNeXt-Tiny + SourceAFIS | 1,49 ± 0,10% | 2,73 ± 0,11% | 4,96 ± 0,25% | 6,86 ± 0,41% |

Intervalos de confiança de 95% por bootstrap sobre dedos (`scripts/bootstrap_ci.py`), diferenças pareadas:

- EER da ConvNeXt-Tiny: 5,6 a 10,0% na FVC2002 DB1, 6,3 a 11,0% nos dedos de teste da FVC2004 DB1 e 9,1 a
  11,9% no protocolo oficial.
- Na FVC2002 DB1 as três CNNs empatam: os intervalos das diferenças para a ConvNeXt-Tiny contêm zero.
- Na FVC2004 DB1 a ResNet-50 fica 4,9 e 4,4 pontos acima da ConvNeXt-Tiny e a EfficientNet-B0 3,9 e 3,7 pontos
  acima, todos com intervalos que excluem zero.
- Sem realce, a EER muda -1,0 ponto na FVC2002 DB1 (-2,8 a 0,7) e +2,0 pontos nos dedos de teste da FVC2004 DB1
  (0,3 a 3,4). No protocolo oficial a diferença é de +0,9 ponto (-0,6 a 2,3), com zero dentro do intervalo. O
  realce custa de 76 a 125 ms por imagem em CPU, contra 20 a 30 ms sem ele.
- Sem os dedos 107 e 109, que foram teste na fase anterior, a EER da ConvNeXt-Tiny é 7,95% na FVC2002 DB1 e
  7,94% na FVC2004 DB1, e a ordem das redes só muda entre a ConvNeXt-Tiny e a EfficientNet-B0 na FVC2002 DB1, que já empatavam
  (`scripts/sensitivity_setb.py`).
- Os intervalos cobrem a amostra de dedos para as três seeds treinadas; não incluem a variação entre treinos.
- O SourceAFIS fica 5,2 pontos abaixo da ConvNeXt-Tiny na FVC2002 DB1 (3,4 a 7,0). Na FVC2004 DB1 a diferença
  inverte de sinal e os intervalos contêm zero.
- A fusão empata com o SourceAFIS na FVC2002 DB1 e fica 4,4 e 5,3 pontos abaixo dele na FVC2004 DB1.

As referências vêm de `scripts/baseline_imagenet.py`, `scripts/baseline_sourceafis.py` e `scripts/fusion.py`
(JDK 17+ e Maven para o SourceAFIS; sem eles o pipeline para com erro, ou segue sem SourceAFIS, fusão e
intervalos com `SKIP_SOURCEAFIS=1`).

O Experimento B do enunciado prevê a CNN de uma segunda equipe. Os scores dela não foram recebidos, então a
comparação cobre as três redes deste repositório; `scripts/import_scores.py` inclui scores externos nos mesmos
pares.

Tabela padronizada (seção 26 do enunciado), ConvNeXt-Tiny, 30 dedos de teste de cada base:

| Informação | FVC2002 DB1 (principal) | FVC2004 DB1 (cross-dataset) |
|---|---|---|
| Limiar | 0,368 ± 0,010 (EER na validação, congelado) | o mesmo |
| EER | 7,87 ± 0,29% | 8,32 ± 0,44% |
| AUC | 0,976 ± 0,001 | 0,966 ± 0,003 |
| FMR no limiar | 6,39 ± 1,02% | 6,92 ± 1,06% |
| FNMR no limiar | 9,56 ± 1,34% | 10,79 ± 2,65% |
| TAR@FAR=1% | 68,81 ± 2,38% | 46,47 ± 2,15% |
| FTA | 0% | 0% |
| Tempo de inferência | média 9,67 ms, mediana 8,87 ms por imagem na rede (batch 1, RTX 4060) | o mesmo |
| Tamanho do modelo | 28,01 M parâmetros, 112,1 MB | o mesmo |

Robustez nos dedos de teste da FVC2002 DB1 (ConvNeXt-Tiny, limiar congelado): blur com σ = 2 px leva o EER a
18,0% e a FNMR a 21%; com σ = 3 px o EER vai a 27,2% e o FMR no limiar a 76%. A hipótese, não testada, é que o
realce produz cristas a partir de uma imagem sem detalhe e os impostores passam a parecer genuínos. O nível
σ = 1 px está dentro da faixa de blur usada no treino. Rotação de 30° leva o EER a 10,7% e
a FNMR a 18,5%. Detalhes em `results/robustness_summary.csv` e `artigo/figures/robustness.pdf`.

## Configuração base e configuração otimizada (ConvNeXt-Tiny)

Uma tentativa separada de reduzir FMR e FNMR da ConvNeXt-Tiny está em [otimizacao/](otimizacao/README.md), em duas
etapas. Os resultados acima continuam os da configuração base.

| Item | Base | Etapa 1 | Etapa 2 |
|---|---|---|---|
| Dados de treino | 60 dedos da FVC2002 DB1 | 390 dedos: os mesmos 60 mais as bases DB2, DB3 e DB4 da FVC2002 | 745 dedos: a etapa 1 mais FVC2004 DB2, DB3 e DB4 e U.are.U |
| Validação | 20 dedos da FVC2002 DB1 | os mesmos | os mesmos mais 20 dedos da FVC2004 DB2 e 20 da DB3 |
| Redes | uma, com realce de Gabor | duas, Gabor e Hong | 12: Gabor e Hong, dois níveis de rotação e translação, três sementes |
| Score | cosseno | soma dos dois cossenos | soma dos 12, cada um normalizado contra os 100 dedos de treino mais próximos |
| EER, FVC2002 DB1 teste | 7,87% | 4,05% | 2,74% |
| FMR / FNMR, FVC2002 DB1 teste | 6,39% / 9,56% | 3,56% / 4,56% | 2,75% / 2,74% |
| EER, FVC2004 DB1 teste | 8,32% | 5,48% | 3,33% |
| FMR / FNMR, FVC2004 DB1 teste | 6,92% / 10,79% | 3,69% / 8,81% | 2,06% / 6,43% |
| EER, FVC2004 DB1-A oficial | 10,42% | 6,23% | 3,82% |
| FMR / FNMR, FVC2004 DB1-A oficial | 7,57% / 14,10% | 3,52% / 10,95% | 1,88% / 7,50% |

Base e etapa 1 são médias de três sementes; a etapa 2 soma as três sementes num só sistema. Na etapa 2 as nove
diferenças para a base têm intervalo de 95% que exclui zero. Com as quatro redes de uma semente só, a
etapa 2 dá em média EER de 3,07%, 3,81% e 4,30%. Dez outras tentativas foram descartadas pela validação e estão listadas em
otimizacao/README.md. As duas etapas treinam em outras bases, então não seguem a regra "só FVC2002 DB1" dos
resultados principais; a etapa 2 usa também a FVC2004 DB2 a DB4, de outros sensores e outros voluntários que a
FVC2004 DB1, que continua fora de todo treino.

## Projeto paralelo: templates protegidos

[template_protection/](template_protection/README.md) implementa o hashing discreto com rótulos de Raja,
Raghavendra e Busch (IET Biometrics, 2022) sobre os embeddings da ConvNeXt-Tiny. É independente do artigo e da
apresentação.

## Estrutura

```
config.yaml              parâmetros de tudo (datasets, pré-processamento, modelo, treino, avaliação, robustez)
make_protocols.py        divisões por identidade e pares genuíno/impostor (split, test, official)
preprocess.py            reamostragem -> QC -> ROI -> normalização -> realce de Gabor -> janela fixa, gera qc.csv (FTA)
model.py                 backbone torchvision -> embedding L2; cabeça ArcFace; congelamento por estágio
train.py                 fine-tuning com sampler P x K, seleção pelo EER de validação
extract_embeddings.py    embeddings + tempo de inferência (batch 1, mediana de 100 repetições)
evaluate.py              scores.csv, metrics.json, threshold.json, roc.png, det.png, distributions.png, errors.csv
robustness.py            experimento D: blur ou rotação em 4 níveis com limiar congelado
fpverif/                 biblioteca: dados, protocolo, imagem, métricas, gráficos, seeds
scripts/                 run_pipeline.sh, baseline_*.py, fusion.py, bootstrap_ci.py, aggregate.py, paper_figures.py, pipeline_figure.py, error_figure.py, import_scores.py
otimizacao/              tentativa de otimização da ConvNeXt-Tiny em duas etapas: mais bases no treino, soma de redes, tentativas descartadas
template_protection/     projeto paralelo: templates protegidos por hashing discreto com rótulos
baselines/sourceafis/    comparador de minúcias de referência (Java, Maven)
protocols/               CSVs de divisão e pares (gerados, versionados)
results/                 scores.csv e metrics.json de cada execução, tabelas agregadas; na raiz, os arquivos do experimento A (ConvNeXt-Tiny, seed 2026)
artigo/                  artigo IEEE (main.tex, refs.bib, figures/)
tests/                   testes unitários (métricas, protocolo, dados, pré-processamento, modelo, avaliação)
```

## Versão avaliada

Repositório: <https://github.com/lucthesloth/biometria26.2>. Versão avaliada: tag `final-v1.0`.
