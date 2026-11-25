# Verificação 1:1 de impressões digitais com embeddings de CNN

Projeto final da disciplina de Biometria (UTFPR Ponta Grossa), Dupla 3. Autor: Lucas Ribeiro de Carvalho.

Esta pasta contém só o que o enunciado pede: um sistema de verificação 1:1 com uma CNN, desenvolvido na base principal e avaliado numa segunda base.

| Item | Valor |
|---|---|
| Modalidade | impressão digital, verificação 1:1 |
| Base principal | FVC2002 DB1 (treino, validação e teste) |
| Segunda base | FVC2004 DB1 (só avaliação, nunca treinada) |
| CNN | ConvNeXt-Tiny da torchvision |
| Origem dos pesos | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` |
| Dataset de pré-treinamento | ImageNet-1k |
| Camadas congeladas | `features.0` a `features.3` (entrada e estágios 1 e 2) |
| Camadas treinadas | `features.4` a `features.7`, LayerNorm final e a camada de embedding |
| Parâmetros | 28.014.432, dos quais 26.782.464 treináveis |
| Dimensão do embedding | 256, norma L2 |
| Função de perda | ArcFace, margem 0,3 e escala 30 |
| Matching | similaridade de cosseno |
| Limiar | ponto de EER nos pares de validação, congelado depois |
| Sementes | 42 para divisão e pares; 2026, 1999 e 1234 para os treinos |

## Resultados

Média ± desvio-padrão amostral de três sementes. FMR e FNMR no limiar congelado (0,308 ± 0,009). Taxas em %.

| Avaliação | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |
|---|---|---|---|---|---|---|
| FVC2002 DB1, 30 dedos de teste (Exp. A) | 9,75 ± 0,63 | 0,9656 ± 0,0038 | 10,45 ± 1,95 | 9,17 ± 2,63 | 55,32 ± 2,94 | 0,00 |
| FVC2004 DB1, 30 dedos de teste (Exp. C) | 14,18 ± 1,49 | 0,9281 ± 0,0094 | 13,61 ± 1,14 | 14,68 ± 2,54 | 25,56 ± 1,25 | 0,00 |
| FVC2004 DB1-A, protocolo oficial (Exp. C) | 14,49 ± 0,97 | 0,9230 ± 0,0083 | 11,91 ± 3,07 | 18,18 ± 2,26 | 22,88 ± 8,12 | 0,00 |

Custo: 6,48 ms por imagem em média (lote 1, NVIDIA L4), 112,1 MB de pesos, 2,6 minutos por treino.

`results/summary.md` traz também a validação, o Experimento D (desfoque e rotação em quatro níveis) e a ablação sem normalização. O artigo está em `artigo/main.pdf` e os slides em `presentation/apresentacao.pdf`.

O Experimento B pede as CNNs de duas equipes nas mesmas bases e pares. Aqui está a linha da nossa CNN. Para acrescentar outra CNN, gere um `scores.csv` para os pares de `protocols/<protocolo>/test_pairs.csv` e calcule as métricas com o limiar da validação dela.

## Dados

As bases não são redistribuídas. As imagens da FVC2002 DB1 e da FVC2004 DB1 (conjuntos A e B) vêm do material suplementar do *Handbook of Fingerprint Recognition* (Maltoni et al., 3ª edição). Coloque-as assim:

```
data/raw/FVC2002/Dbs/Db1_a/1_1.tif ... 100_8.tif
data/raw/FVC2002/Dbs/Db1_b/101_1.tif ... 110_8.tif
data/raw/FVC2004/Dbs/DB1_A/1_1.tif ... 100_8.tif
data/raw/FVC2004/Dbs/DB1_B/101_1.tif ... 110_8.tif
```

O nome do arquivo é `<dedo>_<impressão>.tif`. Cada base tem 110 dedos e 8 impressões por dedo.

## Pré-processamento e formato de entrada

Ordem do enunciado: ROI, escala de cinza, normalização e redimensionamento.

1. ROI: desvio-padrão em blocos de 16 × 16 pixels, limiar de Otsu, fechamento e abertura morfológicos, maior componente conexo.
2. Controle de qualidade: a amostra é uma falha de aquisição (FTA) quando a ROI tem menos de 40 mm².
3. Escala de cinza: as imagens já têm um canal de 8 bits.
4. Normalização de Hong, Wan e Jain: média 128 e variância 2500 dentro da ROI; valor 128 fora dela.
5. Redimensionamento: o menor quadrado em volta da ROI vai para 224 × 224 pixels.

A rede recebe um canal, 224 × 224, com valores em [-1, 1].

## Treino

- Lotes de 16 identidades com 4 impressões cada.
- AdamW, taxa de aprendizado 3e-4, decaimento de pesos 5e-4, decaimento cosseno, 80 épocas, precisão mista.
- Aumento de dados: rotação de até 30°, translação de até 10%, desfoque gaussiano (p = 0,3, σ de 0,1 a 1,5), brilho e contraste de 0,2.
- Fica o modelo da época com menor EER na validação.

Todos os parâmetros estão em `config.yaml`.

## Protocolo

`make_protocols.py` divide os 110 dedos de cada base em 60 de treino, 20 de validação e 30 de teste, com semente 42 e sem dedo repetido entre as partes. Cada pasta de divisão em `protocols/` tem `train.csv`, `validation.csv`, `test.csv`, `genuine_pairs.csv` e `impostor_pairs.csv`.

| Pasta | Uso | Pares genuínos / impostores |
|---|---|---|
| `protocols/fvc2002_db1` | treino e teste da base principal | 840 / 27.840 |
| `protocols/fvc2002_db1_val` | escolha da época e do limiar | 560 / 12.160 |
| `protocols/fvc2004_db1` | teste na segunda base | 840 / 27.840 |
| `protocols/fvc2004_db1_official` | protocolo oficial FVC no conjunto A | 2.800 / 4.950 |

O protocolo oficial da FVC reserva 10 dedos para ajuste, o que não basta para treinar e validar uma CNN com identidades separadas. Por isso a base principal usa o protocolo comum de semente fixa, e o oficial é aplicado na segunda base.

## Como reproduzir

Requisitos: Python 3.12 ou 3.13 e uma GPU com CUDA para o treino.

```bash
pip install -r requirements.txt     # ambiente de CPU; para GPU, instale o torch da sua versão de CUDA
bash scripts/run_pipeline.sh        # protocolos, pré-processamento, 6 treinos, avaliação, robustez, resumo
```

O script aceita etapas separadas:

```bash
STAGE=prep   bash scripts/run_pipeline.sh    # protocolos e pré-processamento
STAGE=train  bash scripts/run_pipeline.sh    # treinos, embeddings, experimentos A, B, C e ablação
STAGE=robust bash scripts/run_pipeline.sh    # experimento D
STAGE=report bash scripts/run_pipeline.sh    # results/summary.md e os arquivos na raiz de results/
```

Os comandos por trás de cada etapa, para uma semente:

```bash
python make_protocols.py --dataset fvc2002_db1 --mode split
python preprocess.py --dataset fvc2002_db1
python train.py --dataset fvc2002_db1 --protocol-dir protocols/fvc2002_db1 --run-name convnext_tiny_hong --seed 2026
python extract_embeddings.py --dataset fvc2002_db1 --checkpoint runs/convnext_tiny_hong/seed2026/best.pt \
    --out runs/convnext_tiny_hong/seed2026/emb_fvc2002_db1.npz
python evaluate.py --embeddings runs/convnext_tiny_hong/seed2026/emb_fvc2002_db1.npz \
    --protocol-dir protocols/fvc2002_db1_val --out results/fvc2002_db1_val/convnext_tiny_hong/seed2026
python evaluate.py --embeddings runs/convnext_tiny_hong/seed2026/emb_fvc2002_db1.npz \
    --protocol-dir protocols/fvc2002_db1 \
    --threshold-json results/fvc2002_db1_val/convnext_tiny_hong/seed2026/threshold.json \
    --out results/fvc2002_db1/convnext_tiny_hong/seed2026
python robustness.py --dataset fvc2002_db1 --checkpoint runs/convnext_tiny_hong/seed2026/best.pt \
    --protocol-dir protocols/fvc2002_db1 \
    --threshold-json results/fvc2002_db1_val/convnext_tiny_hong/seed2026/threshold.json \
    --out results/robustness/blur/convnext_tiny_hong/seed2026 --set robustness.perturbation=blur
```

Figuras do artigo: `python scripts/paper_figures.py`, `scripts/pipeline_figure.py` e `scripts/error_figure.py`. Testes: `pytest`.

## Arquivos

```
config.yaml              parâmetros de todas as etapas
preprocess.py            ROI, controle de qualidade, normalização, redimensionamento
model.py                 ConvNeXt-Tiny com a camada de embedding e a cabeça ArcFace
train.py                 treino com seleção da época pela validação
extract_embeddings.py    embeddings e tempo de inferência
evaluate.py              scores, limiar, métricas e gráficos
robustness.py            experimento D
make_protocols.py        divisão por identidade e pares
protocols/               divisões e pares (semente 42)
results/                 scores.csv, metrics.json, roc.png, distributions.png e resumos
artigo/                  artigo (IEEE, 6 a 8 páginas)
presentation/            slides da apresentação de 10 minutos
```

Em `results/`, os quatro arquivos da raiz são do Experimento A com a semente 2026. Cada avaliação tem os seus em `results/<protocolo>/<modelo>/seed<semente>/`. `convnext_tiny_hong` é o sistema principal e `convnext_tiny_none` é a ablação sem normalização.

## Versões e hardware

Os resultados desta pasta foram gerados no Google Colab:

- GPU NVIDIA L4 (23 GB), CPU Intel Xeon 2,20 GHz, 52 GB de RAM
- Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0), torchvision 0.29.1
- NumPy 2.4.6, OpenCV 4.14.0, pandas 2.2.3, scikit-learn 1.6.1

O pré-processamento das imagens originais rodou numa máquina local com as versões de `requirements.txt`. No Experimento D, o pré-processamento das imagens degradadas rodou no Colab.

## Versão de entrega

Repositório: <https://github.com/lucthesloth/biometria26.2>. Versão: tag `final-v1.0`.
