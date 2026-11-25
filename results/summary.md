# Resultados

Média ± desvio-padrão amostral de três sementes de treino. Taxas em %. FMR e FNMR no limiar congelado na validação.

## Validação, FVC2002 DB1 (20 dedos)

| Modelo | Limiar | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.3079 ± 0.0092 | 6.02 ± 0.37 | 0.9830 ± 0.0014 | 6.01 ± 0.37 | 6.07 ± 0.31 | 69.23 ± 1.50 | 0.00 ± 0.00 |
| ConvNeXt-Tiny sem normalização (ablação) | 0.2678 ± 0.0202 | 6.47 ± 0.72 | 0.9829 ± 0.0026 | 6.47 ± 0.72 | 6.49 ± 0.72 | 68.93 ± 4.04 | 0.00 ± 0.00 |

## Teste, FVC2002 DB1 (30 dedos), exp. A

| Modelo | Limiar | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.3079 ± 0.0092 | 9.75 ± 0.63 | 0.9656 ± 0.0038 | 10.45 ± 1.95 | 9.17 ± 2.63 | 55.32 ± 2.94 | 0.00 ± 0.00 |
| ConvNeXt-Tiny sem normalização (ablação) | 0.2678 ± 0.0202 | 9.61 ± 0.25 | 0.9646 ± 0.0022 | 10.24 ± 1.28 | 8.85 ± 1.32 | 58.97 ± 1.90 | 0.00 ± 0.00 |

## Teste, FVC2004 DB1 (30 dedos), exp. C

| Modelo | Limiar | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.3079 ± 0.0092 | 14.18 ± 1.49 | 0.9281 ± 0.0094 | 13.61 ± 1.14 | 14.68 ± 2.54 | 25.56 ± 1.25 | 0.00 ± 0.00 |
| ConvNeXt-Tiny sem normalização (ablação) | 0.2678 ± 0.0202 | 13.66 ± 0.50 | 0.9294 ± 0.0054 | 11.74 ± 1.76 | 16.63 ± 1.69 | 32.18 ± 3.68 | 0.00 ± 0.00 |

## FVC2004 DB1-A, protocolo oficial, exp. C

| Modelo | Limiar | EER | AUC | FMR | FNMR | TAR@FAR=1% | FTA |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 0.3079 ± 0.0092 | 14.49 ± 0.97 | 0.9230 ± 0.0083 | 11.91 ± 3.07 | 18.18 ± 2.26 | 22.88 ± 8.12 | 0.00 ± 0.00 |
| ConvNeXt-Tiny sem normalização (ablação) | 0.2678 ± 0.0202 | 14.78 ± 0.50 | 0.9197 ± 0.0026 | 12.34 ± 1.39 | 18.37 ± 1.94 | 21.87 ± 1.78 | 0.00 ± 0.00 |

## Experimento D: robustez, 30 dedos de teste da FVC2002 DB1

| Perturbação | Nível | EER | AUC | FMR | FNMR | FTA |
|---|---|---|---|---|---|---|
| blur | original (0) | 9.75 ± 0.63 | 0.9656 ± 0.0038 | 10.45 ± 1.95 | 9.17 ± 2.63 | 0.00 ± 0.00 |
| blur | low (1) | 9.34 ± 0.84 | 0.9688 ± 0.0030 | 11.05 ± 1.98 | 7.98 ± 3.00 | 0.00 ± 0.00 |
| blur | medium (2) | 11.39 ± 0.96 | 0.9535 ± 0.0058 | 13.30 ± 2.01 | 10.44 ± 2.83 | 0.00 ± 0.00 |
| blur | high (3) | 21.19 ± 0.21 | 0.8630 ± 0.0069 | 24.55 ± 5.42 | 18.53 ± 3.70 | 1.25 ± 0.00 |
| rotation | original (0) | 9.75 ± 0.63 | 0.9656 ± 0.0038 | 10.45 ± 1.95 | 9.17 ± 2.63 | 0.00 ± 0.00 |
| rotation | low (10) | 10.33 ± 0.57 | 0.9602 ± 0.0037 | 10.14 ± 2.06 | 10.44 ± 1.67 | 0.00 ± 0.00 |
| rotation | medium (20) | 11.43 ± 1.17 | 0.9547 ± 0.0067 | 9.66 ± 2.17 | 13.10 ± 2.15 | 0.00 ± 0.00 |
| rotation | high (30) | 12.55 ± 1.35 | 0.9455 ± 0.0110 | 9.19 ± 1.69 | 17.06 ± 3.26 | 0.00 ± 0.00 |

## Custo

| Modelo | Parâmetros | Treináveis | Pesos (MB) | Inferência média (ms) | Treino (min) | Hardware |
|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | 28,014,432 | 26,782,464 | 112.1 | 6.48 | 2.6 | NVIDIA L4 |
| ConvNeXt-Tiny sem normalização (ablação) | 28,014,432 | 26,782,464 | 112.1 | 6.48 | 2.6 | NVIDIA L4 |
