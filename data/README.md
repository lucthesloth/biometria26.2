# Dados

Os datasets não são redistribuídos neste repositório (`data/raw/` e `data/processed/` estão no `.gitignore`). Os resultados principais do artigo usam
só as bases FVC2002 DB1 e FVC2004 DB1, conjuntos A e B:

```
data/raw/
  FVC2002/Dbs/Db1_a/1_1.tif ... 100_8.tif       treino, validação e teste (Experimento A)
  FVC2002/Dbs/Db1_b/101_1.tif ... 110_8.tif
  FVC2004/Dbs/DB1_A/1_1.tif ... 100_8.tif       segunda base, nunca treinada (Experimento C)
  FVC2004/Dbs/DB1_B/101_1.tif ... 110_8.tif
```

É a árvore dos DVDs das competições, com as quatro bases de cada uma lado a lado em `Dbs/`. O `config.yaml`
escolhe as pastas da DB1 pela chave `subdirs`; as pastas DB2 a DB4 ficam no lugar e só são lidas pela etapa de `otimizacao/`.

O conjunto B sozinho também é um download gratuito no site de cada competição
(`http://bias.csr.unibo.it/fvc2002/Downloads/DB1_B.zip`, idem para `fvc2004`).

Os conjuntos B da FVC2000 listados em `config.yaml` só serviam ao desenho anterior, que treinava num pool público
enquanto o conjunto A não estava disponível. Eles não entram nos resultados atuais, e as pastas da FVC2000 não estão
mais em `data/raw`. A base U.are.U da Neurotechnology (`data/raw/Neurotechnology/UareU`) entra só na segunda etapa
de `otimizacao/`.

## Conjunto A

O conjunto A de cada base FVC (100 dedos x 8 impressões) e o conjunto B estão no material suplementar do capítulo 4
do *Handbook of Fingerprint Recognition* (3ª ed., Springer, 2022):
`https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4/MediaObjects/74034_3_En_4_MOESM1_ESM.zip`
(870 MB, as 12 bases FVC2000, FVC2002 e FVC2004). O zip traz também executáveis de exemplo da competição: não os
execute. Copie as pastas `FVC2002/Dbs` e `FVC2004/Dbs` do zip para `data/raw/`, com os nomes originais. As 7.040
imagens em `data/raw/FVC2002` e `data/raw/FVC2004` foram conferidas por SHA-256 contra esse zip e são idênticas.

Com os dois conjuntos no lugar, `scripts/run_pipeline.sh` gera a divisão 60/20/30 por identidade e o protocolo
oficial (2.800 genuínos, 4.950 impostores) sem mudar o código. O conjunto B do zip é idêntico ao do site da
competição.

`data/processed/<dataset>/<normalização>/` é produzido por `preprocess.py` e pode ser apagado e regerado.
