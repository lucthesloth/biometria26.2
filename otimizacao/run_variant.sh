#!/usr/bin/env bash
# Trains one ConvNeXt-Tiny variant and evaluates it like scripts/run_pipeline.sh does, into results/otimizacao/.
#   bash otimizacao/run_variant.sh <name> <seed> <normalization> <train dataset> [--set key=value ...]
# <train dataset> is fvc2002_db1 (the base 60 fingers), fvc2002_pool (plus FVC2002 DB2 to DB4) or wide_pool.
# PROTO=<folder under protocols/> overrides the training protocol (wide_val holds validation fingers out of wide_pool).
# The epoch and the threshold come from the 20 FVC2002 DB1 validation fingers; the tests reuse both unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$1 SEED=$2 NORM=$3 TRAIN=$4; shift 4
PY=${PY:-.venv/Scripts/python.exe}
export MPLBACKEND=Agg
S=(--set preprocess.normalization=$NORM "$@")
R=runs/otimizacao/$NAME/seed$SEED
OUT=results/otimizacao
if [ ! -f $R/run.json ]; then
  $PY train.py --dataset $TRAIN --protocol-dir protocols/${PROTO:-$TRAIN} --run-name $NAME --seed $SEED \
      --set train.runs_dir=runs/otimizacao "${S[@]}"
fi
if [ $TRAIN = wide_pool ]; then
  $PY extract_embeddings.py --dataset wide_pool --checkpoint $R/best.pt --out $R/emb_wide_pool.npz "${S[@]}"
fi
for D in fvc2002_db1 fvc2004_db1; do
  $PY extract_embeddings.py --dataset $D --checkpoint $R/best.pt --out $R/emb_$D.npz "${S[@]}"
done
$PY evaluate.py --embeddings $R/emb_fvc2002_db1.npz --protocol-dir protocols/fvc2002_db1_val \
    --qc data/processed/fvc2002_db1/$NORM/qc.csv --out $OUT/fvc2002_db1_val/$NAME/seed$SEED
for P in fvc2002_db1 fvc2004_db1 fvc2004_db1_official; do
  D=${P%_official}
  $PY evaluate.py --embeddings $R/emb_$D.npz --protocol-dir protocols/$P \
      --threshold-json $OUT/fvc2002_db1_val/$NAME/seed$SEED/threshold.json \
      --qc data/processed/$D/$NORM/qc.csv --out $OUT/$P/$NAME/seed$SEED
done
