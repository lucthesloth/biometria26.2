#!/usr/bin/env bash
# Reproduces every number of the paper from the raw FVC2002 DB1 and FVC2004 DB1 images.
#   bash scripts/run_pipeline.sh                    # everything, three seeds
#   SEEDS=2026 STAGE=train bash scripts/run_pipeline.sh
# STAGE: prep (protocols and preprocessing) | train (training, embeddings, experiments A, B, C and ablation)
#        | robust (experiment D) | report (summary tables) | all
set -euo pipefail
cd "$(dirname "$0")/.."
MAIN=fvc2002_db1          # main base: seed-42 identity split 60/20/30 of the 110 fingers
SECOND=fvc2004_db1        # second base, never trained on
VAL=${MAIN}_val           # the 20 validation fingers: epoch selection and the frozen threshold
TESTS="fvc2002_db1 fvc2004_db1 fvc2004_db1_official"
SEEDS=${SEEDS:-"2026 1999 1234"}
PY=${PY:-python}
STAGE=${STAGE:-all}
MAIN_RUN=convnext_tiny_hong
export MPLBACKEND=Agg

step() { echo "[$(date +%H:%M:%S)] $*"; }
base_of() { case $1 in fvc2004_db1*) echo fvc2004_db1 ;; *) echo fvc2002_db1 ;; esac; }

if [ "$STAGE" = prep ] || [ "$STAGE" = all ]; then
  step "protocols (seed 42)"
  $PY make_protocols.py --dataset $MAIN --mode split
  $PY make_protocols.py --dataset $SECOND --mode split
  $PY make_protocols.py --dataset $SECOND --mode official
  step "preprocessing: Hong normalisation (main) and no normalisation (ablation)"
  for NORM in hong none; do
    for D in $MAIN $SECOND; do
      $PY preprocess.py --dataset $D --set preprocess.normalization=$NORM
    done
  done
fi

run_arm() {
  local RUN=$1 NORM=$2 SEED=$3
  local S=(--set preprocess.normalization=$NORM)
  local R=runs/$RUN/seed$SEED
  if [ ! -f $R/run.json ]; then
    step "train $RUN seed$SEED"
    $PY train.py --dataset $MAIN --protocol-dir protocols/$MAIN --run-name $RUN --seed $SEED "${S[@]}"
  fi
  step "evaluate $RUN seed$SEED"
  for D in $MAIN $SECOND; do
    $PY extract_embeddings.py --dataset $D --checkpoint $R/best.pt --out $R/emb_$D.npz "${S[@]}"
  done
  # Threshold: EER point of the validation pairs, frozen for every test below.
  $PY evaluate.py --embeddings $R/emb_$MAIN.npz --protocol-dir protocols/$VAL \
      --qc data/processed/$MAIN/$NORM/qc.csv --out results/$VAL/$RUN/seed$SEED
  for P in $TESTS; do
    local D; D=$(base_of $P)
    $PY evaluate.py --embeddings $R/emb_$D.npz --protocol-dir protocols/$P \
        --threshold-json results/$VAL/$RUN/seed$SEED/threshold.json \
        --qc data/processed/$D/$NORM/qc.csv --out results/$P/$RUN/seed$SEED
  done
}

if [ "$STAGE" = train ] || [ "$STAGE" = all ]; then
  for SEED in $SEEDS; do
    run_arm $MAIN_RUN hong $SEED
    run_arm convnext_tiny_none none $SEED       # ablation: same network and schedule, no normalisation
  done
fi

if [ "$STAGE" = robust ] || [ "$STAGE" = all ]; then
  for SEED in $SEEDS; do
    for PERT in blur rotation; do
      OUT=results/robustness/$PERT/$MAIN_RUN/seed$SEED
      if [ -f $OUT/robustness.json ]; then step "skip $OUT (done)"; continue; fi
      step "robustness $PERT seed$SEED"
      $PY robustness.py --dataset $MAIN --checkpoint runs/$MAIN_RUN/seed$SEED/best.pt --protocol-dir protocols/$MAIN \
          --threshold-json results/$VAL/$MAIN_RUN/seed$SEED/threshold.json \
          --out $OUT --set robustness.perturbation=$PERT
    done
  done
fi

if [ "$STAGE" = report ] || [ "$STAGE" = all ]; then
  step "summary"
  $PY scripts/aggregate.py
  # The four files the assignment names, at the top of results/: experiment A, first seed.
  cp results/$MAIN/$MAIN_RUN/seed${SEEDS%% *}/{scores.csv,metrics.json,roc.png,distributions.png} results/
fi
step "done ($STAGE)"
