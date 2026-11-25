#!/usr/bin/env bash
# Reproduces every number and figure in the paper from the raw data in data/raw/ (see data/README.md).
#   bash scripts/run_pipeline.sh                          # everything (GPU with CUDA 12.4 wheels)
#   STAGE=gpu SKIP_PREP=1 bash scripts/run_pipeline.sh    # trainings, robustness, ImageNet embeddings (e.g. on Colab)
#   STAGE=cpu SKIP_PREP=1 bash scripts/run_pipeline.sh    # SourceAFIS, fusion, tables and figures (needs a JDK)
#   EXTRA=cpu SEEDS=2026 bash scripts/run_pipeline.sh     # CPU only, one seed
# Plain `uv run` would re-sync without the torch extra and uninstall torch, so the extra is always named.
set -euo pipefail
cd "$(dirname "$0")/.."

MAIN=fvc2002_db1          # trained on: seed-42 identity split 60/20/30 of sets A + B (110 fingers)
SECOND=fvc2004_db1        # never trained on: cross-dataset base
VAL=${MAIN}_val           # the 20 validation fingers: model selection and the frozen threshold
TESTS="fvc2002_db1 fvc2004_db1 fvc2004_db1_official"   # protocol folders, see config.yaml `evaluations`
SEEDS=${SEEDS:-"2026 1999 1234"}       # training seeds; protocols and pairs use seed 42 (config.yaml)
EXTRA=${EXTRA:-cu124}
PY=${PY:-"uv run --extra $EXTRA python"}
STAGE=${STAGE:-all}       # gpu | cpu | all
# FORCE=1 retrains arms whose final metrics already exist; by default they are skipped, so a rerun resumes.
FORCE=${FORCE:-0}
R50="--set model.backbone=resnet50 --set model.weights=IMAGENET1K_V2 --set train.freeze_until=layer2"
EFF="--set model.backbone=efficientnet_b0 --set model.weights=IMAGENET1K_V1 --set train.freeze_until=features.3"
MAIN_RUN=convnext_tiny_gabor
IMNET=imagenet_convnext_tiny

# One "[hh:mm:ss] ..." line per step, so a long run can be followed with: grep '^\[' on the log.
step() { echo "[$(date +%H:%M:%S)] $*"; }
base_of() { case $1 in fvc2004_db1*) echo fvc2004_db1 ;; *) echo fvc2002_db1 ;; esac; }

# SKIP_PREP=1 reuses protocols/ and data/processed/ from an earlier run.
if [ "${SKIP_PREP:-0}" != 1 ]; then
  step "protocols and preprocessing"
  # Protocols (seed 42): identity split of both bases (writes <base>_val too) and the official FVC protocol
  #    of the cross-dataset base.
  $PY make_protocols.py --dataset $MAIN --mode split
  $PY make_protocols.py --dataset $SECOND --mode split
  $PY make_protocols.py --dataset $SECOND --mode official
  for NORM in gabor hong; do
    for D in $MAIN $SECOND; do
      $PY preprocess.py --dataset $D --set preprocess.normalization=$NORM
    done
  done
fi

# Train on the 60 training fingers, pick the epoch and freeze the threshold on the 20 validation fingers, then
#    evaluate without any change on the FVC2002 DB1 test fingers (A) and on FVC2004 DB1 (C). B = 3 CNNs x 2 bases.
run_arm() {
  local RUN=$1 NORM=$2 SEED=$3; shift 3
  local S=(--set preprocess.normalization=$NORM "$@")
  local R=runs/$RUN/seed$SEED
  ARM=$((ARM + 1))
  if [ "$FORCE" != 1 ] && [ -f results/fvc2004_db1_official/$RUN/seed$SEED/metrics.json ]; then
    step "skip $RUN seed$SEED ($ARM/$N_ARMS, done)"
    return
  fi
  # run.json is written after the last epoch, so a finished training is reused.
  if [ "$FORCE" = 1 ] || [ ! -f $R/run.json ]; then
    step "train $RUN seed$SEED ($ARM/$N_ARMS)"
    $PY train.py --dataset $MAIN --protocol-dir protocols/$MAIN --run-name $RUN --seed $SEED "${S[@]}"
  fi
  step "evaluate $RUN seed$SEED ($ARM/$N_ARMS)"
  for D in $MAIN $SECOND; do
    $PY extract_embeddings.py --dataset $D --checkpoint $R/best.pt --out $R/emb_$D.npz "${S[@]}"
  done
  $PY evaluate.py --embeddings $R/emb_$MAIN.npz --protocol-dir protocols/$VAL \
      --qc data/processed/$MAIN/$NORM/qc.csv --out results/$VAL/$RUN/seed$SEED
  for P in $TESTS; do
    local D; D=$(base_of $P)
    $PY evaluate.py --embeddings $R/emb_$D.npz --protocol-dir protocols/$P \
        --threshold-json results/$VAL/$RUN/seed$SEED/threshold.json \
        --qc data/processed/$D/$NORM/qc.csv --out results/$P/$RUN/seed$SEED
  done
}

if [ "$STAGE" != cpu ]; then
  ARM=0
  N_ARMS=$((4 * $(echo $SEEDS | wc -w)))
  for SEED in $SEEDS; do
    run_arm $MAIN_RUN gabor $SEED
    run_arm convnext_tiny_hong hong $SEED          # ablation: same network and schedule, no Gabor enhancement
    run_arm resnet50_gabor gabor $SEED $R50
    run_arm efficientnet_b0_gabor gabor $SEED $EFF
  done

  for SEED in $SEEDS; do
    for PERT in blur rotation; do
      OUT=results/robustness/${PERT}_both/$MAIN_RUN/seed$SEED
      if [ "$FORCE" != 1 ] && [ -f $OUT/robustness.json ]; then step "skip $OUT (done)"; continue; fi
      step "robustness $PERT seed$SEED"
      $PY robustness.py --dataset $MAIN --checkpoint runs/$MAIN_RUN/seed$SEED/best.pt --protocol-dir protocols/$MAIN \
          --threshold-json results/$VAL/$MAIN_RUN/seed$SEED/threshold.json \
          --out $OUT --set robustness.perturbation=$PERT --set robustness.apply_to=both
    done
  done

  # Reference: ImageNet ConvNeXt-Tiny features without fine-tuning (Gabor input), on the same pairs.
  step "ImageNet baseline (no fine-tuning)"
  $PY scripts/baseline_imagenet.py
  R=runs/$IMNET/seed0
  $PY evaluate.py --embeddings $R/emb_$MAIN.npz --protocol-dir protocols/$VAL \
      --qc data/processed/$MAIN/gabor/qc.csv --out results/$VAL/$IMNET/seed0
  for P in $TESTS; do
    D=$(base_of $P)
    $PY evaluate.py --embeddings $R/emb_$D.npz --protocol-dir protocols/$P \
        --threshold-json results/$VAL/$IMNET/seed0/threshold.json \
        --qc data/processed/$D/gabor/qc.csv --out results/$P/$IMNET/seed0
  done
fi

if [ "$STAGE" != gpu ]; then
  # Reference: SourceAFIS minutiae matcher (JDK 17+; Maven only to build it), and its score fusion with the
  #     main CNN, both with thresholds from the validation fingers.
  if [ ! -d baselines/sourceafis/target/dependency ] && command -v mvn >/dev/null; then
    mvn -q -B -f baselines/sourceafis/pom.xml package dependency:copy-dependencies
  fi
  if [ -d baselines/sourceafis/target/dependency ] && command -v java >/dev/null; then
    if [ "$FORCE" = 1 ] || [ ! -f results/fvc2004_db1_official/sourceafis/seed0/metrics.json ]; then
      step "SourceAFIS baseline"
      $PY scripts/baseline_sourceafis.py
    fi
    step "fusion with SourceAFIS"
    $PY scripts/fusion.py --run $MAIN_RUN
  elif [ "${SKIP_SOURCEAFIS:-0}" = 1 ]; then
    step "skip SourceAFIS and fusion (SKIP_SOURCEAFIS=1); the bootstrap step needs their scores and is skipped too"
  else
    echo "SourceAFIS needs a JDK 17+ on PATH and Maven to build baselines/sourceafis (or SKIP_SOURCEAFIS=1)" >&2
    exit 1
  fi

  step "tables, bootstrap intervals and figures"
  $PY scripts/aggregate.py
  cp results/$MAIN/$MAIN_RUN/seed${SEEDS%% *}/{scores.csv,metrics.json,roc.png,distributions.png} results/
  if [ "${SKIP_SOURCEAFIS:-0}" != 1 ]; then
    $PY scripts/bootstrap_ci.py
    $PY scripts/sensitivity_setb.py
  fi
  FIG=artigo/figures
  FIG_SEED=${SEEDS%% *}              # single-seed figures (ROC, scores, errors) use the first training seed
  $PY scripts/pipeline_figure.py --dataset $MAIN --sample Db1_b/101_1.tif --dataset $SECOND --sample DB1_B/101_1.tif \
      --out $FIG/pipeline.png
  for D in $MAIN $SECOND; do
    $PY scripts/error_figure.py --result-dir results/$D/$MAIN_RUN/seed$FIG_SEED --dataset $D --out $FIG/errors_$D.png
  done
  $PY scripts/paper_figures.py --seed $FIG_SEED
fi
step "done ($STAGE)"
