#!/usr/bin/env bash
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

if [ "${1:-}" = status ]; then
  for out in runs/[0-9]p-*.out; do
    [ -f "$out" ] || continue
    echo "== $(basename "$out" .out)"
    grep "(eval" "$out" | awk '{
      for (i = 1; i <= NF; i++) {
        if (i == 1 && $i == "iter") it = $(i + 1)
        if ($i ~ /^[0-9]+s$/ && !hours) hours = substr($i, 1, length($i) - 1) / 3600
        if ($i == "previous" && $(i - 1) == "vs") previous = $(i + 1)
        if ($i == "random" && $(i - 1) == "vs") random = $(i + 1)
      }
      printf "iter %6d  %5.1fh  vs previous %s  vs random %s\n", it, hours, previous, random
      hours = 0
    }' | tail -n 8
    tail -n 1 "$out" | grep -o "setup pips.*abandons [0-9.]*" | sed 's/^/habits: /'
  done
  exit 0
fi

ARMS="baseline, no-trading, one-pass, no-hands, no-answers, league, population or exploit"
ARM=${1:?name a run: $ARMS}
SEED=${SEED:?set SEED, e.g. SEED=1}
PLAYERS=${PLAYERS:-2}
if [ "$PLAYERS" = 2 ]; then RULES=${RULES:-colonist-1v1}; else RULES=${RULES:-colonist-4p}; fi
PREFIX="${PLAYERS}p"
case $ARM in
  baseline) EXTRA="" ;;
  no-trading) EXTRA="--no-trading --aux-answers 0" ;;
  one-pass) EXTRA="--epochs 1" ;;
  no-hands) EXTRA="--aux-hands 0" ;;
  no-answers) EXTRA="--aux-answers 0" ;;
  league) EXTRA="--league 0.2 --league-size 8 --league-every 250" ;;
  population) EXTRA="--league 0.5 --league-size 8 --league-every 250 --league-watch runs/$PREFIX-pop --publish runs/$PREFIX-pop/s$SEED.pt" ;;
  exploit) EXTRA="--exploit ${TARGET:?set TARGET to the network to exploit, e.g. TARGET=runs/$PREFIX-baseline-s1/latest.pt}" ;;
  *) echo "the runs are $ARMS"; exit 1 ;;
esac
NAME="$PREFIX-$ARM-s$SEED"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
python -m pip install --quiet ./python
FLAGS="--players $PLAYERS --rules $RULES --games ${GAMES:-2048} --rounds 64 --minibatch ${MINIBATCH:-4096} \
  --width 128 --depth 4 --symmetry --vp-fade 2000 --opening-explore 0.1 --aux-hands 0.1 --aux-answers 0.1 \
  --entropy 0.01 --entropy-floor 0.001 --max-decisions 8000 --turn-budget 15 \
  --eval-every ${EVAL_EVERY:-200} --eval-games 512 --save-every 250 --checkpoint-every 50 \
  --anneal-power 0.5 --anneal-steps 200 --entropy-anneal-steps 20 --iterations ${ITERATIONS:-100000} --seed $SEED"
if [ -n "${WANDB_PROJECT:-}" ]; then
  python -m pip install --quiet wandb
  FLAGS="$FLAGS --wandb $WANDB_PROJECT --wandb-group ${WANDB_GROUP:-$PREFIX-v1}"
fi
mkdir -p runs
echo "=== $NAME: $EXTRA ${MORE:-} ($(date))" >> "runs/$NAME.out"
for attempt in $(seq 1 20); do
  python -u -m train.selfplay $FLAGS $EXTRA ${MORE:-} --out "runs/$NAME" >> "runs/$NAME.out" 2>&1 && exit 0
  echo "!!! $NAME stopped (attempt $attempt); restarting from its latest.pt in a minute" >> "runs/$NAME.out"
  sleep 60
done
