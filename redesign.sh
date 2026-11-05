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

ARMS="long, recipe, lookahead, baseline, no-trading, one-pass, no-hands, no-answers, league, offer-search, trade-search, upgo, random-starts, wide, no-vp, population or exploit"
ARM=${1:?name a run: $ARMS}
SEED=${SEED:?set SEED, e.g. SEED=1}
PLAYERS=${PLAYERS:-2}
if [ "$PLAYERS" = 2 ]; then RULES=${RULES:-colonist-1v1}; else RULES=${RULES:-colonist-4p}; fi
PREFIX="${PLAYERS}p"
LOOKAHEAD="--lookahead 0.25 --lookahead-simulations 16 --lookahead-worlds 2 --lookahead-considered 8"
TRADES="--offer-search 16 --offer-candidates 8 --answer-search --search-after 250 --offer-search-weight 0.3 --aux-accept 0.1 --trade-symmetries 6 --stubborn 0.2"
# The full recipe: searched moves to learn from, a critic that sees every hand valuing the imagined worlds, final
# points to predict in place of a points reward, past selves at a fifth of the tables; in 4p, searched trades too.
if [ "$PLAYERS" = 2 ]; then RECIPE_TRADES="--no-trading --aux-answers 0"; else RECIPE_TRADES="$TRADES --gamma 0.9995"; fi
CORE="--privileged --aux-points 0.1 --vp-reward 0 --league 0.2 --league-size 4 --league-every 250 --league-switch iteration $RECIPE_TRADES"
RECIPE="$CORE $LOOKAHEAD --lookahead-privileged"
# The long run starts without the lookahead: search has not yet beaten the plain network at play time. Once a
# setting does (python -m train.search), restart the run with MORE="<lookahead flags>": it resumes from latest.pt.
case $ARM in
  baseline) EXTRA="" ;;
  long) EXTRA="$CORE" ;;
  recipe) EXTRA="$RECIPE" ;;
  lookahead) EXTRA="$LOOKAHEAD" ;;
  no-trading) EXTRA="--no-trading --aux-answers 0" ;;
  one-pass) EXTRA="--epochs 1" ;;
  no-hands) EXTRA="--aux-hands 0" ;;
  no-answers) EXTRA="--aux-answers 0" ;;
  league) EXTRA="--league 0.2 --league-size 8 --league-every 250" ;;
  offer-search) EXTRA="--offer-search 8 --offer-candidates 8" ;;
  trade-search) EXTRA="$TRADES" ;;
  upgo) EXTRA="--upgo 1.0" ;;
  random-starts) EXTRA="--random-starts 25" ;;
  wide) EXTRA="--width 192" ;;
  no-vp) EXTRA="--vp-reward 0" ;;
  population) EXTRA="--league 0.5 --league-size 8 --league-every 250 --league-watch runs/$PREFIX-pop --publish runs/$PREFIX-pop/s$SEED.pt" ;;
  exploit) EXTRA="--exploit ${TARGET:?set TARGET to the network to exploit, e.g. TARGET=runs/$PREFIX-baseline-s1/latest.pt}"
    if [ "$PLAYERS" != 2 ]; then EXTRA="$EXTRA --exploit-lone"; fi ;;
  *) echo "the runs are $ARMS"; exit 1 ;;
esac
NAME="$PREFIX-$ARM-s$SEED"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
python -m pip install --quiet ./python
FLAGS="--players $PLAYERS --rules $RULES --games ${GAMES:-2048} --rounds 64 --minibatch ${MINIBATCH:-4096} \
  --width ${WIDTH:-128} --depth 4 --symmetry --vp-fade 2000 --opening-explore 0.1 --aux-hands 0.1 --aux-answers 0.1 \
  --entropy 0.01 --entropy-floor 0.001 --max-decisions 8000 --turn-budget 15 \
  --eval-every ${EVAL_EVERY:-200} --eval-games 512 --save-every 250 --checkpoint-every 50 \
  --anneal-power 0.5 --anneal-steps 200 --entropy-anneal-steps 20 --iterations ${ITERATIONS:-100000} --seed $SEED"
if [ -n "${ANCHORS:-}" ]; then FLAGS="$FLAGS --anchors $ANCHORS"; fi
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
