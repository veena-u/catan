#!/usr/bin/env bash
# The final run: the ideas that won overnight, trained much longer, then tested every way we have:
# against the baseline and the bot, with search and deep search, a strength curve over training, and habits.
# A second seed follows if time allows, to show the result is not luck.
# Name the winners:  FLAGS="--symmetry --privileged" tmux new -s final 'bash final.sh'
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

PLAYERS=${PLAYERS:-4}
FLAGS=${FLAGS:-""}
ITERATIONS=${ITERATIONS:-1500}
SEEDS=${SEEDS:-"0 1"}
BASE="--games 2048 --rounds 64 --eval-every 50 --anneal-power 0.5 --anneal-steps 20 --players $PLAYERS"

once() {
  local done=$1 out=$2
  shift 2
  if [ -f "$done" ]; then
    echo "skipping $done: already there"
    return
  fi
  echo "=== $* ($(date))"
  if "$@" > "$out" 2>&1; then
    cp "$out" "$done"
  else
    echo "!!! failed; the last lines of $out:"
    tail -5 "$out"
  fi
}

mkdir -p runs
python -c "import catan; catan.Buffer(1, players=$PLAYERS).track_dice(False)" 2>/dev/null || python -m pip install --quiet ./python
echo "final flags: ${FLAGS:-(none)}; $ITERATIONS iterations; seeds $SEEDS"
for seed in $SEEDS; do
  name=final-$seed
  if [ ! -f "runs/$name/iter_$(printf %05d "$ITERATIONS").pt" ]; then
    echo "=== $name ($(date))"
    python -u -m train.selfplay $BASE --iterations "$ITERATIONS" --seed "$seed" $FLAGS --out "runs/$name" > "runs/$name.out" 2>&1 || {
      echo "!!! $name failed; the last lines of runs/$name.out:"
      tail -5 "runs/$name.out"
      continue
    }
  fi
  once "runs/$name/compare.txt" "runs/$name/compare.out" \
    python -m train.compare runs/base/latest.pt "runs/$name/latest.pt" --players "$PLAYERS" --games 2048
  once "runs/$name/habits.txt" "runs/$name/habits.out" \
    python -m train.habits "runs/$name/latest.pt" --players "$PLAYERS" --games 200
  if [ "$seed" = "$(echo $SEEDS | cut -d' ' -f1)" ]; then
    once "runs/$name/ladder.txt" "runs/$name/ladder.out" \
      python -m train.ladder "runs/$name" --players "$PLAYERS" --checkpoints 8
    once "runs/$name/search.txt" "runs/$name/search.out" \
      python -u -m train.search "runs/$name/latest.pt" --players "$PLAYERS" --games 512 --bot
    once "runs/$name/deep.txt" "runs/$name/deep.out" \
      python -u -m train.search "runs/$name/latest.pt" --players "$PLAYERS" --games 256 --deep
  fi
  tar czf results-final.tar.gz runs/final-*.out $(ls -d runs/final-*/)
done
echo "=== done ($(date))"
