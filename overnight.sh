#!/usr/bin/env bash
# The overnight queue for full Catan: a baseline run first, then each run changes one thing from it.
# After every run: head-to-head against the baseline and per-game habits, so stopping early loses nothing.
# Run from the repo root inside tmux:  bash overnight.sh
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

PLAYERS=${PLAYERS:-4}
BASE=${BASE:-"--games 2048 --rounds 64 --eval-every 10 --anneal-power 0.5 --anneal-steps 20 --players $PLAYERS"}
BIG=${BIG:-"--width 256 --depth 6 --minibatch 2048"}
SHORT=${SHORT:-300}
LONG=${LONG:-600}

run() {
  local name=$1
  shift
  if [ -f "runs/$name/habits.txt" ]; then
    echo "skipping $name: already finished"
    return
  fi
  if [ -f "runs/$name/iter_$(printf %05d "$ITERATIONS").pt" ]; then
    echo "=== $name: trained already; comparing ($(date))"
  else
    echo "=== $name: $* ($(date))"
    if ! python -u -m train.selfplay $BASE --iterations "$ITERATIONS" "$@" --out "runs/$name" > "runs/$name.out" 2>&1; then
      echo "!!! $name failed; the last lines of runs/$name.out:"
      tail -5 "runs/$name.out"
      return
    fi
  fi
  if [ "$name" != base ]; then
    python -m train.compare runs/base/latest.pt "runs/$name/latest.pt" --players "$PLAYERS" --games "${COMPARE_GAMES:-2048}" | tee -a runs/compare.txt
  fi
  python -m train.habits "runs/$name/latest.pt" --players "$PLAYERS" --games "${HABIT_GAMES:-200}" > "runs/$name/habits.txt" 2>&1
  tar czf results-overnight.tar.gz runs/*.out $(ls runs/compare.txt 2>/dev/null) $(ls -d runs/*/)
}

mkdir -p runs
python -c "import catan; catan.Buffer(1, players=$PLAYERS).track_dice(False)" 2>/dev/null || {
  echo "the catan module is out of date; rebuilding it"
  python -m pip install --quiet ./python
}
ITERATIONS=$SHORT run base
[ -f runs/base/latest.pt ] || { echo "the baseline failed; stopping"; exit 1; }
ITERATIONS=$SHORT run dice --average-dice
ITERATIONS=$SHORT run keep --keep 0.3
ITERATIONS=$SHORT run big $BIG
ITERATIONS=$LONG run all-long --average-dice $BIG
echo "=== done ($(date))"
