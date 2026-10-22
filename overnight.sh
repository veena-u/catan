#!/usr/bin/env bash
# The overnight queue: each run changes one thing from runs/anneal (the baseline), then the best ideas combined.
# After every run: head-to-head against the baseline and per-game habits, so stopping early loses nothing.
# Run from the repo root inside tmux:  bash overnight.sh
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

BASE=${BASE:-"--games 2048 --rounds 64 --eval-every 10 --anneal-power 0.5 --anneal-steps 20"}
BIG=${BIG:-"--width 256 --depth 6 --minibatch 2048"}
SHORT=${SHORT:-300}
LONG=${LONG:-600}

run() {
  local name=$1
  shift
  if [ -f "runs/$name/iter_$(printf %05d "$ITERATIONS").pt" ]; then
    echo "skipping $name: already finished"
    return
  fi
  echo "=== $name: $* ($(date))"
  if ! python -u -m train.selfplay $BASE --iterations "$ITERATIONS" "$@" --out "runs/$name" > "runs/$name.out" 2>&1; then
    echo "!!! $name failed; the last lines of runs/$name.out:"
    tail -5 "runs/$name.out"
    return
  fi
  python -m train.compare runs/anneal/latest.pt "runs/$name/latest.pt" | tee -a runs/compare.txt
  python -m train.habits "runs/$name/latest.pt" > "runs/$name/habits.txt" 2>&1
  tar czf results-overnight.tar.gz runs/compare.txt runs/*.out $(ls -d runs/*/ | grep -v anneal)
}

mkdir -p runs
python -c "import catan; catan.Buffer(1).track_dice(False)" 2>/dev/null || {
  echo "the catan module is out of date; rebuilding it"
  python -m pip install --quiet ./python
}
[ -f runs/anneal/latest.pt ] || { echo "runs/anneal/latest.pt is missing: unpack baseline.tar.gz first"; exit 1; }
ITERATIONS=$SHORT run dice --average-dice
ITERATIONS=$SHORT run keep --keep 0.3
ITERATIONS=$SHORT run big $BIG
ITERATIONS=$SHORT run dice-keep --average-dice --keep 0.3
ITERATIONS=$LONG run all-long --average-dice --keep 0.3 $BIG
echo "=== done ($(date))"
