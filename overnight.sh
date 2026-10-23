#!/usr/bin/env bash
# The overnight queue for full Catan: a baseline run first, then search and a belief network on top of it,
# then expert iteration, then each run changes one thing from the baseline, then everything together for twice as long.
# Comparisons are paired: same boards and dice, seats rotated.
# After every run: head-to-head against the baseline and per-game habits, so stopping early loses nothing.
# Run from the repo root inside tmux:  bash overnight.sh
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

PLAYERS=${PLAYERS:-4}
BASE=${BASE:-"--games 2048 --rounds 64 --eval-every 20 --anneal-power 0.5 --anneal-steps 20 --players $PLAYERS"}
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

# Does thinking ahead help? Train a belief network on the baseline's games, then play the baseline with search
# (cards dealt uniformly, then from the belief network) against itself without.
think() {
  if [ -f runs/search.txt ]; then
    echo "skipping search: already finished"
    return
  fi
  mkdir -p runs/belief
  if [ ! -f runs/belief/done ]; then
    echo "=== belief network ($(date))"
    if python -u -m train.belief runs/base/latest.pt --out runs/belief/belief.pt > runs/belief.out 2>&1; then
      touch runs/belief/done
    else
      echo "!!! belief failed; the last lines of runs/belief.out:"
      tail -5 runs/belief.out
    fi
  fi
  local belief=""
  [ -f runs/belief/done ] && belief="--belief runs/belief/belief.pt"
  echo "=== search ($(date))"
  if python -u -m train.search runs/base/latest.pt --players "$PLAYERS" --games "${SEARCH_GAMES:-512}" $belief > runs/search.out 2>&1; then
    cp runs/search.out runs/search.txt
  else
    echo "!!! search failed; the last lines of runs/search.out:"
    tail -5 runs/search.out
  fi
  tar czf results-overnight.tar.gz runs/*.out $(ls runs/compare.txt 2>/dev/null) $(ls -d runs/*/)
}

mkdir -p runs
python -c "import catan; catan.Buffer(1, players=$PLAYERS).track_dice(False); assert hasattr(catan.Buffer(1).worlds([0], [catan.Buffer(1).observe()[1][0].nonzero()[0][0]], 1), 'redealt')" 2>/dev/null || {
  echo "the catan module is out of date; rebuilding it"
  python -m pip install --quiet ./python
}
ITERATIONS=$SHORT run base
[ -f runs/base/latest.pt ] || { echo "the baseline failed; stopping"; exit 1; }
think
ITERATIONS=$SHORT run expert --expert
ITERATIONS=$SHORT run symmetry --symmetry
ITERATIONS=$SHORT run future-dice --future-dice
ITERATIONS=$SHORT run critic --privileged
ITERATIONS=$SHORT run dice --average-dice
ITERATIONS=$LONG run all --symmetry --future-dice --privileged --average-dice
echo "=== done ($(date))"
