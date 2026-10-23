#!/usr/bin/env bash
# One long run on the current rules, then tested every way we have: per-game habits, a strength curve over its
# checkpoints, search and deep search against it, and the bot. The learning rate shrinks slowly enough for days of
# training; stop whenever, and run the same command again to resume. To test a stopped run, rerun with ITERATIONS set
# to the last numbered checkpoint (a multiple of 250).
# Start it inside tmux, e.g.:  NAME=long FLAGS="--symmetry" bash final.sh
# Two long runs play each other with:  python -m train.compare runs/long/latest.pt runs/long-expert/latest.pt
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

PLAYERS=${PLAYERS:-4}
NAME=${NAME:-long}
FLAGS=${FLAGS:-""}
ITERATIONS=${ITERATIONS:-6000}
SEED=${SEED:-0}
SIZE=${SIZE:-"--width 192 --depth 6"}
BASE="--games 2048 --rounds 64 --eval-every 50 --save-every 250 --anneal-power 0.5 --anneal-steps 200 --players $PLAYERS $SIZE"
RUN=runs/$NAME

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
python -m pip install --quiet ./python
echo "$NAME: flags ${FLAGS:-(none)}; $ITERATIONS iterations; seed $SEED"
if [ ! -f "$RUN/iter_$(printf %05d "$ITERATIONS").pt" ]; then
  echo "=== training $NAME ($(date))"
  python -u -m train.selfplay $BASE --iterations "$ITERATIONS" --seed "$SEED" $FLAGS --out "$RUN" >> "$RUN.out" 2>&1 || {
    echo "!!! $NAME failed; the last lines of $RUN.out:"
    tail -5 "$RUN.out"
    exit 1
  }
fi
once "$RUN/habits.txt" "$RUN/habits.out" python -m train.habits "$RUN/latest.pt" --players "$PLAYERS" --games 200
once "$RUN/ladder.txt" "$RUN/ladder.out" python -m train.ladder "$RUN" --players "$PLAYERS" --checkpoints 8
once "$RUN/search.txt" "$RUN/search.out" python -u -m train.search "$RUN/latest.pt" --players "$PLAYERS" --games 512 --bot
once "$RUN/deep.txt" "$RUN/deep.out" python -u -m train.search "$RUN/latest.pt" --players "$PLAYERS" --games 256 --deep
tar czf "results-$NAME.tar.gz" "$RUN.out" "$RUN"
echo "=== done ($(date))"
