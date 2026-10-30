#!/usr/bin/env bash
# One arm of the search-training experiment. Both arms continue training from the same network, runs/anchor.pt, under
# the current rules, with identical settings except that the search arm also learns from Gumbel search. At every
# checkup each arm plays the anchor, so their curves compare directly. Stop with Ctrl-C or tmux kill-session; running
# the same command again resumes.
#   start an arm (inside tmux):   ARM=plain bash experiment.sh      or      ARM=search bash experiment.sh
#   search-turn imagines each future through every other player's turn, until the searcher moves again
#   memory is plain training on rules that remember this turn's last offer; start it from a later network with
#   START=runs/plain-latest.pt ARM=memory bash experiment.sh (the anchor stays the reference for every arm)
#   league learns from search, searches setup placements and robber moves more deeply, and plays half its games
#   against earlier versions of itself; e.g. START=runs/best.pt ANCHOR=runs/best.pt ARM=league bash experiment.sh
#   league-plain is the same without search, so it runs at close to plain speed
#   binding is memory's plain training under the rule that an accepted offer can't be cancelled; start it from
#   memory's final network: START=runs/memory-final.pt ANCHOR=runs/memory-final.pt ARM=binding bash experiment.sh
#   (LEAGUE_FROM adds other saved networks to its pool of opponents; any .pt copied into runs/inbox joins it at a checkup)
#   exploit trains a network only to beat the frozen anchor, then feeds it to the league through runs/inbox:
#   ANCHOR=runs/main.pt ARM=exploit bash experiment.sh
#   see how the arms are doing:   bash experiment.sh status
set -uo pipefail
if [ -f /venv/main/bin/activate ]; then source /venv/main/bin/activate; fi
cd "$(dirname "$0")"

if [ "${1:-}" = status ]; then
  for arm in plain search search-turn memory league league-plain binding exploit; do
    [ -f "runs/$arm.out" ] || continue
    echo "== $arm  (hours count from the latest start)"
    grep "(eval" "runs/$arm.out" | awk '{
      for (i = 1; i <= NF; i++) {
        if (i == 1 && $i == "iter") it = $(i + 1)
        if ($i ~ /^[0-9]+s$/ && !hours) hours = substr($i, 1, length($i) - 1) / 3600
        if ($i == "anchor") anchor = $(i + 1)
        if ($i == "bot" && $(i - 1) == "vs") bot = $(i + 1)
      }
      printf "iter %6d  %5.1fh  vs anchor %s  vs bot %s\n", it, hours, anchor, bot
      hours = 0
    }' | tail -n 12
    tail -n 1 "runs/$arm.out" | grep -o "searched [0-9]* (search changed the top move [0-9.]*" | sed 's/^/latest: /'
  done
  exit 0
fi

ARM=${ARM:?set ARM=plain, search, search-turn, memory or league}
ANCHOR=${ANCHOR:-runs/anchor.pt}
case $ARM in
  plain) EXTRA="" ;;
  search) EXTRA="--expert" ;;
  search-turn) EXTRA="--expert --expert-depth 0 --expert-games 16" ;;
  memory) EXTRA="" ;;
  league) EXTRA="--expert --key-games 8 --league 0.5 --league-from $ANCHOR ${LEAGUE_FROM:-} --league-watch runs/inbox" ;;
  binding) EXTRA="" ;;
  league-plain) EXTRA="--league 0.5 --league-from $ANCHOR ${LEAGUE_FROM:-} --league-watch runs/inbox" ;;
  exploit) EXTRA="--exploit $ANCHOR --bot-games 0" ;;
  *) echo "ARM is plain, search, search-turn, memory, league, league-plain, binding or exploit"; exit 1 ;;
esac
START=${START:-$ANCHOR}
[ -f "$ANCHOR" ] || { echo "put the starting network at $ANCHOR first"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
python -m pip install --quiet ./python
mkdir -p "runs/$ARM"
[ -f "runs/$ARM/latest.pt" ] || cp "$START" "runs/$ARM/latest.pt"
echo "=== $ARM, continuing from $START ($(date))" | tee -a "runs/$ARM.out"
SIZE=$(python -c "import torch; a = torch.load('$START', map_location='cpu')['args']; \
print(f\"--width {a['width']} --depth {a['depth']}\" + (' --entities' if a.get('entities') else ''))")
python -u -m train.selfplay --games "${GAMES:-2048}" --rounds "${ROUNDS:-64}" --minibatch "${MINIBATCH:-4096}" \
  --eval-every "${EVAL_EVERY:-50}" --eval-games "${EVAL_GAMES:-512}" --save-every 250 \
  --anneal-power 0.5 --anneal-steps 200 --magnet-anneal-steps 20 --players 4 $SIZE \
  --symmetry --vp-fade 2000 --iterations "${ITERATIONS:-100000}" --anchor "$ANCHOR" $EXTRA ${MORE:-} \
  --out "runs/$ARM" >> "runs/$ARM.out" 2>&1
