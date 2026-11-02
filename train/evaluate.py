import argparse
import math

import numpy as np

import catan

from .agents import Random

def play(a, b, games=4096, parallel=512, seed=0, max_decisions=5000, players=4, paired=True, rules=None):
    parallel = max(players, parallel // players * players)
    per_slot = math.ceil(games / parallel)
    buffer = catan.Buffer(parallel, max_decisions=max_decisions, seed=seed, players=players, paired=paired, rules=rules)
    seat_a = np.arange(parallel) % players
    finished = np.zeros(parallel, int)
    wins = losses = truncated = first_wins = decided = 0
    while (finished < per_slot).any():
        obs, masks, actors = buffer.observe()
        forbidden, counts = buffer.forbidden()
        a_turn = actors == seat_a
        actions, offers = np.zeros(parallel, np.int64), np.zeros((parallel, 5), np.int8)
        for agent, turn in ((a, a_turn), (b, ~a_turn)):
            if turn.any():
                actions[turn], offers[turn] = agent(obs[turn], masks[turn], forbidden[turn], counts[turn])
        for game, winner, _ in buffer.step(actions, offers):
            if finished[game] >= per_slot:
                continue
            finished[game] += 1
            if winner is None:
                truncated += 1
                continue
            decided += 1
            first_wins += winner == 0
            if winner == seat_a[game]:
                wins += 1
            else:
                losses += 1
    rate = wins / max(decided, 1)
    return {
        "games": int(finished.sum()),
        "wins": wins,
        "losses": losses,
        "truncated": truncated,
        "win_rate": rate,
        "stderr": math.sqrt(rate * (1 - rate) / max(decided, 1)),
        "first_player_win_rate": first_wins / max(decided, 1),
        "players": players,
    }

RULE_SETS = {
    "standard": dict(points=10, discard_limit=7, friendly_robber=False, balanced_dice=False),
    "colonist-1v1": dict(points=15, discard_limit=9, friendly_robber=True, balanced_dice=True),
    "colonist-4p": dict(points=10, discard_limit=7, friendly_robber=True, balanced_dice=True),
}
TURN_BUDGET = 15

def rules_named(name, trading=True, turn_budget=TURN_BUDGET):
    return catan.Rules(**RULE_SETS[name], trading=trading, turn_budget=turn_budget)

def rules_of(args):
    return rules_named(getattr(args, "rules", "standard"), not getattr(args, "no_trading", False),
                       getattr(args, "turn_budget", TURN_BUDGET))

def report(name, result):
    print(
        f"{name:28s} win rate {result['win_rate']:.3f} ± {result['stderr']:.3f} (fair share {1 / result['players']:.2f})  "
        f"({result['wins']}–{result['losses']}, {result['truncated']} truncated, "
        f"first player wins {result['first_player_win_rate']:.3f})"
    )

def main():
    parser = argparse.ArgumentParser(description="Sanity checks for the evaluation harness.")
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()
    report("random vs random", play(Random(1), Random(2), args.games, players=args.players))
    report("random-by-kind vs random", play(Random(1, kinds=True), Random(2), args.games, players=args.players))

if __name__ == "__main__":
    main()
