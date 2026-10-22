import argparse
import math

import numpy as np

import catan

from .agents import Random


def play(a, b, games=4096, parallel=512, seed=0, max_decisions=1000):
    per_slot = math.ceil(games / parallel)
    buffer = catan.Buffer(parallel, max_decisions=max_decisions, seed=seed)
    seat_a = np.arange(parallel) % 2
    finished = np.zeros(parallel, int)
    wins = losses = truncated = first_wins = decided = 0
    while (finished < per_slot).any():
        obs, masks, players = buffer.observe()
        a_turn = players == seat_a
        actions = np.zeros(parallel, np.int64)
        for agent, turn in ((a, a_turn), (b, ~a_turn)):
            if turn.any():
                if getattr(agent, "searches", False):
                    actions[turn] = agent(obs[turn], masks[turn], buffer, np.flatnonzero(turn), players[turn])
                else:
                    actions[turn] = agent(obs[turn], masks[turn])
        for game, winner, _ in buffer.step(actions):
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
    }


def report(name, result):
    print(
        f"{name:28s} win rate {result['win_rate']:.3f} ± {result['stderr']:.3f}  "
        f"({result['wins']}–{result['losses']}, {result['truncated']} truncated, "
        f"first player wins {result['first_player_win_rate']:.3f})"
    )


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for the evaluation harness.")
    parser.add_argument("--games", type=int, default=4096)
    args = parser.parse_args()
    report("random vs random", play(Random(1), Random(2), args.games))
    report("random-by-kind vs random", play(Random(1, kinds=True), Random(2), args.games))


if __name__ == "__main__":
    main()
