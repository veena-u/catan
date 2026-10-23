import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .agents import Network, Random
from .bot import Bot
from .evaluate import play
from .model import device, load


def ratings(wins, prior=0.5):
    """Bradley-Terry strengths from a wins matrix, as Elo with player 0 at 0. The prior keeps 100% records finite."""
    wins = wins + prior * (1 - np.eye(len(wins)))
    games = wins + wins.T
    strength = np.ones(len(wins))
    for _ in range(2000):
        strength = wins.sum(1) / (games / (strength[:, None] + strength[None, :])).sum(1)
        strength /= strength[0]
    return 400 * np.log10(strength)


def main():
    parser = argparse.ArgumentParser(description="Round-robin of saved checkpoints, the bot and random, rated by Elo.")
    parser.add_argument("run", help="a run directory with iter_*.pt checkpoints")
    parser.add_argument("--checkpoints", type=int, default=6, help="how many, evenly spaced, including the last")
    parser.add_argument("--games", type=int, default=256, help="games per pairing")
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()

    at = device()
    saved = sorted(Path(args.run).glob("iter_*.pt"))
    picks = sorted({saved[round(i)] for i in np.linspace(0, len(saved) - 1, min(args.checkpoints, len(saved)))})
    players = {"random": Random(0), "bot": Bot()}
    for path in picks:
        players[path.stem] = Network(load(path, at), at, seed=len(players))
    names = list(players)

    wins = np.zeros((len(names), len(names)))
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        result = play(players[a], players[b], games=args.games, parallel=min(args.games, 512), seed=i * 100 + j,
                      players=args.players)
        wins[i, j], wins[j, i] = result["wins"], result["losses"]
        print(f"{a:>12s} vs {b:<12s} {result['win_rate']:.3f}", flush=True)

    elo = ratings(wins)
    print("\nElo (random = 0)")
    for name, score in sorted(zip(names, elo), key=lambda x: -x[1]):
        print(f"  {name:>12s} {score:7.0f}")
    table = {"players": names, "wins": wins.tolist(), "elo": dict(zip(names, map(float, elo)))}
    (Path(args.run) / "ladder.json").write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
