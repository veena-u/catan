import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .agents import Network, Random
from .evaluate import play, rules_of
from .model import device, load

def ratings(wins, prior=0.5):
    wins = wins + prior * (1 - np.eye(len(wins)))
    games = wins + wins.T
    strength = np.ones(len(wins))
    for _ in range(2000):
        strength = wins.sum(1) / (games / (strength[:, None] + strength[None, :])).sum(1)
        strength /= strength[0]
    return 400 * np.log10(strength)

def main():
    parser = argparse.ArgumentParser(description="Round-robin of saved checkpoints and random play, rated by Elo and robustness.")
    parser.add_argument("runs", nargs="+", help="run directories with iter_*.pt checkpoints, or single .pt files")
    parser.add_argument("--checkpoints", type=int, default=6, help="how many per run directory, evenly spaced, including the last")
    parser.add_argument("--games", type=int, default=256, help="games per pairing")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--rules", choices=["standard", "colonist-1v1"], default="colonist-1v1")
    parser.add_argument("--max-decisions", type=int, default=8000)
    args = parser.parse_args()

    at = device()
    picks = []
    for run in map(Path, args.runs):
        if run.is_dir():
            saved = sorted(run.glob("iter_*.pt"))
            picks += sorted({saved[round(i)] for i in np.linspace(0, len(saved) - 1, min(args.checkpoints, len(saved)))})
        else:
            picks.append(run)
    players = {"random": Random(0)}
    for path in picks:
        name = path.stem if len(args.runs) == 1 else f"{path.parent.name}/{path.stem}"
        players[name] = Network(load(path, at), at, seed=len(players))
    names = list(players)

    wins = np.zeros((len(names), len(names)))
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        result = play(players[a], players[b], games=args.games, parallel=min(args.games, 512), seed=i * 100 + j,
                      players=args.players, rules=rules_of(args), max_decisions=args.max_decisions)
        wins[i, j], wins[j, i] = result["wins"], result["losses"]
        print(f"{a:>24s} vs {b:<24s} {result['win_rate']:.3f}", flush=True)

    elo = ratings(wins)
    rates = wins / np.maximum(wins + wins.T, 1)
    robustness = [float(min(rates[i, j] for j in range(len(names)) if j != i)) for i in range(len(names))]
    print("\nElo (random = 0), and robustness: the win rate in the worst matchup")
    for name, score, robust in sorted(zip(names, elo, robustness), key=lambda x: -x[1]):
        print(f"  {name:>24s} {score:7.0f}   {robust:.2f}")
    table = {"players": names, "wins": wins.tolist(), "elo": dict(zip(names, map(float, elo))),
             "robustness": dict(zip(names, robustness))}
    first = Path(args.runs[0])
    (first if first.is_dir() else first.parent).joinpath("ladder.json").write_text(json.dumps(table, indent=2))

if __name__ == "__main__":
    main()
