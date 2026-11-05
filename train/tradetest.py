import argparse

from .agents import Network, NoTrade
from .evaluate import RULE_SETS, matchup, rules_named
from .model import device, load

def main():
    parser = argparse.ArgumentParser(description="Does trading win games? A network against itself with trading taken away.")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--rules", choices=list(RULE_SETS), default="colonist-4p")
    parser.add_argument("--games", type=int, default=256)
    args = parser.parse_args()
    at = device()
    rules = rules_named(args.rules)
    for path in args.checkpoints:
        model = load(path, at)
        result = matchup(Network(model, at, seed=1), NoTrade(Network(model, at, seed=2)), games=args.games, players=args.players, rules=rules)
        spread = 0.5 / max(result["decided"], 1) ** 0.5
        print(f"{path}: trading scores {result['share']:.3f} ± {spread:.3f} against itself without trading (0.50 is even)", flush=True)

if __name__ == "__main__":
    main()
