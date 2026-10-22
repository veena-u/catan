import argparse

from .agents import Network
from .evaluate import play, report
from .model import device, load


def main():
    parser = argparse.ArgumentParser(description="Play saved networks head to head against a baseline.")
    parser.add_argument("baseline", help="e.g. runs/anneal/latest.pt")
    parser.add_argument("challengers", nargs="+", help="one or more .pt files")
    parser.add_argument("--games", type=int, default=2048)
    args = parser.parse_args()

    at = device()
    baseline = Network(load(args.baseline, at), at, seed=1)
    for i, path in enumerate(args.challengers):
        challenger = Network(load(path, at), at, seed=2 + i)
        report(f"{path} vs {args.baseline}", play(challenger, baseline, games=args.games, parallel=min(args.games, 512), seed=99))


if __name__ == "__main__":
    main()
