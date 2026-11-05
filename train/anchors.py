import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np

from .agents import Network
from .evaluate import RULE_SETS, matchup, rules_named
from .ladder import ratings
from .model import device, load

# A fixed pool of networks with fixed ratings is the yardstick: every run, at every checkpoint, is rated against
# the same opponents, so strength-against-compute curves from different runs can be compared, and an idea that
# pays off late shows up as a curve that is still climbing.

def load_pool(folder, at):
    meta = json.loads((Path(folder) / "anchors.json").read_text())
    pool = [(name, load(Path(folder) / name, at), elo) for name, elo in meta["elo"].items()]
    return pool, meta

def elo_against(shares, elos, games):
    """The rating whose expected scores against anchors of known rating best match the scores seen."""
    clip = 1 / (2 * max(games, 1))
    shares = np.clip(np.asarray(shares, float), clip, 1 - clip)
    elos = np.asarray(elos, float)
    low, high = elos.min() - 2000, elos.max() + 2000
    for _ in range(60):
        mid = (low + high) / 2
        expected = 1 / (1 + 10 ** ((elos - mid) / 400))
        low, high = (mid, high) if (shares - expected).sum() > 0 else (low, mid)
    return (low + high) / 2

def rate(player, pool, games, players, rules, seed=0):
    shares = {}
    for k, (name, anchor, _) in enumerate(pool):
        opponent = Network(anchor, anchor_device(anchor), seed=seed + 7919 * (k + 1))
        shares[name] = matchup(player, opponent, games=games, players=players, rules=rules, seed=seed + 101 * k)["share"]
    return elo_against(list(shares.values()), [elo for _, _, elo in pool], games), shares

def anchor_device(model):
    return next(model.parameters()).device

def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for path in map(Path, args.checkpoints):
        name = f"{path.parent.name}-{path.stem}.pt" if path.parent.name else path.name
        shutil.copy(path, out / name)
        names.append(name)
    at = device()
    rules = rules_named(args.rules, trading=not args.no_trading)
    models = {name: Network(load(out / name, at), at, seed=i) for i, name in enumerate(names)}
    wins = np.zeros((len(names), len(names)))
    for (i, a), (j, b) in itertools.combinations(enumerate(names), 2):
        result = matchup(models[a], models[b], games=args.games, players=args.players, rules=rules, seed=i * 100 + j)
        wins[i, j], wins[j, i] = result["share"] * result["decided"], (1 - result["share"]) * result["decided"]
        print(f"{a:>36s} vs {b:<36s} {result['share']:.3f}", flush=True)
    elo = ratings(wins)
    elo = elo - elo.mean() + 1000
    meta = {"players": args.players, "rules": args.rules, "trading": not args.no_trading, "games": args.games,
            "elo": {name: float(e) for name, e in zip(names, elo)}}
    (out / "anchors.json").write_text(json.dumps(meta, indent=2))
    print("\nanchor ratings (mean 1000):")
    for name, e in sorted(meta["elo"].items(), key=lambda x: -x[1]):
        print(f"  {name:>40s} {e:7.0f}")

def rate_runs(args):
    at = device()
    pool, meta = load_pool(args.pool, at)
    rules = rules_named(meta["rules"], trading=meta["trading"])
    tracker = None
    for run in map(Path, args.runs):
        saved = sorted(run.glob("iter_*.pt"))
        if not saved:
            print(f"{run}: no checkpoints")
            continue
        picks = sorted({saved[round(i)] for i in np.linspace(len(saved) - 1, 0, min(args.checkpoints, len(saved)))})
        if args.wandb:
            import wandb
            tracker = wandb.init(project=args.wandb, group=args.wandb_group, name=f"rated-{run.name}", reinit=True,
                                 config={"run": run.name, "pool": str(args.pool)})
            tracker.define_metric("anchor/*", step_metric="anchor/decisions")
        rows = []
        print(f"{run.name}:")
        for path in picks:
            checkpoint_args = json.loads((run / "args.json").read_text()) if (run / "args.json").exists() else {}
            iteration = int(path.stem.split("_")[1])
            decisions = iteration * checkpoint_args.get("games", 0) * checkpoint_args.get("rounds", 0)
            elo, shares = rate(Network(load(path, at), at, seed=iteration), pool, args.games, meta["players"], rules, seed=iteration)
            rows.append({"iteration": iteration, "decisions": decisions, "elo": elo, "shares": shares})
            print(f"  iter {iteration:5d}  {decisions / 1e6:7.0f}M decisions  Elo {elo:6.0f}", flush=True)
            if tracker is not None:
                tracker.log({"anchor/elo": elo, "anchor/decisions": decisions, "anchor/iteration": iteration})
        (run / "anchor_ratings.json").write_text(json.dumps(rows, indent=2))
        if tracker is not None:
            tracker.finish()

def main():
    parser = argparse.ArgumentParser(description="A fixed pool of rated networks, to rate every run against the same yardstick.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build", help="copy networks into a pool and rate them against each other once")
    make.add_argument("out")
    make.add_argument("checkpoints", nargs="+")
    make.add_argument("--players", type=int, default=2)
    make.add_argument("--rules", choices=list(RULE_SETS), default="colonist-1v1")
    make.add_argument("--no-trading", action="store_true")
    make.add_argument("--games", type=int, default=256)
    score = commands.add_parser("rate", help="rate runs' checkpoints against a pool: a strength-against-training curve per run")
    score.add_argument("pool")
    score.add_argument("runs", nargs="+")
    score.add_argument("--checkpoints", type=int, default=6, help="evenly spaced per run, including the last")
    score.add_argument("--games", type=int, default=128, help="games per anchor (each way, with more than two players)")
    score.add_argument("--wandb", metavar="PROJECT", help="also chart the curves in this W&B project")
    score.add_argument("--wandb-group", default="ratings")
    args = parser.parse_args()
    build(args) if args.command == "build" else rate_runs(args)

if __name__ == "__main__":
    main()
