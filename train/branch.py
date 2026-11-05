import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from .agents import Network
from .anchors import load_pool, rate
from .evaluate import matchup, rules_named
from .model import device, load

# Population-based training, by hand: branch a run from its current weights with some settings changed, train for a
# fixed number of hours, and rate the result on the yardstick, so settings are compared on equal compute without
# starting over.

def command(settings):
    flags = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False or value == []:
            continue
        if value is True:
            flags.append(flag)
        elif isinstance(value, list):
            flags += [flag, *map(str, value)]
        else:
            flags += [flag, str(value)]
    return flags

def parse(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

def main():
    parser = argparse.ArgumentParser(description="Branch a run from its latest checkpoint with changed settings, train for a "
                                                 "fixed time, and rate it against a pool of anchors.")
    parser.add_argument("run", help="the run to branch from")
    parser.add_argument("out", help="the branch's folder")
    parser.add_argument("--hours", type=float, default=3.0)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="settings to change, e.g. lr=0.0003 anneal_power=0")
    parser.add_argument("--pool", help="anchors to rate the branch against at the end")
    parser.add_argument("--games", type=int, default=512, help="games per anchor when rating")
    parser.add_argument("--wandb-group", default="branches")
    args = parser.parse_args()

    run, out = Path(args.run), Path(args.out)
    settings = json.loads((run / "args.json").read_text())
    for pair in args.set:
        key, value = pair.split("=", 1)
        if key.replace("-", "_") not in settings:
            parser.error(f"{run} has no setting {key}")
        settings[key.replace("-", "_")] = parse(value)
    settings.update(out=str(out), checkpoint_every=10, wandb_group=args.wandb_group)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "latest.pt").exists():
        shutil.copy(run / "latest.pt", out / "latest.pt")
    start = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["iteration"]
    print(f"{out.name}: from {run.name} at iteration {start}, changing {' '.join(args.set) or 'nothing'}, for {args.hours:g} hours", flush=True)

    with open(out.with_suffix(".out"), "a") as log:
        if args.hours <= 0:
            child = None
        else:
            child = subprocess.Popen([sys.executable, "-u", "-m", "train.selfplay", *command(settings)], stdout=log, stderr=subprocess.STDOUT)
    if child is not None:
        try:
            child.wait(timeout=args.hours * 3600)
        except subprocess.TimeoutExpired:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
    reached = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)["iteration"]
    print(f"{out.name}: trained iterations {start} to {reached}", flush=True)

    at = device()
    player = Network(load(out / "latest.pt", at), at, seed=reached)
    if settings.get("exploit"):
        target = Network(load(settings["exploit"], at), at, seed=reached + 1)
        players = settings["players"]
        rules = rules_named(settings["rules"], trading=not settings.get("no_trading"))
        share = matchup(player, target, games=args.games, players=players, rules=rules, seed=reached)["share"]
        print(f"{out.name}: the exploiter scores {share:.3f} against its target (0.50 is even)", flush=True)
    if args.pool:
        pool, meta = load_pool(args.pool, at)
        elo, shares = rate(player, pool, args.games, meta["players"], rules_named(meta["rules"], trading=meta["trading"]), seed=reached)
        print(f"{out.name}: Elo {elo:.0f} against the anchors at iteration {reached}  "
              + "  ".join(f"{name.split('-iter_')[-1]} {share:.3f}" for name, share in shares.items()), flush=True)

if __name__ == "__main__":
    main()
