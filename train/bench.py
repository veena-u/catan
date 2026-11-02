import argparse
import time

import numpy as np
import torch

import catan

from .features import SYMMETRIES
from .model import Model, device, load
from .selfplay import policy

def positions(n, seed):
    rng = np.random.default_rng(seed)
    obs, masks = [], []
    game = None
    while len(obs) < n:
        if game is None or game.winner() is not None or game.steps() > 1500:
            game = catan.Game(int(rng.integers(1 << 31)), 2, rules=catan.Rules.colonist_1v1())
            game.settle()
        o, m = game.observe()
        obs.append(o)
        masks.append(m)
        game.step(int(rng.choice(np.flatnonzero(m))))
        game.settle()
    return torch.from_numpy(np.stack(obs)), torch.from_numpy(np.stack(masks))

def seconds_per_step(model, obs, masks, minibatch, steps, symmetry, autocast):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    sync = torch.cuda.synchronize if obs.device.type == "cuda" else (lambda: None)
    for i in range(steps + 3):
        if i == 3:
            sync()
            clock = time.time()
        pick = torch.randint(len(obs), (minibatch,), device=obs.device)
        twist = torch.randint(SYMMETRIES, (minibatch,), device=obs.device) if symmetry else None
        with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
            logits, values, _ = model(obs[pick], masks[pick], twist)
        _, log_probs = policy(logits, masks[pick])
        loss = -log_probs.max(-1).values.mean() + values.float().square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sync()
    return (time.time() - clock) / steps

def main():
    parser = argparse.ArgumentParser(description="Time the learning step under different settings, to pick the fastest.")
    parser.add_argument("checkpoint", nargs="?", help="a saved network; without one, a fresh 128-wide network is timed")
    parser.add_argument("--positions", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--decisions", type=int, default=2048 * 64, help="decisions per update, as in training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatches", type=int, nargs="+", default=[4096, 8192])
    args = parser.parse_args()

    at = device()
    autocast = at.type == "cuda"
    obs, masks = (t.to(at) for t in positions(args.positions, seed=0))
    print(f"{at}, {len(obs)} positions from random play; an update is {args.epochs} passes over {args.decisions} decisions\n")
    print(f"{'setting':32s} {'per step':>9s} {'learn per update':>17s}")
    for compiled in (False, True):
        for minibatch in args.minibatches:
            model = (load(args.checkpoint, at) if args.checkpoint else Model().to(at)).train()
            name = f"minibatch {minibatch}" + (", compiled" if compiled else "")
            try:
                runner = torch.compile(model) if compiled else model
                step = seconds_per_step(runner, obs, masks, minibatch, args.steps, True, autocast)
            except Exception as e:
                print(f"{name:32s} failed: {type(e).__name__}: {str(e).splitlines()[0][:60]}")
                continue
            per_update = step * args.epochs * args.decisions / minibatch
            print(f"{name:32s} {step * 1000:8.0f}ms {per_update:16.1f}s")

if __name__ == "__main__":
    main()
