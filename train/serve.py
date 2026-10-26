import argparse
import sys

import numpy as np
import torch

import catan

from .agents import Network
from .insight import card_values, forecast
from .model import device, load, masked


def main():
    parser = argparse.ArgumentParser(description="Answer the viewer: each line in is one player's view and legal moves, each "
                                                 "line out a move. A line starting with ? asks what the network thinks; one "
                                                 "starting with ! followed by a game record asks for a forecast.")
    parser.add_argument("checkpoint")
    parser.add_argument("--greedy", action="store_true", help="always play the network's top move")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    at = device()
    model = load(args.checkpoint, at)
    network = Network(model, at, greedy=args.greedy, seed=args.seed)
    print("ready", flush=True)
    for line in sys.stdin:
        if line.startswith("!"):
            print("\t".join(forecast(model, at, line[1:].strip(), seed=args.seed)), flush=True)
            continue
        explain = line.startswith("?")
        seen, legal = line.lstrip("?").strip().split(";")
        obs = np.array([int(b) for b in seen.split(",")], np.uint8)[None]
        mask = np.zeros((1, catan.ACTIONS), bool)
        mask[0, [int(c) for c in legal.split(",")]] = True
        if not explain:
            print(int(network(obs, mask)[0]), flush=True)
            continue
        with torch.no_grad():
            logits, values, _ = model(torch.from_numpy(obs).to(at))
            probs = masked(logits.float(), torch.from_numpy(mask).to(at)).softmax(1)[0].cpu()
        top = probs.topk(min(8, int(mask.sum())))
        chances = ",".join(f"{v:.4f}" for v in values[0].float().tolist())
        moves = ",".join(f"{c}:{p:.4f}" for p, c in zip(top.values.tolist(), top.indices.tolist()))
        cards = ",".join(f"{v:.2f}" for v in card_values(model, at, obs[0]))
        print(f"{chances}|{moves}|{cards}", flush=True)


if __name__ == "__main__":
    main()
