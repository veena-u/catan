import argparse
import sys

import numpy as np
import torch

import catan

from .agents import Network
from .insight import card_values, forecast, replay
from .model import device, load, masked
from .search import search_game


def main():
    parser = argparse.ArgumentParser(description="Answer the viewer: each line in is one player's view and legal moves, each "
                                                 "line out a move. A line starting with ? asks what the network thinks; one "
                                                 "starting with !turns| followed by a game record asks for a forecast, and one "
                                                 "starting with * followed by a game record asks for a move chosen by search.")
    parser.add_argument("checkpoint")
    parser.add_argument("--greedy", action="store_true", help="always play the network's top move")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search", action="store_true", help="think ahead before each move; moves come from game records")
    args = parser.parse_args()

    at = device()
    model = load(args.checkpoint, at)
    network = Network(model, at, greedy=args.greedy, seed=args.seed)
    print("ready", flush=True)
    moves = 0
    for line in sys.stdin:
        if line.startswith("*"):
            moves += 1
            print(search_game(model, at, replay(line[1:].strip()), seed=args.seed + moves), flush=True)
            continue
        if line.startswith("!"):
            turns, record = line[1:].strip().split("|", 1)
            lines, plan = forecast(model, at, record, turns=int(turns), seed=args.seed)
            spots = ";".join(f"{kind[0]}:" + ",".join(f"{spot}={share:.3f}" for spot, share in shares.items())
                             for kind, shares in plan.items())
            print("\t".join(lines) + "##" + spots, flush=True)
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
