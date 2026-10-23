import argparse
import sys

import numpy as np

import catan

from .agents import Network
from .model import device, load


def main():
    parser = argparse.ArgumentParser(description="Answer the viewer: each line in is one player's view and legal moves, each line out a move.")
    parser.add_argument("checkpoint")
    parser.add_argument("--greedy", action="store_true", help="always play the network's top move")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    at = device()
    network = Network(load(args.checkpoint, at), at, greedy=args.greedy, seed=args.seed)
    print("ready", flush=True)
    for line in sys.stdin:
        seen, legal = line.strip().split(";")
        obs = np.array([int(b) for b in seen.split(",")], np.uint8)[None]
        mask = np.zeros((1, catan.ACTIONS), bool)
        mask[0, [int(c) for c in legal.split(",")]] = True
        print(int(network(obs, mask)[0]), flush=True)


if __name__ == "__main__":
    main()
