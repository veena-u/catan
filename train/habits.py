import argparse
import collections

import numpy as np

import catan

from .agents import Network
from .bot import Bot
from .model import device, load

KINDS = [("settlement", 0, 54), ("city", 54, 108), ("road", 108, 180), ("robber", 180, 199),
         ("resource pick", 199, 204), ("trade 2:1", 204, 224), ("trade 3:1", 224, 244), ("trade 4:1", 244, 264),
         ("roll", 264, 265), ("end turn", 265, 266), ("buy dev card", 266, 267), ("play knight", 267, 268),
         ("play road building", 268, 269), ("play year of plenty", 269, 270), ("play monopoly", 270, 271),
         ("rob a player", 271, 274), ("offer a trade", 274, 504), ("accept an offer", 504, 505),
         ("decline an offer", 505, 506), ("trade with an accepter", 506, 509), ("take a counter-offer", 509, 512),
         ("cancel a trade", 512, 513), ("counter an offer", None, None)]
PHASE_RESPOND = 13


def kind(code, phase):
    if 274 <= code < 504 and phase == PHASE_RESPOND:
        return "counter an offer"
    return next(name for name, low, high in KINDS if low is not None and low <= code < high)


def main():
    parser = argparse.ArgumentParser(description="What a saved network does, on average, over many games against the bot.")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()

    at = device()
    network, bot = Network(load(args.checkpoint, at), at, seed=0), Bot()
    moves = {"network": collections.Counter(), "bot": collections.Counter()}
    offered = {"network": collections.Counter(), "bot": collections.Counter()}
    end = {"network": collections.Counter(), "bot": collections.Counter()}
    wins = 0
    for seed in range(args.games):
        seat = seed % args.players
        players = {s: ("network", network) if s == seat else ("bot", bot) for s in range(args.players)}
        game = catan.Game(10_000 + seed, args.players)
        game.settle()
        last = {}
        while game.winner() is None and game.steps() < 5000:
            obs, mask = game.observe()
            name, agent = players[game.to_act()]
            for label in ("trade 2:1", "trade 3:1"):
                low, high = next((lo, hi) for n, lo, hi in KINDS if n == label)
                offered[name][label] += bool(mask[low:high].any())
            code = int(agent(obs[None], mask[None])[0])
            moves[name][kind(code, obs[296])] += 1
            last[name] = obs
            game.step(code)
            game.settle()
        wins += players[game.winner()][0] == "network"
        for name, obs in last.items():
            for label, value in count(obs).items():
                end[name][label] += value

    n = args.games
    bots = n * (args.players - 1)
    print(f"network wins {wins}/{n} ({wins / n:.0%}; fair share {1 / args.players:.0%})\n")
    print(f"{'per game':28s} {'network':>9s} {'each bot':>9s}")
    for label, *_ in KINDS:
        if label in ("roll", "end turn", "resource pick"):
            continue
        print(f"{label:28s} {moves['network'][label] / n:9.2f} {moves['bot'][label] / bots:9.2f}")
    print()
    for label in ("trade 2:1", "trade 3:1"):
        print(f"{'decisions where ' + label + ' legal':28s} {offered['network'][label] / n:9.2f} {offered['bot'][label] / bots:9.2f}")
    print()
    for label in ("settlements", "cities", "harbor buildings"):
        print(f"{label + ' near the end':28s} {end['network'][label] / n:9.2f} {end['bot'][label] / bots:9.2f}")


def count(obs):
    bits = lambda at, n: np.unpackbits(obs[at:at + 8], bitorder="little")[:n].astype(bool)
    settlements, cities = bits(0, 54), bits(8, 54)
    harbors = obs[166:220] > 0
    return {"settlements": int(settlements.sum()), "cities": int(cities.sum()),
            "harbor buildings": int(((settlements | cities) & harbors).sum())}


if __name__ == "__main__":
    main()
