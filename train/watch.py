import argparse

import catan

from .agents import Network, Random
from .bot import Bot
from .model import device, load


def main():
    parser = argparse.ArgumentParser(description="Play one game with a saved network and print it for the viewer.")
    parser.add_argument("checkpoint", help="a .pt file, e.g. runs/anneal/latest.pt")
    parser.add_argument("--opponent", choices=["bot", "random", "self"], default="bot")
    parser.add_argument("--seat", type=int, choices=[0, 1], default=0, help="0: the network moves first")
    parser.add_argument("--seed", type=int, default=0, help="the board; the viewer uses the same seed")
    parser.add_argument("--greedy", action="store_true", help="always play the network's top move")
    args = parser.parse_args()

    at = device()
    network = Network(load(args.checkpoint, at), at, greedy=args.greedy, seed=args.seed)
    opponent = {"bot": Bot(), "random": Random(args.seed), "self": network}[args.opponent]
    players = {args.seat: network, 1 - args.seat: opponent}

    game = catan.Game(args.seed)
    game.settle()
    while game.winner() is None and game.steps() < 5000:
        obs, mask = game.observe()
        game.step(int(players[game.to_act()](obs[None], mask[None])[0]))
        game.settle()

    winner = game.winner()
    names = {args.seat: "network", 1 - args.seat: args.opponent}
    result = "no winner" if winner is None else f"{names[winner]} (player {'First' if winner == 0 else 'Second'}) wins"
    print(f"{result} after {game.steps()} steps. Network is {'First' if args.seat == 0 else 'Second'}.")
    print("Paste this into the viewer's game box and press Load:")
    print(game.record())


if __name__ == "__main__":
    main()
