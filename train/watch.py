import argparse

import catan

from .agents import Network, Random
from .bot import Bot
from .model import device, load


def main():
    parser = argparse.ArgumentParser(description="Play one game with a saved network and print it for the viewer.")
    parser.add_argument("checkpoint", help="a .pt file, e.g. runs/anneal/latest.pt")
    parser.add_argument("--opponent", choices=["bot", "random", "self"], default="bot")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--seat", type=int, default=0, help="the network's seat; 0 moves first")
    parser.add_argument("--seed", type=int, default=0, help="the board; the viewer uses the same seed")
    parser.add_argument("--greedy", action="store_true", help="always play the network's top move")
    args = parser.parse_args()

    at = device()
    network = Network(load(args.checkpoint, at), at, greedy=args.greedy, seed=args.seed)
    opponent = {"bot": Bot(), "random": Random(args.seed), "self": network}[args.opponent]
    players = {seat: network if seat == args.seat else opponent for seat in range(args.players)}

    game = catan.Game(args.seed, args.players)
    game.settle()
    while game.winner() is None and game.steps() < 5000:
        obs, mask = game.observe()
        game.step(int(players[game.to_act()](obs[None], mask[None])[0]))
        game.settle()

    winner = game.winner()
    seats = ["First", "Second", "Third", "Fourth"]
    names = {seat: "network" if seat == args.seat else args.opponent for seat in range(args.players)}
    result = "no winner" if winner is None else f"{names[winner]} (player {seats[winner]}) wins"
    print(f"{result} after {game.steps()} steps. Network is {seats[args.seat]} of {args.players}.")
    print("Paste this into the viewer's game box and press Load:")
    print(game.record())


if __name__ == "__main__":
    main()
