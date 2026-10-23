import argparse
import time

import numpy as np
import torch

from .agents import Network
from .belief import load_belief
from .bot import Bot
from .evaluate import play, report
from .model import device, load, masked


class Search:
    searches = True

    def __init__(self, model, at, candidates=8, samples=8, depth=16, eta=10.0, chunk=8192, seed=0, belief=None):
        self.model, self.at, self.belief = model, at, belief
        self.candidates, self.samples, self.depth, self.eta, self.chunk = candidates, samples, depth, eta, chunk
        self.autocast = at.type == "cuda"
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.seed = seed

    @torch.no_grad()
    def evaluate(self, obs, masks=None):
        logits, values = [], []
        for i in range(0, len(obs), self.chunk):
            o = torch.from_numpy(obs[i:i + self.chunk]).to(self.at)
            with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
                l, v, _ = self.model(o)
            if masks is not None:
                l = masked(l.float(), torch.from_numpy(masks[i:i + self.chunk]).to(self.at))
            logits.append(l.float().cpu())
            values.append(v.float().cpu())
        return torch.cat(logits), torch.cat(values)

    def sample(self, obs, masks):
        logits, _ = self.evaluate(obs, masks)
        return torch.multinomial(logits.softmax(1), 1, generator=self.generator).squeeze(1).numpy()

    def improve(self, obs, masks, buffer, games, players):
        """The network's top moves at each position, and how much better search finds each one: (choices, prior, q)."""
        logits, _ = self.evaluate(obs, masks)
        prior = logits.log_softmax(1)
        k = min(self.candidates, prior.shape[1])
        top, choices = prior.topk(k, dim=1)
        valid = torch.isfinite(top)
        rows, ranks = valid.nonzero(as_tuple=True)
        pair_games = games[rows.numpy()]
        pair_actions = choices[rows, ranks].numpy()

        self.seed += 1
        deals = None
        if self.belief is not None:
            deals = self.belief.deals(np.repeat(obs[rows.numpy()], self.samples, axis=0), self.generator).reshape(-1)
        worlds = buffer.worlds(pair_games, pair_actions, self.samples, self.seed, deals)
        for _ in range(self.depth):
            o, m, p = worlds.observe()
            live = p >= 0
            if not live.any():
                break
            actions = np.zeros(len(p), np.int64)
            actions[live] = self.sample(o[live], m[live])
            worlds.step(actions)

        o, m, p = worlds.observe()
        live, winners = p >= 0, worlds.winners()
        root = np.repeat(players[rows.numpy()], self.samples)
        seats = int(obs[0, 221])
        value = np.zeros(len(p), np.float32)
        if live.any():
            _, v = self.evaluate(o[live])
            offset = (root[live] - p[live]) % seats
            value[live] = v.numpy()[np.arange(len(offset)), offset]
        over = winners >= 0
        value[over] = np.where(winners[over] == root[over], 1.0, -1.0 / (seats - 1))

        q = torch.full((len(obs), k), float("-inf"))
        q[rows, ranks] = torch.from_numpy(value.reshape(-1, self.samples).mean(1))
        return choices, top, q

    def __call__(self, obs, masks, buffer, games, players):
        choices, top, q = self.improve(obs, masks, buffer, games, players)
        best = (top + self.eta * q).argmax(1)
        return choices[torch.arange(len(obs)), best].numpy()


def main():
    parser = argparse.ArgumentParser(description="Does search improve a saved network? Plays it with search against itself without.")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=8, help="the network's top moves to consider")
    parser.add_argument("--samples", type=int, default=8, help="random deals of the hidden cards per candidate")
    parser.add_argument("--depth", type=int, default=16, help="decisions played forward in each imagined world")
    parser.add_argument("--eta", type=float, default=10.0, help="how far the search may move away from the network's choice")
    parser.add_argument("--belief", help="deal the hidden cards from this belief network instead of uniformly")
    parser.add_argument("--bot", action="store_true", help="also play search against the bot")
    parser.add_argument("--players", type=int, default=4)
    args = parser.parse_args()

    at = device()
    model = load(args.checkpoint, at)
    settings = (args.candidates, args.samples, args.depth, args.eta)
    search = Search(model, at, *settings)
    parallel = min(args.games, 256)

    def match(name, a, b, seed):
        start = time.time()
        report(name, play(a, b, games=args.games, parallel=parallel, seed=seed, players=args.players))
        print(f"  ({time.time() - start:.0f}s)", flush=True)

    match("search vs network", search, Network(model, at, seed=1), 7)
    if args.belief:
        guided = Search(model, at, *settings, seed=1, belief=load_belief(args.belief, at))
        match("belief search vs network", guided, Network(model, at, seed=1), 7)
        match("belief search vs search", guided, search, 9)
    if args.bot:
        match("search vs bot", search, Bot(), 8)


if __name__ == "__main__":
    main()
