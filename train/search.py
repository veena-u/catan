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


class Node:
    def __init__(self, prior, value):
        self.prior, self.value = prior, value
        self.visits, self.total = np.zeros(len(prior)), np.zeros(len(prior))

    def select(self, legal, c):
        q = np.where(self.visits > 0, self.total / np.maximum(self.visits, 1), self.value)
        score = q + c * self.prior * np.sqrt(self.visits.sum() + 1) / (1 + self.visits)
        action = int(np.where(legal, score, -np.inf).argmax())
        self.visits[action] += 1
        return action


class Deep(Search):
    """Search over the searching player's own moves several decisions deep: a tree whose branches are only that
    player's choices, so it never plans on cards it cannot see. Every simulation re-deals the hidden cards;
    the other players move as the network would."""

    def __init__(self, model, at, simulations=32, wave=16, steps=48, c=1.5, chunk=8192, seed=0):
        super().__init__(model, at, chunk=chunk, seed=seed)
        self.simulations, self.wave, self.steps, self.c = simulations, wave, steps, c

    def __call__(self, obs, masks, buffer, games, players):
        n, seats = len(obs), int(obs[0, 221])
        logits, values = self.evaluate(obs, masks)
        trees = [{(): Node(logits[i].softmax(0).numpy(), values[i, 0].item())} for i in range(n)]
        for _ in range(max(self.simulations // self.wave, 1)):
            owner = np.repeat(np.arange(n), self.wave)
            paths = [[((), trees[i][()].select(masks[i], self.c))] for i in owner]
            self.seed += 1
            worlds = buffer.worlds(games[owner], np.array([path[0][1] for path in paths]), 1, self.seed)
            root = players[owner]
            result = np.full(len(owner), np.nan)
            for _ in range(self.steps):
                o, m, p = worlds.observe()
                over = np.isnan(result) & (p < 0)
                if over.any():
                    winners = worlds.winners()
                    result[over] = np.where(winners[over] == root[over], 1.0, -1.0 / (seats - 1))
                open_ = np.isnan(result) & (p >= 0)
                if not open_.any():
                    break
                actions = m.argmax(1).astype(np.int64)
                rows = np.flatnonzero(open_)
                l, v = self.evaluate(o[rows], m[rows])
                probs = l.softmax(1)
                drawn = torch.multinomial(probs, 1, generator=self.generator).squeeze(1).numpy()
                for k, w in enumerate(rows):
                    if p[w] != root[w] or m[w].sum() == 1:
                        actions[w] = drawn[k]
                        continue
                    key = tuple(a for _, a in paths[w])
                    tree = trees[owner[w]]
                    if key not in tree:
                        tree[key] = Node(probs[k].numpy(), v[k, 0].item())
                        result[w] = v[k, 0].item()
                        continue
                    actions[w] = tree[key].select(m[w], self.c)
                    paths[w].append((key, int(actions[w])))
                worlds.step(actions)
            o, m, p = worlds.observe()
            left = np.flatnonzero(np.isnan(result))
            if len(left):
                live = left[p[left] >= 0]
                if len(live):
                    _, v = self.evaluate(o[live])
                    result[live] = v.numpy()[np.arange(len(live)), (root[live] - p[live]) % seats]
                done = left[p[left] < 0]
                if len(done):
                    winners = worlds.winners()
                    result[done] = np.where(winners[done] == root[done], 1.0, -1.0 / (seats - 1))
            for w, path in enumerate(paths):
                tree = trees[owner[w]]
                for key, action in path:
                    tree[key].total[action] += result[w]
        return np.array([np.where(masks[i], trees[i][()].visits, -1).argmax() for i in range(n)])


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
    parser.add_argument("--belief-only", action="store_true", help="skip the match without the belief network")
    parser.add_argument("--deep", action="store_true",
                        help="instead: search a tree of the player's own moves, several decisions deep, against the network and against plain search")
    parser.add_argument("--simulations", type=int, default=32, help="deep search: imagined games per move")
    parser.add_argument("--steps", type=int, default=48, help="deep search: decisions played forward in each imagined game")
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

    if args.deep:
        deep = Deep(model, at, args.simulations, steps=args.steps, seed=2)
        match("deep search vs network", deep, Network(model, at, seed=1), 7)
        match("deep search vs search", deep, search, 9)
        return
    if not args.belief_only:
        match("search vs network", search, Network(model, at, seed=1), 7)
    if args.belief:
        guided = Search(model, at, *settings, seed=1, belief=load_belief(args.belief, at))
        match("belief search vs network", guided, Network(model, at, seed=1), 7)
        match("belief search vs search", guided, search, 9)
    if args.bot:
        match("search vs bot", search, Bot(), 8)


if __name__ == "__main__":
    main()
