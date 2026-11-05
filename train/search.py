import argparse

import numpy as np
import torch

import catan

from . import deals
from .agents import Network

ACCEPT, DECLINE, END_TURN, PHASE_AT, PHASE_RESPOND = 247, 248, 225, 296, 13
from .evaluate import RULE_SETS, matchup, rules_named
from .model import device, load, load_critics
from .trades import Judge

@torch.no_grad()
def run(search, model, at, autocast=False, critics=None):
    """Answer the search's questions with the network until every root is done, and return its result."""
    while True:
        obs, masks, hidden, _ = search.leaves()
        if len(obs) == 0:
            return search.result()
        with torch.autocast(at.type, dtype=torch.bfloat16, enabled=autocast):
            log_probs, values, summary = model(torch.from_numpy(obs).to(at), torch.from_numpy(masks).to(at))
            if critics is not None:
                values = critics.privileged(summary, torch.from_numpy(hidden).to(at))
        search.expand(log_probs.float().cpu().numpy(), values.float().cpu().numpy())

class Searcher(Network):
    """The network, judging trades and looking ahead before each move. It offers or counters only with the deal the
    judge finds best for both sides (and otherwise doesn't), accepts an offer only when accepting confidently raises
    its own value, and plays every other move it can search as the search picks. With no simulations it judges
    trades and plays the rest by instinct."""
    uses_state = True

    def __init__(self, model, device, simulations=64, worlds=4, considered=16, greedy=False, seed=0, temperature=1.0,
                 critics=None, margin=0.01, value_scale=0.1, rescale=True, symmetries=6):
        super().__init__(model, device, greedy, seed, temperature)
        self.simulations, self.worlds, self.considered, self.critics = simulations, worlds, considered, critics
        self.scaling = dict(value_scale=value_scale, rescale=rescale)
        self.judge = Judge(model, device, margin=margin, symmetries=symmetries, seed=seed)
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, masks, forbidden, counts, buffer=None, games=None):
        actions, offers = super().__call__(obs, masks, forbidden, counts)
        if buffer is None:
            return actions, offers
        if isinstance(buffer, catan.Game):
            single = catan.Buffer(1, max_decisions=1 << 30, players=buffer.players())
            single.load(0, buffer)
            buffer, games = single, np.zeros(1, np.int64)
        picked = np.full(len(obs), -1)
        if self.simulations:
            search = buffer.search(games.astype(np.uint64), self.simulations, self.worlds, self.considered,
                                   seed=int(self.rng.integers(1 << 62)), noise=not self.greedy, **self.scaling)
            _, picked, _ = run(search, self.model, self.device, critics=self.critics)
        searched = (picked >= 0) & ~np.isin(actions, (deals.OFFER, deals.COUNTER))
        actions[searched], offers[searched] = picked[searched], 0
        self.judge_trades(obs, masks, forbidden, counts, buffer, games, actions, offers, picked)
        return actions, offers

    def judge_trades(self, obs, masks, forbidden, counts, buffer, games, actions, offers, picked):
        # The judge reads every game by its slot in the buffer, so the rows here are laid out at their slots.
        size = len(buffer)
        at_slot = lambda rows, shape, dtype: np.zeros((size, *shape), dtype)
        o, f, c = at_slot(None, obs.shape[1:], obs.dtype), at_slot(None, forbidden.shape[1:], forbidden.dtype), np.zeros(size, counts.dtype)
        o[games], f[games], c[games] = obs, forbidden, counts
        players = int(obs[0, 221])
        row_of = {int(g): i for i, g in enumerate(games)}
        answering = np.flatnonzero(masks[:, ACCEPT] & (obs[:, PHASE_AT] == PHASE_RESPOND))
        if len(answering):
            slots, good, bad = self.judge.answers(buffer, games[answering], o)
            for slot, yes, no in zip(slots, good, bad):
                i = row_of[int(slot)]
                if yes:
                    actions[i], offers[i] = ACCEPT, 0
                elif no and actions[i] == ACCEPT:
                    actions[i], offers[i] = DECLINE, 0
        offering = np.flatnonzero(actions == deals.OFFER)
        if len(offering):
            best = self.judge.offers(buffer, games[offering], o, f, c, players)
            for i in offering:
                found = best.get(int(games[i]))
                if found is not None:
                    offers[i] = found[0]
                else:
                    actions[i], offers[i] = (picked[i] if picked[i] >= 0 else END_TURN), 0
        countering = np.flatnonzero(actions == deals.COUNTER)
        if len(countering):
            best = self.judge.counters(buffer, games[countering], o, f, c)
            for i in countering:
                found = best.get(int(games[i]))
                if found is not None:
                    offers[i] = found[0]
                else:
                    actions[i], offers[i] = DECLINE, 0

def main():
    parser = argparse.ArgumentParser(description="How much stronger is a network when it looks ahead? The same checkpoint, "
                                                 "with and without search, head to head.")
    parser.add_argument("checkpoint")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--rules", choices=list(RULE_SETS), default="colonist-1v1")
    parser.add_argument("--no-trading", action="store_true")
    parser.add_argument("--simulations", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--considered", type=int, default=16)
    parser.add_argument("--privileged", action="store_true", help="value the imagined worlds with the critic that sees every hand")
    parser.add_argument("--value-scale", type=float, nargs="+", default=[0.1], help="how strongly better Q-values outweigh the prior")
    parser.add_argument("--fixed", action="store_true", help="read Q-values on their own scale instead of stretching each node's to [0, 1]")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    at = device()
    model = load(args.checkpoint, at)
    critics = load_critics(args.checkpoint, at) if args.privileged else None
    rules = rules_named(args.rules, trading=not args.no_trading)
    plain = Network(model, at, seed=args.seed)
    print(f"{args.checkpoint}: searching against itself without search, {args.players} players, {args.games} games each way"
          + (" (0.50 is even)" if args.players > 2 else ""))
    for simulations in args.simulations:
        for value_scale in args.value_scale:
            searcher = Searcher(model, at, simulations, args.worlds, args.considered, seed=args.seed, critics=critics,
                                value_scale=value_scale, rescale=not args.fixed)
            result = matchup(searcher, plain, games=args.games, players=args.players, rules=rules, seed=args.seed)
            alone = "" if result["alone"][1] is None else f"   alone {result['alone'][0]:.3f} / {result['alone'][1]:.3f}"
            spread = 0.5 / max(result["decided"], 1) ** 0.5
            print(f"  {simulations:4d} simulations, value scale {value_scale:g}{' fixed' if args.fixed else ''}: "
                  f"search scores {result['share']:.3f} ± {spread:.3f}{alone}", flush=True)

if __name__ == "__main__":
    main()
