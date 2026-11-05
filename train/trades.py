import numpy as np
import torch

from . import deals
from .features import SYMMETRIES

SHORTLIST = 8
HAND = slice(222, 227)

class Judge:
    """Values trades with the network's own estimates. Each side's gain is its own view after the trade against its
    own view now, so most of the network's error cancels; the spread of those gains across board symmetries is the
    network's own noise there, and a trade counts only if it clears the margin by two standard errors."""

    def __init__(self, model, at, autocast=False, margin=0.01, symmetries=6, seed=0):
        self.model, self.at, self.autocast, self.margin, self.symmetries = model, at, autocast, margin, symmetries
        self.rng = np.random.default_rng(seed)

    def tensor(self, array):
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.at)

    @torch.no_grad()
    def seat_values(self, views, turns):
        """The network's value for the seat each view belongs to, once per board symmetry: [views, symmetries]."""
        s = len(turns)
        obs = self.tensor(views).repeat_interleave(s, 0)
        symmetry = torch.tensor(turns, device=self.at).repeat(len(views))
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            _, values, _ = self.model(obs, None, symmetry)
        return values.float()[:, 0].view(len(views), s)

    def turns(self):
        others = self.rng.choice(np.arange(1, SYMMETRIES), self.symmetries - 1, replace=False) if self.symmetries > 1 else []
        return [0, *map(int, others)]

    def confident(self, gains):
        """Mean gain, and whether it clears the margin by two standard errors upward, or downward."""
        mean = gains.mean(1)
        error = gains.std(1) / gains.shape[1] ** 0.5 if gains.shape[1] > 1 else torch.zeros_like(mean)
        return mean, mean - 2 * error > self.margin, mean + 2 * error < -self.margin

    def choose(self, groups, ok, mine_after, mine_now, mine_of, theirs_after, theirs_now, theirs_of, weight):
        """For each group of candidate trades, the one with the best confident gain for both sides, weighted: a quick
        look at every candidate, then a careful one across symmetries at each group's most promising few."""
        mine = (self.seat_values(mine_after, [0]) - self.seat_values(mine_now, [0])[self.tensor(mine_of)])[:, 0]
        theirs = (self.seat_values(theirs_after, [0]) - self.seat_values(theirs_now, [0])[self.tensor(theirs_of)])[:, 0]
        quick = torch.where(self.tensor(ok) & (mine > 0) & (theirs > 0), mine * weight, torch.full_like(mine, -1e9)).cpu().numpy()
        shortlist = []
        for group in np.unique(groups):
            members = np.flatnonzero((groups == group) & (quick > -1e9))
            shortlist.extend(members[np.argsort(-quick[members])[:SHORTLIST]])
        if not shortlist:
            return {}
        pick = np.array(shortlist)
        turns = self.turns()
        mine_gain = self.seat_values(mine_after[pick], turns) - self.seat_values(mine_now, turns)[self.tensor(mine_of[pick])]
        theirs_gain = self.seat_values(theirs_after[pick], turns) - self.seat_values(theirs_now, turns)[self.tensor(theirs_of[pick])]
        mine_mean, mine_sure, _ = self.confident(mine_gain)
        theirs_mean, theirs_sure, _ = self.confident(theirs_gain)
        score = torch.where(mine_sure & theirs_sure, mine_mean * weight[self.tensor(pick)], torch.full_like(mine_mean, -1e9)).cpu().numpy()
        best = {}
        for i, candidate in enumerate(pick):
            group = groups[candidate]
            if score[i] > -1e9 and (group not in best or score[i] > score[best[group][0]]):
                best[group] = (i, candidate)
        return {group: (candidate, float(mine_mean[i]), float(theirs_mean[i])) for group, (i, candidate) in best.items()}

    @staticmethod
    def pool(hand, forbidden, extra=None):
        """Every deal of one or two cards for one or two (and any extra ones) this hand can pay for, not yet refused."""
        pool = deals.SMALL if extra is None or len(extra) == 0 else np.unique(np.concatenate([deals.SMALL, extra]), axis=0)
        affordable = (-pool.astype(np.int16) <= hand.astype(np.int16)).all(1)
        fresh = ~(pool[:, None] == forbidden[None]).all(-1).any(1)
        return pool[affordable & (pool < 0).any(1) & (pool > 0).any(1) & fresh]

    def offers(self, buffer, rows, o, forbidden, counts, players, extra=None, accept=None):
        """For each game where the player to act may offer: the best deal, partner and both gains, if any is confident.
        `accept(rows, deals, partners)` weighs each deal by how likely that partner is to take it."""
        partners = players - 1
        pools = [self.pool(o[g, HAND], forbidden[g, :counts[g]], None if extra is None else extra[i]) for i, g in enumerate(rows)]
        sizes = np.array([len(p) for p in pools])
        if sizes.sum() == 0:
            return {}
        flat, owner = np.concatenate(pools), np.repeat(np.arange(len(rows)), sizes)
        pair_row, pair_partner = np.repeat(owner, partners), np.tile(np.arange(partners), len(flat))
        games, offered = rows[pair_row].astype(np.uint64), np.repeat(flat, partners, 0)
        offsets = (pair_partner + 1).astype(np.uint8)
        mine_after, ok = buffer.trade_views(games, offered, offsets)
        theirs_after, _ = buffer.trade_views(games, offered, offsets, as_partner=True)
        theirs_now = buffer.observe_as(np.repeat(rows, partners).astype(np.uint64), np.tile(np.arange(1, partners + 1), len(rows)).astype(np.uint8))
        weight = accept(pair_row, offered, pair_partner) if accept is not None else torch.ones(len(offered), device=self.at)
        best = self.choose(pair_row, ok, mine_after, o[rows], pair_row, theirs_after, theirs_now, pair_row * partners + pair_partner, weight)
        return {int(rows[r]): (offered[c], int(pair_partner[c]) + 1, mine, theirs) for r, (c, mine, theirs) in best.items()}

    def counters(self, buffer, rows, o, forbidden, counts):
        """For each game where the player to act is answering an offer: the best counter-offer and both gains, if any
        is confident. The other side is the player whose turn it is, who would take the counter-offer."""
        pools = [self.pool(o[g, HAND], forbidden[g, :counts[g]]) for g in rows]
        sizes = np.array([len(p) for p in pools])
        if sizes.sum() == 0:
            return {}
        flat, owner = np.concatenate(pools), np.repeat(np.arange(len(rows)), sizes)
        mine_after, theirs_after, ok = buffer.counter_views(rows[owner].astype(np.uint64), flat)
        theirs_now = buffer.observe_as(rows.astype(np.uint64), np.zeros(len(rows), np.uint8))
        best = self.choose(owner, ok, mine_after, o[rows], owner, theirs_after, theirs_now, owner, torch.ones(len(flat), device=self.at))
        return {int(rows[r]): (flat[c], mine, theirs) for r, (c, mine, theirs) in best.items()}

    def answers(self, buffer, rows, o):
        """For each game where the player to act may accept: whether accepting confidently gains, or confidently loses."""
        views, ok = buffer.answer_views(rows.astype(np.uint64))
        rows, views = rows[ok], views[ok]
        if len(rows) == 0:
            return rows, np.zeros(0, bool), np.zeros(0, bool)
        turns = self.turns()
        _, good, bad = self.confident(self.seat_values(views, turns) - self.seat_values(o[rows], turns))
        return rows, good.cpu().numpy(), bad.cpu().numpy()
