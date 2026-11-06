import argparse

import numpy as np
import torch

import catan

from . import deals
from .diagnose import loaded
from .evaluate import rules_of
from .model import device
from .selfplay import OFFER, PHASE_TURN, Rollout
from .trades import HAND, Judge

# Does the deal head offer what the trade judge would? On positions met in self-play where the player may offer, the
# judge picks the best deal both sides confidently gain from (as it does when it teaches the head); the head picks its
# own most likely deal. Reported: how often the two agree, how much probability the head gives the judge's deal, and
# whether the head's own deal gains for it and for its best partner by a quick look of the same values.

@torch.no_grad()
def check(path, at, games, warmup, seed):
    model, critics, args = loaded(path, at)
    autocast = at.type == "cuda"
    buffer = catan.Buffer(games, history=64, max_decisions=args.max_decisions, seed=seed, players=args.players,
                          rules=rules_of(args))
    rollout = Rollout(model, critics, buffer, at, autocast, args.players, False, seed)
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(at)
    totals = dict(positions=0, confident=0, agree=0, head_prob=0.0, head_gains=0, head_mine=0.0, judge_mine=0.0)
    judge = Judge(model, at, autocast, getattr(args, "search_margin", 0.01), getattr(args, "trade_symmetries", 6), seed)
    for _ in range(warmup):
        rollout.collect(args.rounds)
        o, m, _ = buffer.observe()
        forbidden, counts = buffer.forbidden()
        rows = np.flatnonzero((o[:, 296] == PHASE_TURN) & (o[:, 297] == 1) & m[:, OFFER])
        if len(rows) == 0:
            continue
        _, _, summary, goods = model(tensor(o[rows]), tensor(m[rows]), with_goods=True)
        hand = tensor(o[rows, HAND]).long()
        no = torch.zeros(len(rows), dtype=torch.bool, device=at)
        scores = lambda r, c: model.deal_scores(summary[r], goods[r], no[r], c, hand[r])
        head, _, _ = deals.sample(scores, hand, no, tensor(forbidden[rows]), tensor(counts[rows]).long(), greedy=True)
        head = head.cpu().numpy().astype(np.int8)

        def accept(pair_row, offered, partner):
            yes = torch.sigmoid(critics.accept_logits(summary[tensor(pair_row)], tensor(offered).long()).float())
            return yes[torch.arange(len(offered), device=at), tensor(partner)]

        best = judge.offers(buffer, rows, o, forbidden, counts, args.players, accept=accept)
        totals["positions"] += len(rows)
        # The head's own deal, judged by a quick look: its gain, and its best partner's.
        partners = args.players - 1
        valid = ((head < 0).any(1) & (head > 0).any(1))
        games_ = np.repeat(rows, partners).astype(np.uint64)
        offered, offsets = np.repeat(head, partners, 0), np.tile(np.arange(1, partners + 1), len(rows)).astype(np.uint8)
        mine_after, ok = buffer.trade_views(games_, offered, offsets)
        theirs_after, _ = buffer.trade_views(games_, offered, offsets, as_partner=True)
        theirs_now = buffer.observe_as(games_, offsets)
        value = lambda views: judge.seat_values(views, [0])[:, 0].cpu().numpy()
        mine = (value(mine_after) - np.repeat(value(o[rows]), partners)).reshape(len(rows), partners)
        theirs = (value(theirs_after) - value(theirs_now)).reshape(len(rows), partners)
        both = (mine > 0) & (theirs > 0) & ok.reshape(len(rows), partners) & valid[:, None]
        totals["head_gains"] += int(both.any(1).sum())
        index = {g: i for i, g in enumerate(rows)}
        for g, (deal, partner, gain, _) in best.items():
            i = index[g]
            totals["confident"] += 1
            totals["agree"] += int((head[i] == deal).all())
            totals["judge_mine"] += gain
            # Only a partner who gains too would take it; with none, the head's deal gains nothing.
            totals["head_mine"] += float(np.where(both[i], mine[i], 0.0).max())
            choices, masks = deals.walk(tensor(deal[None]).long(), hand[i:i + 1], tensor(forbidden[g:g + 1]), tensor(counts[g:g + 1]).long())
            lp, _ = model.deal_log_probs(summary[i:i + 1], goods[i:i + 1], no[i:i + 1], choices, masks, hand[i:i + 1])
            totals["head_prob"] += float(lp.exp())
    n, c = max(totals["positions"], 1), max(totals["confident"], 1)
    return {
        "positions where an offer is possible": totals["positions"],
        "judge finds a deal both sides gain from": f"{totals['confident'] / n:.2f}",
        "head's own deal gains for both (quick look)": f"{totals['head_gains'] / n:.2f}",
        "head picks exactly the judge's deal": f"{totals['agree'] / c:.2f}",
        "head's probability of the judge's deal": f"{totals['head_prob'] / c:.3f}",
        "proposer's gain with a partner who gains too: judge vs head": f"{totals['judge_mine'] / c:+.4f} vs {totals['head_mine'] / c:+.4f}",
    }

def main():
    parser = argparse.ArgumentParser(description="How closely the deal head offers what the trade judge would.")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=8, help="batches played; positions are read after each")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name, value in check(args.checkpoint, device(), args.games, args.warmup, args.seed).items():
        print(f"  {name:48s} {value}")

if __name__ == "__main__":
    main()
