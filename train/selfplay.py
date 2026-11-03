import argparse
import copy
import json
import subprocess
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

import catan

from . import deals
from .agents import Network, Random
from .evaluate import RULE_SETS, TURN_BUDGET, play, rules_of
from .features import ANSWERS_AT, BUDGET_LEFT, HEX_VERTEX, LAST_ACCEPTED, LAST_COUNTERED, OFFERS_MADE, SEATS, SYMMETRIES
from .model import KIND_COUNT, KIND_MEMBERS, KIND_RUNS, KINDS, Model, device, load

HIDDEN = (SEATS - 1) * 10
TRADED = np.isin(KINDS, (22, 23))
ANSWERS = 3
SETUP_SETTLEMENT, PHASE_TURN, PHASE_RESPOND, PHASE_CONFIRM, PHASE_DRAFT = 0, 2, 13, 14, 15
OFFER, COUNTER, END_TURN, TRADE_WITH, CANCEL, DESERT = 234, 249, 225, 250, 256, 5
CORNER_HEXES = HEX_VERTEX.sum(0)

class Habits:

    def __init__(self):
        self.setups = self.pips = self.desert = self.coast = self.sure = 0
        self.decisions = self.offers = self.counters = self.cancels = self.dropped = 0
        self.ends = self.timeouts = 0
        self.kinds = np.zeros(KIND_COUNT, np.int64)

    def record(self, o, chosen, own, probs):
        phase = o[:, 296]
        self.decisions += int(own.sum())
        self.kinds += np.bincount(KINDS[chosen[own]], minlength=KIND_COUNT)
        self.offers += int((own & (chosen == OFFER)).sum())
        self.counters += int((own & (chosen == COUNTER)).sum())
        self.cancels += int((own & (phase == PHASE_CONFIRM) & (chosen == CANCEL)).sum())
        timed = own & ((phase == PHASE_TURN) & (o[:, 297] == 1) | (phase == PHASE_CONFIRM)) & (o[:, BUDGET_LEFT] == 1) & (chosen != END_TURN)
        self.ends += int((own & (phase == PHASE_TURN) & (chosen == END_TURN)).sum()) + int(timed.sum())
        self.timeouts += int(timed.sum())
        setup = np.flatnonzero(own & (phase == SETUP_SETTLEMENT))
        if len(setup) == 0:
            return
        corner = chosen[setup]
        tokens = o[setup, 147:166].astype(np.int64)
        hex_pips = np.where(tokens == 0, 0, 6 - np.abs(7 - tokens))
        touches = HEX_VERTEX[:, corner].T > 0
        self.setups += len(setup)
        self.pips += int((hex_pips * touches).sum())
        self.desert += int(((o[setup, 128:147] == DESERT) & touches).any(1).sum())
        self.coast += int(((CORNER_HEXES[corner] < 3) & (o[setup, 166 + corner] == 0)).sum())
        self.sure += int((probs[torch.from_numpy(setup).to(probs.device)].max(1).values > 0.99).sum())

    def values(self):
        s, d = max(self.setups, 1), max(self.decisions, 1) / 1000
        shares = {f"kind_{name.replace(' ', '_')}": count / max(self.decisions, 1)
                  for (name, *_), count in zip(KIND_RUNS, self.kinds)}
        return {"setup_pips": self.pips / s, "setup_desert": self.desert / s, "setup_bare_coast": self.coast / s,
                "setup_sure": self.sure / s, "offers_per_1k": self.offers / d, "counters_per_1k": self.counters / d,
                "cancels_per_1k": self.cancels / d, "dropped_offers_per_1k": self.dropped / d,
                "turns_out_of_time": self.timeouts / max(self.ends, 1), **shares}

    def summary(self):
        v = self.values()
        return (f"setup pips {v['setup_pips']:.1f} desert {v['setup_desert']:.2f} bare coast {v['setup_bare_coast']:.2f} "
                f"sure {v['setup_sure']:.2f}  per 1k own moves: offers {v['offers_per_1k']:.1f} "
                f"counters {v['counters_per_1k']:.1f} cancels {v['cancels_per_1k']:.2f} dropped {v['dropped_offers_per_1k']:.2f}  "
                f"turns out of time {v['turns_out_of_time']:.3f}")

def policy(logits, masks):
    logits = logits.float().masked_fill(~masks, -1e9)
    return torch.distributions.Categorical(logits=logits), logits.log_softmax(-1)

def to_absolute(relative, actors, players):
    absolute = torch.zeros_like(relative)
    index = (torch.arange(players, device=relative.device) - actors[:, None]) % players
    absolute[:, :players] = relative[:, :players].gather(1, index)
    return absolute

def to_relative(absolute, actors, players):
    relative = torch.zeros_like(absolute)
    index = (torch.arange(players, device=absolute.device) + actors[:, None]) % players
    relative[:, :players] = absolute[:, :players].gather(1, index)
    return relative

def rewards(winner, players, seats=SEATS, points=None):
    reward = torch.zeros(seats)
    reward[:players] = -1.0 / (players - 1)
    if winner is None:
        return reward
    reward[winner] = 1.0
    if points is not None:
        scale = np.linspace(1.0, -1.0, players)
        order = np.asarray(points[:players], np.float32).copy()
        order[winner] = np.inf
        for seat in range(players):
            if seat != winner:
                above, level = (order > order[seat]).sum(), (order == order[seat]).sum()
                reward[seat] = float(scale[above:above + level].mean())
    return reward

class Critics(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(width + HIDDEN, width), nn.GELU(), nn.Linear(width, SEATS))
        self.hands = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, HIDDEN))
        self.answers = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, (SEATS - 1) * ANSWERS))

    def privileged(self, summary, hidden):
        return torch.tanh(self.hidden(torch.cat([summary.float(), hidden.flatten(1).float() / 5], -1)))

    def guess_hands(self, summary):
        return self.hands(summary.float()).view(-1, SEATS - 1, 10)

    def guess_answers(self, summary):
        return self.answers(summary.float()).view(-1, SEATS - 1, ANSWERS)

class League:

    def __init__(self, size, at, autocast, seed=0):
        self.size, self.at, self.autocast = size, at, autocast
        self.members, self.added = [], 0
        self.rng = np.random.default_rng(seed)
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def add(self, model, name, pinned=False):
        member = {"name": name, "model": model.eval(), "games": 0, "won": 0, "added": self.added, "pinned": pinned}
        self.added += 1
        same = [i for i, m in enumerate(self.members) if m["name"] == name]
        if same:
            self.members[same[0]] = member
            return
        if len(self.members) < self.size:
            self.members.append(member)
            return
        newest = max(range(len(self.members)), key=lambda i: self.members[i]["added"])
        movable = [i for i in range(len(self.members)) if i != newest and not self.members[i]["pinned"]]
        if not movable:
            return
        self.members[max(movable, key=self.win_rate)] = member

    def win_rate(self, i):
        m = self.members[i]
        return (m["won"] + 1) / (m["games"] + 2)

    def pick(self):
        hardness = np.array([(1 - self.win_rate(i)) ** 2 for i in range(len(self.members))]) + 1e-3
        return int(self.rng.choice(len(self.members), p=hardness / hardness.sum()))

    def record(self, i, won):
        self.members[i]["games"] += 1
        self.members[i]["won"] += int(won)

    @torch.no_grad()
    def act(self, i, obs, masks, forbidden, counts, greedy=None):
        model = self.members[i]["model"]
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            logits, _, summary, goods = model(torch.from_numpy(obs).to(self.at), torch.from_numpy(masks).to(self.at), with_goods=True)
        logits = logits.float().cpu()
        moves = torch.multinomial(logits.softmax(1), 1, generator=self.generator).squeeze(1).numpy()
        if greedy is not None and greedy.any():
            moves[greedy] = logits[torch.from_numpy(greedy)].argmax(1).numpy()
        def logits_for(rows, counts):
            at = torch.from_numpy(rows).to(self.at)
            return model.deal_scores(summary[at].float(), goods[at].float(), torch.from_numpy(moves[rows] == COUNTER).to(self.at), counts)
        offers, _, _ = deals.choose(moves, obs, forbidden, counts, logits_for, self.at, generator=self.generator)
        return moves, offers

    def summary(self):
        hardest = min(range(len(self.members)), key=self.win_rate)
        return (f"{len(self.members)} in pool, hardest {self.members[hardest]['name']} "
                f"(network wins {self.win_rate(hardest):.2f} vs it, fair 0.50)")

class Rollout:
    def __init__(self, model, critics, buffer, at, autocast, players, privileged=False, seed=0,
                 vp_reward=0.0, league=None, league_games=0.0, rank=False, greedy_games=0.0, explore_games=0.0,
                 offer_search=0, offer_candidates=8):
        self.model, self.critics, self.buffer, self.at, self.autocast = model, critics, buffer, at, autocast
        self.offer_search, self.offer_candidates = offer_search, offer_candidates
        self.rank, self.greedy_games, self.explore_games = rank, greedy_games, explore_games
        self.games, self.players = len(buffer), players
        self.privileged = privileged
        self.rng = np.random.default_rng(seed)
        self.vp_reward = vp_reward
        game = np.arange(self.games)
        alternate = (np.arange(SEATS)[None, :] + game[:, None]) % 2 == 0
        leagued = game < round(league_games * self.games)
        self.league, self.league_seats = league, leagued[:, None] & alternate
        self.opponent = np.full((self.games, SEATS), -1)
        self.greedy = self.rng.random(self.games) < greedy_games
        self.explore = self.rng.random(self.games) < explore_games

    def reseat(self, game):
        if self.league is not None:
            for seat in np.flatnonzero(self.league_seats[game]):
                self.opponent[game, seat] = self.league.pick()
        self.greedy[game] = self.rng.random() < self.greedy_games
        self.explore[game] = self.rng.random() < self.explore_games

    def explore_opening(self, probs):
        top = probs.topk(min(8, probs.shape[1]), dim=1)
        legal = top.values > 0
        pick = torch.multinomial(legal.float(), 1).squeeze(1)
        return top.indices.gather(1, pick[:, None]).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def search_offers(self, t, rows, o, forbidden, counts, summary, goods, v, search, stats):
        if len(rows) == 0:
            return
        rows = self.rng.choice(rows, min(self.offer_search, len(rows)), replace=False)
        k, at = self.offer_candidates, self.at
        repeat = lambda x: x.repeat_interleave(k, 0)
        index = self.tensor(rows)
        s, g = repeat(summary[index]), repeat(goods[index])
        hand = repeat(self.tensor(o[rows, 222:227]).long())
        with torch.autocast(at.type, dtype=torch.bfloat16, enabled=self.autocast):
            chosen, choices, masks = deals.sample(
                lambda r, c: self.model.deal_scores(s[r], g[r], torch.zeros(len(r), dtype=torch.bool, device=at), c),
                hand, torch.zeros(len(hand), dtype=torch.bool, device=at), repeat(self.tensor(forbidden[rows])),
                repeat(self.tensor(counts[rows]).long()))
        partners = self.players - 1
        games = np.repeat(np.repeat(rows, k), partners)
        offsets = np.tile(np.arange(1, partners + 1), len(rows) * k).astype(np.uint8)
        views, ok = self.buffer.trade_views(games.astype(np.uint64), np.repeat(chosen.cpu().numpy().astype(np.int8), partners, 0), offsets)
        with torch.autocast(at.type, dtype=torch.bfloat16, enabled=self.autocast):
            _, after, _ = self.model(self.tensor(views))
        after = after.float().view(len(rows), k, partners, -1)
        now = v[index].float()[:, None, None]
        mine = after[..., 0] - now[..., 0]
        theirs = (after - now).gather(-1, torch.arange(1, partners + 1, device=at).view(1, 1, -1, 1).expand(len(rows), k, -1, 1)).squeeze(-1)
        agree = torch.sigmoid(theirs / 0.05) * self.tensor(ok).view(len(rows), k, partners).float()
        score = (agree * mine).max(-1).values
        spread = score.std(1, keepdim=True) + 1e-4
        weights = ((score - score.mean(1, keepdim=True)) / spread * 2).softmax(1)
        search.append((self.tensor(o[rows]), choices.view(len(rows), k, -1), masks.view(len(rows), k, *masks.shape[1:]), weights))
        stats["searched_offers"] += len(rows)
        stats["offer_gain"] += float(score.max(1).values.sum())
        stats["acceptable"] += float((agree > 0.5).any(-1).float().mean(1).sum())

    def answered(self, o, p, pending, responses, tally, habits):
        games = np.flatnonzero((pending[:, 0] >= 0) & (p == pending[:, 1]) & np.isin(o[:, 296], (PHASE_TURN, PHASE_CONFIRM)))
        for game in games:
            others = np.arange(1, SEATS) < self.players
            if o[game, 296] == PHASE_TURN and o[game, 297] == 0:
                pending[game] = -1
                continue
            if o[game, OFFERS_MADE] == pending[game, 2]:
                habits.dropped += 1
                pending[game] = -1
                continue
            if o[game, 296] == PHASE_CONFIRM:
                status = o[game, ANSWERS_AT:ANSWERS_AT + 33:11].astype(np.int64)
                labels = np.where(others & (status >= 2), status - 2, -1)
            else:
                seat = np.arange(1, SEATS)
                accepted = o[game, LAST_ACCEPTED] >> seat & 1
                countered = o[game, LAST_COUNTERED] >> seat & 1
                labels = np.where(others, np.where(accepted == 1, 0, np.where(countered == 1, 2, 1)), -1)
            responses[pending[game, 0], game] = self.tensor(labels)
            tally += np.bincount(labels[labels >= 0], minlength=ANSWERS)
            pending[game] = -1

    @torch.no_grad()
    def evaluate(self, obs, masks, hidden=None):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            logits, values, summary, goods = self.model(obs, masks, with_goods=True)
        if hidden is not None:
            values = self.critics.privileged(summary, hidden)
        dist, _ = policy(logits, masks)
        return dist, values.float(), summary.float(), goods.float()

    @torch.no_grad()
    def value(self, obs, actors):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            _, values, _ = self.model(obs)
        return to_absolute(values.float(), actors, self.players)

    def tensor(self, array, dtype=None):
        return torch.from_numpy(array.astype(dtype) if dtype else array).to(self.at)

    def collect(self, rounds):
        n, at = self.games, self.at
        obs = torch.zeros(rounds, n, catan.OBSERVATION_BYTES, dtype=torch.uint8, device=at)
        masks = torch.zeros(rounds, n, catan.ACTIONS, dtype=torch.bool, device=at)
        actions = torch.zeros(rounds, n, dtype=torch.long, device=at)
        log_probs = torch.zeros(rounds, n, device=at)
        values = torch.zeros(rounds, n, SEATS, device=at)
        actors = torch.zeros(rounds, n, dtype=torch.long, device=at)
        hidden = torch.zeros(rounds, n, SEATS - 1, 10, dtype=torch.uint8, device=at)
        done, ends = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, SEATS, device=at)
        progress = torch.zeros(rounds, n, SEATS, device=at)
        responses = torch.full((rounds, n, SEATS - 1), -1, dtype=torch.long, device=at)
        offered = torch.zeros(rounds, n, dtype=torch.bool, device=at)
        offer_choices = torch.full((rounds, n, deals.STEPS), -1, dtype=torch.long, device=at)
        offer_masks = torch.ones(rounds, n, deals.STEPS, deals.CHOICES, dtype=torch.bool, device=at)
        stats = {"won": 0, "truncated": 0, "first_won": 0, "trades": 0, "league": 0, "league_network_won": 0, "explored": 0,
                 "offer_cards": 0, "offers_built": 0}
        habits = Habits()
        learn = torch.ones(rounds, n, dtype=torch.bool, device=at)
        pending, tally, trades = np.full((n, 3), -1), np.zeros(ANSWERS, np.int64), []
        search = []
        stats.update(searched_offers=0, offer_gain=0.0, acceptable=0.0)
        if self.league is not None:
            for game in np.flatnonzero((self.league_seats & (self.opponent < 0)).any(1)):
                self.reseat(game)

        for t in range(rounds):
            o, m, p = self.buffer.observe()
            obs[t], masks[t], actors[t] = self.tensor(o), self.tensor(m), self.tensor(p, np.int64)
            forbidden, counts = self.buffer.forbidden()
            self.answered(o, p, pending, responses, tally, habits)
            hidden[t] = self.tensor(self.buffer.hidden())
            dist, v, summary, goods = self.evaluate(obs[t], masks[t], hidden[t] if self.privileged else None)
            a = dist.sample()
            actions[t], log_probs[t], values[t] = a, dist.log_prob(a), to_absolute(v, actors[t], self.players)
            chosen = a.cpu().numpy()
            opponent = self.opponent[np.arange(n), p]
            league_turn = self.league_seats[np.arange(n), p] & (opponent >= 0)
            offers = np.zeros((n, deals.RESOURCES), np.int8)
            for k in np.unique(opponent[league_turn]):
                rows = league_turn & (opponent == k)
                chosen[rows], offers[rows] = self.league.act(k, o[rows], m[rows], forbidden[rows], counts[rows], self.greedy[rows])
            own = ~league_turn
            mine = np.where(own, chosen, 0)
            def logits_for(rows, counts):
                at_rows = self.tensor(rows)
                return self.model.deal_scores(summary[at_rows], goods[at_rows], self.tensor(mine[rows] == COUNTER), counts)
            with torch.no_grad(), torch.autocast(at.type, dtype=torch.bfloat16, enabled=self.autocast):
                own_offers, rows, sampled = deals.choose(mine, o, forbidden, counts, logits_for, at)
            if sampled is not None:
                offers[rows] = own_offers[rows]
                at_rows = self.tensor(rows)
                offered[t, at_rows], offer_choices[t, at_rows], offer_masks[t, at_rows] = True, sampled[0], sampled[1]
                stats["offer_cards"] += int((sampled[0] > deals.STOP).sum())
                stats["offers_built"] += len(rows)
                with torch.no_grad(), torch.autocast(at.type, dtype=torch.bfloat16, enabled=self.autocast):
                    deal_lp, _ = self.model.deal_log_probs(summary[at_rows], goods[at_rows], self.tensor(mine[rows] == COUNTER),
                                                           sampled[0], sampled[1])
                log_probs[t, at_rows] += deal_lp.float()
                if self.offer_search:
                    self.search_offers(t, rows[mine[rows] == OFFER], o, forbidden, counts, summary, goods, v, search, stats)
            exploring = own & self.explore & (o[:, 296] == SETUP_SETTLEMENT) & (m.sum(1) > 1)
            if exploring.any():
                chosen[exploring] = self.explore_opening(dist.probs[self.tensor(exploring)])
                stats["explored"] += int(exploring.sum())
            others = ~own | exploring
            if others.any():
                actions[t] = self.tensor(chosen, np.int64)
                learn[t] = self.tensor(~others)
            habits.record(o, chosen, own & ~exploring, dist.probs)
            stats["trades"] += int(TRADED[chosen].sum())
            sent = np.flatnonzero(chosen == OFFER)
            pending[sent] = np.stack([np.full(len(sent), t), p[sent], o[sent, OFFERS_MADE]], 1)
            for game in np.flatnonzero(TRADED[chosen]):
                trades.append((t, game, p[game], (p[game] + (chosen[game] - TRADE_WITH) % (SEATS - 1) + 1) % self.players))
            before = self.buffer.points()
            endings = self.buffer.step(chosen, offers)
            progress[t] = self.tensor((self.buffer.points().astype(np.float32) - before))
            for game, winner, _ in endings:
                done[t, game] = True
                ends[t, game] = rewards(winner, self.players, points=before[game] if self.rank else None).to(at)
                if winner is None:
                    stats["truncated"] += 1
                else:
                    stats["won"] += 1
                    stats["first_won"] += winner == 0
                pending[game] = -1
                seated = self.opponent[game][self.opponent[game] >= 0]
                if len(seated):
                    won = winner is not None and not self.league_seats[game, winner]
                    for k in seated:
                        self.league.record(k, won)
                    stats["league"] += 1
                    stats["league_network_won"] += won
                self.reseat(game)

        o, m, p = self.buffer.observe()
        if self.privileged:
            _, last = self.evaluate(self.tensor(o), self.tensor(m), self.tensor(self.buffer.hidden()))
            last = to_absolute(last, self.tensor(p, np.int64), self.players)
        else:
            last = self.value(self.tensor(o), self.tensor(p, np.int64))
        batch = dict(obs=obs, masks=masks, actions=actions, log_probs=log_probs, actors=actors, hidden=hidden,
                     learn=learn, responses=responses, offered=offered, offer_choices=offer_choices, offer_masks=offer_masks)
        if search:
            batch["search"] = tuple(torch.cat(part) for part in zip(*search))
        stats["answers"] = tally
        stats["trade_advantage"] = trade_advantage(trades, values, done)
        shaped = self.vp_reward * relative(progress, self.players) * (~done)[..., None]
        stats["habits"], stats["habit_values"] = habits.summary(), habits.values()
        return batch, values, (done, ends, last, shaped), stats

def sync(at):
    if at.type == "cuda":
        torch.cuda.synchronize()
    elif at.type == "mps":
        torch.mps.synchronize()

def trade_advantage(trades, values, done):
    gains = [(values[t + 1, g, a] - values[t, g, a] - values[t + 1, g, b] + values[t, g, b]).item()
             for t, g, a, b in trades if t + 1 < len(values) and not done[t, g]]
    return (float(np.mean(gains)), len(gains)) if gains else (None, 0)

def relative(gains, players):
    gains = gains[..., :players]
    others = (gains.sum(-1, keepdim=True) - gains) / (players - 1)
    out = torch.zeros(*gains.shape[:-1], SEATS, device=gains.device)
    out[..., :players] = gains - others
    return out

def advantages(values, done, ends, last, shaped, lam):
    result = torch.zeros_like(values)
    following = torch.zeros_like(last)
    for t in reversed(range(values.shape[0])):
        next_values = last if t == values.shape[0] - 1 else values[t + 1]
        delta = torch.where(done[t, :, None], ends[t] - values[t], shaped[t] + next_values - values[t])
        following = torch.where(done[t, :, None], delta, delta + lam * following)
        result[t] = following
    return result

def upgo_advantages(values, done, ends, last, shaped):
    steps = values.shape[0]
    following = torch.zeros_like(values)
    for t in reversed(range(steps)):
        next_values = last if t == steps - 1 else values[t + 1]
        if t == steps - 1:
            carry = next_values
        else:
            after = last if t + 1 == steps - 1 else values[t + 2]
            q = torch.where(done[t + 1, :, None], ends[t + 1], shaped[t + 1] + after)
            carry = torch.where(q >= values[t + 1], following[t + 1], values[t + 1])
        following[t] = torch.where(done[t, :, None], ends[t], shaped[t] + carry)
    return following - values

def entropies(log_probs):
    probs = log_probs.exp()
    total = -(probs * log_probs).sum(-1)
    kinds = probs @ KIND_MEMBERS.to(probs.device)
    kind = -(kinds * kinds.clamp(min=1e-12).log()).sum(-1)
    return kind, total - kind

def update(model, critics, optimizer, batch, adv, returns, players, args, autocast, entropy_weight=0.0, upgo=None):
    search = batch.get("search")
    flat = {k: v.flatten(0, 1) for k, v in batch.items() if k != "search"}
    actors = flat["actors"]
    own = adv.flatten(0, 1).gather(1, actors[:, None]).squeeze(1)
    own_upgo = None if upgo is None else upgo.flatten(0, 1).gather(1, actors[:, None]).squeeze(1)
    targets = to_relative(returns.flatten(0, 1), actors, players)
    total = len(own)
    seats = slice(0, players)
    threshold = own.abs().float().quantile(1 - args.keep) if args.keep < 1 else -1.0
    informative = own.abs() >= threshold
    logs = {"upgo": 0.0, "offer_search": 0.0, "policy": 0.0, "value": 0.0, "approx_kl": 0.0, "clipped": 0.0, "hands": 0.0, "answers": 0.0,
            "answers_right": 0.0, "answers_seen": 0.0, "entropy_kind": 0.0, "entropy_target": 0.0,
            "kept": informative.float().mean().item()}
    steps = 0
    for _ in range(args.epochs):
        for pick in torch.randperm(total, device=own.device).split(args.minibatch):
            obs, masks = flat["obs"][pick], flat["masks"][pick]
            symmetry = torch.randint(SYMMETRIES, (len(pick),), device=obs.device) if args.symmetry else None
            with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                logits, values, summary, goods = model(obs, masks, symmetry, with_goods=True)
                offered = flat["offered"][pick]
                rows = torch.nonzero(offered).squeeze(1)
                deal_lp, deal_entropy = torch.zeros(len(pick), device=obs.device), torch.zeros(len(pick), device=obs.device)
                if len(rows):
                    lp, h = model.deal_log_probs(summary[rows], goods[rows], flat["actions"][pick][rows] == COUNTER,
                                                 flat["offer_choices"][pick][rows], flat["offer_masks"][pick][rows])
                    deal_lp, deal_entropy = deal_lp.index_put((rows,), lp.float()), deal_entropy.index_put((rows,), h.float())
            dist, log_probs = policy(logits, masks)
            new = dist.log_prob(flat["actions"][pick]) + deal_lp
            ratio = (new - flat["log_probs"][pick]).exp()
            target = targets[pick]
            value_loss = nn.functional.mse_loss(values.float()[:, seats], target[:, seats])
            a = own[pick]
            if args.privileged:
                value_loss = value_loss + nn.functional.mse_loss(critics.privileged(summary, flat["hidden"][pick])[:, seats], target[:, seats])
            kept = informative[pick] & flat["learn"][pick]
            ratio, a = ratio[kept], a[kept]
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            if own_upgo is not None:
                u = own_upgo[pick][kept]
                u = (u - u.mean()) / (u.std() + 1e-8)
                upgo_loss = -torch.min(ratio * u, ratio.clamp(1 - args.clip, 1 + args.clip) * u).mean()
                policy_loss = policy_loss + args.upgo * upgo_loss
                logs["upgo"] += upgo_loss.item()
            kind_h, target_h = entropies(log_probs)
            target_h = target_h + deal_entropy
            entropy_kind, entropy_target = (h[flat["learn"][pick]].mean() for h in (kind_h, target_h))
            loss = policy_loss + args.value_coef * value_loss
            loss = loss - entropy_weight * (entropy_kind + args.target_entropy_share * entropy_target)
            if args.aux_hands:
                hands_loss = nn.functional.mse_loss(critics.guess_hands(summary), flat["hidden"][pick].float() / 5)
                loss = loss + args.aux_hands * hands_loss
                logs["hands"] += hands_loss.item()
            labels = flat["responses"][pick]
            if args.aux_answers and (labels >= 0).any():
                guesses = critics.guess_answers(summary)
                answers_loss = nn.functional.cross_entropy(guesses.flatten(0, 1), labels.flatten(), ignore_index=-1)
                loss = loss + args.aux_answers * answers_loss
                seen = labels >= 0
                logs["answers"] += answers_loss.item()
                logs["answers_right"] += (guesses.argmax(-1) == labels)[seen].sum().item()
                logs["answers_seen"] += seen.sum().item()
            if search is not None:
                s_obs, s_choices, s_masks, s_weights = search
                take = torch.randint(len(s_obs), (min(len(s_obs), max(len(pick) // 16, 1)),), device=s_obs.device)
                k = s_choices.shape[1]
                with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                    _, _, s_summary, s_goods = model(s_obs[take], with_goods=True)
                    lp, _ = model.deal_log_probs(s_summary.repeat_interleave(k, 0), s_goods.repeat_interleave(k, 0),
                                                 torch.zeros(len(take) * k, dtype=torch.bool, device=obs.device),
                                                 s_choices[take].flatten(0, 1), s_masks[take].flatten(0, 1))
                search_loss = -(s_weights[take] * lp.float().view(len(take), k)).sum(1).mean()
                loss = loss + args.offer_search_weight * search_loss
                logs["offer_search"] += search_loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *critics.parameters()], args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                logs["policy"] += policy_loss.item()
                logs["value"] += value_loss.item()
                logs["entropy_kind"] += entropy_kind.item()
                logs["entropy_target"] += entropy_target.item()
                logs["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                logs["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item()
            steps += 1
    kept, right, seen = logs.pop("kept"), logs.pop("answers_right"), logs.pop("answers_seen")
    return {**{k: v / steps for k, v in logs.items()}, "kept": kept, "answers_accuracy": right / seen if seen else None}

def code_version():
    here = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "uncommitted_changes": dirty}
    except (OSError, subprocess.CalledProcessError):
        stamp = here / "COMMIT"
        return {"commit": stamp.read_text().strip() if stamp.exists() else "unknown"}

def main():
    parser = argparse.ArgumentParser(description="Self-play PPO with a win/loss reward.")
    parser.add_argument("--games", type=int, default=1024, help="games played side by side")
    parser.add_argument("--players", type=int, default=4, help="players per game, 2 to 4")
    parser.add_argument("--rounds", type=int, default=64, help="decisions per game between updates")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=4096)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--entropy", type=float, default=0.01,
                        help="entropy bonus, annealed down to --entropy-floor (OpenAI Five: 0.01 to 0.001)")
    parser.add_argument("--entropy-floor", type=float, default=0.001)
    parser.add_argument("--target-entropy-share", type=float, default=1.0,
                        help="weight of the targets' entropy relative to the kinds'")
    parser.add_argument("--rules", choices=list(RULE_SETS), default="standard",
                        help="colonist-1v1: 15 points, discard over 9; colonist-4p: 10 points, discard over 7; both with the friendly robber and balanced dice")
    parser.add_argument("--no-trading", action="store_true", help="no trades between players, only with the bank")
    parser.add_argument("--turn-budget", type=int, default=TURN_BUDGET, help="decisions a turn may take after the roll: a turn timer")
    parser.add_argument("--anneal-power", type=float, default=0.0,
                        help="shrink the learning rate and entropy bonus as (1 + iteration / anneal-steps) ** -power; 0 is off")
    parser.add_argument("--anneal-steps", type=float, default=20.0)
    parser.add_argument("--entropy-anneal-steps", type=float, default=None,
                        help="shrink the entropy bonus on its own schedule; defaults to --anneal-steps")
    parser.add_argument("--keep", type=float, default=1.0,
                        help="train the policy only on this fraction of moves, those with the largest advantage magnitude")
    parser.add_argument("--symmetry", action="store_true", help="turn each training sample by a random board symmetry")
    parser.add_argument("--privileged", action="store_true",
                        help="score moves with a critic that also sees every hidden card (training only)")
    parser.add_argument("--league", type=float, default=0.0,
                        help="share of training games with earlier versions of the network in half the seats; 0 is off")
    parser.add_argument("--league-size", type=int, default=8, help="earlier versions kept in the pool")
    parser.add_argument("--league-every", type=int, default=250, help="add the current network to the pool every this many iterations")
    parser.add_argument("--league-from", nargs="*", default=[], help="saved networks to seed the pool with, e.g. other runs' checkpoints")
    parser.add_argument("--league-watch", help="a folder checked at every checkup: any new .pt in it joins the pool, e.g. an exploiter")
    parser.add_argument("--publish", help="a population: write this network here every --publish-every iterations, for the "
                                          "others to play against (they watch the folder with --league-watch)")
    parser.add_argument("--publish-every", type=int, default=25,
                        help="how often to publish, and to pick up the other networks' newest versions from --league-watch")
    parser.add_argument("--reward", choices=["win", "rank"], default="win",
                        help="win: +1 to the winner, the same loss for everyone else; rank: losers are paid by finishing place")
    parser.add_argument("--greedy-opponents", type=float, default=0.0,
                        help="share of games whose league opponents play their top move instead of sampling, so a "
                             "question asked twice gets the same answer")
    parser.add_argument("--opening-explore", type=float, default=0.0,
                        help="share of games whose setup settlements are drawn from the network's eight favourites; "
                             "those moves are not learned from")
    parser.add_argument("--aux-hands", type=float, default=0.0,
                        help="weight of learning to guess every opponent's hidden cards (training only); 0 is off")
    parser.add_argument("--upgo", type=float, default=0.0,
                        help="weight of a second policy loss on UPGO advantages (AlphaStar): returns that keep following moves which did better than expected")
    parser.add_argument("--offer-search", type=int, default=0,
                        help="offer decisions per step whose candidate offers are scored with the network's own values; 0 is off")
    parser.add_argument("--offer-candidates", type=int, default=8, help="offers sampled and scored per searched decision")
    parser.add_argument("--offer-search-weight", type=float, default=1.0)
    parser.add_argument("--random-starts", type=int, default=0,
                        help="percent of training games where every player starts with up to 3 extra cards and maybe a development card")
    parser.add_argument("--aux-answers", type=float, default=0.0,
                        help="weight of learning to predict how each opponent answers an offer (training only); 0 is off")
    parser.add_argument("--wandb", metavar="PROJECT", help="also log every metric to this Weights & Biases project")
    parser.add_argument("--wandb-group", help="W&B group, e.g. one population, so its networks chart together")
    parser.add_argument("--exploit", help="train an exploiter: every game puts this frozen network in half the seats, and "
                                          "nothing else ever joins the pool, so learning goes only into beating it")
    parser.add_argument("--save-every", type=int, default=250, help="keep a numbered checkpoint every this many iterations")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="overwrite latest.pt every this many iterations, so a restart resumes from there whatever --eval-every is")
    parser.add_argument("--compile", action="store_true", help="compile the network for the learning step (CUDA; see train.bench)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=3000, help="training games are cut off (a loss for all) after this many")
    parser.add_argument("--vp-fade", type=float, default=0,
                        help="fade the victory-point reward linearly to nothing over this many iterations, so later only winning counts; 0 keeps it")
    parser.add_argument("--vp-reward", type=float, default=0.03,
                        help="reward each victory point gained relative to the other players at once; 0 rewards winning alone")
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=512)
    parser.add_argument("--anchor", help="also play this fixed network at every checkup, to compare runs that start from it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/selfplay")
    args = parser.parse_args()
    if args.exploit:
        if args.league or args.league_from or args.league_watch:
            parser.error("--exploit sets its own pool; drop --league, --league-from and --league-watch")
        args.league, args.league_from = 1.0, [args.exploit]
    if args.league > 1:
        parser.error("--league is a share of games, at most 1")
    if args.league_watch and not args.league:
        parser.error("--league-watch adds to the pool; add --league")
    if args.greedy_opponents and not args.league:
        parser.error("--greedy-opponents applies to league opponents; add --league")

    torch.manual_seed(args.seed)
    at = device()
    autocast = at.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    rules = rules_of(args)
    model = Model(args.width, args.depth).to(at)
    critics = Critics(args.width).to(at)
    optimizer = torch.optim.AdamW([*model.parameters(), *critics.parameters()], lr=args.lr)
    buffer = catan.Buffer(args.games, history=64, max_decisions=args.max_decisions, seed=args.seed, players=args.players,
                          rules=rules, random_starts=args.random_starts)
    league = League(args.league_size, at, autocast, args.seed) if args.league else None
    rollout = Rollout(model, critics, buffer, at, autocast, args.players, args.privileged, args.seed,
                      args.vp_reward, league, args.league, args.reward == "rank", args.greedy_opponents, args.opening_explore,
                      args.offer_search, args.offer_candidates)
    log = open(out / "log.txt", "a")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {at}; "
        f"{args.games} games × {args.rounds} decisions = {args.games * args.rounds} per update")
    tracker = None
    if args.wandb:
        import wandb
        name = out.resolve().name
        tracker = wandb.init(project=args.wandb, group=args.wandb_group, name=name, config={**vars(args), **code_version()},
                             dir=str(out),
                             id=f"{name}-{zlib.crc32(str(out.resolve()).encode()):08x}", resume="allow")

    anchor = load(args.anchor, at) if args.anchor else None
    decisions, first, previous = 0, 1, None
    if (out / "latest.pt").exists():
        saved = torch.load(out / "latest.pt", map_location=at)
        model.load_state_dict(saved["model"])
        critics.load_state_dict(saved["critics"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        first = saved["iteration"] + 1
        decisions = saved["iteration"] * args.games * args.rounds
        previous = copy.deepcopy(model).eval()
        say(f"resuming after iteration {saved['iteration']}")
    watched = set()
    published = Path(args.publish).resolve() if args.publish else None

    def watch(quiet=False):
        for path in sorted(Path(args.league_watch).glob("*.pt")):
            if path.resolve() == published:
                continue
            key = (path.name, path.stat().st_mtime)
            if key in watched:
                continue
            try:
                member = load(path, at)
            except Exception as error:
                say(f"league: could not load {path} yet ({type(error).__name__}); will retry")
                continue
            watched.add(key)
            league.add(member, path.stem, pinned=True)
            if not quiet:
                say(f"league: added {path}")

    def publish(iteration):
        published.parent.mkdir(parents=True, exist_ok=True)
        part = published.with_suffix(".part")
        torch.save({"model": model.state_dict(), "args": vars(args), "iteration": iteration}, part)
        part.replace(published)

    if league is not None:
        for path in args.league_from:
            league.add(load(path, at), path)
        if not args.exploit:
            saved = sorted(out.glob("iter_*.pt"))
            room = max(args.league_size - len(league.members) - 1, 0)
            if room and saved:
                for i in np.unique(np.linspace(0, len(saved) - 1, min(room, len(saved))).round().astype(int)):
                    league.add(load(saved[i], at), saved[i].stem)
            league.add(copy.deepcopy(model).eval(), f"start {first - 1}")
        if args.league_watch:
            watch()
        say(f"league: {', '.join(m['name'] for m in league.members)}")
    learner = torch.compile(model) if args.compile else model
    start = time.time()
    for iteration in range(first, args.iterations + 1):
        model.eval()
        clock = time.time()
        rollout.vp_reward = args.vp_reward * max(0.0, 1 - (iteration - 1) / args.vp_fade) if args.vp_fade else args.vp_reward
        batch, values, scoring, stats = rollout.collect(args.rounds)
        sync(at)
        collect_time, clock = time.time() - clock, time.time()
        adv = advantages(values, *scoring, args.lam)
        returns = adv + values
        scale = (1 + (iteration - 1) / args.anneal_steps) ** -args.anneal_power
        pull = (1 + (iteration - 1) / (args.entropy_anneal_steps or args.anneal_steps)) ** -args.anneal_power
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        model.train()
        entropy_weight = max(args.entropy * pull, args.entropy_floor)
        upgo = upgo_advantages(values, *scoring) if args.upgo else None
        logs = update(learner, critics, optimizer, batch, adv, returns, args.players, args, autocast, entropy_weight, upgo)
        sync(at)
        update_time = time.time() - clock
        decisions += args.games * args.rounds

        finished = stats["won"] + stats["truncated"]
        line = (f"iter {iteration:4d}  {decisions / 1e6:6.2f}M decisions  {time.time() - start:6.0f}s  "
                f"games {finished:5d} ({stats['truncated']} cut)  first seat wins {stats['first_won'] / max(stats['won'], 1):.2f}  "
                f"trades per 1k decisions {1000 * stats['trades'] / (args.games * args.rounds):.1f}  "
                f"value {logs['value']:.3f}  approx_kl {logs['approx_kl']:.4f}  "
                f"clipped {logs['clipped']:.2f}  kept {logs['kept']:.2f}  lr {args.lr * scale:.1e}  "
                f"play {collect_time:.1f}s learn {update_time:.1f}s  {stats['habits']}")
        metrics = {
            "decisions": decisions, "hours": (time.time() - start) / 3600, "games/finished": finished,
            "games/cut": stats["truncated"], "games/first_seat_wins": stats["first_won"] / max(stats["won"], 1),
            "games/trades_per_1k": 1000 * stats["trades"] / (args.games * args.rounds),
            "loss/policy": logs["policy"], "loss/value": logs["value"],
            "ppo/approx_kl": logs["approx_kl"], "ppo/clipped": logs["clipped"], "ppo/lr": args.lr * scale,
            "ppo/entropy_kind": logs["entropy_kind"], "ppo/entropy_target": logs["entropy_target"],
            "ppo/entropy_weight": entropy_weight,
            "time/play": collect_time, "time/learn": update_time,
            **{f"habits/{k}": v for k, v in stats["habit_values"].items()},
        }
        if args.aux_hands:
            line += f"  hands guess error {logs['hands']:.4f}"
            metrics["loss/hands"] = logs["hands"]
        if stats["explored"]:
            line += f"  explored openings {stats['explored']}"
            metrics["games/explored_openings"] = stats["explored"]
        answered = stats["answers"].sum()
        if answered:
            share = stats["answers"] / answered
            line += f"  answers accept {share[0]:.2f} decline {share[1]:.2f} counter {share[2]:.2f}"
            metrics.update({"trade/answers": int(answered), "trade/accepted": share[0], "trade/declined": share[1],
                            "trade/countered": share[2]})
        if stats["offers_built"]:
            cards = stats["offer_cards"] / stats["offers_built"]
            line += f"  cards per offer {cards:.1f}"
            metrics["trade/cards_per_offer"] = cards
        if stats["searched_offers"]:
            m = stats["searched_offers"]
            line += f"  offer search: best gain {stats['offer_gain'] / m:+.4f}, acceptable {stats['acceptable'] / m:.2f}"
            metrics.update({"search/best_gain": stats["offer_gain"] / m, "search/acceptable": stats["acceptable"] / m,
                            "loss/offer_search": logs["offer_search"]})
        if args.upgo:
            metrics["loss/upgo"] = logs["upgo"]
        advantage, counted = stats["trade_advantage"]
        if counted:
            line += f"  trade advantage {advantage:+.3f}"
            metrics.update({"trade/advantage": advantage, "trade/scored": counted})
        if args.aux_answers and logs["answers_accuracy"] is not None:
            line += f"  answer guess right {logs['answers_accuracy']:.2f}"
            metrics.update({"loss/answers": logs["answers"], "trade/answer_guess_accuracy": logs["answers_accuracy"]})
        if league is not None:
            line += f"  network seats win vs league {stats['league_network_won'] / max(stats['league'], 1):.2f} (fair 0.50)"
            metrics["train/vs_league"] = stats["league_network_won"] / max(stats["league"], 1)
            if iteration % args.league_every == 0 and not args.exploit:
                league.add(copy.deepcopy(model).eval(), f"iter {iteration}")
        if iteration % args.publish_every == 0:
            if published is not None:
                publish(iteration)
            if args.league_watch:
                watch(quiet=True)
        last = iteration == args.iterations
        if iteration % args.checkpoint_every == 0 or iteration % args.save_every == 0 or last:
            checkpoint = {"model": model.state_dict(), "critics": critics.state_dict(), "optimizer": optimizer.state_dict(),
                          "args": vars(args), "iteration": iteration}
            torch.save(checkpoint, out / "latest.pt")
            if iteration % args.save_every == 0 or last:
                torch.save({k: v for k, v in checkpoint.items() if k != "optimizer"}, out / f"iter_{iteration:05d}.pt")
        if iteration % args.eval_every == 0 or last:
            model.eval()
            clock = time.time()
            player = Network(model, at, seed=iteration)
            opponents = {"random": Random(iteration)} if previous is None else {}
            if anchor is not None:
                opponents["anchor"] = Network(anchor, at, seed=-2 * iteration)
            if previous is not None:
                opponents["previous"] = Network(previous, at, seed=-iteration)
            for name, opponent in opponents.items():
                result = play(player, opponent, games=args.eval_games, parallel=min(args.eval_games, 512), rules=rules,
                              max_decisions=args.max_decisions,
                              seed=10_000 + iteration, players=args.players)
                line += f"  vs {name} {result['win_rate']:.3f}"
                metrics[f"eval/vs_{name}"] = result["win_rate"]
            previous = copy.deepcopy(model).eval()
            if league is not None:
                line += f"  league: {league.summary()}"
                metrics["league/size"] = len(league.members)
            line += f"  (eval {time.time() - clock:.0f}s)"
            metrics["time/eval"] = time.time() - clock
        say(line)
        if tracker is not None:
            tracker.log(metrics, step=iteration)

if __name__ == "__main__":
    main()
