import collections

import numpy as np
import torch

import catan

from .agents import Network

RESOURCES = ["brick", "lumber", "wool", "grain", "ore"]
SETTLEMENT, CITY, ROAD, BANK, END_TURN, BUY_CARD, PLAY_KNIGHT = 0, 54, 108, 204, 265, 266, 267
DEAL, TRADE_WITH, CANCEL = 274, 506, 512
PHASE_TURN, PHASE_RESPOND = 2, 13


def chance(values):
    """The value head predicts +1 for a win and -1/3 for a loss; this is the implied chance of winning."""
    return ((3 * values + 1) / 4).clamp(0, 1)


@torch.no_grad()
def card_values(model, at, obs):
    """How much one more card of each resource would change the player's own chance of winning, in percentage points."""
    variants = np.repeat(obs[None], 6, axis=0)
    for r in range(5):
        variants[r + 1, 222 + r] += 1
        variants[r + 1, 253 + r] = max(int(variants[r + 1, 253 + r]) - 1, 0)
    _, values, _ = model(torch.from_numpy(variants).to(at))
    mine = chance(values[:, 0].float().cpu())
    return (100 * (mine[1:] - mine[0])).tolist()


def replay(record):
    players, seed, codes = record.split(";")
    game = catan.Game(int(seed), int(players))
    for code in codes.split(","):
        if code.strip():
            game.step(int(code))
    return game


@torch.no_grad()
def forecast(model, at, record, samples=64, turns=2, limit=None, seed=0):
    """What the player to act tends to do over their next `turns` turns, from `samples` imagined futures in which the
    hidden cards are re-dealt from their own view and every player is played by the network. Returns a summary in words
    and the plan on the board: for each spot, the share of futures in which they build a settlement, city or road there."""
    limit = limit or 150 * turns
    game = replay(record)
    me = game.to_act()
    worlds = game.imagine(samples, seed)
    agent = Network(model, at, seed=seed)
    ended = np.zeros(samples, int)
    won = np.zeros(samples, bool)
    did = [collections.Counter() for _ in range(samples)]
    spots = {"settlement": collections.Counter(), "city": collections.Counter()}
    built = [{"settlement": set(), "city": set(), "road": set()} for _ in range(samples)]
    for _ in range(limit):
        obs, masks, actors = worlds.observe()
        watching = (actors == me) & (ended < turns)
        live = actors >= 0
        if not (live & (ended < turns)).any():
            break
        moves = np.zeros(len(actors), np.int64)
        if live.any():
            moves[live] = agent(obs[live], masks[live])
        for w in np.flatnonzero(watching):
            code, phase = int(moves[w]), int(obs[w, 296])
            if code < CITY:
                did[w]["settlement"] += 1
                spots["settlement"][f"v{code}"] += 1
                built[w]["settlement"].add(code)
            elif code < ROAD:
                did[w]["city"] += 1
                spots["city"][f"v{code - CITY}"] += 1
                built[w]["city"].add(code - CITY)
            elif code < ROAD + 72:
                did[w]["road"] += 1
                built[w]["road"].add(code - ROAD)
            elif BANK <= code < END_TURN - 1:
                did[w]["bank or harbor trade"] += 1
            elif code == BUY_CARD:
                did[w]["development card"] += 1
            elif code == PLAY_KNIGHT:
                did[w]["knight"] += 1
            elif DEAL <= code < DEAL + catan.DEALS and phase == PHASE_TURN:
                did[w]["trade offer"] += 1
            elif TRADE_WITH <= code < CANCEL:
                did[w]["trade with a player"] += 1
            elif code == END_TURN:
                ended[w] += 1
        worlds.step(moves)
        winners = worlds.winners()
        won |= (winners == me) & (ended < turns)
    lines = [f"over its next {turns} turns, in {samples} imagined futures:"]
    for name in ("city", "settlement", "road", "development card", "knight", "bank or harbor trade", "trade offer",
                 "trade with a player"):
        share = np.mean([d[name] > 0 for d in did])
        if share > 0:
            detail = ""
            if name in spots and spots[name]:
                where, count = spots[name].most_common(1)[0]
                detail = f", most often at {where} ({count / samples:.0%})"
            lines.append(f"{share:.0%} {name}{detail}")
    if won.any():
        lines.append(f"{won.mean():.0%} wins")
    plan = {}
    for kind in ("settlement", "city", "road"):
        counts = collections.Counter(spot for b in built for spot in b[kind])
        plan[kind] = {spot: n / samples for spot, n in counts.items()}
    return lines, plan
