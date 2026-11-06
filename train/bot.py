import numpy as np
import torch

from .features import EDGE_ENDS, RATES_AT, SEATS, TRADE_AT, decode

# A hand-written opponent, for measuring only: it is never learned from. It builds the most valuable thing it can
# afford, trades with the bank only to afford something, accepts an offer only if it lets it build, and robs the leader.

SETTLEMENT, CITY, ROAD, ROBBER, RESOURCE, BANK = 0, 54, 108, 180, 199, 204
ROLL, END_TURN, BUY_CARD, PLAY_KNIGHT, PLAY_ROAD_BUILDING, PLAY_YEAR_OF_PLENTY, PLAY_MONOPOLY = range(224, 231)
ROB, ACCEPT, DECLINE = 231, 247, 248
PHASE_DISCARD, PHASE_MONOPOLY = 4, 10
PAIRS = [(give, get) for give in range(5) for get in range(5) if get != give]

COSTS = [  # brick, lumber, wool, grain, ore
    (CITY, np.array([0, 0, 0, 2, 3])),
    (SETTLEMENT, np.array([1, 1, 1, 1, 0])),
    (BUY_CARD, np.array([0, 0, 1, 1, 1])),
    (ROAD, np.array([1, 1, 0, 0, 0])),
]

def affordable(hand):
    return next((a for a, cost in COSTS[:3] if (hand >= cost).all()), None)

class Bot:
    def __call__(self, obs, masks, forbidden=None, counts=None):
        hexes, vertices, _, _ = (x.numpy() for x in decode(torch.from_numpy(obs)))
        actions = np.array([self.choose(o, m, h, v) for o, m, h, v in zip(obs, masks, hexes, vertices)])
        return actions, np.zeros((len(obs), 5), np.int8)

    def choose(self, obs, legal, hexes, vertices):
        hand = obs[222:227].astype(int)
        phase = obs[296]
        pips = vertices[:, 16:21] * 5
        value = pips.sum(1) + 0.5 * (pips > 0).sum(1)
        best = lambda options, score: max(options, key=score)
        among = lambda start, count: [i for i in range(start, start + count) if legal[i]]
        players = int(obs[221])

        if legal[ACCEPT] or legal[DECLINE]:
            after = hand - obs[TRADE_AT + 5:TRADE_AT + 10] + obs[TRADE_AT:TRADE_AT + 5]
            return ACCEPT if legal[ACCEPT] and affordable(hand) is None and affordable(after) is not None else DECLINE

        resources = among(RESOURCE, 5)
        if resources:
            if phase == PHASE_DISCARD:
                return best(resources, lambda i: hand[i - RESOURCE])
            if phase == PHASE_MONOPOLY:
                held = lambda r: sum(int(obs[232 + 7 * (s - 1) + r]) for s in range(1, players))
                return best(resources, lambda i: held(i - RESOURCE))
            return best(resources, lambda i: -hand[i - RESOURCE])

        victims = among(ROB, SEATS - 1)
        if victims:
            return best(victims, lambda i: obs[288 + i - ROB + 1])

        robber = among(ROBBER, 19)
        if robber:
            weight = lambda h, s: hexes[h, 21 + s] + 2 * hexes[h, 21 + SEATS + s]
            others = lambda h: sum(weight(h, s) for s in range(1, SEATS))
            return best(robber, lambda i: hexes[i - ROBBER, 19] * 5 * (others(i - ROBBER) - 3 * weight(i - ROBBER, 0)))

        for start, count in ((CITY, 54), (SETTLEMENT, 54)):
            spots = among(start, count)
            if spots:
                return best(spots, lambda i: value[i - start])

        if legal[PLAY_KNIGHT] and legal[ROLL]:
            return PLAY_KNIGHT
        if legal[ROLL]:
            return ROLL

        for action in (PLAY_YEAR_OF_PLENTY, PLAY_MONOPOLY, PLAY_ROAD_BUILDING):
            if legal[action]:
                return action

        if affordable(hand) is None:
            for k, (give, get) in sorted(enumerate(PAIRS), key=lambda p: obs[RATES_AT + p[1][0]]):
                if legal[BANK + k]:
                    after = hand.copy()
                    after[give] -= obs[RATES_AT + give]
                    after[get] += 1
                    if affordable(after) is not None:
                        return BANK + k

        if legal[BUY_CARD]:
            return BUY_CARD
        roads = among(ROAD, 72)
        if roads:
            free = lambda v: vertices[v, :2 * SEATS].sum() == 0 and vertices[v, 21:21 + 2 * SEATS].sum() == 0
            reach = lambda e: max(value[v] * free(v) for v in EDGE_ENDS[e])
            return best(roads, lambda i: reach(i - ROAD))
        if legal[END_TURN]:
            return END_TURN
        return int(np.flatnonzero(legal)[0])
