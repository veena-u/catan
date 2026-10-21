import numpy as np
import torch

from .features import EDGE_ENDS, HEX_CORNERS, decode

SETTLEMENT, CITY, ROAD, ROBBER, RESOURCE, TRADE = 0, 54, 108, 180, 199, 204
ROLL, END_TURN, BUY_CARD, PLAY_KNIGHT, PLAY_ROAD_BUILDING, PLAY_YEAR_OF_PLENTY, PLAY_MONOPOLY = range(264, 271)
PHASE_DISCARD, PHASE_YEAR_OF_PLENTY, PHASE_MONOPOLY = 4, 9, 10

COSTS = [  # brick, lumber, wool, grain, ore
    (CITY, np.array([0, 0, 0, 2, 3])),
    (SETTLEMENT, np.array([1, 1, 1, 1, 0])),
    (BUY_CARD, np.array([0, 0, 1, 1, 1])),
    (ROAD, np.array([1, 1, 0, 0, 0])),
]
TRADES = [(TRADE + 20 * (rate - 2) + 4 * give + get, give, get + (get >= give), rate)
          for rate in (2, 3, 4) for give in range(5) for get in range(4)]


class Bot:
    """Builds the most valuable thing it can afford, trades only to afford something, robs the opponent's best hex."""

    def __call__(self, obs, masks):
        hexes, vertices, _, _ = (x.numpy() for x in decode(torch.from_numpy(obs)))
        return np.array([self.choose(o, m, h, v) for o, m, h, v in zip(obs, masks, hexes, vertices)])

    def choose(self, obs, legal, hexes, vertices):
        hand = obs[157:162].astype(int)
        phase = obs[205]
        pips = vertices[:, 12:17] * 5
        value = pips.sum(1) + 0.5 * (pips > 0).sum(1)
        best = lambda options, score: max(options, key=score)
        among = lambda start, count: [i for i in range(start, start + count) if legal[i]]

        resources = among(RESOURCE, 5)
        if resources:
            if phase == PHASE_DISCARD:
                return best(resources, lambda i: hand[i - RESOURCE])
            if phase == PHASE_MONOPOLY:
                return best(resources, lambda i: obs[162 + i - RESOURCE])
            return best(resources, lambda i: -hand[i - RESOURCE])

        robber = among(ROBBER, 19)
        if robber:
            weight = lambda h, me: hexes[h, 21 + me] + 2 * hexes[h, 23 + me]
            return best(robber, lambda i: hexes[i - ROBBER, 19] * 5 * (weight(i - ROBBER, 1) - 3 * weight(i - ROBBER, 0)))

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

        affordable = lambda h: next((a for a, cost in COSTS[:3] if (h >= cost).all()), None)
        if affordable(hand) is None:
            for action, give, get, rate in TRADES:
                if legal[action]:
                    after = hand.copy()
                    after[give] -= rate
                    after[get] += 1
                    if affordable(after) is not None:
                        return action

        if legal[BUY_CARD]:
            return BUY_CARD
        roads = among(ROAD, 72)
        if roads:
            reach = lambda e: max(value[v] * (vertices[v, :4].sum() == 0) * (vertices[v, 17:21].sum() == 0)
                                  for v in EDGE_ENDS[e])
            return best(roads, lambda i: reach(i - ROAD))
        if legal[END_TURN]:
            return END_TURN
        return int(np.flatnonzero(legal)[0])
