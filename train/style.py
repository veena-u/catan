import argparse
import json
from pathlib import Path

import numpy as np

import catan

from .agents import Network
from .evaluate import rules_named
from .model import device, load

# How a network plays, in the terms VINCO's write-up uses for its own play and for top human players, so the three can
# be put side by side: how games are won, whether the first build is a city, how late the first new settlement comes,
# how often ports are used, and how many development cards and roads are played. It reads self-play, games against
# another network (--opponent), or games logged by the viewer against a person (--log).

KNIGHT, MONOPOLY = 227, 230
SETTLEMENTS, CITIES, ROADS, BANK = range(0, 54), range(54, 108), range(108, 180), range(204, 224)
GIVES = [give for give in range(5) for get in range(5) if get != give]
PHASE, SETUP_SETTLEMENT, SETUP_ROAD, DICE_PHASE, RATES, LONGEST, LARGEST = 296, 0, 1, 3, 360, 292, 293
REFERENCE = {
    # (VINCO, top human players) from the write-up; None where it gives no number.
    "wins by largest army": (0.63, None), "wins by longest road": (0.33, None), "wins by both": (0.20, None),
    "saves for a city first": (0.40, 0.18), "uses a port": (0.50, 0.70), "uses a 2:1 port": (0.19, 0.47),
}

class Seat:
    def __init__(self):
        self.first_build, self.first_settlement = None, None
        self.port, self.special_port, self.cards_played, self.roads = False, False, 0, 0

    def note(self, o, action, turn):
        if action in CITIES and self.first_build is None:
            self.first_build = "city"
        elif action in SETTLEMENTS and o[PHASE] != SETUP_SETTLEMENT:
            self.first_build = self.first_build or "settlement"
            if self.first_settlement is None:
                self.first_settlement = turn
        elif action in ROADS and o[PHASE] != SETUP_ROAD:
            self.roads += 1
        elif action in BANK:
            rate = o[RATES + GIVES[action - BANK.start]]
            self.port |= rate < 4
            self.special_port |= rate == 2
        elif KNIGHT <= action <= MONOPOLY:
            self.cards_played += 1

def holders(o, observer, players=2):
    return [(observer + o[k] - 1) % players if o[k] else -1 for k in (LONGEST, LARGEST)]

def play(agent, games, seed, rules, max_decisions, opponent=None):
    """Finished games as (seats, winner, longest road holder, largest army holder, turns, the agent's seat or None
    when it played both)."""
    parallel = min(games, 512) // 2 * 2
    buffer = catan.Buffer(parallel, max_decisions=max_decisions, seed=seed, players=2, rules=rules)
    mine = np.arange(parallel) % 2
    seats = [[Seat(), Seat()] for _ in range(parallel)]
    titles = np.full((parallel, 2), -1)
    done, finished = [], 0
    while finished < games:
        # Rolls the network never sees (a forced roll is played for it) still count: one roll per turn.
        turns = buffer.rolls() / 2
        obs, masks, actors = buffer.observe()
        forbidden, counts = buffer.forbidden()
        actions, offers = np.zeros(parallel, np.int64), np.zeros((parallel, 5), np.int8)
        ours = actors == mine if opponent is not None else np.ones(parallel, bool)
        for player, rows in ((agent, ours), (opponent, ~ours)):
            if rows.any():
                actions[rows], offers[rows] = player(obs[rows], masks[rows], forbidden[rows], counts[rows])
        for game, (o, action, actor) in enumerate(zip(obs, actions, actors)):
            seats[game][actor].note(o, action, turns[game])
            titles[game] = holders(o, actor)
        for game, winner, _ in buffer.step(actions, offers):
            if finished < games:
                # Titles as last seen; a winning road or knight scores only by taking a title, so it took that one.
                road, army = titles[game]
                if winner is not None and actions[game] in ROADS:
                    road = winner
                if winner is not None and actions[game] == KNIGHT:
                    army = winner
                done.append((seats[game], winner, road, army, turns[game], None if opponent is None else mine[game]))
                finished += 1
            seats[game], titles[game] = [Seat(), Seat()], -1
    return done

def replay(entry, rules):
    """A game logged by the viewer, played back move by move, in the same form as `play`'s, focused on the network."""
    players, seed, codes = entry["record"].split(";")
    game = catan.Game(int(seed), int(players), rules=rules)
    seats, turns = [Seat() for _ in range(int(players))], 0
    for code in map(int, filter(None, codes.split(","))):
        actor = game.to_act()
        if actor is None:
            turns += game.observe_as(0)[PHASE] == DICE_PHASE
        else:
            seats[actor].note(game.observe_as(actor), code, turns / 2)
        game.step(code, None)
    winner = game.winner()
    road, army = holders(game.observe_as(0), 0, int(players))
    return seats, winner, road, army, turns / 2, 1 - entry["human"]

def summarize(done):
    """Style of the focus seats (the agent's, or every seat in self-play)."""
    focus = lambda mine: (0, 1) if mine is None else (mine,)
    rows = [(seats[s], winner == s) for seats, winner, *_, mine in done for s in focus(mine)]
    share = lambda xs: sum(xs) / max(len(xs), 1)
    won = [(road == winner, army == winner) for _, winner, road, army, _, mine in done
           if winner is not None and winner in focus(mine)]
    out = {"games won": len(won) / max(len(rows), 1),
           "wins by largest army": share([army for _, army in won]), "wins by longest road": share([road for road, _ in won]),
           "wins by both": share([road and army for road, army in won]), "wins by neither": share([not (road or army) for road, army in won])}
    out["saves for a city first"] = share([s.first_build == "city" for s, _ in rows if s.first_build])
    out["uses a port"] = share([s.port for s, _ in rows])
    out["uses a 2:1 port"] = share([s.special_port for s, _ in rows])
    out["development cards played per game"] = share([s.cards_played for s, _ in rows])
    out["roads built per game"] = share([s.roads for s, _ in rows])
    out["roads built per game, winning"] = share([s.roads for s, w in rows if w])
    out["roads built per game, losing"] = share([s.roads for s, w in rows if not w])
    out["turn of first new settlement"] = share([s.first_settlement for s, _ in rows if s.first_settlement is not None])
    out["turns per player per game"] = share([turns for *_, turns, _ in done])
    return out

def report(done, label):
    print(f"{len(done)} games, {sum(winner is None for _, winner, *_ in done)} cut off")
    print(f"{'':38s}{label:>9s}{'VINCO':>9s}{'humans':>9s}")
    for name, value in summarize(done).items():
        print(f"{name:38s}{value:9.2f}" + "".join(f"{'' if r is None else f'{r:.2f}':>9s}" for r in REFERENCE.get(name, (None, None))))

def main():
    parser = argparse.ArgumentParser(description="Report a 2p network's style next to VINCO's and top human players' numbers.")
    parser.add_argument("checkpoint", nargs="?", help="the network to play (not needed with --log)")
    parser.add_argument("--opponent", help="another network for it to play, instead of itself")
    parser.add_argument("--log", help="games the viewer logged against a person, instead of playing any")
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    rules = rules_named("colonist-1v1", trading=False)
    if args.log:
        entries = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
        done = [replay(e, rules) for e in entries if e.get("winner") is not None]
        print(f"the network against you, from {args.log}")
        return report(done, "network")
    at = device()
    agent = Network(load(args.checkpoint, at), at, greedy=args.greedy, seed=args.seed)
    opponent = Network(load(args.opponent, at), at, greedy=args.greedy, seed=args.seed + 1) if args.opponent else None
    report(play(agent, args.games, args.seed, rules, max_decisions=8000, opponent=opponent), "network")

if __name__ == "__main__":
    main()
