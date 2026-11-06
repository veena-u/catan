import argparse
import json
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import numpy as np

import catan
from train.agents import Network, Random
from train.search import Searcher
from train.evaluate import RULE_SETS, rules_of
from train.features import (
    ANSWERS_AT, BUDGET_LEFT, LAST_AT, LAST_TRADED, OFFERS_MADE, PROTECTED_AT, RATES_AT, TRADE_AT,
)
from train.model import KIND_RUNS, KINDS, device, load

RESOURCES = ["brick", "lumber", "wool", "grain", "ore"]
TERRAINS = ["hills", "forest", "pasture", "fields", "mountains", "desert"]
DEV_CARDS = ["knight", "victory point", "road building", "year of plenty", "monopoly"]
PHASES = [
    "place settlement", "place road", "turn", "dice", "discard", "move robber", "steal", "draw", "road building",
    "year of plenty", "monopoly", "over", "choose victim", "respond", "confirm", "draft",
]
NAMES = ["Blue", "Red", "Orange", "Green"]
BANK_PAIRS = [(give, get) for give in range(5) for get in range(5) if get != give]
MAX_PROPOSED = 40
PHASE_AT = 296
END_TURN, OFFER, SEND, ACCEPT, DECLINE, COUNTER, TRADE_WITH, TAKE_COUNTER, CANCEL = 225, 234, 245, 247, 248, 249, 250, 253, 256
HEXES, EDGE_ENDS = (a.tolist() for a in catan.topology())
HEX_POINTS, VERTEX_POINTS, COAST = (a.tolist() for a in catan.geometry())
GEOMETRY = {"hexes": HEX_POINTS, "vertices": VERTEX_POINTS, "edges": EDGE_ENDS, "corners": HEXES, "coast": COAST}

def bitset(raw, count):
    value = int.from_bytes(bytes(raw), "little")
    return [i for i in range(count) if value >> i & 1]

def sides(counts):
    return [max(-c, 0) for c in counts], [max(c, 0) for c in counts]

def cards(counts):
    named = [f"{n} {RESOURCES[r]}" for r, n in enumerate(counts) if n]
    return ", ".join(named) if named else "anything"

class Session:
    def __init__(self, args, agent):
        self.args, self.agent = args, agent
        self.lock = threading.Lock()
        self.rules = rules_of(args)
        self.new(args.seed, args.seat)

    def new(self, seed=None, seat=None):
        if getattr(self, "game", None) is not None and not self.logged and self.game.steps() > 0:
            self.log()
        self.logged = False
        self.seed = random.randrange(1 << 31) if seed is None else int(seed)
        self.human = random.randrange(self.args.players) if seat is None else int(seat)
        self.game = catan.Game(self.seed, self.args.players, rules=self.rules)
        self.game.settle()
        self.events, self.read, self.last_roll, self.victim, self.mover = [], 0, None, None, 0
        self.trade = None

    def absolute(self, relative):
        return (self.human + relative) % self.args.players

    def act(self, code, deal=None):
        if self.game.to_act() != self.human or code not in self.game.legal():
            raise ValueError("that move is not yours to make now")
        if deal is not None and (code not in (OFFER, COUNTER) or len(deal) != 5):
            raise ValueError("a deal goes with an offer or a counter-offer, as five counts")
        self.take(code, None if deal is None else [int(c) for c in deal])

    def step(self):
        actor = self.game.to_act()
        if actor is None or actor == self.human:
            return
        obs, mask = self.game.observe()
        listed = self.game.forbidden()
        forbidden = np.zeros((1, MAX_PROPOSED, 5), np.int8)
        if listed:
            forbidden[0, :len(listed)] = listed
        state = (self.game, None) if getattr(self.agent, "uses_state", False) else ()
        actions, offers = self.agent(obs[None], mask[None], forbidden, np.array([len(listed)], np.uint8), *state)
        self.take(int(actions[0]), [int(c) for c in offers[0]])

    def take(self, code, deal=None):
        actor, phase = self.game.to_act(), PHASES[self.game.observe_as(self.human)[PHASE_AT]]
        self.countering = None
        self.track(actor, phase, code, deal, self.game.draft())
        self.game.step(code, deal)
        if self.countering and self.trade:
            seat, proposer = self.countering[0], self.trade["proposer"]
            relative = (seat - proposer) % self.args.players
            if self.game.observe_as(proposer)[ANSWERS_AT + 11 * (relative - 1)] == 4:
                self.answer(seat, "countered", *self.countering[1:])
        self.countering = None
        self.game.settle()
        self.narrate()
        self.settled()
        if self.game.winner() is not None and not self.logged:
            self.log()

    def log(self):
        """Every game against a person, finished or left, kept so its style can be measured (python -m train.style --log)."""
        if not self.args.log:
            return
        a, n = self.args, self.args.players
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checkpoint": a.checkpoint, "rules": a.rules,
                 "trading": self.rules.trading, "greedy": a.greedy, "temperature": a.temperature, "search": a.search,
                 "seed": self.seed, "players": n, "human": self.human, "winner": self.game.winner(),
                 "points": [int(self.game.observe_as(s)[288]) for s in range(n)], "record": self.game.record()}
        path = Path(a.log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as out:
            out.write(json.dumps(entry) + "\n")
        self.logged = True

    def say(self, seat, text):
        self.events.append({"seat": seat, "text": text})

    def track(self, actor, phase, code, deal, drafted):
        n = self.args.players
        if code == END_TURN:
            self.trade = None
        if code == OFFER and deal is not None:
            self.open_offer(actor, *sides(deal))
        elif phase == "draft" and code == SEND and drafted:
            self.open_offer(actor, list(drafted[1]), list(drafted[2]))
        elif phase == "respond" and self.trade:
            if code == ACCEPT:
                self.answer(actor, "accepted")
            elif code == DECLINE:
                self.answer(actor, "declined")
            elif code == COUNTER and deal is not None:
                self.countering = (actor, *sides(deal))
            elif code == SEND and drafted and drafted[0] == 2:
                self.answer(actor, "countered", list(drafted[1]), list(drafted[2]))
        elif phase == "confirm" and self.trade:
            if TRADE_WITH <= code < TAKE_COUNTER:
                partner = (actor + code - TRADE_WITH + 1) % n
                self.close(f"traded with {NAMES[partner]}", partner)
            elif TAKE_COUNTER <= code < CANCEL:
                partner = (actor + code - TAKE_COUNTER + 1) % n
                self.close(f"took {NAMES[partner]}'s counter-offer", partner)
            elif code == CANCEL:
                self.close("called off the trade")

    def open_offer(self, proposer, give, get):
        others = [(proposer + k) % self.args.players for k in range(1, self.args.players)]
        self.trade = {"proposer": proposer, "give": give, "get": get, "result": None, "partner": None,
                      "answers": {s: {"seat": s, "status": "waiting", "counter": None} for s in others}}
        self.say(proposer, f"offered {cards(give)} for {cards(get)}")

    def answer(self, seat, status, give=None, get=None):
        entry = self.trade["answers"][seat]
        entry["status"] = status
        if give is not None:
            entry["counter"] = {"give": give, "get": get}
            self.say(seat, f"declined, and offered {cards(give)} for {cards(get)} instead")
        else:
            self.say(seat, f"{status} the offer")

    def close(self, text, partner=None):
        self.trade["result"], self.trade["partner"] = text, partner
        self.say(self.trade["proposer"], text)

    def settled(self):
        if not self.trade or self.trade["result"]:
            return
        phase = PHASES[self.game.observe_as(self.human)[PHASE_AT]]
        if phase in ("draft", "respond", "confirm"):
            return
        for entry in self.trade["answers"].values():
            if entry["status"] == "waiting":
                entry["status"] = "declined"
        countered = any(e["status"] == "countered" for e in self.trade["answers"].values())
        self.close("couldn't take the counter-offer, so the trade fell through" if countered else "got no takers")

    def narrate(self):
        history = self.game.history()
        seen = self.game.observe_as(self.human)
        for actor, phase, code in history[self.read:]:
            text = self.describe(actor, phase, code, seen)
            if text:
                self.events.append({"seat": actor if actor >= 0 else self.mover, "text": text})
            if actor >= 0:
                self.mover = actor
        self.read = len(history)

    def describe(self, actor, phase, code, seen):
        if actor < 0:
            mover = self.mover
            if PHASES[phase] == "dice":
                self.last_roll = code + 2
                return f"rolled {code + 2}"
            if PHASES[phase] == "steal":
                victim = self.victim
                if victim is None and self.args.players == 2:
                    victim = 1 - mover
                self.victim = None
                if victim is None:
                    return "stole a card"
                if self.human in (mover, victim):
                    return f"stole 1 {RESOURCES[code]} from {NAMES[victim]}"
                return f"stole a card from {NAMES[victim]}"
            if PHASES[phase] == "draw":
                return f"bought a {DEV_CARDS[code]}" if mover == self.human else "bought a development card"
            return None
        kind, start = KIND_RUNS[KINDS[code]][0], KIND_RUNS[KINDS[code]][1]
        i, n = code - start, self.args.players
        if kind in ("settlement", "city", "road"):
            return f"built a {kind}"
        if kind == "robber":
            return "moved the robber"
        if kind == "rob":
            self.victim = (actor + i + 1) % n
            return None
        if kind == "pick resource":
            return {
                "discard": "discarded a card",
                "year of plenty": f"took 1 {RESOURCES[i]} from the bank",
                "monopoly": f"called a monopoly on {RESOURCES[i]}",
            }.get(PHASES[phase])
        if kind == "bank trade":
            give, get = BANK_PAIRS[i]
            return f"traded {RESOURCES[give]} for 1 {RESOURCES[get]} with the bank"
        if kind.startswith("play "):
            return f"played {kind[5:].title()}"
        if kind == "end turn":
            return "ended their turn"
        return None

    def moves(self):
        if self.game.to_act() != self.human:
            return []
        n, out = self.args.players, []
        for code in self.game.legal():
            name, start, _, target = KIND_RUNS[KINDS[code]]
            i, move = code - start, {"code": code, "kind": name}
            if target == "corner":
                move["vertex"] = i
            elif target == "edge":
                move["edge"] = i
            elif target == "hex":
                move["hex"] = i
            elif target == "resource":
                move["resource"] = RESOURCES[i]
            elif target == "player":
                move["seat"] = (self.human + i + 1) % n
            elif target == "resource pair":
                move["give"], move["get"] = (RESOURCES[r] for r in BANK_PAIRS[i])
            out.append(move)
        return out

    def view(self):
        o = [int(b) for b in self.game.observe_as(self.human)]
        n, t = self.args.players, TRADE_AT
        seat_of = lambda relative: self.absolute(relative)
        settlements, cities, roads = [], [], []
        for s in range(n):
            base = 32 * s
            settlements += [{"vertex": v, "seat": seat_of(s)} for v in bitset(o[base:base + 8], 54)]
            cities += [{"vertex": v, "seat": seat_of(s)} for v in bitset(o[base + 8:base + 16], 54)]
            roads += [{"edge": e, "seat": seat_of(s)} for e in bitset(o[base + 16:base + 32], 72)]
        holder = lambda value: None if value == 0 else seat_of(value - 1)
        seats = []
        for s in range(n):
            if s == 0:
                hand, devs = sum(o[222:227]), sum(o[258:263])
            else:
                hand, devs = o[232 + 7 * (s - 1) + 5], o[268 + 2 * (s - 1)]
            seat = {
                "seat": seat_of(s), "name": NAMES[seat_of(s)], "you": s == 0, "points": o[288 + s], "cards": hand,
                "devCards": devs, "knights": o[280 + s], "roadLength": o[284 + s],
                "longestRoad": holder(o[292]) == seat_of(s), "largestArmy": holder(o[293]) == seat_of(s),
                "protected": bool(o[PROTECTED_AT + s]),
            }
            if s > 0:
                at = ANSWERS_AT + 11 * (s - 1)
                seat["answer"] = ["", "waiting", "accepted", "declined", "countered"][min(o[at], 4)]
                counter = o[at + 1:at + 11]
                seat["counter"] = {"give": counter[:5], "get": counter[5:]} if any(counter) else None
            seats.append(seat)
        current = seat_of(o[294])
        offer = o[t:t + 10]
        drafting, give, get, closed = self.game.draft() or (0, [0] * 5, [0] * 5, False)
        if self.game.to_act() != self.human:
            drafting, give, get = 0, [0] * 5, [0] * 5
        draft = list(give) + list(get)
        last = o[LAST_AT:LAST_AT + 10]
        winner = self.game.winner()
        return {
            "seed": self.seed, "players": n, "human": self.human, "toAct": self.game.to_act(), "winner": winner,
            "geometry": GEOMETRY,
            "rules": {"points": self.rules.points, "discardLimit": self.rules.discard_limit,
                      "friendlyRobber": self.rules.friendly_robber, "balancedDice": self.rules.balanced_dice,
                      "trading": self.rules.trading},
            "board": {
                "terrain": [TERRAINS[c] for c in o[128:147]], "tokens": o[147:166], "robber": o[220],
                "harbors": [None if h == 0 else ("3:1" if h == 1 else RESOURCES[h - 2]) for h in o[166:220]],
                "settlements": settlements, "cities": cities, "roads": roads,
            },
            "seats": seats,
            "hand": o[222:227], "devCards": o[258:263], "bought": o[263:268], "rates": o[RATES_AT:RATES_AT + 5],
            "bank": o[253:258], "deck": o[274],
            "phase": PHASES[o[296]], "rolled": bool(o[297]), "count": o[298], "current": current,
            "phasePlayer": None if o[299] == 255 else seat_of(o[299]), "cardPlayed": bool(o[295]),
            "trade": {
                "offer": {"give": offer[:5], "get": offer[5:]} if any(offer) else None,
                "draft": {"give": draft[:5], "get": draft[5:]},
                "drafting": ["", "offer", "counter"][drafting],
                "offers": o[OFFERS_MADE], "budgetLeft": o[BUDGET_LEFT], "closed": closed,
                "last": {"give": last[:5], "get": last[5:], "traded": bool(o[LAST_TRADED])} if any(last) else None,
            },
            "lastRoll": self.last_roll,
            "moves": self.moves(), "events": self.events[-60:], "record": self.game.record(),
            "forbidden": self.game.forbidden() if self.game.to_act() == self.human else [],
            "offerBoard": None if self.trade is None else {**self.trade, "answers": sorted(self.trade["answers"].values(), key=lambda e: e["seat"])},
        }

class Handler(BaseHTTPRequestHandler):
    session = None
    page = Path(__file__).with_name("index.html")

    def log_message(self, *args):
        pass

    def reply(self, status, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.reply(200, self.page.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/view":
            with self.session.lock:
                return self.reply(200, self.session.view())
        self.reply(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        session = self.session
        try:
            with session.lock:
                if self.path == "/api/new":
                    session.new(body.get("seed"), body.get("seat"))
                elif self.path == "/api/act":
                    session.act(int(body["code"]), body.get("deal"))
                elif self.path == "/api/step":
                    session.step()
                else:
                    return self.reply(404, {"error": "not found"})
                return self.reply(200, session.view())
        except (ValueError, KeyError) as error:
            self.reply(400, {"error": str(error)})

def main():
    parser = argparse.ArgumentParser(description="Play the network in the browser, on a Colonist-style board.")
    parser.add_argument("--checkpoint", help="a trained network to play against; without one, a random player")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--rules", choices=list(RULE_SETS), default="colonist-1v1")
    parser.add_argument("--no-trading", action="store_true", help="no trades between players, as in Colonist 1v1")
    parser.add_argument("--seat", type=int, help="your seat; random if not given")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=0.8, help="how freely the network samples its moves")
    parser.add_argument("--greedy", action="store_true", help="the network always plays its top move")
    parser.add_argument("--search", type=int, default=0, metavar="SIMULATIONS",
                        help="the network looks ahead this many simulations before each move; 0 plays its instinct")
    parser.add_argument("--instinct", action="store_true",
                        help="the network trades by instinct too, instead of offering, countering and accepting only what it judges good for both sides")
    parser.add_argument("--log", default="games/played.jsonl", help="where every game against you is kept; '' keeps none")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and not args.instinct:
        at = device()
        agent = Searcher(load(args.checkpoint, at), at, args.search, greedy=args.greedy, temperature=args.temperature)
    elif args.checkpoint:
        at = device()
        agent = Network(load(args.checkpoint, at), at, greedy=args.greedy, temperature=args.temperature)
    else:
        agent = Random(0, kinds=True)
    Handler.session = Session(args, agent)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"playing at {url} ({'against ' + args.checkpoint if args.checkpoint else 'against a random player'})")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()

if __name__ == "__main__":
    main()
