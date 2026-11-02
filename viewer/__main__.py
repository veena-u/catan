import argparse
import json
import random
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


import catan
from train.agents import Network, Random
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
TRADE_AT = catan.TRADE_AT
HEXES, EDGE_ENDS = (a.tolist() for a in catan.topology())
HEX_POINTS, VERTEX_POINTS, COAST = (a.tolist() for a in catan.geometry())
GEOMETRY = {"hexes": HEX_POINTS, "vertices": VERTEX_POINTS, "edges": EDGE_ENDS, "corners": HEXES, "coast": COAST}

def bitset(raw, count):
    value = int.from_bytes(bytes(raw), "little")
    return [i for i in range(count) if value >> i & 1]

def cards(counts):
    named = [f"{n} {RESOURCES[r]}" for r, n in enumerate(counts) if n]
    return ", ".join(named) if named else "anything"

class Session:
    def __init__(self, args, agent):
        self.args, self.agent = args, agent
        self.lock = threading.Lock()
        self.rules = catan.Rules.colonist_1v1() if args.rules == "colonist-1v1" else catan.Rules()
        self.new(args.seed, args.seat)

    def new(self, seed=None, seat=None):
        self.seed = random.randrange(1 << 31) if seed is None else int(seed)
        self.human = random.randrange(self.args.players) if seat is None else int(seat)
        self.game = catan.Game(self.seed, self.args.players, rules=self.rules)
        self.game.settle()
        self.events, self.read, self.last_roll, self.victim, self.mover = [], 0, None, None, 0

    def absolute(self, relative):
        return (self.human + relative) % self.args.players

    def act(self, code):
        if self.game.to_act() != self.human or code not in self.game.legal():
            raise ValueError("that move is not yours to make now")
        self.take(code)

    def step(self):
        actor = self.game.to_act()
        if actor is None or actor == self.human:
            return
        obs, mask = self.game.observe()
        self.take(int(self.agent(obs[None], mask[None])[0]))

    def take(self, code):
        self.game.step(code)
        self.game.settle()
        self.narrate()

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
        if kind == "send" and PHASES[phase] == "draft":
            offer = seen[TRADE_AT:TRADE_AT + 10]
            return f"offered {cards(offer[:5])} for {cards(offer[5:])}"
        if kind == "send":
            return "made a counter-offer"
        if kind in ("accept", "decline"):
            return f"{kind}d the offer"
        if kind == "trade with":
            return f"traded with {NAMES[(actor + i + 1) % n]}"
        if kind == "take counter":
            return f"took {NAMES[(actor + i + 1) % n]}'s counter-offer"
        if kind == "cancel":
            return "called off the trade"
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
                "protected": bool(o[t + 92 + s]),
            }
            if s > 0:
                at = t + 21 + 11 * (s - 1)
                seat["answer"] = ["", "waiting", "accepted", "declined", "countered"][min(o[at], 4)]
                counter = o[at + 1:at + 11]
                seat["counter"] = {"give": counter[:5], "get": counter[5:]} if any(counter) else None
            seats.append(seat)
        current = seat_of(o[294])
        offer = o[t:t + 10]
        draft = o[t + 10:t + 20]
        last = o[t + 58:t + 68]
        winner = self.game.winner()
        return {
            "seed": self.seed, "players": n, "human": self.human, "toAct": self.game.to_act(), "winner": winner,
            "geometry": GEOMETRY,
            "rules": {"points": o[t + 71], "discardLimit": o[t + 72], "friendlyRobber": bool(o[t + 73]),
                      "balancedDice": bool(o[t + 74]), "trading": bool(o[t + 75])},
            "board": {
                "terrain": [TERRAINS[c] for c in o[128:147]], "tokens": o[147:166], "robber": o[220],
                "harbors": [None if h == 0 else ("3:1" if h == 1 else RESOURCES[h - 2]) for h in o[166:220]],
                "settlements": settlements, "cities": cities, "roads": roads,
            },
            "seats": seats,
            "hand": o[222:227], "devCards": o[258:263], "bought": o[263:268], "rates": o[t + 87:t + 92],
            "bank": o[253:258], "deck": o[274],
            "phase": PHASES[o[296]], "rolled": bool(o[297]), "count": o[298], "current": current,
            "phasePlayer": None if o[299] == 255 else seat_of(o[299]), "cardPlayed": bool(o[295]),
            "trade": {
                "offer": {"give": offer[:5], "get": offer[5:]} if any(offer) else None,
                "draft": {"give": draft[:5], "get": draft[5:]},
                "drafting": ["", "offer", "counter"][o[t + 20]],
                "offers": o[t + 54], "budgetLeft": o[t + 56], "closed": bool(o[t + 57]),
                "last": {"give": last[:5], "get": last[5:], "traded": bool(o[t + 70])} if any(last) else None,
            },
            "dice": o[t + 76:t + 87], "lastRoll": self.last_roll,
            "moves": self.moves(), "events": self.events[-60:], "record": self.game.record(),
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
                    session.act(int(body["code"]))
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
    parser.add_argument("--rules", choices=["standard", "colonist-1v1"], default="colonist-1v1")
    parser.add_argument("--seat", type=int, help="your seat; random if not given")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--temperature", type=float, default=0.8, help="how freely the network samples its moves")
    parser.add_argument("--greedy", action="store_true", help="the network always plays its top move")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if args.checkpoint:
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
