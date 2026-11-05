import argparse
import math

METRICS = [
    ("eval/vs_previous", "vs prev"),
    ("eval/vs_random", "vs rand"),
    ("games/finished", "finished"),
    ("games/first_seat_wins", "seat1 wins"),
    ("habits/setup_pips", "pips"),
    ("habits/setup_desert", "desert"),
    ("habits/kind_road", "road%"),
    ("habits/kind_settlement", "settle%"),
    ("habits/kind_city", "city%"),
    ("habits/kind_buy_card", "card%"),
    ("habits/kind_offer", "offer%"),
    ("trade/cards_per_offer", "cards/offer"),
    ("trade/accepted", "accept"),
    ("trade/countered", "counter"),
    ("trade/advantage", "trade adv"),
    ("games/trades_per_1k", "trades/1k"),
    ("habits/turns_out_of_time", "out of time"),
    ("search/acceptable", "search ok"),
    ("search/best_gain", "search gain"),
    ("ppo/entropy_kind", "H kind"),
    ("ppo/entropy_target", "H target"),
    ("ppo/approx_kl", "kl"),
    ("loss/value", "value loss"),
]

def series(run, key):
    rows = [(r["decisions"], r[key]) for r in run.scan_history(keys=["decisions", key])
            if r.get(key) is not None and r.get("decisions") is not None]
    return [(d, v) for d, v in rows if not (isinstance(v, float) and math.isnan(v))]

def near(points, at, window):
    chosen = [v for d, v in points if at - window <= d <= at]
    if not chosen:
        before = [v for d, v in points if d <= at]
        chosen = before[-1:]
    return sum(chosen) / len(chosen) if chosen else None

def show(value):
    if value is None:
        return "-"
    return f"{value:.3f}" if abs(value) < 10 else f"{value:.0f}"

def main():
    parser = argparse.ArgumentParser(description="Summarize every run in W&B groups at the same amount of training.")
    parser.add_argument("groups", nargs="+", help="e.g. 4p-v2 2p-v6")
    parser.add_argument("--project", default="catan")
    parser.add_argument("--entity", help="your W&B user or team; defaults to your login")
    args = parser.parse_args()
    import wandb
    api = wandb.Api()
    entity = args.entity or api.default_entity
    for group in args.groups:
        runs = [r for r in api.runs(f"{entity}/{args.project}", filters={"group": group}) if r.state != "crashed" or r.lastHistoryStep > 0]
        if not runs:
            print(f"## {group}: no runs found")
            continue
        data = {r.name: {key: series(r, key) for key, _ in METRICS} for r in runs}
        reach = {r.name: max((d for pts in data[r.name].values() for d, _ in pts), default=0) for r in runs}
        common = min(v for v in reach.values() if v > 0)
        names = sorted(data, key=lambda n: (not n.endswith("baseline-s1"), n))
        print(f"## {group}: compared at {common / 1e6:.0f}M decisions (the most every run reached)\n")
        print("| run | reached | state | " + " | ".join(label for _, label in METRICS) + " |")
        print("|" + "---|" * (len(METRICS) + 3))
        for name in names:
            run = next(r for r in runs if r.name == name)
            cells = [show(near(data[name][key], common, common * 0.05)) for key, _ in METRICS]
            print(f"| {name} | {reach[name] / 1e6:.0f}M | {run.state} | " + " | ".join(cells) + " |")
        print()
        print(f"### {group}: latest values, whatever each run reached\n")
        print("| run | " + " | ".join(label for _, label in METRICS) + " |")
        print("|" + "---|" * (len(METRICS) + 1))
        for name in names:
            cells = [show(near(data[name][key], reach[name], reach[name] * 0.05)) for key, _ in METRICS]
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()

if __name__ == "__main__":
    main()
