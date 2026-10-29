import argparse
import re
import signal
import time
from pathlib import Path

BARS = "▁▂▃▄▅▆▇█"
LINE = re.compile(r"^iter\s+(\d+)\s+[\d.]+M decisions\s+(\d+)s")
FIELDS = ("random", "bot", "previous", "anchor")


def spark(values, low, high, least=0.1):
    middle = (low + high) / 2
    low, high = min(low, middle - least / 2), max(high, middle + least / 2)
    return "".join(BARS[min(7, max(0, int((v - low) / (high - low) * 8)))] for v in values)


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    parser = argparse.ArgumentParser(description="Summarise a training log: speed, time left, and checkup curves.")
    parser.add_argument("log", help="e.g. runs/four-scratch.out")
    parser.add_argument("--total", type=int, default=12000, help="the run's last iteration")
    parser.add_argument("--rows", type=int, default=8, help="how many recent checkups to list")
    parser.add_argument("--curve", type=int, default=60, help="how many checkups in each curve")
    args = parser.parse_args()

    path = Path(args.log)
    steps, evals = [], []
    offset = previous = run = 0
    for line in path.read_text(errors="ignore").splitlines():
        found = LINE.match(line)
        if not found:
            continue
        it, seconds = int(found.group(1)), int(found.group(2))
        if seconds < previous:
            offset += previous
            run += 1
        previous = seconds
        steps.append((it, offset + seconds, run))
        if "(eval" in line:
            row = {"iter": it, "hours": (offset + seconds) / 3600}
            for name in FIELDS:
                value = re.search(rf"vs {name} ([\d.]+)", line)
                if value:
                    row[name] = float(value.group(1))
            evals.append(row)
    if not steps:
        print(f"no iterations in {path} yet")
        return

    it, seconds, run = steps[-1]
    recent = [s for s in steps[-50:] if s[2] == run]
    per = (recent[-1][1] - recent[0][1]) / max(recent[-1][0] - recent[0][0], 1) if len(recent) > 1 else float("nan")
    left = max(args.total - it, 0) * per / 3600
    idle = (time.time() - path.stat().st_mtime) / 60
    restarts = f", restarted {run}x" if run else ""
    print(f"{path.name}: iteration {it} of {args.total} ({100 * it / args.total:.0f}%), {seconds / 3600:.1f}h trained{restarts}")
    print(f"speed {per:.1f}s per iteration, about {left:.0f}h left")
    if idle > 10:
        print(f"WARNING: nothing written for {idle:.0f} minutes; the run has probably stopped")
    else:
        print(f"last line {idle:.0f} minutes ago: running")
    if not evals:
        return

    print()
    names = [n for n in FIELDS if any(n in r for r in evals)]
    print("iter    hours  " + "  ".join(f"{n:>8}" for n in names))
    for row in evals[-args.rows:]:
        print(f"{row['iter']:<7} {row['hours']:5.1f}  " + "  ".join(f"{row.get(n, float('nan')):8.3f}" for n in names))

    print()
    for name in names:
        values = [r[name] for r in evals if name in r][-args.curve:]
        print(f"{name:>8}  {spark(values, min(values), max(values))}  {min(values):.2f} to {max(values):.2f}, last {values[-1]:.3f}")
    print(f"(curves cover the last {min(args.curve, len(evals))} checkups; each spans at least 0.1, so noise stays flat)")


if __name__ == "__main__":
    main()
