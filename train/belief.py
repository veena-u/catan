import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import catan

from .agents import Network
from .model import Tokens, device, load

OTHERS = 3


def pools(obs):
    """What one player can work out from their own view: the cards of each kind that must be in other hands,
    how many cards of each hand are not yet pinned down, and the same for development cards."""
    o = obs.long()
    known = torch.stack([o[:, 232 + 7 * k:237 + 7 * k] for k in range(OTHERS)], 1)
    sizes = torch.stack([o[:, 237 + 7 * k] for k in range(OTHERS)], 1)
    hand_pool = 19 - o[:, 253:258] - o[:, 222:227] - o[:, 227:232] - known.sum(1)
    unknown = sizes - known.sum(2)
    dev_pool = o[:, 275:280]
    dev_counts = torch.stack([o[:, 268 + 2 * k] for k in range(OTHERS)], 1)
    return known, hand_pool.clamp(min=0), unknown.clamp(min=0), dev_pool, dev_counts


def card_log_probs(logits, pool):
    """Chance that one unseen card is of each kind: the share of the pool, tilted by the network. Zero logits is the baseline."""
    tilted = logits.float() + pool.float().clamp(min=1e-6).log()[:, None]
    return tilted.masked_fill((pool == 0)[:, None], -1e4).log_softmax(-1)


class Belief(nn.Module):
    def __init__(self, width=128, depth=4):
        super().__init__()
        self.trunk = Tokens(width, depth)
        self.head = nn.Linear(width, OTHERS * 2 * 5)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.width, self.depth = width, depth

    def forward(self, obs):
        _, _, summary = self.trunk(obs)
        return self.head(summary.float()).view(-1, OTHERS, 2, 5).clamp(-10, 10)

    @torch.no_grad()
    def deals(self, obs, generator, chunk=8192):
        """A guess at every other hand and development card for each observation, as counts (n, 3, 10)."""
        at = next(self.parameters()).device
        logits = torch.cat([self(torch.from_numpy(obs[i:i + chunk]).to(at)).float().cpu() for i in range(0, len(obs), chunk)])
        known, hand_pool, unknown, dev_pool, dev_counts = pools(torch.from_numpy(obs))
        hands = known + draw(logits[:, :, 0], hand_pool, unknown, generator)
        devs = draw(logits[:, :, 1], dev_pool, dev_counts, generator)
        return torch.cat([hands, devs], 2).clamp(0, 255).to(torch.uint8).numpy()


def draw(logits, pool, counts, generator):
    pool, dealt = pool.clone(), torch.zeros(*logits.shape, dtype=torch.long)
    rows = torch.arange(len(pool))
    for k in range(OTHERS):
        left = counts[:, k].clone()
        while (left > 0).any():
            active = left > 0
            weights = logits[:, k].exp() * pool.float()
            weights[~active | (weights.sum(1) <= 0)] = 1.0
            pick = torch.multinomial(weights, 1, generator=generator).squeeze(1)
            r, c = rows[active], pick[active]
            dealt[r, k, c] += 1
            pool[r, c] = (pool[r, c] - 1).clamp(min=0)
            left[active] -= 1
    return dealt


def positions(model, at, count, games, seed, warmup=200, keep=0.02):
    """Positions from the network playing itself, with every hidden card written down."""
    buffer = catan.Buffer(games, seed=seed, players=4)
    agent, rng = Network(model, at, seed=seed), np.random.default_rng(seed)
    seen, truth, kept, step = [], [], 0, 0
    while kept < count:
        obs, masks, _ = buffer.observe()
        if step >= warmup:
            pick = rng.random(len(obs)) < keep
            seen.append(obs[pick])
            truth.append(buffer.hidden()[pick])
            kept += int(pick.sum())
        buffer.step(agent(obs, masks))
        step += 1
    return torch.from_numpy(np.concatenate(seen)[:count]), torch.from_numpy(np.concatenate(truth)[:count])


def losses(belief, obs, hidden):
    """Surprise per unseen card, in nats, for the network and for the baseline that only counts what is left."""
    known, hand_pool, _, dev_pool, _ = pools(obs)
    hand_target = (hidden[:, :, :5].long() - known).clamp(min=0).float()
    dev_target = hidden[:, :, 5:].float()
    logits = belief(obs)
    zero = torch.zeros_like(logits)
    out = {}
    for name, l in (("network", logits), ("baseline", zero)):
        out[f"{name} hands"] = -(hand_target * card_log_probs(l[:, :, 0], hand_pool)).sum()
        out[f"{name} devs"] = -(dev_target * card_log_probs(l[:, :, 1], dev_pool)).sum()
    return out, hand_target.sum(), dev_target.sum()


@torch.no_grad()
def score(belief, obs, hidden, at, batch=4096):
    totals, hand_cards, dev_cards = {}, 0.0, 0.0
    for i in range(0, len(obs), batch):
        out, h, d = losses(belief, obs[i:i + batch].to(at), hidden[i:i + batch].to(at))
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        hand_cards += h.item()
        dev_cards += d.item()
    return {k: v / max(hand_cards if "hands" in k else dev_cards, 1) for k, v in totals.items()}


def load_belief(path, at):
    saved = torch.load(path, map_location=at)
    belief = Belief(saved["width"], saved["depth"]).to(at)
    belief.load_state_dict(saved["belief"])
    return belief.eval()


def main():
    parser = argparse.ArgumentParser(description="Train a network to guess the cards other players hold, from one player's view.")
    parser.add_argument("checkpoint", help="the policy whose games to learn from; its trunk is the starting point")
    parser.add_argument("--positions", type=int, default=500_000)
    parser.add_argument("--test-positions", type=int, default=50_000)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--keep", type=float, default=0.02, help="share of decisions kept, so the positions come from many games")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", default="runs/belief/belief.pt")
    args = parser.parse_args()

    at = device()
    torch.manual_seed(0)
    policy = load(args.checkpoint, at)
    start = time.time()
    data = positions(policy, at, args.positions, args.games, seed=1, keep=args.keep)
    test = positions(policy, at, args.test_positions, args.games, seed=2, keep=args.keep)
    print(f"{len(data[0])} training and {len(test[0])} held-out positions ({time.time() - start:.0f}s)", flush=True)

    belief = Belief(policy.width, len(policy.blocks)).to(at)
    belief.trunk.load_state_dict(policy.state_dict())
    optimizer = torch.optim.AdamW(belief.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    obs, hidden = data
    best = float("inf")
    for step in range(1, args.steps + 1):
        pick = torch.randint(len(obs), (args.batch,))
        out, h, d = losses(belief, obs[pick].to(at), hidden[pick].to(at))
        loss = (out["network hands"] + out["network devs"]) / (h + d).clamp(min=1)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(belief.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if step % max(args.steps // 10, 1) == 0 or step == args.steps:
            belief.eval()
            result = score(belief, *test, at)
            belief.train()
            surprise = result["network hands"] + result["network devs"]
            kept = surprise < best
            print(f"step {step:6d}  {time.time() - start:5.0f}s  nats per unseen card: "
                  f"hands {result['network hands']:.3f} (baseline {result['baseline hands']:.3f})  "
                  f"dev cards {result['network devs']:.3f} (baseline {result['baseline devs']:.3f})"
                  f"{'  saved' if kept else ''}", flush=True)
            if kept:
                best = surprise
                torch.save({"belief": belief.state_dict(), "width": belief.width, "depth": belief.depth}, args.out)


if __name__ == "__main__":
    main()
