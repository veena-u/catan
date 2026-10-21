import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import catan

from .agents import KIND_COUNT, KINDS, Network, Random
from .evaluate import play
from .model import Tokens, device

KIND_MATRIX = torch.from_numpy(np.eye(KIND_COUNT, dtype=np.float32)[KINDS])


def policy(logits, masks):
    logits = logits.float().masked_fill(~masks, -1e9)
    return torch.distributions.Categorical(logits=logits), logits.log_softmax(-1)


def magnet_log_probs(masks):
    counts = masks.float() @ KIND_MATRIX.to(masks.device)
    kinds = (counts > 0).sum(-1, keepdim=True).float()
    per_action = counts @ KIND_MATRIX.to(masks.device).T
    return -(kinds.log() + per_action.clamp(min=1).log())


def kl_to_magnet(log_probs, masks):
    terms = log_probs.exp() * (log_probs - magnet_log_probs(masks))
    return torch.where(masks, terms, 0.0).sum(-1)


class Rollout:
    def __init__(self, model, buffer, at, autocast):
        self.model, self.buffer, self.at, self.autocast = model, buffer, at, autocast
        self.games = len(buffer)

    @torch.no_grad()
    def evaluate(self, obs, masks):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            logits, values, _ = self.model(obs)
        dist, _ = policy(logits, masks)
        return dist, values.float()

    def collect(self, rounds):
        n, at = self.games, self.at
        obs = torch.zeros(rounds, n, catan.OBSERVATION_BYTES, dtype=torch.uint8, device=at)
        masks = torch.zeros(rounds, n, catan.ACTIONS, dtype=torch.bool, device=at)
        actions = torch.zeros(rounds, n, dtype=torch.long, device=at)
        log_probs, values = torch.zeros(rounds, n, device=at), torch.zeros(rounds, n, device=at)
        players = torch.zeros(rounds, n, dtype=torch.long, device=at)
        done, ends = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, device=at)
        stats = {"won": 0, "truncated": 0, "first_won": 0}

        for t in range(rounds):
            o, m, p = self.buffer.observe()
            obs[t], masks[t] = torch.from_numpy(o).to(at), torch.from_numpy(m).to(at)
            players[t] = torch.from_numpy(p.astype(np.int64)).to(at)
            dist, v = self.evaluate(obs[t], masks[t])
            a = dist.sample()
            actions[t], log_probs[t], values[t] = a, dist.log_prob(a), v
            endings = self.buffer.step(a.cpu().numpy())
            truncated = []
            for game, winner, time_ in endings:
                done[t, game] = True
                if winner is None:
                    truncated.append((game, time_))
                    stats["truncated"] += 1
                else:
                    ends[t, game] = 1.0 if winner == p[game] else -1.0
                    stats["won"] += 1
                    stats["first_won"] += winner == 0
            if truncated:
                games, times = zip(*truncated)
                o, m, q = self.buffer.observe_at(list(games), list(times))
                _, v = self.evaluate(torch.from_numpy(o).to(at), torch.from_numpy(m).to(at))
                same = torch.from_numpy(q.astype(np.int64) == p[list(games)]).to(at)
                ends[t, list(games)] = torch.where(same, v, -v)

        o, m, p = self.buffer.observe()
        _, last_values = self.evaluate(torch.from_numpy(o).to(at), torch.from_numpy(m).to(at))
        last_players = torch.from_numpy(p.astype(np.int64)).to(at)
        batch = dict(obs=obs, masks=masks, actions=actions, log_probs=log_probs, values=values)
        return batch, (players, done, ends, last_values, last_players), stats


def advantages(values, players, done, ends, last_values, last_players, lam):
    rounds = values.shape[0]
    result = torch.zeros_like(values)
    following = torch.zeros_like(last_values)
    for t in reversed(range(rounds)):
        next_values = last_values if t == rounds - 1 else values[t + 1]
        next_players = last_players if t == rounds - 1 else players[t + 1]
        sign = torch.where(players[t] == next_players, 1.0, -1.0)
        delta = torch.where(done[t], ends[t] - values[t], sign * next_values - values[t])
        following = torch.where(done[t], delta, delta + lam * sign * following)
        result[t] = following
    return result


def update(model, optimizer, batch, adv, returns, args, autocast):
    flat = {k: v.flatten(0, 1) for k, v in batch.items()}
    adv, returns = adv.flatten(), returns.flatten()
    total = len(adv)
    logs = {"policy": 0.0, "value": 0.0, "kl_magnet": 0.0, "approx_kl": 0.0, "clipped": 0.0}
    steps = 0
    for _ in range(args.epochs):
        for pick in torch.randperm(total, device=adv.device).split(args.minibatch):
            obs, masks = flat["obs"][pick], flat["masks"][pick]
            with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                logits, values, _ = model(obs)
            dist, log_probs = policy(logits, masks)
            new = dist.log_prob(flat["actions"][pick])
            ratio = (new - flat["log_probs"][pick]).exp()
            a = adv[pick]
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            value_loss = nn.functional.mse_loss(values.float(), returns[pick])
            magnet = kl_to_magnet(log_probs, masks).mean()
            loss = policy_loss + args.value_coef * value_loss + args.magnet * magnet
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                logs["policy"] += policy_loss.item()
                logs["value"] += value_loss.item()
                logs["kl_magnet"] += magnet.item()
                logs["approx_kl"] += ((ratio - 1) - ratio.log()).mean().item()
                logs["clipped"] += ((ratio - 1).abs() > args.clip).float().mean().item()
            steps += 1
    return {k: v / steps for k, v in logs.items()}


def main():
    parser = argparse.ArgumentParser(description="Self-play PPO with a win/loss reward.")
    parser.add_argument("--games", type=int, default=1024, help="games played side by side")
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
    parser.add_argument("--magnet", type=float, default=0.01, help="weight of the pull toward random-by-kind play")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/selfplay")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    at = device()
    autocast = at.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    model = Tokens(args.width, args.depth).to(at)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    buffer = catan.Buffer(args.games, history=64, max_decisions=args.max_decisions, seed=args.seed)
    rollout = Rollout(model, buffer, at, autocast)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {at}; "
          f"{args.games} games × {args.rounds} decisions = {args.games * args.rounds} per update")

    decisions = 0
    start = time.time()
    for iteration in range(1, args.iterations + 1):
        model.eval()
        batch, scoring, stats = rollout.collect(args.rounds)
        adv = advantages(batch["values"], *scoring, args.lam)
        returns = adv + batch["values"]
        model.train()
        logs = update(model, optimizer, batch, adv, returns, args, autocast)
        decisions += args.games * args.rounds

        finished = stats["won"] + stats["truncated"]
        line = (f"iter {iteration:4d}  {decisions / 1e6:6.2f}M decisions  {time.time() - start:6.0f}s  "
                f"games {finished:5d} ({stats['truncated']} cut)  first wins {stats['first_won'] / max(stats['won'], 1):.2f}  "
                f"value {logs['value']:.3f}  kl→magnet {logs['kl_magnet']:.3f}  approx_kl {logs['approx_kl']:.4f}  "
                f"clipped {logs['clipped']:.2f}")
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            model.eval()
            result = play(Network(model, at, seed=iteration), Random(iteration), games=args.eval_games,
                          parallel=min(args.eval_games, 512), seed=10_000 + iteration)
            line += f"  vs random {result['win_rate']:.3f} ± {result['stderr']:.3f}"
            torch.save({"model": model.state_dict(), "args": vars(args), "iteration": iteration}, out / "latest.pt")
        print(line, flush=True)


if __name__ == "__main__":
    main()
