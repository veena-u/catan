import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import catan

from .agents import KIND_COUNT, KINDS, Network, Random
from .bot import Bot
from .evaluate import play
from .features import SEATS
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


def to_absolute(relative, actors, players):
    absolute = torch.zeros_like(relative)
    index = (torch.arange(players, device=relative.device) - actors[:, None]) % players
    absolute[:, :players] = relative[:, :players].gather(1, index)
    return absolute


def to_relative(absolute, actors, players):
    relative = torch.zeros_like(absolute)
    index = (torch.arange(players, device=absolute.device) + actors[:, None]) % players
    relative[:, :players] = absolute[:, :players].gather(1, index)
    return relative


def rewards(winner, players, seats=SEATS):
    reward = torch.zeros(seats)
    reward[:players] = -1.0 / (players - 1)
    reward[winner] = 1.0
    return reward


class Rollout:
    def __init__(self, model, buffer, at, autocast, players, average_dice=False):
        self.model, self.buffer, self.at, self.autocast = model, buffer, at, autocast
        self.games, self.players = len(buffer), players
        self.average_dice = average_dice
        buffer.track_dice(average_dice)

    @torch.no_grad()
    def evaluate(self, obs, masks):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            logits, values, _ = self.model(obs)
        dist, _ = policy(logits, masks)
        return dist, values.float()

    @torch.no_grad()
    def value(self, obs, actors):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            _, values, _ = self.model(obs)
        return to_absolute(values.float(), actors, self.players)

    def tensor(self, array, dtype=None):
        return torch.from_numpy(array.astype(dtype) if dtype else array).to(self.at)

    def expected_after_roll(self):
        has, obs, actors, weights = self.buffer.dice()
        games = np.flatnonzero(has)
        if len(games) == 0:
            return games, None
        outcomes = obs.shape[1]
        flat_actors = self.tensor(actors[games].reshape(-1), np.int64)
        values = self.value(self.tensor(obs[games].reshape(-1, obs.shape[-1])), flat_actors).view(len(games), outcomes, -1)
        return games, (self.tensor(weights[games])[..., None] * values).sum(1)

    def collect(self, rounds):
        n, at = self.games, self.at
        obs = torch.zeros(rounds, n, catan.OBSERVATION_BYTES, dtype=torch.uint8, device=at)
        masks = torch.zeros(rounds, n, catan.ACTIONS, dtype=torch.bool, device=at)
        actions = torch.zeros(rounds, n, dtype=torch.long, device=at)
        log_probs = torch.zeros(rounds, n, device=at)
        values = torch.zeros(rounds, n, SEATS, device=at)
        actors = torch.zeros(rounds, n, dtype=torch.long, device=at)
        done, ends = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, SEATS, device=at)
        rolled, expected = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, SEATS, device=at)
        stats = {"won": 0, "truncated": 0, "first_won": 0}

        for t in range(rounds):
            o, m, p = self.buffer.observe()
            obs[t], masks[t], actors[t] = self.tensor(o), self.tensor(m), self.tensor(p, np.int64)
            dist, v = self.evaluate(obs[t], masks[t])
            a = dist.sample()
            actions[t], log_probs[t], values[t] = a, dist.log_prob(a), to_absolute(v, actors[t], self.players)
            endings = self.buffer.step(a.cpu().numpy())
            if self.average_dice:
                games, value = self.expected_after_roll()
                if value is not None:
                    rolled[t, games], expected[t, games] = True, value
            truncated = []
            for game, winner, time_ in endings:
                done[t, game] = True
                if winner is None:
                    truncated.append((game, time_))
                    stats["truncated"] += 1
                else:
                    ends[t, game] = rewards(winner, self.players).to(at)
                    stats["won"] += 1
                    stats["first_won"] += winner == 0
            if truncated:
                games, times = zip(*truncated)
                o, _, q = self.buffer.observe_at(list(games), list(times))
                ends[t, list(games)] = self.value(self.tensor(o), self.tensor(q, np.int64))

        o, _, p = self.buffer.observe()
        last = self.value(self.tensor(o), self.tensor(p, np.int64))
        batch = dict(obs=obs, masks=masks, actions=actions, log_probs=log_probs, actors=actors)
        return batch, values, (done, ends, rolled, expected, last), stats


def sync(at):
    if at.type == "cuda":
        torch.cuda.synchronize()
    elif at.type == "mps":
        torch.mps.synchronize()


def advantages(values, done, ends, rolled, expected, last, lam):
    result = torch.zeros_like(values)
    following = torch.zeros_like(last)
    for t in reversed(range(values.shape[0])):
        next_values = last if t == values.shape[0] - 1 else values[t + 1]
        target = torch.where(rolled[t, :, None], expected[t], next_values)
        delta = torch.where(done[t, :, None], ends[t] - values[t], target - values[t])
        following = torch.where(done[t, :, None], delta, delta + lam * following)
        result[t] = following
    return result


def update(model, optimizer, batch, adv, returns, players, args, autocast, magnet_weight):
    flat = {k: v.flatten(0, 1) for k, v in batch.items()}
    actors = flat["actors"]
    own = adv.flatten(0, 1).gather(1, actors[:, None]).squeeze(1)
    targets = to_relative(returns.flatten(0, 1), actors, players)
    total = len(own)
    threshold = own.abs().float().quantile(1 - args.keep) if args.keep < 1 else -1.0
    informative = own.abs() >= threshold
    logs = {"policy": 0.0, "value": 0.0, "kl_magnet": 0.0, "approx_kl": 0.0, "clipped": 0.0,
            "kept": informative.float().mean().item()}
    steps = 0
    for _ in range(args.epochs):
        for pick in torch.randperm(total, device=own.device).split(args.minibatch):
            obs, masks = flat["obs"][pick], flat["masks"][pick]
            with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                logits, values, _ = model(obs)
            dist, log_probs = policy(logits, masks)
            new = dist.log_prob(flat["actions"][pick])
            ratio = (new - flat["log_probs"][pick]).exp()
            kept = informative[pick]
            ratio, a = ratio[kept], own[pick][kept]
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            value_loss = nn.functional.mse_loss(values.float()[:, :players], targets[pick][:, :players])
            magnet = kl_to_magnet(log_probs, masks).mean()
            loss = policy_loss + args.value_coef * value_loss + magnet_weight * magnet
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
    kept = logs.pop("kept")
    return {**{k: v / steps for k, v in logs.items()}, "kept": kept}


def main():
    parser = argparse.ArgumentParser(description="Self-play PPO with a win/loss reward.")
    parser.add_argument("--games", type=int, default=1024, help="games played side by side")
    parser.add_argument("--players", type=int, default=4, help="players per game, 2 to 4")
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
    parser.add_argument("--anneal-power", type=float, default=0.0,
                        help="shrink the learning rate and magnet weight as (1 + iteration / anneal-steps) ** -power; 0 is off")
    parser.add_argument("--anneal-steps", type=float, default=20.0)
    parser.add_argument("--keep", type=float, default=1.0,
                        help="train the policy only on this fraction of moves, those with the largest advantage magnitude")
    parser.add_argument("--average-dice", action="store_true",
                        help="score a move followed by a roll against the average over all 11 rolls, not the one rolled")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-decisions", type=int, default=5000)
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
    buffer = catan.Buffer(args.games, history=64, max_decisions=args.max_decisions, seed=args.seed, players=args.players)
    rollout = Rollout(model, buffer, at, autocast, args.players, args.average_dice)
    log = open(out / "log.txt", "a")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {at}; "
        f"{args.games} games × {args.rounds} decisions = {args.games * args.rounds} per update")

    decisions = 0
    previous = None
    start = time.time()
    for iteration in range(1, args.iterations + 1):
        model.eval()
        clock = time.time()
        batch, values, scoring, stats = rollout.collect(args.rounds)
        sync(at)
        collect_time, clock = time.time() - clock, time.time()
        adv = advantages(values, *scoring, args.lam)
        returns = adv + values
        scale = (1 + (iteration - 1) / args.anneal_steps) ** -args.anneal_power
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        model.train()
        logs = update(model, optimizer, batch, adv, returns, args.players, args, autocast, args.magnet * scale)
        sync(at)
        update_time = time.time() - clock
        decisions += args.games * args.rounds

        finished = stats["won"] + stats["truncated"]
        line = (f"iter {iteration:4d}  {decisions / 1e6:6.2f}M decisions  {time.time() - start:6.0f}s  "
                f"games {finished:5d} ({stats['truncated']} cut)  first seat wins {stats['first_won'] / max(stats['won'], 1):.2f}  "
                f"value {logs['value']:.3f}  kl→magnet {logs['kl_magnet']:.3f}  approx_kl {logs['approx_kl']:.4f}  "
                f"clipped {logs['clipped']:.2f}  kept {logs['kept']:.2f}  lr {args.lr * scale:.1e}  "
                f"play {collect_time:.1f}s learn {update_time:.1f}s")
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            model.eval()
            clock = time.time()
            player = Network(model, at, seed=iteration)
            opponents = {"random": Random(iteration), "bot": Bot()}
            if previous is not None:
                opponents["previous"] = Network(previous, at, seed=-iteration)
            for name, opponent in opponents.items():
                result = play(player, opponent, games=args.eval_games, parallel=min(args.eval_games, 512),
                              seed=10_000 + iteration, players=args.players)
                line += f"  vs {name} {result['win_rate']:.3f}"
            checkpoint = {"model": model.state_dict(), "args": vars(args), "iteration": iteration}
            torch.save(checkpoint, out / f"iter_{iteration:05d}.pt")
            torch.save(checkpoint, out / "latest.pt")
            previous = copy.deepcopy(model).eval()
            line += f"  (eval {time.time() - clock:.0f}s)"
        say(line)


if __name__ == "__main__":
    main()
