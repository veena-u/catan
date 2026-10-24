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
from .features import SEATS, SYMMETRIES
from .model import Tokens, device
from .search import Search

HIDDEN = (SEATS - 1) * 10
TRADED = np.isin(KINDS, (17, 18))
EXPERT_VISITS = 50
DICE_SUMS = 11

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
    """+1 to the winner and -1/(players-1) to everyone else. A game cut off without a winner is a loss for everyone,
    so stalling is never as good as playing to win."""
    reward = torch.zeros(seats)
    reward[:players] = -1.0 / (players - 1)
    if winner is not None:
        reward[winner] = 1.0
    return reward


class Critics(nn.Module):
    """Value heads used only in training: one that also sees every hidden card, one that also sees the dice to come."""

    def __init__(self, width):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(width + HIDDEN, width), nn.GELU(), nn.Linear(width, SEATS))
        self.future = nn.Sequential(nn.Linear(width + DICE_SUMS, width), nn.GELU(), nn.Linear(width, SEATS))

    def privileged(self, summary, hidden):
        return torch.tanh(self.hidden(torch.cat([summary.float(), hidden.flatten(1).float() / 5], -1)))

    def with_future(self, summary, rolls):
        return torch.tanh(self.future(torch.cat([summary.float(), rolls.float()], -1)))


class Rollout:
    def __init__(self, model, critics, buffer, at, autocast, players, average_dice=False, privileged=False,
                 search=None, expert_games=0, expert_every=1, expert_scale=0.1, seed=0, vp_reward=0.0, bot_games=0.0):
        self.model, self.critics, self.buffer, self.at, self.autocast = model, critics, buffer, at, autocast
        self.games, self.players = len(buffer), players
        self.average_dice, self.privileged = average_dice, privileged
        self.search, self.expert_games, self.expert_every, self.expert_scale = search, expert_games, expert_every, expert_scale
        self.rng = np.random.default_rng(seed)
        self.vp_reward = vp_reward
        self.bot = Bot()
        mixed = np.arange(self.games) < round(bot_games * self.games)
        self.bot_seats = mixed[:, None] & ((np.arange(SEATS)[None, :] + np.arange(self.games)[:, None]) % 2 == 0)
        buffer.track_dice(average_dice)

    def expert(self, o, m, p):
        """Gumbel AlphaZero's policy improvement on a few positions with a real choice. Search a sample of moves; every
        move keeps the network's preference, and each searched one moves up or down by how much better or worse search
        found it than the network expected the position to be worth."""
        choosing = np.flatnonzero(m.sum(1) > 1)
        games = self.rng.choice(choosing, min(self.expert_games, len(choosing)), replace=False)
        if len(games) == 0:
            return None
        choices, q, prior, value = self.search.improve(o[games], m[games], self.buffer, games, p[games], gumbel=True)
        completed = value[:, None].expand_as(prior).clone()
        completed.scatter_(1, choices, torch.where(torch.isfinite(q), q, value[:, None]))
        target = (prior + self.expert_scale * (EXPERT_VISITS + self.search.samples) * completed).softmax(1)
        changed = (target.argmax(1) != prior.argmax(1)).float().sum().item()
        return self.tensor(o[games]), self.tensor(m[games]), target.to(self.at), changed

    @torch.no_grad()
    def evaluate(self, obs, masks, hidden=None):
        with torch.autocast(self.at.type, dtype=torch.bfloat16, enabled=self.autocast):
            logits, values, summary = self.model(obs)
        if hidden is not None:
            values = self.critics.privileged(summary, hidden)
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
        hidden = torch.zeros(rounds, n, SEATS - 1, 10, dtype=torch.uint8, device=at)
        rolls = torch.zeros(rounds, n, DICE_SUMS, device=at)
        done, ends = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, SEATS, device=at)
        progress = torch.zeros(rounds, n, SEATS, device=at)
        rolled, expected = torch.zeros(rounds, n, dtype=torch.bool, device=at), torch.zeros(rounds, n, SEATS, device=at)
        stats = {"won": 0, "truncated": 0, "first_won": 0, "searched": 0, "changed": 0, "trades": 0,
                 "mixed": 0, "mixed_network_won": 0}
        learn = torch.ones(rounds, n, dtype=torch.bool, device=at)
        expert = []

        for t in range(rounds):
            o, m, p = self.buffer.observe()
            obs[t], masks[t], actors[t] = self.tensor(o), self.tensor(m), self.tensor(p, np.int64)
            if self.search is not None and t % self.expert_every == 0:
                found = self.expert(o, m, p)
                if found is not None:
                    expert.append(found[:3])
                    stats["searched"] += len(found[0])
                    stats["changed"] += found[3]
            hidden[t] = self.tensor(self.buffer.hidden())
            dist, v = self.evaluate(obs[t], masks[t], hidden[t] if self.privileged else None)
            a = dist.sample()
            actions[t], log_probs[t], values[t] = a, dist.log_prob(a), to_absolute(v, actors[t], self.players)
            chosen = a.cpu().numpy()
            bot_turn = self.bot_seats[np.arange(n), p]
            if bot_turn.any():
                chosen[bot_turn] = self.bot(o[bot_turn], m[bot_turn])
                actions[t] = self.tensor(chosen, np.int64)
                learn[t] = self.tensor(~bot_turn)
            stats["trades"] += int(TRADED[chosen].sum())
            before = self.buffer.points()
            endings = self.buffer.step(chosen)
            progress[t] = self.tensor((self.buffer.points().astype(np.float32) - before))
            rolls[t] = self.tensor(self.buffer.rolls().astype(np.float32))
            if self.average_dice:
                games, value = self.expected_after_roll()
                if value is not None:
                    rolled[t, games], expected[t, games] = True, value
            for game, winner, _ in endings:
                done[t, game] = True
                ends[t, game] = rewards(winner, self.players).to(at)
                if winner is None:
                    stats["truncated"] += 1
                else:
                    stats["won"] += 1
                    stats["first_won"] += winner == 0
                if self.bot_seats[game].any():
                    stats["mixed"] += 1
                    stats["mixed_network_won"] += winner is not None and not self.bot_seats[game, winner]

        o, m, p = self.buffer.observe()
        if self.privileged:
            _, last = self.evaluate(self.tensor(o), self.tensor(m), self.tensor(self.buffer.hidden()))
            last = to_absolute(last, self.tensor(p, np.int64), self.players)
        else:
            last = self.value(self.tensor(o), self.tensor(p, np.int64))
        future = upcoming(rolls, done)
        batch = dict(obs=obs, masks=masks, actions=actions, log_probs=log_probs, actors=actors, hidden=hidden, future=future,
                     learn=learn)
        if expert:
            batch["expert"] = tuple(torch.cat(part) for part in zip(*expert))
        shaped = self.vp_reward * relative(progress, self.players) * (~done)[..., None]
        return batch, values, (done, ends, rolled, expected, last, shaped), stats


def upcoming(rolls, done, decay=0.9):
    """For each decision, the dice rolled from it on in its game, nearer rolls counting more."""
    future = torch.zeros_like(rolls)
    following = torch.zeros_like(rolls[0])
    for t in reversed(range(rolls.shape[0])):
        following = rolls[t] + decay * following * (~done[t])[:, None]
        future[t] = following
    return future / 3


def sync(at):
    if at.type == "cuda":
        torch.cuda.synchronize()
    elif at.type == "mps":
        torch.mps.synchronize()


def relative(gains, players):
    """Each seat's gain minus the average gain of the other seats, so progress rewards sum to zero."""
    gains = gains[..., :players]
    others = (gains.sum(-1, keepdim=True) - gains) / (players - 1)
    out = torch.zeros(*gains.shape[:-1], SEATS, device=gains.device)
    out[..., :players] = gains - others
    return out


def advantages(values, done, ends, rolled, expected, last, shaped, lam):
    result = torch.zeros_like(values)
    following = torch.zeros_like(last)
    for t in reversed(range(values.shape[0])):
        next_values = last if t == values.shape[0] - 1 else values[t + 1]
        target = torch.where(rolled[t, :, None], expected[t], next_values)
        delta = torch.where(done[t, :, None], ends[t] - values[t], shaped[t] + target - values[t])
        following = torch.where(done[t, :, None], delta, delta + lam * following)
        result[t] = following
    return result


def update(model, critics, optimizer, batch, adv, returns, players, args, autocast, magnet_weight):
    flat = {k: v.flatten(0, 1) for k, v in batch.items() if k != "expert"}
    expert = batch.get("expert")
    actors = flat["actors"]
    own = adv.flatten(0, 1).gather(1, actors[:, None]).squeeze(1)
    targets = to_relative(returns.flatten(0, 1), actors, players)
    total = len(own)
    seats = slice(0, players)
    threshold = own.abs().float().quantile(1 - args.keep) if args.keep < 1 else -1.0
    informative = own.abs() >= threshold
    logs = {"policy": 0.0, "value": 0.0, "kl_magnet": 0.0, "approx_kl": 0.0, "clipped": 0.0, "expert": 0.0,
            "kept": informative.float().mean().item()}
    steps = 0
    for _ in range(args.epochs):
        for pick in torch.randperm(total, device=own.device).split(args.minibatch):
            obs, masks = flat["obs"][pick], flat["masks"][pick]
            symmetry = torch.randint(SYMMETRIES, (len(pick),), device=obs.device) if args.symmetry else None
            with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                logits, values, summary = model(obs, symmetry)
            dist, log_probs = policy(logits, masks)
            new = dist.log_prob(flat["actions"][pick])
            ratio = (new - flat["log_probs"][pick]).exp()
            target = targets[pick]
            value_loss = nn.functional.mse_loss(values.float()[:, seats], target[:, seats])
            a = own[pick]
            if args.privileged:
                value_loss = value_loss + nn.functional.mse_loss(critics.privileged(summary, flat["hidden"][pick])[:, seats], target[:, seats])
            if args.future_dice:
                baseline = critics.with_future(summary, flat["future"][pick])
                value_loss = value_loss + nn.functional.mse_loss(baseline[:, seats], target[:, seats])
                a = target[:, 0] - baseline[:, 0].detach()
            kept = informative[pick] & flat["learn"][pick]
            ratio, a = ratio[kept], a[kept]
            a = (a - a.mean()) / (a.std() + 1e-8)
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            magnet = kl_to_magnet(log_probs, masks)[flat["learn"][pick]].mean()
            loss = policy_loss + args.value_coef * value_loss + magnet_weight * magnet
            if expert is not None:
                e_obs, e_masks, e_target = expert
                chosen = torch.randint(len(e_obs), (max(len(pick) // 4, 1),), device=e_obs.device)
                e_symmetry = torch.randint(SYMMETRIES, (len(chosen),), device=e_obs.device) if args.symmetry else None
                with torch.autocast(e_obs.device.type, dtype=torch.bfloat16, enabled=autocast):
                    e_logits, _, _ = model(e_obs[chosen], e_symmetry)
                _, e_log_probs = policy(e_logits, e_masks[chosen])
                expert_loss = -(e_target[chosen] * e_log_probs).sum(-1).mean()
                loss = loss + args.expert_weight * expert_loss
                logs["expert"] += expert_loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *critics.parameters()], args.max_grad_norm)
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
    parser.add_argument("--entities", action="store_true",
                        help="one token per player, and trades scored from what they give and get")
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
    parser.add_argument("--magnet-anneal-steps", type=float, default=None,
                        help="shrink the magnet weight on its own schedule; defaults to --anneal-steps")
    parser.add_argument("--keep", type=float, default=1.0,
                        help="train the policy only on this fraction of moves, those with the largest advantage magnitude")
    parser.add_argument("--average-dice", action="store_true",
                        help="score a move followed by a roll against the average over all 11 rolls, not the one rolled")
    parser.add_argument("--symmetry", action="store_true", help="turn each training sample by a random board symmetry")
    parser.add_argument("--privileged", action="store_true",
                        help="score moves with a critic that also sees every hidden card (training only)")
    parser.add_argument("--future-dice", action="store_true",
                        help="score moves against a baseline that also knows the dice rolled next (ICLR 2019)")
    parser.add_argument("--expert", action="store_true",
                        help="expert iteration (Gumbel AlphaZero): search a few positions and also train the policy toward what search finds")
    parser.add_argument("--expert-games", type=int, default=128, help="positions searched each time")
    parser.add_argument("--expert-every", type=int, default=8, help="search every this many rounds")
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--expert-candidates", type=int, default=8, help="moves searched at each position")
    parser.add_argument("--expert-samples", type=int, default=8, help="deals of the hidden cards per move")
    parser.add_argument("--expert-depth", type=int, default=16, help="decisions played forward")
    parser.add_argument("--expert-scale", type=float, default=0.1, help="how far search may move the target from the network")
    parser.add_argument("--save-every", type=int, default=250, help="keep a numbered checkpoint every this many iterations")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bot-games", type=float, default=0.25,
                        help="share of training games with the hand-written bot in two seats; only the network's own moves are learned from")
    parser.add_argument("--max-decisions", type=int, default=3000, help="training games are cut off (a loss for all) after this many")
    parser.add_argument("--vp-fade", type=float, default=0,
                        help="fade the victory-point reward linearly to nothing over this many iterations, so later only winning counts; 0 keeps it")
    parser.add_argument("--vp-reward", type=float, default=0.03,
                        help="reward each victory point gained relative to the other players at once; 0 rewards winning alone")
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

    model = Tokens(args.width, args.depth, entities=args.entities).to(at)
    critics = Critics(args.width).to(at)
    optimizer = torch.optim.AdamW([*model.parameters(), *critics.parameters()], lr=args.lr)
    buffer = catan.Buffer(args.games, history=64, max_decisions=args.max_decisions, seed=args.seed, players=args.players)
    search = None
    if args.expert:
        search = Search(model, at, args.expert_candidates, args.expert_samples, args.expert_depth, seed=args.seed)
    rollout = Rollout(model, critics, buffer, at, autocast, args.players, args.average_dice, args.privileged,
                      search, args.expert_games, args.expert_every, args.expert_scale, args.seed, args.vp_reward, args.bot_games)
    log = open(out / "log.txt", "a")

    def say(line):
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {at}; "
        f"{args.games} games × {args.rounds} decisions = {args.games * args.rounds} per update")

    decisions, first, previous = 0, 1, None
    if (out / "latest.pt").exists():
        saved = torch.load(out / "latest.pt", map_location=at)
        model.load_state_dict(saved["model"])
        critics.load_state_dict(saved["critics"])
        if "optimizer" in saved:
            optimizer.load_state_dict(saved["optimizer"])
        first = saved["iteration"] + 1
        decisions = saved["iteration"] * args.games * args.rounds
        previous = copy.deepcopy(model).eval()
        say(f"resuming after iteration {saved['iteration']}")
    start = time.time()
    for iteration in range(first, args.iterations + 1):
        model.eval()
        clock = time.time()
        rollout.vp_reward = args.vp_reward * max(0.0, 1 - (iteration - 1) / args.vp_fade) if args.vp_fade else args.vp_reward
        batch, values, scoring, stats = rollout.collect(args.rounds)
        sync(at)
        collect_time, clock = time.time() - clock, time.time()
        adv = advantages(values, *scoring, args.lam)
        returns = adv + values
        scale = (1 + (iteration - 1) / args.anneal_steps) ** -args.anneal_power
        pull = (1 + (iteration - 1) / (args.magnet_anneal_steps or args.anneal_steps)) ** -args.anneal_power
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        model.train()
        logs = update(model, critics, optimizer, batch, adv, returns, args.players, args, autocast, args.magnet * pull)
        sync(at)
        update_time = time.time() - clock
        decisions += args.games * args.rounds

        finished = stats["won"] + stats["truncated"]
        line = (f"iter {iteration:4d}  {decisions / 1e6:6.2f}M decisions  {time.time() - start:6.0f}s  "
                f"games {finished:5d} ({stats['truncated']} cut)  first seat wins {stats['first_won'] / max(stats['won'], 1):.2f}  "
                f"trades per 1k decisions {1000 * stats['trades'] / (args.games * args.rounds):.1f}  "
                f"network seats win vs bots {stats['mixed_network_won'] / max(stats['mixed'], 1):.2f} (fair 0.50)  "
                f"value {logs['value']:.3f}  kl→magnet {logs['kl_magnet']:.3f}  approx_kl {logs['approx_kl']:.4f}  "
                f"clipped {logs['clipped']:.2f}  kept {logs['kept']:.2f}  lr {args.lr * scale:.1e}  "
                f"play {collect_time:.1f}s learn {update_time:.1f}s")
        if stats["searched"]:
            line += (f"  searched {stats['searched']} (search changed the top move {stats['changed'] / stats['searched']:.2f}, "
                     f"expert loss {logs['expert']:.3f})")
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
            checkpoint = {"model": model.state_dict(), "critics": critics.state_dict(), "optimizer": optimizer.state_dict(),
                          "args": vars(args), "iteration": iteration}
            torch.save(checkpoint, out / "latest.pt")
            if iteration % args.save_every == 0 or iteration == args.iterations:
                torch.save({k: v for k, v in checkpoint.items() if k != "optimizer"}, out / f"iter_{iteration:05d}.pt")
            previous = copy.deepcopy(model).eval()
            line += f"  (eval {time.time() - clock:.0f}s)"
        say(line)


if __name__ == "__main__":
    main()
