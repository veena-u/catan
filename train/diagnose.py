import argparse

import numpy as np
import torch

import catan

from .agents import Network
from .evaluate import rules_of
from .model import Model, device, widen
from .selfplay import Critics, Rollout, advantages, policy, to_relative

# Two measurements of where self-play training is stuck.
#
# noise: how much of each gradient step is signal (McCandlish et al. 2018, "An Empirical Model of Large-Batch
# Training"). From one batch of self-play, the gradient of many small minibatches and of the whole batch give the
# critical batch size B = tr(Σ) / |G|²: steps on far fewer samples than B are mostly noise.
#
# value: how well the network knows who is winning. From positions met in self-play, the game is played out many times
# with the same policy; the share won is the position's true value under that policy, against which the value head (and
# the critic that sees every hand) is scored, after taking out the noise of the playouts themselves.

def loaded(path, at):
    checkpoint = torch.load(path, map_location=at, weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    model = Model(args.width, args.depth, pooled=getattr(args, "pooled_kind", False)).to(at)
    model.load_state_dict(widen(checkpoint["model"], model)[0], strict=not getattr(args, "pooled_kind", False))
    critics = Critics(args.width).to(at)
    if "critics" in checkpoint:
        critics.load_state_dict(checkpoint["critics"], strict=False)
    return model.eval(), critics.eval(), args

def noise(path, at, games, warmup, batches, minibatch, seed):
    """Critical batch of the policy and value gradients, as trained and with each roll's, steal's and draw's luck taken
    out of the advantages (--luck), on the same batches and minibatches so the two differ only in the luck."""
    model, critics, args = loaded(path, at)
    privileged = bool(getattr(args, "privileged", False))
    buffer = catan.Buffer(games, history=64, max_decisions=args.max_decisions, seed=seed, players=args.players,
                          rules=rules_of(args))
    rollout = Rollout(model, critics, buffer, at, at.type == "cuda", args.players, privileged, seed,
                      explore_games=getattr(args, "opening_explore", 0.0), luck=True)
    players, seats = args.players, slice(0, args.players)
    lam, gamma = getattr(args, "lam", 0.95), getattr(args, "gamma", 1.0)
    for _ in range(warmup):
        rollout.collect(args.rounds)
    variants = ("as trained", "luck out")
    results = {(v, name): [] for v in variants for name in ("policy", "value")}
    spreads = {v: [] for v in variants}
    params = {"policy": list(model.parameters()), "value": [*model.parameters(), *critics.parameters()]}
    for _ in range(batches):
        batch, values, scoring, _ = rollout.collect(args.rounds)
        luck = batch.pop("luck")
        flat = {k: v.flatten(0, 1) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        actors = flat["actors"]
        total = len(actors) // minibatch * minibatch
        order = torch.randperm(len(actors), device=at)[:total].split(minibatch)
        for variant in variants:
            adv = advantages(values, *scoring, lam, gamma, luck=luck if variant == "luck out" else None)
            returns = adv + values
            own = adv.flatten(0, 1).gather(1, actors[:, None]).squeeze(1)
            spreads[variant].append(float(own.var()))
            own = (own - own.mean()) / (own.std() + 1e-8)
            targets = to_relative(returns.flatten(0, 1), actors, players)
            sums = {k: torch.zeros(sum(p.numel() for p in ps), device=at) for k, ps in params.items()}
            squares = {k: 0.0 for k in params}
            for pick in order:
                logits, v, summary = model(flat["obs"][pick], flat["masks"][pick])
                dist, _ = policy(logits, flat["masks"][pick])
                ratio = (dist.log_prob(flat["actions"][pick]) - flat["log_probs"][pick]).exp()
                losses = {
                    "policy": -(ratio * own[pick] * flat["learn"][pick].float()).mean(),
                    "value": torch.nn.functional.mse_loss(v.float()[:, seats], targets[pick][:, seats]),
                }
                if privileged:
                    losses["value"] = losses["value"] + torch.nn.functional.mse_loss(
                        critics.privileged(summary, flat["hidden"][pick])[:, seats], targets[pick][:, seats])
                for name, loss in losses.items():
                    grads = torch.autograd.grad(loss, params[name], retain_graph=name == "policy", allow_unused=True)
                    g = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(grads, params[name])])
                    sums[name] += g
                    squares[name] += float((g * g).sum())
            k, small, big = len(order), minibatch, total
            for name in params:
                whole = float((sums[name] / k).pow(2).sum())
                spread = (squares[name] / k - whole) / (1 / small - 1 / big)
                results[(variant, name)].append((spread, whole - spread / big))
    report = {}
    for (variant, name), pairs in results.items():
        spread, signal = (np.mean([p[i] for p in pairs]) for i in (0, 1))
        # With no signal left after taking out the whole batch's own noise, the batch is too small to tell.
        report[f"{variant}: {name}"] = f"{spread / signal:,.0f}" if signal > 0 else f"more than {total:,}"
    for variant in variants:
        report[f"{variant}: advantage variance"] = f"{np.mean(spreads[variant]):.4f}"
    return report

@torch.no_grad()
def value(path, at, positions, playouts, seed):
    model, critics, args = loaded(path, at)
    rules = rules_of(args)
    rng = np.random.default_rng(seed)
    agent = Network(model, at, seed=seed)
    games = [catan.Game(seed * 100_003 + i, args.players, rules=rules) for i in range(positions)]
    for g in games:
        g.settle()
    # Each position is met after a random number of the game's decisions, as positions are met in training.
    when, taken = rng.integers(0, 240, positions), np.zeros(positions, int)
    buffer = catan.Buffer(positions * playouts, history=64, max_decisions=args.max_decisions, seed=seed + 1,
                          players=args.players, rules=rules)
    copied = np.zeros(positions, bool)
    while True:
        live = [i for i in range(positions) if not copied[i] and games[i].winner() is None]
        for i in [i for i in live if taken[i] >= when[i]]:
            for k in range(playouts):
                buffer.load(i * playouts + k, games[i])
            copied[i] = True
        live = [i for i in live if not copied[i]]
        if not live:
            break
        obs, masks = map(np.stack, zip(*(games[i].observe() for i in live)))
        actions, _ = agent(obs, masks, np.zeros((len(live), 1, 5), np.int8), np.zeros(len(live), np.uint8))
        for i, a in zip(live, actions):
            games[i].step(int(a))
            games[i].settle()
            taken[i] += 1
    obs, masks, actors = buffer.observe()
    hidden = buffer.hidden()
    first = np.arange(positions) * playouts
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(at)
    _, v, summary = model(tensor(obs[first]), None)
    plain = (v.float()[:, 0].cpu().numpy() + 1) / 2
    seen = (critics.privileged(summary, tensor(hidden[first]))[:, 0].cpu().numpy() + 1) / 2
    mover = actors[first]
    wins, ended = np.zeros(positions * playouts), np.zeros(positions * playouts, bool)
    forbidden = np.zeros((len(obs), 1, 5), np.int8)
    while not ended[np.repeat(copied, playouts)].all():
        obs, masks, _ = buffer.observe()
        actions, offers = agent(obs, masks, forbidden, np.zeros(len(obs), np.uint8))
        for slot, winner, _ in buffer.step(actions, offers):
            if not ended[slot]:
                ended[slot] = True
                wins[slot] = winner is not None and winner == mover[slot // playouts]
    keep = copied
    share = wins.reshape(positions, playouts).mean(1)[keep]
    playout_noise = (share * (1 - share) / (playouts - 1)).mean()
    spread = share.var() - playout_noise
    stage = taken[keep]
    out = {"positions": int(keep.sum()), "playouts": playouts, "true spread (sd)": spread ** 0.5}
    for name, guess in (("value head", plain[keep]), ("critic seeing every hand", seen[keep])):
        error = ((guess - share) ** 2).mean() - playout_noise
        out[name] = {"error (rms)": max(error, 0) ** 0.5, "explained": 1 - error / spread}
        for label, rows in (("early", stage < 80), ("middle", (stage >= 80) & (stage < 160)), ("late", stage >= 160)):
            e = ((guess[rows] - share[rows]) ** 2).mean() - (share[rows] * (1 - share[rows]) / (playouts - 1)).mean()
            out[name][f"error {label} (rms)"] = max(e, 0) ** 0.5
    return out

def main():
    parser = argparse.ArgumentParser(description="Measure where self-play training is stuck: gradient noise and value accuracy.")
    parser.add_argument("test", choices=["noise", "value"])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--games", type=int, default=2048, help="noise: games in the batch, as in training")
    parser.add_argument("--warmup", type=int, default=6, help="noise: batches played first, so games are at every stage")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--playouts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    at = device()
    for path in args.checkpoints:
        if args.test == "noise":
            report = noise(path, at, args.games, args.warmup, args.batches, args.minibatch, args.seed)
            print(f"{path}: critical batch (samples per step where signal matches noise), and advantage variance", flush=True)
            for k, v in report.items():
                print(f"  {k:32s} {v}", flush=True)
        else:
            print(f"{path}: {value(path, at, args.positions, args.playouts, args.seed)}", flush=True)

if __name__ == "__main__":
    main()
