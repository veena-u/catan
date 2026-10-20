import argparse
import time

import numpy as np
import torch
from torch import nn

import catan

from .agents import Random
from .features import decode
from .model import Flat, Tokens, device

HEX_CORNERS, EDGE_ENDS = (torch.from_numpy(a.astype(np.int64)) for a in catan.topology())
TERRAIN_IN_BOX = [3, 4, 4, 4, 3, 1]
SUMS = 11


def positions(count, games=4096, every=40, seed=0):
    buffer, agent = catan.Buffer(games, seed=seed), Random(seed, kinds=True)
    rng = np.random.default_rng(seed)
    seen, legal, kept = [], [], 0
    while kept < count:
        for _ in range(rng.integers(1, 2 * every)):
            obs, masks, _ = buffer.observe()
            buffer.step(agent(obs, masks))
        obs, masks, _ = buffer.observe()
        seen.append(obs)
        legal.append(masks)
        kept += len(obs)
    obs, masks = np.concatenate(seen)[:count], np.concatenate(legal)[:count]
    return torch.from_numpy(obs), torch.from_numpy(masks)


def production(obs):
    hexes, vertices, _, _ = decode(obs)
    weight = vertices[..., [0, 1]] + 2 * vertices[..., [2, 3]]
    per_hex = weight[:, HEX_CORNERS.to(obs.device)].sum(2).transpose(1, 2)
    terrain, tokens, robber = hexes[..., :6], hexes[..., 6:19], hexes[..., 20]
    producing = (1 - robber) * (1 - terrain[..., 5])
    return torch.einsum("bph,bhd,bhr->bpdr", per_hex * producing.unsqueeze(1), tokens[..., 2:], terrain[..., :5])


def check(obs, masks):
    hexes, vertices, edges, _ = decode(obs)
    assert (hexes[..., :6].sum(1) == torch.tensor(TERRAIN_IN_BOX, dtype=torch.float)).all(), "terrain"
    assert (hexes[..., 20].sum(1) == 1).all(), "one robber"
    assert (vertices[..., 4:11].argmax(-1).gt(0).sum(1) == 18).all(), "18 harbor vertices"
    assert (vertices[..., :4].sum(-1) <= 1).all(), "one building per vertex"
    assert (edges[..., :2].sum(-1) <= 1).all(), "one road per edge"
    built = vertices[..., :4].sum(-1) > 0
    neighbours = torch.zeros_like(built)
    for a, b in EDGE_ENDS.tolist():
        neighbours[:, a] |= built[:, b]
        neighbours[:, b] |= built[:, a]
    setup = obs[:, 205] == 0
    expected = ~built & ~neighbours
    assert (masks[setup, :54] == expected[setup]).all(), "setup settlement spots follow the distance rule"
    return int(setup.sum())


class Quiz(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.production = nn.Linear(model.width, 2 * SUMS * 5)

    def forward(self, obs):
        logits, _, summary = self.model(obs)
        return logits, self.production(summary).view(-1, 2, SUMS, 5)


@torch.no_grad()
def score(quiz, obs, masks, at, batch=2048):
    right = exact = settle_right = settle_total = produce_exact = produce_error = 0
    for i in range(0, len(obs), batch):
        o, m = obs[i:i + batch].to(at), masks[i:i + batch].to(at)
        logits, produced = quiz(o)
        guess = logits > 0
        right += (guess == m).float().mean(1).sum().item()
        exact += (guess == m).all(1).sum().item()
        settling = m[:, :54].any(1)
        settle_right += (guess[settling, :54] == m[settling, :54]).all(1).sum().item()
        settle_total += settling.sum().item()
        target = production(o)
        produce_exact += (produced.round() == target).flatten(1).all(1).sum().item()
        produce_error += (produced - target).abs().mean((1, 2, 3)).sum().item()
    n = len(obs)
    return {
        "action accuracy": right / n,
        "whole mask exact": exact / n,
        "settlement spots exact": settle_right / max(settle_total, 1),
        "production exact": produce_exact / n,
        "production mean error": produce_error / n,
    }


def train(name, model, data, test, steps, lr, batch, at):
    quiz = Quiz(model).to(at)
    optimizer = torch.optim.AdamW(quiz.parameters(), lr=lr)
    obs, masks = data
    parameters = sum(p.numel() for p in quiz.parameters())
    print(f"\n{name}: {parameters / 1e6:.2f}M parameters")
    start = time.time()
    for step in range(1, steps + 1):
        pick = torch.randint(len(obs), (batch,))
        o, m = obs[pick].to(at), masks[pick].to(at)
        logits, produced = quiz(o)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, m.float())
        loss = loss + nn.functional.mse_loss(produced, production(o))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % (steps // 5) == 0 or step == steps:
            result = score(quiz, *test, at)
            shown = "  ".join(f"{k} {v:.3f}" for k, v in result.items())
            print(f"  step {step:5d}  loss {loss.item():.4f}  {time.time() - start:5.0f}s  {shown}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Can each network learn facts it will need, with the answers given?")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--train-positions", type=int, default=200_000)
    parser.add_argument("--models", default="flat,tokens")
    parser.add_argument("--width", type=int, default=128, help="token network width")
    parser.add_argument("--depth", type=int, default=4, help="token network layers")
    args = parser.parse_args()
    at = device()
    torch.manual_seed(0)

    data = positions(args.train_positions, seed=1)
    test = positions(16_384, seed=2)
    checked = check(*data) + check(*test)
    print(f"{len(data[0])} training and {len(test[0])} held-out positions; decoder checks passed "
          f"(distance rule matched on {checked} setup positions)")
    illegal = 1 - test[1].float().mean().item()
    target = production(test[0])
    average = (target - production(data[0]).mean(0)).abs().mean().item()
    print(f"for scale: always guessing 'illegal' scores action accuracy {illegal:.3f}; guessing the average production "
          f"for every position scores production mean error {average:.3f} and production exact 0.000")

    choices = {"flat": (Flat, 1e-3), "tokens": (lambda: Tokens(args.width, args.depth), 3e-4)}
    for name in args.models.split(","):
        build, lr = choices[name]
        train(name, build(), data, test, args.steps, lr, args.batch, at)


if __name__ == "__main__":
    main()
