import argparse
import collections

import numpy as np
import torch

import catan

from .agents import Network, Random
from .insight import card_values, forecast
from .model import Tokens, device, load
from .search import Search
from .selfplay import SEATS, advantages, improved_policy, relative, rewards, to_absolute, to_relative


def check_learning_math():
    """Hand-made cases with known answers for the pieces that turn games into training signal."""
    assert torch.allclose(rewards(2, 4), torch.tensor([-1 / 3, -1 / 3, 1.0, -1 / 3])), "a win: +1 and -1/3 each"
    assert torch.allclose(rewards(None, 4), torch.full((SEATS,), -1 / 3)), "a cut-off game is a loss for all"
    assert torch.allclose(rewards(1, 3), torch.tensor([-0.5, 1.0, -0.5, 0.0])), "absent seats score nothing"

    gains = torch.tensor([[[2.0, 0, 1, 0]]])
    shares = relative(gains, 4)
    assert torch.allclose(shares.sum(-1), torch.zeros(1, 1), atol=1e-6), "progress rewards sum to zero"
    assert torch.allclose(shares, torch.tensor([[[2 - 1 / 3, -1.0, 1 - 2 / 3, -1.0]]]))

    rows = torch.randn(5, SEATS)
    for players in (2, 3, 4):
        actors = torch.randint(players, (5,))
        back = to_absolute(to_relative(rows, actors, players), actors, players)
        assert torch.allclose(back[:, :players], rows[:, :players]), "seat conversions undo each other"
    relative_values = torch.tensor([[0.9, 0.1, 0.2, 0.3]])
    assert to_absolute(relative_values, torch.tensor([2]), 4)[0, 2] == 0.9, "the player to move sits at its own seat"

    rounds, lam = 4, 0.5
    values = torch.zeros(rounds, 1, SEATS)
    done = torch.zeros(rounds, 1, dtype=torch.bool)
    done[2] = True
    ends = torch.zeros(rounds, 1, SEATS)
    ends[2, 0] = rewards(0, 4)
    rolled, expected = torch.zeros(rounds, 1, dtype=torch.bool), torch.zeros(rounds, 1, SEATS)
    last, shaped = torch.zeros(1, SEATS), torch.zeros(rounds, 1, SEATS)
    shaped[3, 0, 1] = 0.25
    adv = advantages(values, done, ends, rolled, expected, last, shaped, lam)
    assert torch.allclose(adv[2, 0], rewards(0, 4)), "the final move is credited with the result"
    assert torch.allclose(adv[1, 0], lam * rewards(0, 4)) and torch.allclose(adv[0, 0], lam ** 2 * rewards(0, 4)), \
        "credit flows back, fading by lambda"
    assert adv[3, 0, 1] == 0.25 and adv[3, 0, 0] == 0, "a new game starts clean, and progress counts at once"
    print("learning math: all checks passed")


def check_search_training():
    """The search target, sequential halving, and the planning tools, on a small untrained network."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0, float("-inf")]])
    prior = logits.log_softmax(1)
    choices, value, visits = torch.tensor([[0, 2]]), torch.tensor([0.1]), torch.tensor([[8.0, 8.0]])
    same = improved_policy(prior, choices, torch.tensor([[0.1, 0.1]]), value, visits, 0.1)
    assert torch.allclose(same, prior.exp(), atol=1e-6), "when search finds nothing new, the target is the network itself"
    better = improved_policy(prior, choices, torch.tensor([[0.1, 0.4]]), value, visits, 0.1)
    assert better[0, 2] > prior.exp()[0, 2] and better[0, 0] < prior.exp()[0, 0], "a move search likes more gains"
    assert torch.isclose(better[0, 1] / better[0, 3], prior.exp()[0, 1] / prior.exp()[0, 3]), "unsearched moves keep their ratio"
    assert better[0, 4] == 0, "an illegal move stays impossible"

    torch.manual_seed(0)
    at = device()
    model = Tokens(32, 1, entities=True).to(at).eval()
    buffer, agent = catan.Buffer(16, seed=3), Random(0, kinds=True)
    for _ in range(300):
        obs, masks, _ = buffer.observe()
        buffer.step(agent(obs, masks))
    obs, masks, actors = buffer.observe()
    rows = np.flatnonzero(masks.sum(1) >= 8)
    assert len(rows), "some position with at least 8 moves"
    search = Search(model, at, candidates=8, samples=2, depth=4)
    choices, q, prior, value, visits = search.improve(obs[rows], masks[rows], buffer, rows, actors[rows], gumbel=True, rounds=3)
    for row in visits:
        assert sorted(row.tolist(), reverse=True) == [6, 6, 4, 4, 2, 2, 2, 2], f"halving 8 -> 4 -> 2: {row.tolist()}"
    assert torch.isfinite(q).all() and masks[rows][np.arange(len(rows))[:, None], choices.numpy()].all()

    game = catan.Game(5, 4)
    game.settle()
    for _ in range(150):
        obs, mask = game.observe()
        game.step(int(agent(obs[None], mask[None])[0]))
        game.settle()
    worth = card_values(model, at, game.observe()[0])
    assert len(worth) == 5 and all(np.isfinite(worth))
    lines = forecast(model, at, game.record(), samples=8, limit=120)
    assert lines[0].startswith("over its next"), lines
    print("search training and planning tools: all checks passed")


@torch.no_grad()
def check_values(path, games, positions, seed=0, steps=20_000):
    """Does the network's value head pick the eventual winner more often than chance, and more so near the end?"""
    at = device()
    model = load(path, at)
    agent = Network(model, at, seed=seed)
    buffer = catan.Buffer(games, seed=seed, players=4)
    recent = [collections.deque(maxlen=50) for _ in range(games)]
    early, late, done, cut = [], [], 0, 0
    for _ in range(steps):
        if done >= positions:
            break
        obs, masks, actors = buffer.observe()
        _, values, _ = model(torch.from_numpy(obs).to(at))
        absolute = to_absolute(values.float().cpu(), torch.from_numpy(actors.astype(np.int64)), 4).numpy()
        for g in range(games):
            recent[g].append(absolute[g])
        for game, winner, _ in buffer.step(agent(obs, masks)):
            cut += winner is None
            if winner is not None and len(recent[game]) == recent[game].maxlen:
                early.append(int(np.argmax(recent[game][0]) == winner))
                late.append(int(np.argmax(recent[game][-1]) == winner))
                done += 1
            recent[game].clear()
    if not done:
        print(f"value check: no game was won in {steps} steps ({cut} cut off); the network may be stalling")
        return
    print(f"value head picks the winner: 50 decisions before the end {np.mean(early):.2f}, "
          f"at the last decision {np.mean(late):.2f}  (chance 0.25, {done} games won, {cut} cut off)")


def main():
    parser = argparse.ArgumentParser(description="Quick checks that training turns games into the right signal.")
    parser.add_argument("checkpoint", nargs="?", help="also check this network's value head against real outcomes")
    parser.add_argument("--games", type=int, default=256, help="games played side by side for the value check")
    parser.add_argument("--positions", type=int, default=300, help="finished games to score")
    args = parser.parse_args()
    check_learning_math()
    check_search_training()
    if args.checkpoint:
        check_values(args.checkpoint, args.games, args.positions)


if __name__ == "__main__":
    main()
