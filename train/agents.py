import numpy as np
import torch

from .model import masked

KINDS = np.array([0] * 54 + [1] * 54 + [2] * 72 + [3] * 19 + [4] * 5 + [5] * 60 + list(range(6, 13)))
KIND_COUNT = 13
_ONE_HOT = np.eye(KIND_COUNT)[KINDS]


def sample(probabilities, rng):
    cumulative = probabilities.cumsum(1)
    draws = rng.random((len(probabilities), 1)) * cumulative[:, -1:]
    return (cumulative < draws).sum(1)


def uniform(masks):
    return masks / masks.sum(1, keepdims=True)


def by_kind(masks):
    counts = masks @ _ONE_HOT
    kinds = (counts > 0).sum(1, keepdims=True)
    per_action = np.take_along_axis(counts, np.broadcast_to(KINDS, masks.shape), 1)
    return np.where(masks, 1 / (kinds * np.maximum(per_action, 1)), 0)


class Random:
    def __init__(self, seed=0, kinds=False):
        self.rng = np.random.default_rng(seed)
        self.kinds = kinds

    def __call__(self, obs, masks):
        return sample(by_kind(masks) if self.kinds else uniform(masks), self.rng)


class Network:
    def __init__(self, model, device, greedy=False, seed=0):
        self.model, self.device, self.greedy = model, device, greedy
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    @torch.no_grad()
    def __call__(self, obs, masks):
        logits, _, _ = self.model(torch.from_numpy(obs).to(self.device))
        logits = masked(logits, torch.from_numpy(masks).to(self.device)).float().cpu()
        if self.greedy:
            return logits.argmax(1).numpy()
        return torch.multinomial(logits.softmax(1), 1, generator=self.generator).squeeze(1).numpy()
