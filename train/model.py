import math

import torch
from torch import nn

import catan

from .features import (
    EDGE_FEATURES, EDGES, ENTITY_RELATIONS, FLAT_FEATURES, GLOBAL_FEATURES, HEX_FEATURES, HEXES, PLAYER_FEATURES,
    RELATIONS, SEATS, TOKENS, VERTEX_FEATURES, VERTICES, decode, entity_relations, flat, players, relations,
    symmetry_tables,
)

ACTIONS = catan.ACTIONS
OTHER = ACTIONS - 2 * VERTICES - EDGES - HEXES
FIRST_DEAL = 274 - (2 * VERTICES + EDGES + HEXES)


class Flat(nn.Module):
    def __init__(self, width=512, depth=3):
        super().__init__()
        layers, size = [], FLAT_FEATURES
        for _ in range(depth):
            layers += [nn.Linear(size, width), nn.ReLU()]
            size = width
        self.trunk = nn.Sequential(*layers)
        self.policy = nn.Linear(width, ACTIONS)
        self.value = nn.Linear(width, SEATS)
        self.width = width

    def forward(self, obs, symmetry=None):
        summary = self.trunk(flat(obs, symmetry))
        return untwist(self.policy(summary), symmetry), torch.tanh(self.value(summary)), summary


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.attend_norm, self.feed_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.feed = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, bias):
        b, t, d = x.shape
        q, k, v = self.qkv(self.attend_norm(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attended = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(attended.transpose(1, 2).reshape(b, t, d))
        return x + self.feed(self.feed_norm(x))


class Tokens(nn.Module):
    """A transformer over the board's hexes, corners and edges plus one summary token. With `entities`, also one
    token per player, and trades scored from what they give and get rather than each learned separately."""

    def __init__(self, width=128, depth=4, heads=4, entities=False):
        super().__init__()
        self.entities = entities
        self.embed = nn.ModuleList(nn.Linear(n, width) for n in (GLOBAL_FEATURES, HEX_FEATURES, VERTEX_FEATURES, EDGE_FEATURES))
        self.position = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.register_buffer("relations", entity_relations() if entities else relations(), persistent=False)
        self.relation_bias = nn.Parameter(torch.zeros(depth, heads, ENTITY_RELATIONS if entities else RELATIONS))
        with torch.no_grad():
            self.relation_bias[:, :, 0] = -2.0
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.queries = nn.Linear(width, 4 * width)
        self.keys = nn.Linear(width, width)
        self.other = nn.Linear(width, OTHER - catan.DEALS if entities else OTHER)
        self.value = nn.Linear(width, SEATS)
        self.width = width
        if entities:
            self.embed_player = nn.Linear(PLAYER_FEATURES, width)
            self.seat = nn.Parameter(torch.randn(SEATS, width) * 0.02)
            self.register_buffer("deals", torch.from_numpy(catan.deals()).float() / 2, persistent=False)
            self.embed_deal = nn.Sequential(nn.Linear(10, width), nn.GELU(), nn.Linear(width, width))
            self.deal_query = nn.Linear(width, width)

    def forward(self, obs, symmetry=None):
        hexes, vertices, edges, globals_ = decode(obs, symmetry)
        board = torch.cat([self.embed[1](hexes), self.embed[2](vertices), self.embed[3](edges)], 1) + self.position[1:]
        parts = [(self.embed[0](globals_) + self.position[0]).unsqueeze(1)]
        if self.entities:
            parts.append(self.embed_player(players(obs)) + self.seat)
        x = torch.cat([*parts, board], 1)
        for block, bias in zip(self.blocks, self.relation_bias):
            x = block(x, bias[:, self.relations].unsqueeze(0))
        x = self.norm(x)
        summary = x[:, 0]
        keys = self.keys(x[:, -(HEXES + VERTICES + EDGES):])
        hex_keys, vertex_keys, edge_keys = keys.split([HEXES, VERTICES, EDGES], 1)
        settle, city, road, robber = self.queries(summary).view(-1, 4, self.width).unbind(1)
        score = lambda k, q: torch.einsum("bnd,bd->bn", k, q) / math.sqrt(self.width)
        other = self.other(summary)
        if self.entities:
            deals = self.embed_deal(self.deals)
            dealt = torch.einsum("nd,bd->bn", deals, self.deal_query(summary)) / math.sqrt(self.width)
            other = torch.cat([other[:, :FIRST_DEAL], dealt, other[:, FIRST_DEAL:]], -1)
        logits = torch.cat([
            score(vertex_keys, settle),
            score(vertex_keys, city),
            score(edge_keys, road),
            score(hex_keys, robber),
            other,
        ], -1)
        return untwist(logits, symmetry), torch.tanh(self.value(summary)), summary


def untwist(logits, symmetry):
    if symmetry is None:
        return logits
    return logits.gather(1, symmetry_tables(logits.device)[3][symmetry])


def masked(logits, mask):
    return logits.masked_fill(~mask, float("-inf"))


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(path, at):
    checkpoint = torch.load(path, map_location=at)
    args = checkpoint["args"]
    model = Tokens(args["width"], args["depth"], entities=args.get("entities", False)).to(at)
    model.load_state_dict(checkpoint["model"])
    return model.eval()
