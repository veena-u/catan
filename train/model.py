import math

import numpy as np
import torch
from torch import nn

import catan

from .deals import COUNT_OFFSET, STOP, replay, step_log_probs
from .features import (
    EDGE_FEATURES, EDGES, GLOBAL_FEATURES, HEX_FEATURES, HEXES, PLAYER_FEATURES, RESOURCE_FEATURES, RESOURCES, SEATS,
    TOKEN_RELATIONS, TOKENS, VERTEX_FEATURES, VERTICES, decode, players, resources, symmetry_tables, token_relations,
)

ACTIONS = catan.ACTIONS
UNLIKELY = -1e9

KIND_RUNS = [
    ("settlement", 0, VERTICES, "corner"),
    ("city", 54, VERTICES, "corner"),
    ("road", 108, EDGES, "edge"),
    ("robber", 180, HEXES, "hex"),
    ("pick resource", 199, RESOURCES, "resource"),
    ("bank trade", 204, 20, "resource pair"),
    ("roll", 224, 1, None),
    ("end turn", 225, 1, None),
    ("buy card", 226, 1, None),
    ("play knight", 227, 1, None),
    ("play road building", 228, 1, None),
    ("play year of plenty", 229, 1, None),
    ("play monopoly", 230, 1, None),
    ("rob", 231, SEATS - 1, "player"),
    ("offer", 234, 1, None),
    ("give", 235, RESOURCES, "resource"),
    ("ask", 240, RESOURCES, "resource"),
    ("send", 245, 1, None),
    ("abandon", 246, 1, None),
    ("accept", 247, 1, None),
    ("decline", 248, 1, None),
    ("counter", 249, 1, None),
    ("trade with", 250, SEATS - 1, "player"),
    ("take counter", 253, SEATS - 1, "player"),
    ("cancel", 256, 1, None),
]
KIND_COUNT = len(KIND_RUNS)
KINDS = np.concatenate([np.full(count, k) for k, (_, _, count, _) in enumerate(KIND_RUNS)])
assert len(KINDS) == ACTIONS and all(start == sum(c for _, _, c, _ in KIND_RUNS[:k]) for k, (_, start, _, _) in enumerate(KIND_RUNS))
POINTED = [k for k, run in enumerate(KIND_RUNS) if run[3] is not None]

_BANK = [(give, get) for give in range(RESOURCES) for get in range(RESOURCES) if get != give]

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

class Model(nn.Module):

    def __init__(self, width=128, depth=4, heads=4, pooled=False):
        super().__init__()
        self.width = width
        # Optionally the kind of move also sees every group of rows pooled (mean and max), as AlphaStar's action type
        # sees its pooled units, not the summary row alone. It starts at zero, so a network gaining it plays as before.
        self.pooled = nn.Linear(2 * 5 * width, KIND_COUNT) if pooled else None
        if pooled:
            nn.init.zeros_(self.pooled.weight)
            nn.init.zeros_(self.pooled.bias)
        self.embed_global = nn.Linear(GLOBAL_FEATURES, width)
        self.embed_hex, self.embed_vertex, self.embed_edge = (nn.Linear(n, width) for n in (HEX_FEATURES, VERTEX_FEATURES, EDGE_FEATURES))
        self.embed_player, self.embed_resource = nn.Linear(PLAYER_FEATURES, width), nn.Linear(RESOURCE_FEATURES, width)
        self.position = nn.Parameter(torch.randn(TOKENS, width) * 0.02)
        self.seat = nn.Parameter(torch.randn(SEATS, width) * 0.02)
        self.register_buffer("relations", token_relations(), persistent=False)
        self.relation_bias = nn.Parameter(torch.zeros(depth, heads, TOKEN_RELATIONS))
        with torch.no_grad():
            self.relation_bias[:, :, 0] = -2.0
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(width)
        self.kind = nn.Linear(width, KIND_COUNT)
        questions = len(POINTED) + 1
        self.questions = nn.Linear(width, questions * width)
        self.gates = nn.Parameter(torch.zeros(questions, width))
        self.keys = nn.Linear(width, width)
        self.value = nn.Linear(width, SEATS)
        self.deal_query = nn.Linear(width, 2 * width)
        self.count_embed = nn.Parameter(torch.randn(RESOURCES, 2 * COUNT_OFFSET + 1, width) * 0.02)
        self.stop_hidden, self.stop_out = nn.Linear(width, width), nn.Linear(width, 1)
        self.card_hidden, self.card_out = nn.Linear(2 * width, width), nn.Linear(width, 2)
        self.register_buffer("kind_of", torch.from_numpy(KINDS), persistent=False)

    def forward(self, obs, masks=None, symmetry=None, with_goods=False):
        hexes, vertices, edges, globals_ = decode(obs, symmetry)
        board = torch.cat([self.embed_hex(hexes), self.embed_vertex(vertices), self.embed_edge(edges)], 1) + self.position[1:]
        summary = (self.embed_global(globals_) + self.position[0]).unsqueeze(1)
        people = self.embed_player(players(obs)) + self.seat
        goods = self.embed_resource(resources(obs))
        x = torch.cat([summary, people, goods, board], 1)
        for block, bias in zip(self.blocks, self.relation_bias):
            x = block(x, bias[:, self.relations].unsqueeze(0).to(x.dtype))
        x = self.norm(x)
        summary = x[:, 0]
        values = torch.tanh(self.value(summary))
        log_probs = None if masks is None else self.policy(x, summary, masks, symmetry)
        if with_goods:
            return log_probs, values, summary, x[:, 1 + SEATS:1 + SEATS + RESOURCES]
        return log_probs, values, summary

    def policy(self, x, summary, masks, symmetry):
        n, d = len(x), self.width
        people, goods = x[:, 1:1 + SEATS], x[:, 1 + SEATS:1 + SEATS + RESOURCES]
        board = x[:, 1 + SEATS + RESOURCES:]
        hexes, corners, edges = board.split([HEXES, VERTICES, EDGES], 1)
        targets = {"corner": corners, "edge": edges, "hex": hexes, "resource": goods, "player": people[:, 1:]}
        keys = {name: self.keys(rows) for name, rows in targets.items()}
        questions = self.questions(summary).view(n, -1, d)
        gates = torch.sigmoid(self.gates)
        point = lambda q, target: torch.einsum("ntd,nd->nt", keys[target] * gates[q], questions[:, q]) / math.sqrt(d)
        scores = torch.zeros(n, ACTIONS, device=x.device, dtype=summary.dtype)
        for q, k in enumerate(POINTED):
            _, start, count, target = KIND_RUNS[k]
            if target == "resource pair":
                give, get = point(q, "resource"), point(len(POINTED), "resource")
                pairs = torch.tensor(_BANK, device=x.device)
                scores[:, start:start + count] = give[:, pairs[:, 0]] + get[:, pairs[:, 1]]
            else:
                scores[:, start:start + count] = point(q, target)
        scores = untwist(scores, symmetry)
        kind_logits = self.kind(summary)
        if self.pooled is not None:
            groups = (people, goods, hexes, corners, edges)
            kind_logits = kind_logits + self.pooled(torch.cat([f(g, 1) for g in groups for f in (lambda g, d: g.mean(d), lambda g, d: g.amax(d))], -1))
        return factored(kind_logits, scores, masks, self.kind_of)

    def deal_scores(self, summary, goods, counter, counts):
        n, d = len(summary), self.width
        query = self.deal_query(summary).view(n, 2, d)[torch.arange(n, device=summary.device), counter.long()]
        shape = counts.shape[:-1]
        query = query.view(n, *([1] * (len(shape) - 1)), d)
        h = query + self.count_embed[torch.arange(RESOURCES, device=counts.device), counts + COUNT_OFFSET].sum(-2)
        stop = self.stop_out(nn.functional.gelu(self.stop_hidden(h))).squeeze(-1)
        rows = goods.view(n, *([1] * (len(shape) - 1)), RESOURCES, d).expand(*shape, RESOURCES, d)
        both = torch.cat([h.unsqueeze(-2).expand(*shape, RESOURCES, d), rows], -1)
        cards = self.card_out(nn.functional.gelu(self.card_hidden(both)))
        return stop, torch.cat([cards[..., 0], cards[..., 1]], -1)

    def deal_log_probs(self, summary, goods, counter, choices, masks):
        stop, cards = self.deal_scores(summary, goods, counter, replay(choices))
        n, steps = choices.shape
        log_probs = step_log_probs(stop.reshape(-1), cards.reshape(n * steps, -1), masks.reshape(n * steps, -1)).view(n, steps, -1)
        taken = choices >= 0
        chosen = log_probs.gather(-1, choices.clamp(min=STOP)[..., None]).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        steps_taken = taken.sum(1).clamp(min=1)
        return torch.where(taken, chosen, 0.0).sum(1), torch.where(taken, entropy, 0.0).sum(1) / steps_taken

KIND_MEMBERS = torch.from_numpy(np.eye(KIND_COUNT, dtype=np.float32)[KINDS])

def factored(kind_logits, scores, masks, kind_of):
    kind_logits, scores = kind_logits.float(), scores.float().masked_fill(~masks, UNLIKELY)
    kind_legal = masks.float() @ KIND_MEMBERS.to(masks.device) > 0
    kind_log = kind_logits.masked_fill(~kind_legal, UNLIKELY).log_softmax(-1)
    within = torch.empty_like(scores)
    for _, start, count, _ in KIND_RUNS:
        run = scores[:, start:start + count]
        within[:, start:start + count] = run - run.logsumexp(1, keepdim=True)
    log_probs = kind_log.gather(1, kind_of.expand_as(masks)) + within
    return log_probs.masked_fill(~masks, UNLIKELY)

def untwist(scores, symmetry):
    if symmetry is None:
        return scores
    return scores.gather(1, symmetry_tables(scores.device)[3][symmetry])

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
    model = Model(args["width"], args["depth"], pooled=args.get("pooled_kind", False)).to(at)
    model.load_state_dict(checkpoint["model"], strict=not args.get("pooled_kind", False))
    return model.eval()

def load_critics(path, at):
    from .selfplay import Critics
    checkpoint = torch.load(path, map_location=at)
    if "critics" not in checkpoint or not checkpoint["args"].get("privileged"):
        raise ValueError(f"{path} was not trained with --privileged, so it has no critic that sees every hand")
    critics = Critics(checkpoint["args"]["width"]).to(at)
    critics.load_state_dict(checkpoint["critics"], strict=False)
    return critics.eval()
