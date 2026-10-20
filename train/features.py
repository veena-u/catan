import numpy as np
import torch

import catan

HEXES, VERTICES, EDGES = 19, 54, 72
TOKENS = 1 + HEXES + VERTICES + EDGES
HEX_FEATURES = 6 + 13 + 1 + 1 + 4
VERTEX_FEATURES = 4 + 7 + 1 + 5 + 4 + 2
EDGE_FEATURES = 2 + 4 + 2
GLOBAL_FEATURES = 44 + 3 + 3 + 2 + 12 + 1 + 1 + 3 + 6
FLAT_FEATURES = HEXES * HEX_FEATURES + VERTICES * VERTEX_FEATURES + EDGES * EDGE_FEATURES + GLOBAL_FEATURES

NONE = 255

HEX_CORNERS, EDGE_ENDS = (a.astype(np.int64) for a in catan.topology())
HEX_VERTEX = np.zeros((HEXES, VERTICES), np.float32)
HEX_VERTEX[np.arange(HEXES)[:, None], HEX_CORNERS] = 1
EDGE_VERTEX = np.zeros((EDGES, VERTICES), np.float32)
EDGE_VERTEX[np.arange(EDGES)[:, None], EDGE_ENDS] = 1
VERTEX_VERTEX = EDGE_VERTEX.T @ EDGE_VERTEX * (1 - np.eye(VERTICES, dtype=np.float32))
EDGE_EDGE = (EDGE_VERTEX @ EDGE_VERTEX.T > 0) * (1 - np.eye(EDGES, dtype=np.float32))

_cache = {}


def adjacency(device):
    if device not in _cache:
        tables = (HEX_VERTEX, EDGE_VERTEX, VERTEX_VERTEX, EDGE_EDGE.astype(np.float32))
        _cache[device] = tuple(torch.from_numpy(t).to(device) for t in tables)
    return _cache[device]


def bits(obs, start, length, count):
    raw = obs[:, start:start + length].long()
    shifts = torch.arange(8, device=obs.device)
    return ((raw.unsqueeze(-1) >> shifts) & 1).flatten(1)[:, :count].float()


def one_hot(values, classes):
    return torch.nn.functional.one_hot(values.long(), classes).float()


def optional(values, classes):
    return one_hot(torch.where(values == NONE, classes, values.long()), classes + 1)


def decode(obs):
    obs = obs.to(torch.uint8)
    hex_vertex, edge_vertex, vertex_vertex, edge_edge = adjacency(obs.device)

    terrain = one_hot(obs[:, 64:83], 6)
    tokens = obs[:, 83:102].long()
    pips = torch.where(tokens == 0, 0, 6 - (7 - tokens).abs()).float() / 5
    robber = one_hot(obs[:, 156], HEXES).unsqueeze(-1)
    buildings = torch.stack([bits(obs, at, 8, VERTICES) for at in (0, 8, 16, 24)], -1)
    roads = torch.stack([bits(obs, at, 16, EDGES) for at in (32, 48)], -1)

    hexes = torch.cat([terrain, one_hot(tokens, 13), pips.unsqueeze(-1), robber, hex_vertex @ buildings], -1)

    harbors = one_hot(obs[:, 102:156], 7)
    phase_vertex = one_hot(torch.where(obs[:, 209] == NONE, VERTICES, obs[:, 209].long()), VERTICES + 1)[:, :VERTICES]
    yields = hex_vertex.T @ (terrain[..., :5] * pips.unsqueeze(-1))
    vertices = torch.cat([
        buildings, harbors, phase_vertex.unsqueeze(-1), yields, vertex_vertex @ buildings, edge_vertex.T @ roads,
    ], -1)

    edges = torch.cat([roads, edge_vertex @ buildings, edge_edge @ roads], -1)

    globals_ = torch.cat([
        obs[:, 157:201].float() / 10,
        one_hot(obs[:, 201], 3),
        one_hot(obs[:, 202], 3),
        obs[:, 203:205].float(),
        one_hot(obs[:, 205], 12),
        obs[:, 206:207].float(),
        obs[:, 207:208].float() / 4,
        optional(obs[:, 208], 2),
        optional(obs[:, 210], 5),
    ], -1)
    return hexes, vertices, edges, globals_


def flat(obs):
    hexes, vertices, edges, globals_ = decode(obs)
    return torch.cat([hexes.flatten(1), vertices.flatten(1), edges.flatten(1), globals_], -1)


def relations():
    kinds = np.zeros((TOKENS, TOKENS), np.int64)
    h, v, e = 1, 1 + HEXES, 1 + HEXES + VERTICES
    kinds[0, :] = kinds[:, 0] = 1
    kinds[h:v, v:e][HEX_VERTEX > 0] = 2
    kinds[v:e, h:v][HEX_VERTEX.T > 0] = 2
    kinds[v:e, v:e][VERTEX_VERTEX > 0] = 3
    kinds[e:, v:e][EDGE_VERTEX > 0] = 4
    kinds[v:e, e:][EDGE_VERTEX.T > 0] = 4
    kinds[e:, e:][EDGE_EDGE > 0] = 5
    kinds[np.arange(TOKENS), np.arange(TOKENS)] = 6
    return torch.from_numpy(kinds)


RELATIONS = 7
