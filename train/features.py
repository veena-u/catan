import numpy as np
import torch

import catan

SEATS = catan.MAX_PLAYERS
HEXES, VERTICES, EDGES = 19, 54, 72
TOKENS = 1 + HEXES + VERTICES + EDGES
HEX_FEATURES = 6 + 13 + 1 + 1 + 2 * SEATS
VERTEX_FEATURES = 2 * SEATS + 7 + 1 + 5 + 2 * SEATS + SEATS
EDGE_FEATURES = SEATS + 2 * SEATS + SEATS
GLOBAL_FEATURES = 133
FLAT_FEATURES = HEXES * HEX_FEATURES + VERTICES * VERTEX_FEATURES + EDGES * EDGE_FEATURES + GLOBAL_FEATURES

NONE = 255

HEX_CORNERS, EDGE_ENDS = (a.astype(np.int64) for a in catan.topology())
HEX_VERTEX = np.zeros((HEXES, VERTICES), np.float32)
HEX_VERTEX[np.arange(HEXES)[:, None], HEX_CORNERS] = 1
EDGE_VERTEX = np.zeros((EDGES, VERTICES), np.float32)
EDGE_VERTEX[np.arange(EDGES)[:, None], EDGE_ENDS] = 1
VERTEX_VERTEX = EDGE_VERTEX.T @ EDGE_VERTEX * (1 - np.eye(VERTICES, dtype=np.float32))
EDGE_EDGE = (EDGE_VERTEX @ EDGE_VERTEX.T > 0) * (1 - np.eye(EDGES, dtype=np.float32))

SYMMETRY_HEX, SYMMETRY_VERTEX, SYMMETRY_EDGE, SYMMETRY_ACTION = (a.astype(np.int64) for a in catan.symmetries())
SYMMETRIES = len(SYMMETRY_HEX)
_INVERSE = tuple(np.argsort(t, axis=1) for t in (SYMMETRY_HEX, SYMMETRY_VERTEX, SYMMETRY_EDGE))

_cache = {}
_symmetry_cache = {}


def symmetry_tables(device):
    if device not in _symmetry_cache:
        _symmetry_cache[device] = tuple(torch.from_numpy(t).to(device) for t in (*_INVERSE, SYMMETRY_ACTION))
    return _symmetry_cache[device]


def turn(rows, inverse):
    return rows.gather(1, inverse[..., None].expand(-1, -1, rows.shape[-1]))


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


def seats_present(obs):
    return (torch.arange(SEATS, device=obs.device) < obs[:, 221:222].long()).float()


def decode(obs, symmetry=None):
    obs = obs.to(torch.uint8)
    hex_vertex, edge_vertex, vertex_vertex, edge_edge = adjacency(obs.device)

    settlements = torch.stack([bits(obs, 32 * s, 8, VERTICES) for s in range(SEATS)], -1)
    cities = torch.stack([bits(obs, 32 * s + 8, 8, VERTICES) for s in range(SEATS)], -1)
    roads = torch.stack([bits(obs, 32 * s + 16, 16, EDGES) for s in range(SEATS)], -1)
    buildings = torch.cat([settlements, cities], -1)

    terrain = one_hot(obs[:, 128:147], 6)
    tokens = obs[:, 147:166].long()
    pips = torch.where(tokens == 0, 0, 6 - (7 - tokens).abs()).float() / 5
    robber = one_hot(obs[:, 220], HEXES).unsqueeze(-1)
    hexes = torch.cat([terrain, one_hot(tokens, 13), pips.unsqueeze(-1), robber, hex_vertex @ buildings], -1)

    harbors = one_hot(obs[:, 166:220], 7)
    phase_vertex = one_hot(torch.where(obs[:, 300] == NONE, VERTICES, obs[:, 300].long()), VERTICES + 1)[:, :VERTICES]
    yields = hex_vertex.T @ (terrain[..., :5] * pips.unsqueeze(-1))
    vertices = torch.cat([
        buildings, harbors, phase_vertex.unsqueeze(-1), yields, vertex_vertex @ buildings, edge_vertex.T @ roads,
    ], -1)

    edges = torch.cat([roads, edge_vertex @ buildings, edge_edge @ roads], -1)

    globals_ = torch.cat([
        seats_present(obs),
        obs[:, 222:232].float() / 10,
        obs[:, 232:253].float() / 10,
        obs[:, 253:258].float() / 10,
        obs[:, 258:292].float() / 10,
        one_hot(obs[:, 292], SEATS + 1),
        one_hot(obs[:, 293], SEATS + 1),
        one_hot(obs[:, 294], SEATS),
        obs[:, 295:296].float(),
        one_hot(obs[:, 296], 16),
        obs[:, 297:298].float(),
        obs[:, 298:299].float() / 4,
        optional(obs[:, 299], SEATS),
        optional(obs[:, 301], 5),
        obs[:, 302:312].float() / 4,
        bits(obs, 312, 1, SEATS),
        obs[:, 313:314].float() / 3,
    ], -1)
    if symmetry is not None:
        hex_inverse, vertex_inverse, edge_inverse, _ = symmetry_tables(obs.device)
        hexes = turn(hexes, hex_inverse[symmetry])
        vertices = turn(vertices, vertex_inverse[symmetry])
        edges = turn(edges, edge_inverse[symmetry])
    return hexes, vertices, edges, globals_


def flat(obs, symmetry=None):
    hexes, vertices, edges, globals_ = decode(obs, symmetry)
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
