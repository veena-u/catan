use std::sync::OnceLock;

use crate::ids::{Edge, EdgeKind, Hex, HexKind, Set, Vertex, VertexKind, EDGES, HEXES, VERTICES};

pub const COAST: usize = 30;
pub type Point = (i32, i32);

const CORNER_OFFSETS: [Point; 6] = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)];

// cached relationships between hexes, vertices, and edges
pub struct Topology {
    hex_points: [Point; HEXES],
    vertex_points: [Point; VERTICES],
    corners: [[Vertex; 6]; HEXES],
    ends: [[Vertex; 2]; EDGES],
    coast: [Edge; COAST],
    coast_slots: [Option<u8>; EDGES],
    corner_set: [Set<VertexKind>; HEXES],
    hexes_at: [Set<HexKind>; VERTICES],
    adjacent: [Set<VertexKind>; VERTICES],
    edges_at: [Set<EdgeKind>; VERTICES],
}

pub fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(Topology::build)
}

impl Topology {
    pub fn corners(&self, hex: Hex) -> [Vertex; 6] {
        self.corners[hex.index()]
    }

    pub fn corner_set(&self, hex: Hex) -> Set<VertexKind> {
        self.corner_set[hex.index()]
    }

    pub fn hexes_at(&self, vertex: Vertex) -> Set<HexKind> {
        self.hexes_at[vertex.index()]
    }

    pub fn adjacent(&self, vertex: Vertex) -> Set<VertexKind> {
        self.adjacent[vertex.index()]
    }

    pub fn edges_at(&self, vertex: Vertex) -> Set<EdgeKind> {
        self.edges_at[vertex.index()]
    }

    pub fn ends(&self, edge: Edge) -> [Vertex; 2] {
        self.ends[edge.index()]
    }

    pub fn edge_between(&self, a: Vertex, b: Vertex) -> Option<Edge> {
        if a == b {
            return None;
        }
        (self.edges_at(a) & self.edges_at(b)).iter().next()
    }

    pub fn coast(&self) -> &[Edge; COAST] {
        &self.coast
    }

    pub fn coast_slot(&self, edge: Edge) -> Option<usize> {
        self.coast_slots[edge.index()].map(usize::from)
    }

    pub fn hex_point(&self, hex: Hex) -> Point {
        self.hex_points[hex.index()]
    }

    pub fn vertex_point(&self, vertex: Vertex) -> Point {
        self.vertex_points[vertex.index()]
    }

    fn build() -> Topology {
        let mut hex_points = Vec::new();
        for r in -2i32..=2 {
            for q in -2i32..=2 {
                if (q + r).abs() <= 2 {
                    hex_points.push((2 * q + r, 3 * r));
                }
            }
        }

        let mut vertex_points: Vec<Point> = Vec::new();
        let mut corners = Vec::new();
        for &(cx, cy) in &hex_points {
            corners.push(CORNER_OFFSETS.map(|(dx, dy)| {
                let point = (cx + dx, cy + dy);
                let index = vertex_points.iter().position(|&p| p == point).unwrap_or_else(|| {
                    vertex_points.push(point);
                    vertex_points.len() - 1
                });
                Vertex::new(index)
            }));
        }

        let mut ends: Vec<[Vertex; 2]> = Vec::new();
        let mut bordering: Vec<u8> = Vec::new();
        for hex in &corners {
            for i in 0..6 {
                let mut pair = [hex[i], hex[(i + 1) % 6]];
                pair.sort();
                let index = ends.iter().position(|&e| e == pair).unwrap_or_else(|| {
                    ends.push(pair);
                    bordering.push(0);
                    ends.len() - 1
                });
                bordering[index] += 1;
            }
        }
        let corners: [[Vertex; 6]; HEXES] = exactly(corners);
        let ends: [[Vertex; 2]; EDGES] = exactly(ends);
        let vertex_points: [Point; VERTICES] = exactly(vertex_points);

        let corner_set: [Set<VertexKind>; HEXES] =
            std::array::from_fn(|h| corners[h].iter().copied().collect());
        let hexes_at = std::array::from_fn(|v| {
            Hex::all().filter(|h| corner_set[h.index()].contains(Vertex::new(v))).collect()
        });
        let edges_at: [Set<EdgeKind>; VERTICES] = std::array::from_fn(|v| {
            Edge::all().filter(|e| ends[e.index()].contains(&Vertex::new(v))).collect()
        });
        let adjacent = std::array::from_fn(|v| {
            edges_at[v]
                .iter()
                .map(|e| ends[e.index()].into_iter().find(|w| w.index() != v).expect("two ends"))
                .collect()
        });

        let coastal: Set<EdgeKind> = Edge::all().filter(|e| bordering[e.index()] == 1).collect();
        let coast = trace_coast(coastal, &ends, &edges_at, &vertex_points);
        let mut coast_slots = [None; EDGES];
        for (slot, edge) in coast.iter().enumerate() {
            coast_slots[edge.index()] = Some(slot as u8);
        }

        Topology {
            hex_points: exactly(hex_points),
            vertex_points,
            corners,
            ends,
            coast,
            coast_slots,
            corner_set,
            hexes_at,
            adjacent,
            edges_at,
        }
    }
}

fn trace_coast(
    coastal: Set<EdgeKind>,
    ends: &[[Vertex; 2]; EDGES],
    edges_at: &[Set<EdgeKind>; VERTICES],
    points: &[Point; VERTICES],
) -> [Edge; COAST] {
    let start = coastal.iter().next().expect("the board has a coast");
    let [a, b] = ends[start.index()];
    let ((px, py), (qx, qy)) = (points[a.index()], points[b.index()]);
    let turn = px * qy - qx * py;
    assert_ne!(turn, 0, "a coastal edge does not point at the center");
    let mut at = if turn > 0 { b } else { a };
    let mut coast = vec![start];
    loop {
        let onward = (edges_at[at.index()] & coastal) - Set::single(coast[coast.len() - 1]);
        assert_eq!(onward.len(), 1, "the coast neither ends nor forks");
        let next = onward.iter().next().expect("one edge");
        if next == start {
            break;
        }
        assert!(coast.len() < EDGES, "the coast closes");
        coast.push(next);
        at = ends[next.index()].into_iter().find(|&v| v != at).expect("two ends");
    }
    exactly(coast)
}

fn exactly<T, const N: usize>(items: Vec<T>) -> [T; N] {
    let len = items.len();
    items.try_into().unwrap_or_else(|_| panic!("expected {N} items, found {len}"))
}
