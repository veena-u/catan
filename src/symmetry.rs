// for augmentation of training data
use std::sync::OnceLock;

use crate::ids::{Edge, Hex, Vertex, EDGES, HEXES, VERTICES};
use crate::topology::{topology, Point, COAST};

// k = 0 is the identity
// k = 1 to 5 are rotations
// k = 6 to 11 are a mirror followed by a rotation
pub const SYMMETRIES: usize = 12;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Symmetry(u8);

struct Maps {
    hex: [Hex; HEXES],
    vertex: [Vertex; VERTICES],
    edge: [Edge; EDGES],
    coast_slot: [u8; COAST],
}

fn maps() -> &'static [Maps; SYMMETRIES] {
    static MAPS: OnceLock<[Maps; SYMMETRIES]> = OnceLock::new();
    MAPS.get_or_init(|| std::array::from_fn(build))
}

impl Symmetry {
    pub const IDENTITY: Symmetry = Symmetry(0);

    pub fn all() -> impl Iterator<Item = Symmetry> {
        (0..SYMMETRIES).map(|k| Symmetry(k as u8))
    }

    pub fn new(index: usize) -> Symmetry {
        assert!(index < SYMMETRIES, "there are {SYMMETRIES} symmetries");
        Symmetry(index as u8)
    }

    pub fn index(self) -> usize {
        usize::from(self.0)
    }

    pub fn hex(self, hex: Hex) -> Hex {
        maps()[self.index()].hex[hex.index()]
    }

    pub fn vertex(self, vertex: Vertex) -> Vertex {
        maps()[self.index()].vertex[vertex.index()]
    }

    pub fn edge(self, edge: Edge) -> Edge {
        maps()[self.index()].edge[edge.index()]
    }

    pub fn coast_slot(self, slot: usize) -> usize {
        usize::from(maps()[self.index()].coast_slot[slot])
    }
}

fn transform(k: usize, (mut x, mut y): Point) -> Point {
    if k >= 6 {
        x = -x;
    }
    for _ in 0..k % 6 {
        assert_eq!((x - y) % 2, 0, "x - y is even on the lattice");
        (x, y) = ((x - y) / 2, (3 * x + y) / 2);
    }
    (x, y)
}

fn build(k: usize) -> Maps {
    let t = topology();
    let hex = Hex::all().map(|h| {
        let image = transform(k, t.hex_point(h));
        Hex::all().find(|&g| t.hex_point(g) == image).expect("a hex goes to a hex")
    });
    let vertex: [Vertex; VERTICES] = exactly(Vertex::all().map(|v| {
        let image = transform(k, t.vertex_point(v));
        Vertex::all().find(|&w| t.vertex_point(w) == image).expect("a vertex goes to a vertex")
    }));
    let edge: [Edge; EDGES] = exactly(Edge::all().map(|e| {
        let [a, b] = t.ends(e);
        t.edge_between(vertex[a.index()], vertex[b.index()]).expect("an edge goes to an edge")
    }));
    let coast_slot = exactly(t.coast().iter().map(|e| {
        t.coast_slot(edge[e.index()]).expect("the coast goes to the coast") as u8
    }));
    Maps { hex: exactly(hex), vertex, edge, coast_slot }
}

fn exactly<T, const N: usize>(items: impl Iterator<Item = T>) -> [T; N] {
    let items: Vec<T> = items.collect();
    items.try_into().unwrap_or_else(|_| panic!("expected {N} items"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn every_symmetry_keeps_every_adjacency() {
        let t = topology();
        for s in Symmetry::all() {
            for hex in Hex::all() {
                let image: Vec<_> = t.corners(hex).iter().map(|&v| s.vertex(v)).collect();
                assert_eq!(image.iter().copied().collect::<crate::ids::Set<_>>(), t.corner_set(s.hex(hex)));
            }
            for e in Edge::all() {
                let [a, b] = t.ends(e);
                assert_eq!(t.edge_between(s.vertex(a), s.vertex(b)), Some(s.edge(e)));
            }
            for v in Vertex::all() {
                let adjacent: crate::ids::Set<_> = t.adjacent(v).iter().map(|w| s.vertex(w)).collect();
                assert_eq!(adjacent, t.adjacent(s.vertex(v)));
            }
            let vertices: HashSet<_> = Vertex::all().map(|v| s.vertex(v)).collect();
            assert_eq!(vertices.len(), VERTICES, "a permutation");
        }
    }

    #[test]
    fn the_twelve_are_distinct_a_group_and_start_with_the_identity() {
        let as_vertices = |f: &dyn Fn(Vertex) -> Vertex| Vertex::all().map(f).collect::<Vec<_>>();
        let all: Vec<Vec<Vertex>> = Symmetry::all().map(|s| as_vertices(&|v| s.vertex(v))).collect();
        assert_eq!(all[0], Vertex::all().collect::<Vec<_>>());
        assert_eq!(all.iter().collect::<HashSet<_>>().len(), SYMMETRIES);
        for a in Symmetry::all() {
            for b in Symmetry::all() {
                assert!(all.contains(&as_vertices(&|v| a.vertex(b.vertex(v)))), "closed");
            }
        }
    }

    #[test]
    fn a_rotation_turns_the_coast_by_a_sixth_and_a_mirror_reverses_it() {
        let turn = Symmetry::new(1);
        let shift = turn.coast_slot(0);
        assert_eq!(shift % 5, 0);
        assert!((0..COAST).all(|s| turn.coast_slot(s) == (s + shift) % COAST));
        assert_eq!(shift, 5, "one sixth of the 30 coastal edges");
        let mirror = Symmetry::new(6);
        let start = mirror.coast_slot(0);
        assert!((0..COAST).all(|s| mirror.coast_slot(s) == (start + COAST - s) % COAST));
    }
}
