// constructs the terrain, tokens and harbors of a board
use std::fmt;

use crate::ids::{Hex, Set, Vertex, VertexKind, HEXES};
use crate::resource::Resource;
use crate::rng::Rng;
use crate::symmetry::{Symmetry, SYMMETRIES};
use crate::topology::{topology, COAST};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Terrain {
    Hills,
    Forest,
    Pasture,
    Fields,
    Mountains,
    Desert,
}

impl Terrain {
    pub const fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Hills => Some(Resource::Brick),
            Terrain::Forest => Some(Resource::Lumber),
            Terrain::Pasture => Some(Resource::Wool),
            Terrain::Fields => Some(Resource::Grain),
            Terrain::Mountains => Some(Resource::Ore),
            Terrain::Desert => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Harbor {
    Generic,
    Special(Resource),
}

impl Harbor {
    pub const ALL: [Harbor; 6] = [
        Harbor::Generic,
        Harbor::Special(Resource::Brick),
        Harbor::Special(Resource::Lumber),
        Harbor::Special(Resource::Wool),
        Harbor::Special(Resource::Grain),
        Harbor::Special(Resource::Ore),
    ];

    pub const fn index(self) -> usize {
        match self {
            Harbor::Generic => 0,
            Harbor::Special(resource) => 1 + resource.index(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoardError(&'static str);

impl fmt::Display for BoardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for BoardError {}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Board {
    terrain: [Terrain; HEXES],
    tokens: [u8; HEXES],
    harbors: [Set<VertexKind>; 6],
}

pub const TERRAIN: [Terrain; HEXES] = {
    use Terrain::*;
    [
        Forest, Forest, Forest, Forest, Pasture, Pasture, Pasture, Pasture, Fields, Fields,
        Fields, Fields, Hills, Hills, Hills, Mountains, Mountains, Mountains, Desert,
    ]
};

pub const TOKENS: [u8; HEXES - 1] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

pub const HARBORS: [Harbor; 9] = [
    Harbor::Generic,
    Harbor::Generic,
    Harbor::Generic,
    Harbor::Generic,
    Harbor::Special(Resource::Brick),
    Harbor::Special(Resource::Lumber),
    Harbor::Special(Resource::Wool),
    Harbor::Special(Resource::Grain),
    Harbor::Special(Resource::Ore),
];

pub const PRINTED_FRAME: [usize; 9] = [2, 6, 9, 12, 16, 19, 22, 26, 29];

impl Board {
    pub fn new(
        terrain: [Terrain; HEXES],
        tokens: [u8; HEXES],
        harbors: [Option<Harbor>; COAST],
    ) -> Result<Board, BoardError> {
        let t = topology();
        let mut sets = [Set::EMPTY; 6];
        for (slot, harbor) in harbors.iter().enumerate() {
            if let Some(harbor) = harbor {
                for vertex in t.ends(t.coast()[slot]) {
                    sets[harbor.index()].insert(vertex);
                }
            }
        }
        let board = Board { terrain, tokens, harbors: sets };
        board.check()?;
        Ok(board)
    }

    pub fn random(rng: &mut Rng) -> Board {
        let mut terrain = TERRAIN;
        rng.shuffle(&mut terrain);
        let tokens = loop {
            let mut deck = TOKENS;
            rng.shuffle(&mut deck);
            let mut deck = deck.into_iter();
            let tokens = terrain.map(|t| if t == Terrain::Desert { 0 } else { deck.next().unwrap_or(0) });
            if !red_numbers_touch(&tokens) {
                break tokens;
            }
        };
        let frame = Symmetry::new(rng.below(SYMMETRIES));
        let mut kinds = HARBORS;
        rng.shuffle(&mut kinds);
        let mut harbors = [None; COAST];
        for (slot, kind) in PRINTED_FRAME.into_iter().zip(kinds) {
            harbors[frame.coast_slot(slot)] = Some(kind);
        }
        Board::new(terrain, tokens, harbors).expect("a random board follows the rules")
    }

    pub fn terrain(&self, hex: Hex) -> Terrain {
        self.terrain[hex.index()]
    }

    pub fn token(&self, hex: Hex) -> Option<u8> {
        Some(self.tokens[hex.index()]).filter(|&n| n != 0)
    }

    pub fn desert(&self) -> Hex {
        Hex::all().find(|&h| self.terrain(h) == Terrain::Desert).expect("a board has a desert")
    }

    pub fn harbor_at(&self, vertex: Vertex) -> Option<Harbor> {
        Harbor::ALL.into_iter().find(|h| self.harbors[h.index()].contains(vertex))
    }

    pub fn harbor_vertices(&self, harbor: Harbor) -> Set<VertexKind> {
        self.harbors[harbor.index()]
    }

    pub fn harbor_slots(&self) -> [Option<Harbor>; COAST] {
        let t = topology();
        std::array::from_fn(|slot| {
            let [a, b] = t.ends(t.coast()[slot]);
            Harbor::ALL
                .into_iter()
                .find(|h| self.harbors[h.index()].contains(a) && self.harbors[h.index()].contains(b))
        })
    }

    pub fn transformed(&self, symmetry: Symmetry) -> Board {
        let mut moved = *self;
        for hex in Hex::all() {
            moved.terrain[symmetry.hex(hex).index()] = self.terrain[hex.index()];
            moved.tokens[symmetry.hex(hex).index()] = self.tokens[hex.index()];
        }
        moved.harbors = self.harbors.map(|set| set.iter().map(|v| symmetry.vertex(v)).collect());
        moved
    }

    fn check(&self) -> Result<(), BoardError> {
        if sorted(self.terrain) != sorted(TERRAIN) {
            return Err(BoardError("the terrain is not the terrain in the box"));
        }
        for hex in Hex::all() {
            if (self.terrain(hex) == Terrain::Desert) != self.token(hex).is_none() {
                return Err(BoardError("a token on the desert, or none on another hex"));
            }
        }
        let mut tokens: Vec<u8> = self.tokens.iter().copied().filter(|&n| n != 0).collect();
        tokens.sort();
        if tokens != TOKENS {
            return Err(BoardError("the tokens are not the tokens in the box"));
        }
        if red_numbers_touch(&self.tokens) {
            return Err(BoardError("two red numbers touch"));
        }
        let slots = self.harbor_slots();
        let used: Vec<usize> = (0..COAST).filter(|&s| slots[s].is_some()).collect();
        let frames = [Symmetry::IDENTITY, Symmetry::new(1)].map(|s| {
            let mut frame = PRINTED_FRAME.map(|slot| s.coast_slot(slot));
            frame.sort();
            frame
        });
        if !frames.iter().any(|f| f[..] == used[..]) {
            return Err(BoardError("the harbors are not on the harbor frame"));
        }
        let kinds: Vec<Harbor> = slots.iter().flatten().copied().collect();
        if sorted_vec(kinds) != sorted_vec(HARBORS.to_vec()) {
            return Err(BoardError("the harbors are not the harbors in the box"));
        }
        Ok(())
    }
}

fn sorted<T: Ord, const N: usize>(mut items: [T; N]) -> [T; N] {
    items.sort();
    items
}

fn sorted_vec<T: Ord>(mut items: Vec<T>) -> Vec<T> {
    items.sort();
    items
}

fn red_numbers_touch(tokens: &[u8; HEXES]) -> bool {
    let t = topology();
    let red = |h: Hex| matches!(tokens[h.index()], 6 | 8);
    Hex::all().any(|a| {
        red(a) && Hex::all().any(|b| a < b && red(b) && (t.corner_set(a) & t.corner_set(b)).len() == 2)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn random_boards(count: u64) -> impl Iterator<Item = Board> {
        (0..count).map(|seed| Board::random(&mut Rng::new(seed)))
    }

    fn neighbors(a: Hex, b: Hex) -> bool {
        let t = topology();
        let ((ax, ay), (bx, by)) = (t.hex_point(a), t.hex_point(b));
        matches!(((ax - bx).abs(), (ay - by).abs()), (2, 0) | (1, 3))
    }

    #[test]
    fn the_printed_frame_is_where_a_real_game_puts_its_harbors() {
        let sides = [(0, 5), (1, 0), (6, 0), (11, 1), (15, 2), (17, 2), (16, 3), (12, 4), (3, 4)];
        let t = topology();
        let mut slots: Vec<usize> = sides
            .iter()
            .map(|&(h, i)| {
                let c = t.corners(Hex::new(h));
                t.coast_slot(t.edge_between(c[i], c[(i + 1) % 6]).unwrap()).expect("coastal")
            })
            .collect();
        slots.sort();
        assert_eq!(slots, PRINTED_FRAME);
        let frames: HashSet<Vec<usize>> = Symmetry::all()
            .map(|s| sorted_vec(PRINTED_FRAME.iter().map(|&slot| s.coast_slot(slot)).collect()))
            .collect();
        assert_eq!(frames.len(), 2, "the symmetries make exactly two frames");
    }

    #[test]
    fn red_numbers_never_touch_by_an_independent_neighbor_test() {
        for board in random_boards(2_000) {
            for a in Hex::all() {
                for b in Hex::all() {
                    let red = |h| matches!(board.token(h), Some(6 | 8));
                    assert!(!(neighbors(a, b) && red(a) && red(b)));
                }
            }
        }
        let count = Hex::all().map(|a| Hex::all().filter(|&b| neighbors(a, b)).count()).sum::<usize>();
        assert_eq!(count, 2 * 42, "a 19-hex board has 42 pairs of neighbors");
    }

    #[test]
    fn each_harbor_serves_the_two_ends_of_its_edge_and_no_other_vertex() {
        let t = topology();
        for board in random_boards(100) {
            let mut served = HashSet::new();
            for (slot, harbor) in board.harbor_slots().iter().enumerate() {
                if let Some(harbor) = harbor {
                    for v in t.ends(t.coast()[slot]) {
                        assert_eq!(board.harbor_at(v), Some(*harbor));
                        served.insert(v);
                    }
                }
            }
            assert_eq!(served.len(), 18);
            assert!(Vertex::all().filter(|v| !served.contains(v)).all(|v| board.harbor_at(v).is_none()));
        }
    }

    #[test]
    fn boards_that_break_a_rule_are_refused() {
        let board = Board::random(&mut Rng::new(1));
        let (terrain, tokens, harbors) = (board.terrain, board.tokens, board.harbor_slots());
        assert_eq!(Board::new(terrain, tokens, harbors), Ok(board));
        let desert = board.desert().index();
        let other = (desert + 1) % HEXES;

        let mut t = terrain;
        t[other] = Terrain::Desert;
        assert!(Board::new(t, tokens, harbors).is_err(), "two deserts");
        let mut k = tokens;
        k.swap(desert, other);
        assert!(Board::new(terrain, k, harbors).is_err(), "a token on the desert");
        let mut k = tokens;
        k[other] = if k[other] == 2 { 3 } else { 2 };
        assert!(Board::new(terrain, k, harbors).is_err(), "the wrong tokens");
        let mut h = harbors;
        h.rotate_left(1);
        assert!(Board::new(terrain, tokens, h).is_err(), "off the frame");
        let mut h = harbors;
        let first = h.iter().position(Option::is_some).unwrap();
        h[first] = Some(if h[first] == Some(Harbor::Generic) { Harbor::Special(Resource::Ore) } else { Harbor::Generic });
        assert!(Board::new(terrain, tokens, h).is_err(), "the wrong harbors");

        let reds: Vec<usize> = (0..HEXES).filter(|&i| matches!(tokens[i], 6 | 8)).collect();
        let next_to_red = (0..HEXES)
            .find(|&i| tokens[i] != 0 && !reds.contains(&i) && neighbors(Hex::new(i), Hex::new(reds[0])))
            .unwrap();
        let mut k = tokens;
        k.swap(next_to_red, reds[1]);
        assert_eq!(Board::new(terrain, k, harbors), Err(BoardError("two red numbers touch")));
    }

    #[test]
    fn random_boards_are_spread_evenly_and_repeatable() {
        let mut desert = [0u32; HEXES];
        let mut frames: HashMap<Vec<usize>, u32> = HashMap::new();
        let (mut used, mut generic) = ([0u32; COAST], [0u32; COAST]);
        for board in random_boards(19_000) {
            desert[board.desert().index()] += 1;
            let slots = board.harbor_slots();
            *frames.entry((0..COAST).filter(|&s| slots[s].is_some()).collect()).or_default() += 1;
            for s in 0..COAST {
                used[s] += u32::from(slots[s].is_some());
                generic[s] += u32::from(slots[s] == Some(Harbor::Generic));
            }
        }
        assert!(desert.iter().all(|&n| (850..=1_150).contains(&n)), "{desert:?}");
        assert_eq!(frames.len(), 2);
        assert!(frames.values().all(|&n| (9_150..=9_850).contains(&n)), "{frames:?}");
        for s in (0..COAST).filter(|&s| used[s] > 0) {
            let share = f64::from(generic[s]) / f64::from(used[s]);
            assert!((share - 4.0 / 9.0).abs() < 0.03, "slot {s}: {share}");
        }
        assert_eq!(Board::random(&mut Rng::new(5)), Board::random(&mut Rng::new(5)));
        assert_eq!(random_boards(200).collect::<HashSet<_>>().len(), 200);
    }
}
