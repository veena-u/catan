// primitive for hexes, vertices, and edges (which are all represented as bitsets)
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{BitAnd, BitOr, Not, Sub};

pub const HEXES: usize = 19;
pub const VERTICES: usize = 54;
pub const EDGES: usize = 72;

const _: () = assert!(HEXES <= 64 && VERTICES <= 64 && EDGES <= 128);

pub trait Kind: Copy + Eq + Ord + Hash + fmt::Debug {
    const COUNT: usize;
    const NAME: &'static str;
    type Bits: Bits;
}


pub trait Bits:
    Copy + Eq + Hash + fmt::Debug + BitAnd<Output = Self> + BitOr<Output = Self> + Not<Output = Self>
{

    const ZERO: Self;
    fn bit(index: usize) -> Self;
    fn low(count: usize) -> Self;
    fn count(self) -> u32;
    fn lowest(self) -> usize;
}

impl Bits for u64 {
    const ZERO: Self = 0;
    fn bit(index: usize) -> Self {
        1 << index
    }
    fn low(count: usize) -> Self {
        if count == Self::BITS as usize { Self::MAX } else { (1 << count) - 1 }
    }
    fn count(self) -> u32 {
        self.count_ones()
    }
    fn lowest(self) -> usize {
        self.trailing_zeros() as usize
    }
}

impl Bits for u128 {
    const ZERO: Self = 0;
    fn bit(index: usize) -> Self {
        1 << index
    }
    fn low(count: usize) -> Self {
        if count == Self::BITS as usize { Self::MAX } else { (1 << count) - 1 }
    }
    fn count(self) -> u32 {
        self.count_ones()
    }
    fn lowest(self) -> usize {
        self.trailing_zeros() as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum HexKind {}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum VertexKind {}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum EdgeKind {}

impl Kind for HexKind {
    const COUNT: usize = HEXES;
    const NAME: &'static str = "Hex";
    type Bits = u64;
}
impl Kind for VertexKind {
    const COUNT: usize = VERTICES;
    const NAME: &'static str = "Vertex";
    type Bits = u64;
}
impl Kind for EdgeKind {
    const COUNT: usize = EDGES;
    const NAME: &'static str = "Edge";
    type Bits = u128;
}

pub type Hex = Id<HexKind>;
pub type Vertex = Id<VertexKind>;
pub type Edge = Id<EdgeKind>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Id<K: Kind> {
    index: u8,
    kind: PhantomData<K>,
}

impl<K: Kind> Id<K> {
    pub const fn new(index: usize) -> Self {
        assert!(index < K::COUNT, "id index out of range");
        Self { index: index as u8, kind: PhantomData }
    }

    pub const fn index(self) -> usize {
        self.index as usize
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (0..K::COUNT).map(Self::new)
    }
}

impl<K: Kind> fmt::Debug for Id<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", K::NAME, self.index)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Set<K: Kind> {
    bits: K::Bits,
    kind: PhantomData<K>,
}

impl<K: Kind> Set<K> {
    pub const EMPTY: Self = Self { bits: K::Bits::ZERO, kind: PhantomData };

    fn from_bits(bits: K::Bits) -> Self {
        Self { bits, kind: PhantomData }
    }

    pub fn full() -> Self {
        Self::from_bits(K::Bits::low(K::COUNT))
    }

    pub fn single(id: Id<K>) -> Self {
        Self::from_bits(K::Bits::bit(id.index()))
    }

    pub fn contains(self, id: Id<K>) -> bool {
        self.bits & K::Bits::bit(id.index()) != K::Bits::ZERO
    }

    pub fn insert(&mut self, id: Id<K>) {
        self.bits = self.bits | K::Bits::bit(id.index());
    }

    pub fn remove(&mut self, id: Id<K>) {
        self.bits = self.bits & !K::Bits::bit(id.index());
    }

    pub fn len(self) -> u8 {
        self.bits.count() as u8
    }

    pub fn is_empty(self) -> bool {
        self.bits == K::Bits::ZERO
    }

    pub fn iter(self) -> impl Iterator<Item = Id<K>> {
        let mut bits = self.bits;
        std::iter::from_fn(move || {
            if bits == K::Bits::ZERO {
                return None;
            }
            let index = bits.lowest();
            bits = bits & !K::Bits::bit(index);
            Some(Id::new(index))
        })
    }

    pub fn bits(self) -> K::Bits {
        self.bits
    }
}

impl<K: Kind> Default for Set<K> {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<K: Kind> BitOr for Set<K> {
    type Output = Self;
    fn bitor(self, other: Self) -> Self {
        Self::from_bits(self.bits | other.bits)
    }
}

impl<K: Kind> BitAnd for Set<K> {
    type Output = Self;
    fn bitand(self, other: Self) -> Self {
        Self::from_bits(self.bits & other.bits)
    }
}

impl<K: Kind> Sub for Set<K> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::from_bits(self.bits & !other.bits)
    }
}

impl<K: Kind> FromIterator<Id<K>> for Set<K> {
    fn from_iter<I: IntoIterator<Item = Id<K>>>(ids: I) -> Self {
        let mut set = Self::EMPTY;
        for id in ids {
            set.insert(id);
        }
        set
    }
}

impl<K: Kind> fmt::Debug for Set<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.iter()).finish()
    }
}
