#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Player {
    First,
    Second,
}

impl Player {
    pub const BOTH: [Player; 2] = [Player::First, Player::Second];

    pub const fn other(self) -> Player {
        match self {
            Player::First => Player::Second,
            Player::Second => Player::First,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}
