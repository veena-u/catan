pub const MAX_PLAYERS: usize = 4;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Player {
    First,
    Second,
    Third,
    Fourth,
}

impl Player {
    pub const ALL: [Player; MAX_PLAYERS] = [Player::First, Player::Second, Player::Third, Player::Fourth];

    pub const fn new(index: usize) -> Player {
        Player::ALL[index]
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn after(self, offset: usize, players: usize) -> Player {
        Player::new((self.index() + offset) % players)
    }

    pub const fn offset_from(self, other: Player, players: usize) -> usize {
        (self.index() + players - other.index()) % players
    }
}
