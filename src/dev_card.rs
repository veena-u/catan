#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}

impl DevCard {
    pub const ALL: [DevCard; 5] = [
        DevCard::Knight,
        DevCard::VictoryPoint,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

pub const DEV_CARDS: [u8; 5] = [14, 5, 2, 2, 2];
