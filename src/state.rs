mod info;
mod rules;
mod validate;

pub use info::Hidden;
pub use rules::{longest_road, Action, Chances, Invalid, Mask, Outcome, ACTIONS, WINNING_POINTS};
pub use validate::Violation;

use crate::board::Board;
use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{EdgeKind, Hex, Set, Vertex, VertexKind};
use crate::player::Player;
use crate::resource::{Resource, RESOURCE_CARDS};
use crate::symmetry::Symmetry;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Actor {
    Player(Player),
    Chance,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    PlaceSettlement, // waiting for a setup settlement to be placed
    PlaceRoad { // waiting for a setup road to be placed
        settlement: Vertex, // the road must touch this settlement
    },
    Turn { // waiting for opponent's next move
        rolled: bool, // dice roll yet? 
    },
    Dice, // waiting for dice to be rolled
    Discard { // waiting for player to discard half their hand
        player: Player, // who is discarding
        owed: u8, // how many cards left to discard
        from: Resource, // lowest card allowed
    },
    MoveRobber { // waiting for player to move robber
        rolled: bool, // dice roll yet?
    },
    Steal { // waiting for chance to pick a stolen card
        rolled: bool, // dice roll yet?
    },
    Draw, // waiting for player to draw a dev card
    RoadBuilding { // waiting for player to place two roads (dev card)
        rolled: bool, // dice roll yet?
        roads: u8, // how many roads left to place
    },
    YearOfPlenty { // waiting for player to pick two resources (dev card)
        rolled: bool, // dice roll yet?
        picks: u8, // how many resources left to pick
    },
    Monopoly { // waiting for player to pick a resource (dev card)
        rolled: bool, // dice roll yet?
    },
    Over { // end of game
        winner: Player,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State { // most fields are indexed by player
    board: Board,
    settlements: [Set<VertexKind>; 2],
    cities: [Set<VertexKind>; 2],
    roads: [Set<EdgeKind>; 2],
    road_lengths: [u8; 2],
    robber: Hex,
    hands: [[u8; 5]; 2],
    set_aside: [u8; 5],
    dev_cards: [[u8; 5]; 2],
    deck: [u8; 5],
    bought: [u8; 5],
    card_played: bool,
    knights: [u8; 2],
    longest_road: Option<Player>,
    largest_army: Option<Player>,
    current: Player,
    phase: Phase,
}

impl State {
    pub fn new(board: Board) -> State {
        State {
            robber: board.desert(),
            board,
            settlements: [Set::EMPTY; 2],
            cities: [Set::EMPTY; 2],
            roads: [Set::EMPTY; 2],
            road_lengths: [0; 2],
            hands: [[0; 5]; 2],
            set_aside: [0; 5],
            dev_cards: [[0; 5]; 2],
            deck: DEV_CARDS,
            bought: [0; 5],
            card_played: false,
            knights: [0; 2],
            longest_road: None,
            largest_army: None,
            current: Player::First,
            phase: Phase::PlaceSettlement,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn settlements(&self, player: Player) -> Set<VertexKind> {
        self.settlements[player.index()]
    }

    pub fn cities(&self, player: Player) -> Set<VertexKind> {
        self.cities[player.index()]
    }

    pub fn buildings(&self, player: Player) -> Set<VertexKind> {
        self.settlements[player.index()] | self.cities[player.index()]
    }

    pub fn roads(&self, player: Player) -> Set<EdgeKind> {
        self.roads[player.index()]
    }

    pub fn road_length(&self, player: Player) -> u8 {
        self.road_lengths[player.index()]
    }

    pub fn robber(&self) -> Hex {
        self.robber
    }

    pub fn resources(&self, player: Player, resource: Resource) -> u8 {
        self.hands[player.index()][resource.index()]
    }

    pub fn hand_size(&self, player: Player) -> u8 {
        self.hands[player.index()].iter().sum()
    }

    pub fn set_aside(&self, resource: Resource) -> u8 {
        self.set_aside[resource.index()]
    }

    pub fn bank(&self, resource: Resource) -> u8 {
        let r = resource.index();
        RESOURCE_CARDS - self.hands[0][r] - self.hands[1][r] - self.set_aside[r]
    }

    pub fn dev_cards(&self, player: Player, card: DevCard) -> u8 {
        self.dev_cards[player.index()][card.index()]
    }

    pub fn dev_card_count(&self, player: Player) -> u8 {
        self.dev_cards[player.index()].iter().sum()
    }

    pub fn deck_size(&self) -> u8 {
        self.deck.iter().sum()
    }

    pub fn played(&self, card: DevCard) -> u8 {
        let c = card.index();
        DEV_CARDS[c] - self.deck[c] - self.dev_cards[0][c] - self.dev_cards[1][c]
    }

    pub fn bought(&self, card: DevCard) -> u8 {
        self.bought[card.index()]
    }

    pub fn card_played(&self) -> bool {
        self.card_played
    }

    pub fn knights(&self, player: Player) -> u8 {
        self.knights[player.index()]
    }

    pub fn longest_road(&self) -> Option<Player> {
        self.longest_road
    }

    pub fn largest_army(&self) -> Option<Player> {
        self.largest_army
    }

    pub fn current(&self) -> Player {
        self.current
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn actor(&self) -> Option<Actor> {
        match self.phase {
            Phase::Dice | Phase::Steal { .. } | Phase::Draw => Some(Actor::Chance),
            Phase::Discard { player, .. } => Some(Actor::Player(player)),
            Phase::Over { .. } => None,
            _ => Some(Actor::Player(self.current)),
        }
    }

    pub fn winner(&self) -> Option<Player> {
        match self.phase {
            Phase::Over { winner } => Some(winner),
            _ => None,
        }
    }

    pub fn victory_points(&self, player: Player) -> u8 {
        self.public_victory_points(player) + self.dev_cards(player, DevCard::VictoryPoint)
    }

    pub fn public_victory_points(&self, player: Player) -> u8 {
        let p = player.index();
        let buildings = self.settlements[p].len() + 2 * self.cities[p].len();
        let titles = u8::from(self.longest_road == Some(player)) + u8::from(self.largest_army == Some(player));
        buildings + 2 * titles
    }

    pub fn transformed(&self, symmetry: Symmetry) -> State {
        let vertices = |set: Set<VertexKind>| set.iter().map(|v| symmetry.vertex(v)).collect();
        let mut moved = *self;
        moved.board = self.board.transformed(symmetry);
        moved.settlements = self.settlements.map(vertices);
        moved.cities = self.cities.map(vertices);
        moved.roads = self.roads.map(|set| set.iter().map(|e| symmetry.edge(e)).collect());
        moved.robber = symmetry.hex(self.robber);
        if let Phase::PlaceRoad { settlement } = self.phase {
            moved.phase = Phase::PlaceRoad { settlement: symmetry.vertex(settlement) };
        }
        moved
    }
}
