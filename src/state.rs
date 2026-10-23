mod info;
mod rules;
mod validate;

pub use info::Hidden;
pub use rules::{longest_road, Action, Chances, Invalid, Mask, Outcome, ACTIONS, WINNING_POINTS};
pub use validate::Violation;

use crate::board::Board;
use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{EdgeKind, Hex, Set, Vertex, VertexKind};
use crate::player::{Player, MAX_PLAYERS};
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
    Turn { // waiting for the current player's next move
        rolled: bool, // dice roll yet?
    },
    Dice, // waiting for dice to be rolled
    Discard { // waiting for a player over 7 cards to discard one
        player: Player, // who is discarding
        owed: u8, // how many cards left to discard
        from: Resource, // lowest card allowed
    },
    MoveRobber { // waiting for player to move robber
        rolled: bool, // dice roll yet?
    },
    ChooseVictim { // waiting for the robber's mover to pick whom to rob
        rolled: bool, // dice roll yet?
    },
    Steal { // waiting for chance to pick a stolen card
        rolled: bool, // dice roll yet?
        victim: Player, // whose hand the card comes from
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
    Offer { // the current player is building a trade offer, card by card
        give: [u8; 5], // what they would give
        get: [u8; 5], // what they want in return
    },
    Respond { // each other player in turn accepts or declines the offer
        give: [u8; 5],
        get: [u8; 5],
        responder: Player, // who answers now
        accepted: u8, // bit p set: player p accepted
    },
    Confirm { // the current player picks a partner among those who accepted, or cancels
        give: [u8; 5],
        get: [u8; 5],
        accepted: u8,
    },
}

type PerPlayer<T> = [T; MAX_PLAYERS];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State { // most fields are indexed by player; seats past `players` stay empty
    board: Board,
    players: u8,
    settlements: PerPlayer<Set<VertexKind>>,
    cities: PerPlayer<Set<VertexKind>>,
    roads: PerPlayer<Set<EdgeKind>>,
    road_lengths: PerPlayer<u8>,
    robber: Hex,
    hands: PerPlayer<[u8; 5]>,
    known: PerPlayer<PerPlayer<[u8; 5]>>, // known[o][p]: the fewest of each resource o is sure p holds
    set_aside: PerPlayer<[u8; 5]>,
    dev_cards: PerPlayer<[u8; 5]>,
    deck: [u8; 5],
    bought: [u8; 5],
    card_played: bool,
    offers: u8, // trade offers made this turn
    knights: PerPlayer<u8>,
    longest_road: Option<Player>,
    largest_army: Option<Player>,
    current: Player,
    phase: Phase,
}

impl State {
    pub fn new(board: Board, players: usize) -> State {
        assert!((2..=MAX_PLAYERS).contains(&players), "2 to {MAX_PLAYERS} players");
        State {
            robber: board.desert(),
            board,
            players: players as u8,
            settlements: [Set::EMPTY; MAX_PLAYERS],
            cities: [Set::EMPTY; MAX_PLAYERS],
            roads: [Set::EMPTY; MAX_PLAYERS],
            road_lengths: [0; MAX_PLAYERS],
            hands: [[0; 5]; MAX_PLAYERS],
            known: [[[0; 5]; MAX_PLAYERS]; MAX_PLAYERS],
            set_aside: [[0; 5]; MAX_PLAYERS],
            dev_cards: [[0; 5]; MAX_PLAYERS],
            deck: DEV_CARDS,
            bought: [0; 5],
            card_played: false,
            offers: 0,
            knights: [0; MAX_PLAYERS],
            longest_road: None,
            largest_army: None,
            current: Player::First,
            phase: Phase::PlaceSettlement,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn players(&self) -> usize {
        usize::from(self.players)
    }

    pub fn seats(&self) -> impl Iterator<Item = Player> + use<> {
        Player::ALL.into_iter().take(self.players())
    }

    pub fn others(&self, player: Player) -> impl Iterator<Item = Player> + use<> {
        let n = self.players();
        (1..n).map(move |offset| player.after(offset, n))
    }

    pub fn next(&self, player: Player) -> Player {
        player.after(1, self.players())
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

    pub fn all_buildings(&self) -> Set<VertexKind> {
        self.seats().fold(Set::EMPTY, |all, p| all | self.buildings(p))
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

    pub fn hand(&self, player: Player) -> [u8; 5] {
        self.hands[player.index()]
    }

    pub fn hand_size(&self, player: Player) -> u8 {
        self.hands[player.index()].iter().sum()
    }

    pub fn known(&self, observer: Player, player: Player) -> [u8; 5] {
        if observer == player { self.hands[player.index()] } else { self.known[observer.index()][player.index()] }
    }

    pub fn set_aside(&self, player: Player) -> [u8; 5] {
        self.set_aside[player.index()]
    }

    pub fn bank(&self, resource: Resource) -> u8 {
        let r = resource.index();
        let held: u8 = self.seats().map(|p| self.hands[p.index()][r] + self.set_aside[p.index()][r]).sum();
        RESOURCE_CARDS - held
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
        DEV_CARDS[c] - self.deck[c] - self.seats().map(|p| self.dev_cards[p.index()][c]).sum::<u8>()
    }

    pub fn bought(&self, card: DevCard) -> u8 {
        self.bought[card.index()]
    }

    pub fn card_played(&self) -> bool {
        self.card_played
    }

    pub fn offers(&self) -> u8 {
        self.offers
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
            Phase::Discard { player, .. } | Phase::Respond { responder: player, .. } => Some(Actor::Player(player)),
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
