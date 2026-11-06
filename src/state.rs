mod info;
mod rules;
mod validate;

pub use info::Hidden;
pub use rules::{longest_road, Action, Chances, Deal, Invalid, Mask, Outcome, ACTIONS, DICE, MASK_WORDS, MAX_PROPOSED};
pub use validate::Violation;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Rules {
    pub points: u8,
    pub discard_limit: u8,
    pub friendly_robber: bool,
    pub balanced_dice: bool,
    pub trading: bool,
    pub turn_budget: u8,
}

impl Rules {
    pub const STANDARD: Rules =
        Rules { points: 10, discard_limit: 7, friendly_robber: false, balanced_dice: false, trading: true, turn_budget: 40 };
    pub const COLONIST_1V1: Rules =
        Rules { points: 15, discard_limit: 9, friendly_robber: true, balanced_dice: true, trading: true, turn_budget: 40 };

    pub fn check(&self) -> Result<(), Violation> {
        let fine = (3..=20).contains(&self.points)
            && self.discard_limit >= 2
            && (4..=MAX_PROPOSED as u8).contains(&self.turn_budget);
        if fine { Ok(()) } else { Err(Violation(format!("unplayable rules {self:?}"))) }
    }
}

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
    PlaceSettlement,
    PlaceRoad {
        settlement: Vertex,
    },
    Turn {
        rolled: bool,
    },
    Dice,
    Discard {
        player: Player,
        owed: u8,
        from: Resource,
    },
    MoveRobber {
        rolled: bool,
    },
    ChooseVictim {
        rolled: bool,
    },
    Steal {
        rolled: bool,
        victim: Player,
    },
    Draw,
    RoadBuilding {
        rolled: bool,
        roads: u8,
    },
    YearOfPlenty {
        rolled: bool,
        picks: u8,
    },
    Monopoly {
        rolled: bool,
    },
    Over {
        winner: Player,
    },
    Draft {
        deal: Deal,
    },
    Respond {
        deal: Deal,
        responder: Player,
        accepted: u8,
        counters: [Option<Deal>; MAX_PLAYERS],
        draft: Option<Deal>,
        closed: bool,
        clicks: u8,
    },
    Confirm {
        deal: Deal,
        accepted: u8,
        counters: [Option<Deal>; MAX_PLAYERS],
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LastOffer {
    pub deal: Deal,
    pub accepted: u8,
    pub countered: u8,
    pub traded: bool,
}

type PerPlayer<T> = [T; MAX_PLAYERS];

/// Every roll so far, as anyone at the table could count it: what is left of the balanced dice deck since its last
/// reshuffle, the last sum, and who has had the sevens, which Colonist's dice also lean on.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DiceHistory {
    drawn: [u8; 11],
    last: u8,
    sevens: PerPlayer<u8>,
    seven_by: Option<Player>,
    streak: u8,
    rolls: u16,
}

impl DiceHistory {
    const NEW: DiceHistory = DiceHistory { drawn: [0; 11], last: 0, sevens: [0; MAX_PLAYERS], seven_by: None, streak: 0, rolls: 0 };

    /// The deck's cards still to come for each sum from 2 to 12.
    pub fn remaining(&self) -> [u8; 11] {
        std::array::from_fn(|i| DICE[i] - self.drawn[i])
    }

    pub fn last(&self) -> Option<u8> {
        (self.last > 0).then_some(self.last)
    }

    pub fn sevens(&self, player: Player) -> u8 {
        self.sevens[player.index()]
    }

    /// The player who rolled the latest sevens, and how many of the sevens in a row were theirs.
    pub fn streak(&self) -> Option<(Player, u8)> {
        self.seven_by.map(|p| (p, self.streak))
    }

    pub fn rolls(&self) -> u16 {
        self.rolls
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct State {
    board: Board,
    rules: Rules,
    players: u8,
    settlements: PerPlayer<Set<VertexKind>>,
    cities: PerPlayer<Set<VertexKind>>,
    roads: PerPlayer<Set<EdgeKind>>,
    road_lengths: PerPlayer<u8>,
    robber: Hex,
    hands: PerPlayer<[u8; 5]>,
    known: PerPlayer<PerPlayer<[u8; 5]>>,
    set_aside: PerPlayer<[u8; 5]>,
    dev_cards: PerPlayer<[u8; 5]>,
    deck: [u8; 5],
    bought: [u8; 5],
    card_played: bool,
    moves: u8,
    proposed: u8,
    proposals: [Deal; MAX_PROPOSED],
    last_offer: Option<LastOffer>,
    knights: PerPlayer<u8>,
    longest_road: Option<Player>,
    largest_army: Option<Player>,
    current: Player,
    phase: Phase,
    dice: DiceHistory,
}

impl State {
    pub fn new(board: Board, players: usize) -> State {
        State::with_rules(board, players, Rules::STANDARD)
    }

    pub fn with_rules(board: Board, players: usize, rules: Rules) -> State {
        assert!((2..=MAX_PLAYERS).contains(&players), "2 to {MAX_PLAYERS} players");
        rules.check().expect("playable rules");
        State {
            robber: board.desert(),
            board,
            rules,
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
            moves: 0,
            proposed: 0,
            proposals: [Deal::EMPTY; MAX_PROPOSED],
            last_offer: None,
            knights: [0; MAX_PLAYERS],
            longest_road: None,
            largest_army: None,
            current: Player::First,
            phase: Phase::PlaceSettlement,
            dice: DiceHistory::NEW,
        }
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn dice(&self) -> &DiceHistory {
        &self.dice
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

    pub fn last_offer(&self) -> Option<LastOffer> {
        self.last_offer
    }

    pub fn offers(&self) -> u8 {
        self.proposed
    }

    pub fn rules(&self) -> Rules {
        self.rules
    }

    pub fn moves(&self) -> u8 {
        self.moves
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
