use std::fmt;

use crate::board::Harbor;
use crate::dev_card::DevCard;
use crate::ids::{Edge, EdgeKind, Hex, Set, Vertex, VertexKind, EDGES, VERTICES};
use crate::player::Player;
use crate::resource::Resource;
use crate::rng::Rng;
use crate::symmetry::Symmetry;
use crate::topology::topology;

use super::{Actor, Phase, State};

pub const ACTIONS: usize = 271;

pub const WINNING_POINTS: u8 = 10;

const SETTLEMENT: usize = 0;
const CITY: usize = 54;
const ROAD: usize = 108;
const ROBBER: usize = 180;
const RESOURCE: usize = 199;
const TRADE: usize = 204;
const ROLL: usize = 264;
const END_TURN: usize = 265;
const BUY_CARD: usize = 266;
const PLAY_KNIGHT: usize = 267;
const PLAY_ROAD_BUILDING: usize = 268;
const PLAY_YEAR_OF_PLENTY: usize = 269;
const PLAY_MONOPOLY: usize = 270;

const _: () = assert!(CITY == SETTLEMENT + VERTICES && ROAD == CITY + VERTICES);
const _: () = assert!(ROBBER == ROAD + EDGES && RESOURCE == ROBBER + 19 && TRADE == RESOURCE + 5);
const _: () = assert!(ROLL == TRADE + 3 * 5 * 4 && PLAY_MONOPOLY + 1 == ACTIONS);

const ROAD_COST: [u8; 5] = [1, 1, 0, 0, 0];
const SETTLEMENT_COST: [u8; 5] = [1, 1, 1, 1, 0];
const CITY_COST: [u8; 5] = [0, 0, 0, 2, 3];
const CARD_COST: [u8; 5] = [0, 0, 1, 1, 1];

const ROADS: u8 = 15;
const SETTLEMENTS: u8 = 5;
const CITIES: u8 = 4;

const DICE: [u8; 11] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Action {
    Settlement(Vertex),
    City(Vertex),
    Road(Edge),
    Robber(Hex),
    Resource(Resource),
    Trade {
        give: Resource,
        get: Resource,
        rate: u8,
    },
    Roll,
    EndTurn,
    BuyCard,
    PlayKnight,
    PlayRoadBuilding,
    PlayYearOfPlenty,
    PlayMonopoly,
}

impl Action {
    pub fn index(self) -> u16 {
        let index = match self {
            Action::Settlement(v) => SETTLEMENT + v.index(),
            Action::City(v) => CITY + v.index(),
            Action::Road(e) => ROAD + e.index(),
            Action::Robber(h) => ROBBER + h.index(),
            Action::Resource(r) => RESOURCE + r.index(),
            Action::Trade { give, get, rate } => {
                debug_assert!((2..=4).contains(&rate) && give != get);
                let get = get.index() - usize::from(get > give);
                TRADE + 20 * usize::from(rate - 2) + 4 * give.index() + get
            }
            Action::Roll => ROLL,
            Action::EndTurn => END_TURN,
            Action::BuyCard => BUY_CARD,
            Action::PlayKnight => PLAY_KNIGHT,
            Action::PlayRoadBuilding => PLAY_ROAD_BUILDING,
            Action::PlayYearOfPlenty => PLAY_YEAR_OF_PLENTY,
            Action::PlayMonopoly => PLAY_MONOPOLY,
        };
        index as u16
    }

    pub fn from_index(index: u16) -> Option<Action> {
        let i = usize::from(index);
        Some(match i {
            _ if i < CITY => Action::Settlement(Vertex::new(i - SETTLEMENT)),
            _ if i < ROAD => Action::City(Vertex::new(i - CITY)),
            _ if i < ROBBER => Action::Road(Edge::new(i - ROAD)),
            _ if i < RESOURCE => Action::Robber(Hex::new(i - ROBBER)),
            _ if i < TRADE => Action::Resource(Resource::ALL[i - RESOURCE]),
            _ if i < ROLL => {
                let (rate, rest) = ((i - TRADE) / 20, (i - TRADE) % 20);
                let (give, get) = (rest / 4, rest % 4);
                Action::Trade {
                    give: Resource::ALL[give],
                    get: Resource::ALL[get + usize::from(get >= give)],
                    rate: rate as u8 + 2,
                }
            }
            ROLL => Action::Roll,
            END_TURN => Action::EndTurn,
            BUY_CARD => Action::BuyCard,
            PLAY_KNIGHT => Action::PlayKnight,
            PLAY_ROAD_BUILDING => Action::PlayRoadBuilding,
            PLAY_YEAR_OF_PLENTY => Action::PlayYearOfPlenty,
            PLAY_MONOPOLY => Action::PlayMonopoly,
            _ => return None,
        })
    }

    pub fn transformed(self, symmetry: Symmetry) -> Action {
        match self {
            Action::Settlement(v) => Action::Settlement(symmetry.vertex(v)),
            Action::City(v) => Action::City(symmetry.vertex(v)),
            Action::Road(e) => Action::Road(symmetry.edge(e)),
            Action::Robber(h) => Action::Robber(symmetry.hex(h)),
            other => other,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Mask {
    words: [u64; 5],
}

const _: () = assert!(ACTIONS <= 5 * 64);

impl Mask {
    pub const EMPTY: Mask = Mask { words: [0; 5] };

    pub fn contains(&self, action: Action) -> bool {
        let i = usize::from(action.index());
        self.words[i / 64] >> (i % 64) & 1 == 1
    }

    pub fn insert(&mut self, action: Action) {
        let i = usize::from(action.index());
        self.words[i / 64] |= 1 << (i % 64);
    }

    pub fn len(&self) -> u32 {
        self.words.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words == [0; 5]
    }

    pub fn iter(&self) -> impl Iterator<Item = Action> + '_ {
        (0..5).flat_map(move |w| {
            let mut word = self.words[w];
            std::iter::from_fn(move || {
                if word == 0 {
                    return None;
                }
                let bit = word.trailing_zeros() as usize;
                word &= word - 1;
                Action::from_index((64 * w + bit) as u16)
            })
        })
    }

    pub fn words(&self) -> [u64; 5] {
        self.words
    }

    fn insert_bits(&mut self, offset: usize, bits: u128) {
        let shifted = bits << (offset % 64);
        self.words[offset / 64] |= shifted as u64;
        self.words[offset / 64 + 1] |= (shifted >> 64) as u64;
    }
}

const _: () = assert!(SETTLEMENT % 64 + VERTICES <= 128 && SETTLEMENT / 64 + 1 < 5);
const _: () = assert!(CITY % 64 + VERTICES <= 128 && CITY / 64 + 1 < 5);
const _: () = assert!(ROAD % 64 + EDGES <= 128 && ROAD / 64 + 1 < 5);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Outcome {
    Dice(u8),
    Steal(Resource),
    Draw(DevCard),
}

impl Outcome {
    pub fn index(self) -> u16 {
        match self {
            Outcome::Dice(sum) => u16::from(sum - 2),
            Outcome::Steal(resource) => resource.index() as u16,
            Outcome::Draw(card) => card.index() as u16,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Chances {
    weights: [u8; 11],
    count: u8,
}

impl Chances {
    pub fn weights(&self) -> &[u8] {
        &self.weights[..usize::from(self.count)]
    }

    pub fn total(&self) -> u32 {
        self.weights().iter().map(|&w| u32::from(w)).sum()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Invalid {
    pub code: u16,
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "step {} cannot be taken in this state", self.code)
    }
}

impl std::error::Error for Invalid {}

impl State {
    pub fn legal_actions(&self) -> Mask {
        let player = self.current;
        let p = player.index();
        let mut mask = Mask::EMPTY;
        match self.phase {
            Phase::PlaceSettlement => {
                let free = Set::<VertexKind>::full() - self.blocked();
                mask.insert_bits(SETTLEMENT, free.bits().into());
            }
            Phase::PlaceRoad { settlement } => {
                let edges = topology().edges_at(settlement) - self.all_roads();
                mask.insert_bits(ROAD, edges.bits());
            }
            Phase::Turn { rolled: false } => {
                mask.insert(Action::Roll);
                self.insert_playable_cards(&mut mask);
            }
            Phase::Turn { rolled: true } => {
                if self.can_pay(player, SETTLEMENT_COST) {
                    mask.insert_bits(SETTLEMENT, self.settlement_spots(player).bits().into());
                }
                if self.can_pay(player, CITY_COST) {
                    mask.insert_bits(CITY, self.city_spots(player).bits().into());
                }
                if self.can_pay(player, ROAD_COST) {
                    mask.insert_bits(ROAD, self.road_spots(player).bits());
                }
                for give in Resource::ALL {
                    for rate in 2..=4 {
                        if self.hands[p][give.index()] < rate || !self.has_rate(player, give, rate) {
                            continue;
                        }
                        for get in Resource::ALL {
                            if get != give && self.bank(get) > 0 {
                                mask.insert(Action::Trade { give, get, rate });
                            }
                        }
                    }
                }
                if self.can_pay(player, CARD_COST) && self.deck_size() > 0 {
                    mask.insert(Action::BuyCard);
                }
                self.insert_playable_cards(&mut mask);
                mask.insert(Action::EndTurn);
            }
            Phase::Discard { player, owed, from } => {
                let hand = &self.hands[player.index()];
                for resource in Resource::ALL.into_iter().filter(|&r| r >= from) {
                    let from_here: u8 = hand[resource.index()..].iter().sum();
                    if hand[resource.index()] > 0 && from_here >= owed {
                        mask.insert(Action::Resource(resource));
                    }
                }
            }
            Phase::MoveRobber { .. } => {
                for hex in Hex::all().filter(|&h| h != self.robber) {
                    mask.insert(Action::Robber(hex));
                }
            }
            Phase::RoadBuilding { .. } => mask.insert_bits(ROAD, self.road_spots(player).bits()),
            Phase::YearOfPlenty { .. } => {
                for resource in Resource::ALL.into_iter().filter(|&r| self.bank(r) > 0) {
                    mask.insert(Action::Resource(resource));
                }
            }
            Phase::Monopoly { .. } => {
                for resource in Resource::ALL {
                    mask.insert(Action::Resource(resource));
                }
            }
            Phase::Dice | Phase::Steal { .. } | Phase::Draw | Phase::Over { .. } => {}
        }
        mask
    }

    pub fn apply(&mut self, action: Action) {
        debug_assert!(self.legal_actions().contains(action), "{action:?} is not legal in {:?}", self.phase);
        self.transition(action);
        self.settle();
    }

    pub fn chances(&self) -> Chances {
        let mut weights = [0; 11];
        let count = match self.phase {
            Phase::Dice => {
                weights = DICE;
                11
            }
            Phase::Steal { .. } => {
                weights[..5].copy_from_slice(&self.hands[self.current.other().index()]);
                5
            }
            Phase::Draw => {
                weights[..5].copy_from_slice(&self.deck);
                5
            }
            phase => panic!("{phase:?} is not a chance node"),
        };
        Chances { weights, count }
    }

    pub fn outcome(&self, index: u16) -> Option<Outcome> {
        let i = usize::from(index);
        match self.phase {
            Phase::Dice if i < DICE.len() => Some(Outcome::Dice(i as u8 + 2)),
            Phase::Steal { .. } if i < 5 => Some(Outcome::Steal(Resource::ALL[i])),
            Phase::Draw if i < 5 => Some(Outcome::Draw(DevCard::ALL[i])),
            _ => None,
        }
    }

    pub fn sample_outcome(&self, rng: &mut Rng) -> Outcome {
        let index = rng.weighted(self.chances().weights());
        self.outcome(index as u16).expect("an outcome of this node")
    }

    pub fn resolve(&mut self, outcome: Outcome) {
        debug_assert!(
            self.chances().weights()[usize::from(outcome.index())] > 0,
            "{outcome:?} cannot happen in {:?}",
            self.phase
        );
        self.happen(outcome);
        self.settle();
    }

    fn happen(&mut self, outcome: Outcome) {
        let player = self.current;
        match (self.phase, outcome) {
            (Phase::Dice, Outcome::Dice(7)) => self.start_discards(),
            (Phase::Dice, Outcome::Dice(sum)) => {
                self.produce(sum);
                self.phase = Phase::Turn { rolled: true };
            }
            (Phase::Steal { rolled }, Outcome::Steal(resource)) => {
                let r = resource.index();
                self.hands[player.other().index()][r] -= 1;
                self.hands[player.index()][r] += 1;
                self.phase = Phase::Turn { rolled };
            }
            (Phase::Draw, Outcome::Draw(card)) => {
                let c = card.index();
                self.deck[c] -= 1;
                self.dev_cards[player.index()][c] += 1;
                self.bought[c] += 1;
                self.phase = Phase::Turn { rolled: true };
                self.finish_if_won();
            }
            (phase, outcome) => panic!("{outcome:?} cannot happen in {phase:?}"),
        }
    }

    pub fn step(&mut self, code: u16) -> Result<(), Invalid> {
        match self.actor() {
            Some(Actor::Chance) => match self.outcome(code) {
                Some(outcome) if self.chances().weights()[usize::from(code)] > 0 => self.resolve(outcome),
                _ => return Err(Invalid { code }),
            },
            Some(Actor::Player(_)) => match Action::from_index(code) {
                Some(action) if self.legal_actions().contains(action) => self.apply(action),
                _ => return Err(Invalid { code }),
            },
            None => return Err(Invalid { code }),
        }
        Ok(())
    }

    fn settle(&mut self) {
        while let Some(Actor::Player(_)) = self.actor() {
            let legal = self.legal_actions();
            debug_assert!(!legal.is_empty(), "a player to act has an action in {:?}", self.phase);
            if legal.len() != 1 {
                return;
            }
            self.transition(legal.iter().next().expect("one action"));
        }
    }

    fn transition(&mut self, action: Action) {
        let player = self.current;
        let p = player.index();
        match (self.phase, action) {
            (Phase::PlaceSettlement, Action::Settlement(vertex)) => {
                self.place_settlement(vertex);
                if self.settlements[p].len() == 2 {
                    for hex in topology().hexes_at(vertex).iter() {
                        if let Some(resource) = self.board.terrain(hex).resource() {
                            self.hands[p][resource.index()] += 1;
                        }
                    }
                }
                self.phase = Phase::PlaceRoad { settlement: vertex };
            }
            (Phase::PlaceRoad { .. }, Action::Road(edge)) => {
                self.place_road(edge);
                let placed = self.settlements[0].len() + self.settlements[1].len();
                self.current = match placed {
                    1 | 2 => Player::Second,
                    _ => Player::First,
                };
                self.phase = if placed < 4 { Phase::PlaceSettlement } else { Phase::Turn { rolled: false } };
            }
            (Phase::Turn { rolled: false }, Action::Roll) => self.phase = Phase::Dice,
            (Phase::Turn { rolled: true }, Action::Settlement(vertex)) => {
                self.pay(player, SETTLEMENT_COST);
                self.place_settlement(vertex);
                self.finish_if_won();
            }
            (Phase::Turn { rolled: true }, Action::City(vertex)) => {
                self.pay(player, CITY_COST);
                self.settlements[p].remove(vertex);
                self.cities[p].insert(vertex);
                self.finish_if_won();
            }
            (Phase::Turn { rolled: true }, Action::Road(edge)) => {
                self.pay(player, ROAD_COST);
                self.place_road(edge);
                self.finish_if_won();
            }
            (Phase::Turn { rolled: true }, Action::Trade { give, get, rate }) => {
                self.hands[p][give.index()] -= rate;
                self.hands[p][get.index()] += 1;
            }
            (Phase::Turn { rolled: true }, Action::BuyCard) => {
                self.pay(player, CARD_COST);
                self.phase = Phase::Draw;
            }
            (Phase::Turn { rolled: true }, Action::EndTurn) => {
                self.current = player.other();
                self.bought = [0; 5];
                self.card_played = false;
                self.phase = Phase::Turn { rolled: false };
            }
            (Phase::Turn { rolled }, Action::PlayKnight) => {
                self.play_card(DevCard::Knight);
                self.knights[p] += 1;
                self.largest_army = award(self.largest_army, self.knights, 3);
                self.phase = Phase::MoveRobber { rolled };
                self.finish_if_won();
            }
            (Phase::Turn { rolled }, Action::PlayRoadBuilding) => {
                self.play_card(DevCard::RoadBuilding);
                self.continue_road_building(rolled, 2);
            }
            (Phase::Turn { rolled }, Action::PlayYearOfPlenty) => {
                self.play_card(DevCard::YearOfPlenty);
                self.continue_year_of_plenty(rolled, 2);
            }
            (Phase::Turn { rolled }, Action::PlayMonopoly) => {
                self.play_card(DevCard::Monopoly);
                self.phase = Phase::Monopoly { rolled };
            }
            (Phase::Discard { player, owed, .. }, Action::Resource(resource)) => {
                let r = resource.index();
                self.hands[player.index()][r] -= 1;
                if player == self.current {
                    self.set_aside[r] += 1;
                }
                let other = player.other();
                self.phase = if owed > 1 {
                    Phase::Discard { player, owed: owed - 1, from: resource }
                } else if player == self.current && self.hand_size(other) > 7 {
                    Phase::Discard { player: other, owed: self.hand_size(other) / 2, from: Resource::Brick }
                } else {
                    self.set_aside = [0; 5];
                    Phase::MoveRobber { rolled: true }
                };
            }
            (Phase::MoveRobber { rolled }, Action::Robber(hex)) => {
                self.robber = hex;
                let victim = player.other();
                let exposed = !(self.buildings(victim) & topology().corner_set(hex)).is_empty();
                self.phase = if exposed && self.hand_size(victim) > 0 {
                    Phase::Steal { rolled }
                } else {
                    Phase::Turn { rolled }
                };
            }
            (Phase::RoadBuilding { rolled, roads }, Action::Road(edge)) => {
                self.place_road(edge);
                self.continue_road_building(rolled, roads - 1);
                self.finish_if_won();
            }
            (Phase::YearOfPlenty { rolled, picks }, Action::Resource(resource)) => {
                self.hands[p][resource.index()] += 1;
                self.continue_year_of_plenty(rolled, picks - 1);
            }
            (Phase::Monopoly { rolled }, Action::Resource(resource)) => {
                let (r, other) = (resource.index(), player.other().index());
                self.hands[p][r] += self.hands[other][r];
                self.hands[other][r] = 0;
                self.phase = Phase::Turn { rolled };
            }
            (phase, action) => panic!("{action:?} is not legal in {phase:?}"),
        }
    }

    fn place_settlement(&mut self, vertex: Vertex) {
        let player = self.current;
        self.settlements[player.index()].insert(vertex);
        let other = player.other();
        if (topology().edges_at(vertex) & self.roads[other.index()]).len() >= 2 {
            self.road_lengths[other.index()] = self.find_road_length(other);
            self.longest_road = award(self.longest_road, self.road_lengths, 5);
        }
    }

    fn place_road(&mut self, edge: Edge) {
        let player = self.current;
        self.roads[player.index()].insert(edge);
        self.road_lengths[player.index()] = self.find_road_length(player);
        self.longest_road = award(self.longest_road, self.road_lengths, 5);
    }

    pub(super) fn find_road_length(&self, player: Player) -> u8 {
        longest_road(self.roads[player.index()], self.buildings(player.other()))
    }

    fn can_pay(&self, player: Player, cost: [u8; 5]) -> bool {
        self.hands[player.index()].iter().zip(cost).all(|(&held, needed)| held >= needed)
    }

    fn pay(&mut self, player: Player, cost: [u8; 5]) {
        for (held, needed) in self.hands[player.index()].iter_mut().zip(cost) {
            *held -= needed;
        }
    }

    fn play_card(&mut self, card: DevCard) {
        self.dev_cards[self.current.index()][card.index()] -= 1;
        self.card_played = true;
    }

    fn insert_playable_cards(&self, mask: &mut Mask) {
        if self.card_played {
            return;
        }
        let p = self.current.index();
        for (card, action) in [
            (DevCard::Knight, Action::PlayKnight),
            (DevCard::RoadBuilding, Action::PlayRoadBuilding),
            (DevCard::YearOfPlenty, Action::PlayYearOfPlenty),
            (DevCard::Monopoly, Action::PlayMonopoly),
        ] {
            if self.dev_cards[p][card.index()] > self.bought[card.index()] {
                mask.insert(action);
            }
        }
    }

    fn has_rate(&self, player: Player, give: Resource, rate: u8) -> bool {
        let harbor = match rate {
            4 => return true,
            3 => Harbor::Generic,
            _ => Harbor::Special(give),
        };
        !(self.buildings(player) & self.board.harbor_vertices(harbor)).is_empty()
    }

    fn all_roads(&self) -> Set<EdgeKind> {
        self.roads[0] | self.roads[1]
    }

    fn blocked(&self) -> Set<VertexKind> {
        let t = topology();
        let occupied = self.buildings(Player::First) | self.buildings(Player::Second);
        occupied.iter().fold(occupied, |blocked, v| blocked | t.adjacent(v))
    }

    fn road_ends(&self, player: Player) -> Set<VertexKind> {
        let t = topology();
        self.roads[player.index()].iter().flat_map(|e| t.ends(e)).collect()
    }

    fn settlement_spots(&self, player: Player) -> Set<VertexKind> {
        if self.settlements[player.index()].len() >= SETTLEMENTS {
            return Set::EMPTY;
        }
        self.road_ends(player) - self.blocked()
    }

    fn city_spots(&self, player: Player) -> Set<VertexKind> {
        if self.cities[player.index()].len() >= CITIES {
            return Set::EMPTY;
        }
        self.settlements[player.index()]
    }

    pub(super) fn road_spots(&self, player: Player) -> Set<EdgeKind> {
        if self.roads[player.index()].len() >= ROADS {
            return Set::EMPTY;
        }
        let t = topology();
        let reach = self.buildings(player) | (self.road_ends(player) - self.buildings(player.other()));
        reach.iter().fold(Set::EMPTY, |edges, v| edges | t.edges_at(v)) - self.all_roads()
    }

    fn produce(&mut self, sum: u8) {
        let t = topology();
        let mut owed = [[0u8; 5]; 2];
        for hex in Hex::all() {
            if self.board.token(hex) != Some(sum) || hex == self.robber {
                continue;
            }
            let resource = self.board.terrain(hex).resource().expect("a hex with a token produces");
            let corners = t.corner_set(hex);
            for (p, owed) in owed.iter_mut().enumerate() {
                owed[resource.index()] += (self.settlements[p] & corners).len() + 2 * (self.cities[p] & corners).len();
            }
        }
        for resource in Resource::ALL {
            let (r, bank) = (resource.index(), self.bank(resource));
            let (first, second) = (owed[0][r], owed[1][r]);
            if first + second <= bank {
                self.hands[0][r] += first;
                self.hands[1][r] += second;
            } else if second == 0 {
                self.hands[0][r] += bank;
            } else if first == 0 {
                self.hands[1][r] += bank;
            }
        }
    }

    fn start_discards(&mut self) {
        let (first, second) = (self.current, self.current.other());
        self.phase = match [first, second].into_iter().find(|&p| self.hand_size(p) > 7) {
            Some(player) => Phase::Discard { player, owed: self.hand_size(player) / 2, from: Resource::Brick },
            None => Phase::MoveRobber { rolled: true },
        };
    }

    fn continue_road_building(&mut self, rolled: bool, roads: u8) {
        self.phase = if roads > 0 && !self.road_spots(self.current).is_empty() {
            Phase::RoadBuilding { rolled, roads }
        } else {
            Phase::Turn { rolled }
        };
    }

    fn continue_year_of_plenty(&mut self, rolled: bool, picks: u8) {
        self.phase = if picks > 0 && Resource::ALL.iter().any(|&r| self.bank(r) > 0) {
            Phase::YearOfPlenty { rolled, picks }
        } else {
            Phase::Turn { rolled }
        };
    }

    fn finish_if_won(&mut self) {
        if self.victory_points(self.current) >= WINNING_POINTS {
            self.phase = Phase::Over { winner: self.current };
        }
    }
}

pub(super) fn award(holder: Option<Player>, counts: [u8; 2], least: u8) -> Option<Player> {
    let count = |p: Player| counts[p.index()];
    match holder {
        Some(h) if count(h) >= least && count(h) >= count(h.other()) => Some(h),
        _ => Player::BOTH.into_iter().find(|&p| count(p) >= least && count(p) > count(p.other())),
    }
}

pub fn longest_road(roads: Set<EdgeKind>, blocked: Set<VertexKind>) -> u8 {
    let t = topology();
    let starts: Set<VertexKind> = roads.iter().flat_map(|e| t.ends(e)).collect();
    starts.iter().map(|start| longest_trail(start, roads, blocked)).max().unwrap_or(0)
}

fn longest_trail(at: Vertex, unused: Set<EdgeKind>, blocked: Set<VertexKind>) -> u8 {
    let t = topology();
    (t.edges_at(at) & unused)
        .iter()
        .map(|edge| {
            let [a, b] = t.ends(edge);
            let next = if a == at { b } else { a };
            let mut rest = unused;
            rest.remove(edge);
            1 + if blocked.contains(next) { 0 } else { longest_trail(next, rest, blocked) }
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests;
