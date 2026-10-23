use std::fmt;

use crate::board::Harbor;
use crate::dev_card::DevCard;
use crate::ids::{Edge, EdgeKind, Hex, Set, Vertex, VertexKind, EDGES, VERTICES};
use crate::player::{Player, MAX_PLAYERS};
use crate::resource::{Resource, RESOURCE_CARDS};
use crate::rng::Rng;
use crate::symmetry::Symmetry;
use crate::topology::topology;

use super::{Actor, Phase, State};

pub const ACTIONS: usize = 292;

pub const MAX_OFFERS: u8 = 3;
pub const MAX_OFFER_CARDS: u8 = 4;

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
const ROB: usize = 271;
const OFFER: usize = 274;
const GIVE: usize = 275;
const WANT: usize = 280;
const PROPOSE: usize = 285;
const CANCEL: usize = 286;
const ACCEPT: usize = 287;
const DECLINE: usize = 288;
const TRADE_WITH: usize = 289;

const _: () = assert!(CITY == SETTLEMENT + VERTICES && ROAD == CITY + VERTICES);
const _: () = assert!(ROBBER == ROAD + EDGES && RESOURCE == ROBBER + 19 && TRADE == RESOURCE + 5);
const _: () = assert!(ROLL == TRADE + 3 * 5 * 4 && PLAY_MONOPOLY + 1 == ROB && ROB + MAX_PLAYERS - 1 == OFFER);
const _: () = assert!(GIVE == OFFER + 1 && WANT == GIVE + 5 && PROPOSE == WANT + 5 && TRADE_WITH == DECLINE + 1);
const _: () = assert!(TRADE_WITH + MAX_PLAYERS - 1 == ACTIONS);

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
    Rob(u8),
    Offer,
    Give(Resource),
    Want(Resource),
    Propose,
    Cancel,
    Accept,
    Decline,
    TradeWith(u8),
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
            Action::Rob(offset) => ROB + usize::from(offset) - 1,
            Action::Offer => OFFER,
            Action::Give(r) => GIVE + r.index(),
            Action::Want(r) => WANT + r.index(),
            Action::Propose => PROPOSE,
            Action::Cancel => CANCEL,
            Action::Accept => ACCEPT,
            Action::Decline => DECLINE,
            Action::TradeWith(offset) => TRADE_WITH + usize::from(offset) - 1,
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
            _ if i < OFFER => Action::Rob((i - ROB + 1) as u8),
            OFFER => Action::Offer,
            _ if i < WANT => Action::Give(Resource::ALL[i - GIVE]),
            _ if i < PROPOSE => Action::Want(Resource::ALL[i - WANT]),
            PROPOSE => Action::Propose,
            CANCEL => Action::Cancel,
            ACCEPT => Action::Accept,
            DECLINE => Action::Decline,
            _ if i < ACTIONS => Action::TradeWith((i - TRADE_WITH + 1) as u8),
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
                if self.offers < MAX_OFFERS && self.hand_size(player) > 0 {
                    mask.insert(Action::Offer);
                }
                mask.insert(Action::EndTurn);
            }
            Phase::Offer { give, get } => {
                let (given, wanted) = (give.iter().sum::<u8>(), get.iter().sum::<u8>());
                if wanted == 0 && given < MAX_OFFER_CARDS {
                    for resource in Resource::ALL.into_iter().filter(|&r| r >= highest(give).unwrap_or(Resource::Brick)) {
                        if self.hands[p][resource.index()] > give[resource.index()] {
                            mask.insert(Action::Give(resource));
                        }
                    }
                }
                let largest_hand = self.others(player).map(|o| self.hand_size(o)).max().unwrap_or(0);
                if given > 0 && wanted < MAX_OFFER_CARDS && wanted < largest_hand {
                    for resource in Resource::ALL.into_iter().filter(|&r| r >= highest(get).unwrap_or(Resource::Brick)) {
                        let r = resource.index();
                        let elsewhere = RESOURCE_CARDS - self.bank(resource) - self.hands[p][r] - self.set_aside[p][r];
                        if give[r] == 0 && get[r] < elsewhere {
                            mask.insert(Action::Want(resource));
                        }
                    }
                }
                if wanted > 0 {
                    mask.insert(Action::Propose);
                }
                mask.insert(Action::Cancel);
            }
            Phase::Respond { get, responder, .. } => {
                if self.can_pay(responder, get) {
                    mask.insert(Action::Accept);
                }
                mask.insert(Action::Decline);
            }
            Phase::Confirm { accepted, .. } => {
                for offset in 1..self.players() {
                    if accepted >> player.after(offset, self.players()).index() & 1 == 1 {
                        mask.insert(Action::TradeWith(offset as u8));
                    }
                }
                mask.insert(Action::Cancel);
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
            Phase::ChooseVictim { .. } => {
                for offset in 1..self.players() {
                    if self.can_rob(player.after(offset, self.players())) {
                        mask.insert(Action::Rob(offset as u8));
                    }
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
            Phase::Steal { victim, .. } => {
                weights[..5].copy_from_slice(&self.hands[victim.index()]);
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
            (Phase::Steal { rolled, victim }, Outcome::Steal(resource)) => {
                self.steal(player, victim, resource.index());
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
                            self.gain(player, resource.index(), 1);
                        }
                    }
                }
                self.phase = Phase::PlaceRoad { settlement: vertex };
            }
            (Phase::PlaceRoad { .. }, Action::Road(edge)) => {
                self.place_road(edge);
                let placed: usize = self.seats().map(|q| usize::from(self.settlements[q.index()].len())).sum();
                if placed < 2 * self.players() {
                    self.current = self.placer(placed);
                    self.phase = Phase::PlaceSettlement;
                } else {
                    self.current = Player::First;
                    self.phase = Phase::Turn { rolled: false };
                }
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
                self.spend(player, give.index(), rate);
                self.gain(player, get.index(), 1);
            }
            (Phase::Turn { rolled: true }, Action::BuyCard) => {
                self.pay(player, CARD_COST);
                self.phase = Phase::Draw;
            }
            (Phase::Turn { rolled: true }, Action::Offer) => {
                self.offers += 1;
                self.phase = Phase::Offer { give: [0; 5], get: [0; 5] };
            }
            (Phase::Offer { mut give, get }, Action::Give(resource)) => {
                give[resource.index()] += 1;
                self.phase = Phase::Offer { give, get };
            }
            (Phase::Offer { give, mut get }, Action::Want(resource)) => {
                get[resource.index()] += 1;
                self.phase = Phase::Offer { give, get };
            }
            (Phase::Offer { give, get }, Action::Propose) => {
                self.phase = Phase::Respond { give, get, responder: self.next(player), accepted: 0 };
            }
            (Phase::Offer { .. } | Phase::Confirm { .. }, Action::Cancel) => self.phase = Phase::Turn { rolled: true },
            (Phase::Respond { give, get, responder, accepted }, Action::Accept | Action::Decline) => {
                let mut accepted = accepted;
                if action == Action::Accept {
                    accepted |= 1 << responder.index();
                    for observer in self.others(responder) {
                        let known = &mut self.known[observer.index()][responder.index()];
                        (0..5).for_each(|r| known[r] = known[r].max(get[r]));
                    }
                }
                let next = self.next(responder);
                self.phase = if next != player {
                    Phase::Respond { give, get, responder: next, accepted }
                } else if accepted != 0 {
                    Phase::Confirm { give, get, accepted }
                } else {
                    Phase::Turn { rolled: true }
                };
            }
            (Phase::Confirm { give, get, .. }, Action::TradeWith(offset)) => {
                let partner = player.after(usize::from(offset), self.players());
                for r in 0..5 {
                    self.spend(player, r, give[r]);
                    self.spend(partner, r, get[r]);
                    self.gain(player, r, get[r]);
                    self.gain(partner, r, give[r]);
                }
                self.phase = Phase::Turn { rolled: true };
            }
            (Phase::Turn { rolled: true }, Action::EndTurn) => {
                self.current = self.next(player);
                self.bought = [0; 5];
                self.card_played = false;
                self.offers = 0;
                self.phase = Phase::Turn { rolled: false };
                self.finish_if_won();
            }
            (Phase::Turn { rolled }, Action::PlayKnight) => {
                self.play_card(DevCard::Knight);
                self.knights[p] += 1;
                self.largest_army = award(self.largest_army, &self.knights[..self.players()], 3);
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
            (Phase::Discard { player: discarder, owed, .. }, Action::Resource(resource)) => {
                let (d, r) = (discarder.index(), resource.index());
                self.hands[d][r] -= 1;
                self.set_aside[d][r] += 1;
                self.phase = if owed > 1 {
                    Phase::Discard { player: discarder, owed: owed - 1, from: resource }
                } else if let Some(next) = self.next_discarder(discarder) {
                    Phase::Discard { player: next, owed: self.hand_size(next) / 2, from: Resource::Brick }
                } else {
                    self.reveal_discards();
                    Phase::MoveRobber { rolled: true }
                };
            }
            (Phase::MoveRobber { rolled }, Action::Robber(hex)) => {
                self.robber = hex;
                let any = self.others(player).any(|victim| self.can_rob(victim));
                self.phase = if any { Phase::ChooseVictim { rolled } } else { Phase::Turn { rolled } };
            }
            (Phase::ChooseVictim { rolled }, Action::Rob(offset)) => {
                let victim = player.after(usize::from(offset), self.players());
                self.phase = Phase::Steal { rolled, victim };
            }
            (Phase::RoadBuilding { rolled, roads }, Action::Road(edge)) => {
                self.place_road(edge);
                self.continue_road_building(rolled, roads - 1);
                self.finish_if_won();
            }
            (Phase::YearOfPlenty { rolled, picks }, Action::Resource(resource)) => {
                self.gain(player, resource.index(), 1);
                self.continue_year_of_plenty(rolled, picks - 1);
            }
            (Phase::Monopoly { rolled }, Action::Resource(resource)) => {
                let r = resource.index();
                let mut taken = 0;
                for other in self.others(player) {
                    let held = self.hands[other.index()][r];
                    self.spend(other, r, held);
                    taken += held;
                }
                self.gain(player, r, taken);
                self.phase = Phase::Turn { rolled };
            }
            (phase, action) => panic!("{action:?} is not legal in {phase:?}"),
        }
    }

    fn placer(&self, placed: usize) -> Player {
        let n = self.players();
        Player::new(if placed < n { placed } else { 2 * n - 1 - placed })
    }

    fn gain(&mut self, player: Player, r: usize, count: u8) {
        let p = player.index();
        self.hands[p][r] += count;
        for observer in self.others(player) {
            self.known[observer.index()][p][r] += count;
        }
    }

    fn spend(&mut self, player: Player, r: usize, count: u8) {
        let p = player.index();
        self.hands[p][r] -= count;
        for observer in self.others(player) {
            let known = &mut self.known[observer.index()][p][r];
            *known = known.saturating_sub(count);
        }
    }

    fn steal(&mut self, thief: Player, victim: Player, r: usize) {
        let (t, v) = (thief.index(), victim.index());
        self.hands[v][r] -= 1;
        self.hands[t][r] += 1;
        self.known[t][v][r] = self.known[t][v][r].saturating_sub(1);
        self.known[v][t][r] += 1;
        for observer in self.others(thief).filter(|&o| o != victim) {
            for known in self.known[observer.index()][v].iter_mut() {
                *known = known.saturating_sub(1);
            }
        }
    }

    fn reveal_discards(&mut self) {
        for player in self.seats() {
            let p = player.index();
            for observer in self.others(player) {
                for r in 0..5 {
                    let known = &mut self.known[observer.index()][p][r];
                    *known = known.saturating_sub(self.set_aside[p][r]);
                }
            }
        }
        self.set_aside = [[0; 5]; MAX_PLAYERS];
    }

    fn next_discarder(&self, after: Player) -> Option<Player> {
        let mut candidate = self.next(after);
        while candidate != self.current {
            if self.hand_size(candidate) > 7 {
                return Some(candidate);
            }
            candidate = self.next(candidate);
        }
        None
    }

    pub(super) fn can_rob(&self, victim: Player) -> bool {
        victim != self.current
            && self.hand_size(victim) > 0
            && !(self.buildings(victim) & topology().corner_set(self.robber)).is_empty()
    }

    fn place_settlement(&mut self, vertex: Vertex) {
        let player = self.current;
        self.settlements[player.index()].insert(vertex);
        let mut cut = false;
        for other in self.others(player) {
            if (topology().edges_at(vertex) & self.roads[other.index()]).len() >= 2 {
                self.road_lengths[other.index()] = self.find_road_length(other);
                cut = true;
            }
        }
        if cut {
            self.longest_road = award(self.longest_road, &self.road_lengths[..self.players()], 5);
        }
    }

    fn place_road(&mut self, edge: Edge) {
        let player = self.current;
        self.roads[player.index()].insert(edge);
        self.road_lengths[player.index()] = self.find_road_length(player);
        self.longest_road = award(self.longest_road, &self.road_lengths[..self.players()], 5);
    }

    pub(super) fn find_road_length(&self, player: Player) -> u8 {
        longest_road(self.roads[player.index()], self.all_buildings() - self.buildings(player))
    }

    pub(super) fn can_pay(&self, player: Player, cost: [u8; 5]) -> bool {
        self.hands[player.index()].iter().zip(cost).all(|(&held, needed)| held >= needed)
    }

    fn pay(&mut self, player: Player, cost: [u8; 5]) {
        for (r, needed) in cost.into_iter().enumerate() {
            self.spend(player, r, needed);
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
        self.seats().fold(Set::EMPTY, |all, p| all | self.roads[p.index()])
    }

    fn blocked(&self) -> Set<VertexKind> {
        let t = topology();
        let occupied = self.all_buildings();
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
        let theirs = self.all_buildings() - self.buildings(player);
        let reach = self.buildings(player) | (self.road_ends(player) - theirs);
        reach.iter().fold(Set::EMPTY, |edges, v| edges | t.edges_at(v)) - self.all_roads()
    }

    fn produce(&mut self, sum: u8) {
        let t = topology();
        let mut owed = [[0u8; 5]; MAX_PLAYERS];
        for hex in Hex::all() {
            if self.board.token(hex) != Some(sum) || hex == self.robber {
                continue;
            }
            let resource = self.board.terrain(hex).resource().expect("a hex with a token produces");
            let corners = t.corner_set(hex);
            for player in self.seats() {
                let p = player.index();
                owed[p][resource.index()] += (self.settlements[p] & corners).len() + 2 * (self.cities[p] & corners).len();
            }
        }
        for resource in Resource::ALL {
            let (r, bank) = (resource.index(), self.bank(resource));
            let total: u8 = self.seats().map(|p| owed[p.index()][r]).sum();
            let owed_to: Vec<Player> = self.seats().filter(|p| owed[p.index()][r] > 0).collect();
            if total <= bank {
                for player in owed_to {
                    self.gain(player, r, owed[player.index()][r]);
                }
            } else if let [only] = owed_to[..] {
                self.gain(only, r, bank);
            }
        }
    }

    fn start_discards(&mut self) {
        let n = self.players();
        let first = (0..n).map(|offset| self.current.after(offset, n)).find(|&p| self.hand_size(p) > 7);
        self.phase = match first {
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

pub(super) fn highest(counts: [u8; 5]) -> Option<Resource> {
    (0..5).rev().find(|&r| counts[r] > 0).map(|r| Resource::ALL[r])
}

pub(super) fn award(holder: Option<Player>, counts: &[u8], least: u8) -> Option<Player> {
    let leads = |p: usize, strictly: bool| (0..counts.len()).all(|q| q == p || if strictly { counts[p] > counts[q] } else { counts[p] >= counts[q] });
    match holder {
        Some(h) if counts[h.index()] >= least && leads(h.index(), false) => Some(h),
        _ => (0..counts.len()).find(|&p| counts[p] >= least && leads(p, true)).map(Player::new),
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
