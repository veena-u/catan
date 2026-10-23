use std::fmt;

use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{Set, VertexKind};
use crate::player::{Player, MAX_PLAYERS};
use crate::resource::{Resource, RESOURCE_CARDS};
use crate::topology::topology;

use super::rules::{award, longest_road, Deal, DEALS, MAX_OFFERS};
use super::{Actor, Phase, State, WINNING_POINTS};

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Violation(pub String);

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Violation {}

fn ensure(condition: bool, message: impl FnOnce() -> String) -> Result<(), Violation> {
    if condition { Ok(()) } else { Err(Violation(message())) }
}

impl State {
    pub fn validate(&self) -> Result<(), Violation> {
        self.validate_seats()?;
        self.validate_board()?;
        self.validate_cards()?;
        self.validate_knowledge()?;
        self.validate_turn()
    }

    fn validate_seats(&self) -> Result<(), Violation> {
        let n = self.players();
        ensure((2..=MAX_PLAYERS).contains(&n), || format!("{n} players"))?;
        for p in n..MAX_PLAYERS {
            ensure(
                self.settlements[p].is_empty()
                    && self.cities[p].is_empty()
                    && self.roads[p].is_empty()
                    && self.hands[p] == [0; 5]
                    && self.dev_cards[p] == [0; 5]
                    && self.set_aside[p] == [0; 5]
                    && self.knights[p] == 0
                    && self.road_lengths[p] == 0
                    && self.known[p] == [[0; 5]; MAX_PLAYERS]
                    && self.known.iter().all(|row| row[p] == [0; 5]),
                || format!("seat {p} is empty in a {n}-player game but holds something"),
            )?;
        }
        let seated = |p: Option<Player>| p.is_none_or(|p| p.index() < n);
        ensure(
            self.current.index() < n && seated(self.longest_road) && seated(self.largest_army),
            || "a player who is not seated".into(),
        )?;
        let in_phase = match self.phase {
            Phase::Discard { player, .. } | Phase::Steal { victim: player, .. } | Phase::Over { winner: player } => Some(player),
            _ => None,
        };
        ensure(seated(in_phase), || "the phase names a player who is not seated".into())
    }

    fn validate_board(&self) -> Result<(), Violation> {
        let t = topology();
        let all: Vec<Set<VertexKind>> = self.seats().flat_map(|p| [self.settlements(p), self.cities(p)]).collect();
        for i in 0..all.len() {
            for j in i + 1..all.len() {
                ensure((all[i] & all[j]).is_empty(), || "a vertex holds two buildings".into())?;
            }
        }
        for a in self.seats() {
            for b in self.seats().filter(|&b| b > a) {
                ensure((self.roads(a) & self.roads(b)).is_empty(), || "an edge holds two roads".into())?;
            }
        }
        let occupied = self.all_buildings();
        for v in occupied.iter() {
            ensure(
                (t.adjacent(v) & occupied).is_empty(),
                || format!("buildings on adjacent vertices at {v:?}"),
            )?;
        }
        for player in self.seats() {
            let p = player.index();
            ensure(
                self.settlements[p].len() <= 5 && self.cities[p].len() <= 4,
                || "too many buildings".into(),
            )?;
            ensure(self.roads[p].len() <= 15, || "too many roads".into())?;
            let buildings = self.buildings(player);
            let mut reached = Set::EMPTY;
            let mut frontier = buildings;
            while !frontier.is_empty() {
                let edges = frontier.iter().fold(Set::EMPTY, |e, v| e | t.edges_at(v)) & self.roads[p];
                let new = edges - reached;
                reached = reached | new;
                frontier = new.iter().flat_map(|e| t.ends(e)).collect::<Set<VertexKind>>() - buildings;
            }
            ensure(
                reached == self.roads[p],
                || format!("{player:?} has a road not connected to a building"),
            )?;
            let waiting = match self.phase {
                Phase::PlaceRoad { settlement } if self.current == player => Set::single(settlement),
                _ => Set::EMPTY,
            };
            for v in (buildings - waiting).iter() {
                ensure(
                    !(t.edges_at(v) & self.roads[p]).is_empty(),
                    || format!("{player:?}'s building at {v:?} has no road"),
                )?;
            }
            ensure(
                self.road_lengths[p] == longest_road(self.roads[p], self.all_buildings() - buildings),
                || format!("{player:?}'s road length is stale"),
            )?;
        }
        Ok(())
    }

    fn validate_cards(&self) -> Result<(), Violation> {
        for resource in Resource::ALL {
            let r = resource.index();
            let held: u32 = self.seats().map(|p| u32::from(self.hands[p.index()][r] + self.set_aside[p.index()][r])).sum();
            ensure(held <= u32::from(RESOURCE_CARDS), || format!("more than 19 {resource:?} in play"))?;
        }
        for card in DevCard::ALL {
            let c = card.index();
            let held: u32 = u32::from(self.deck[c]) + self.seats().map(|p| u32::from(self.dev_cards[p.index()][c])).sum::<u32>();
            ensure(held <= u32::from(DEV_CARDS[c]), || format!("more than {} {card:?} cards", DEV_CARDS[c]))?;
            ensure(
                self.bought[c] <= self.dev_cards[self.current.index()][c],
                || format!("bought {card:?} not held"),
            )?;
        }
        let knights: u8 = self.seats().map(|p| self.knights[p.index()]).sum();
        ensure(
            self.played(DevCard::Knight) == knights,
            || "the knights played do not match the knights missing".into(),
        )?;
        ensure(self.played(DevCard::VictoryPoint) == 0, || "a victory point card was played".into())?;
        let n = self.players();
        ensure(
            self.longest_road == award(self.longest_road, &self.road_lengths[..n], 5),
            || "the longest road is not where the lengths put it".into(),
        )?;
        ensure(
            self.longest_road.is_some() || self.road_lengths[..n].iter().all(|&l| l < 5) || !leader(&self.road_lengths[..n]),
            || "an unheld road of 5 with a single leader".into(),
        )?;
        ensure(
            self.largest_army == award(self.largest_army, &self.knights[..n], 3),
            || "the largest army is not where the knights put it".into(),
        )?;
        ensure(
            self.largest_army.is_some() || self.knights[..n].iter().all(|&k| k < 3) || !leader(&self.knights[..n]),
            || "an unheld army of 3 with a single leader".into(),
        )?;
        Ok(())
    }

    fn validate_knowledge(&self) -> Result<(), Violation> {
        for observer in self.seats() {
            ensure(
                self.known[observer.index()][observer.index()] == [0; 5],
                || "a player's knowledge of their own hand is stored".into(),
            )?;
            for player in self.others(observer) {
                let (o, p) = (observer.index(), player.index());
                for r in 0..5 {
                    ensure(
                        self.known[o][p][r] <= self.hands[p][r] + self.set_aside[p][r],
                        || format!("{observer:?} is sure {player:?} holds more {:?} than they do", Resource::ALL[r]),
                    )?;
                }
            }
        }
        if self.players() == 2 && !matches!(self.phase, Phase::Discard { .. }) {
            for player in self.seats() {
                let other = self.next(player);
                ensure(
                    self.known(player, other) == self.hand(other),
                    || "with two players every hand is known".into(),
                )?;
            }
        }
        Ok(())
    }

    fn validate_turn(&self) -> Result<(), Violation> {
        let n = self.players();
        let placed: usize = self.seats().map(|p| usize::from(self.buildings(p).len())).sum();
        let in_setup = matches!(self.phase, Phase::PlaceSettlement | Phase::PlaceRoad { .. });
        let fresh_turn = matches!(self.phase, Phase::Turn { rolled: false }) && !self.card_played;
        let set_aside: u8 = self.seats().flat_map(|p| self.set_aside[p.index()]).sum();
        let bought: u8 = self.bought.iter().sum();
        if in_setup {
            let roads: usize = self.seats().map(|p| usize::from(self.roads(p).len())).sum();
            let (roads_expected, placing) = match self.phase {
                Phase::PlaceSettlement => (placed, placed),
                _ => (placed.saturating_sub(1), placed.saturating_sub(1)),
            };
            ensure(
                placed >= 1 || matches!(self.phase, Phase::PlaceSettlement),
                || "a setup road with no settlement".into(),
            )?;
            ensure(roads == roads_expected && placing < 2 * n, || "setup roads out of step".into())?;
            let expected = Player::new(if placing < n { placing } else { 2 * n - 1 - placing });
            ensure(self.current == expected, || "the wrong player is placing".into())?;
            for player in self.seats() {
                let p = player.index();
                let cards: u8 = self.hands[p].iter().sum();
                let earned = if self.settlements[p].len() == 2 { 3 } else { 0 };
                ensure(cards <= earned, || format!("{player:?} holds {cards} cards in setup"))?;
            }
            ensure(
                self.dev_cards == [[0; 5]; MAX_PLAYERS] && !self.card_played && bought == 0,
                || "cards in setup".into(),
            )?;
        } else {
            for player in self.seats() {
                let p = player.index();
                ensure(
                    self.buildings(player).len() >= 2 && self.roads[p].len() >= 2,
                    || "setup unfinished".into(),
                )?;
            }
        }
        ensure(!fresh_turn || bought == 0, || "cards bought before the roll".into())?;
        ensure(self.offers <= MAX_OFFERS, || "more offers than allowed".into())?;
        let before_roll = match self.phase {
            Phase::PlaceSettlement | Phase::PlaceRoad { .. } | Phase::Dice => true,
            Phase::Turn { rolled }
            | Phase::MoveRobber { rolled }
            | Phase::ChooseVictim { rolled }
            | Phase::Steal { rolled, .. }
            | Phase::RoadBuilding { rolled, .. }
            | Phase::YearOfPlenty { rolled, .. }
            | Phase::Monopoly { rolled } => !rolled,
            _ => false,
        };
        ensure(!before_roll || self.offers == 0, || "offers made before the roll".into())?;
        match self.phase {
            Phase::PlaceRoad { settlement } => {
                ensure(
                    self.settlements[self.current.index()].contains(settlement),
                    || "no settlement to build from".into(),
                )?;
            }
            Phase::Discard { player, owed, from } => {
                let hand = &self.hands[player.index()];
                let from_here: u8 = hand[from.index()..].iter().sum();
                ensure(owed >= 1 && from_here >= owed, || "a discard that cannot be completed".into())?;
                let aside = self.set_aside[player.index()];
                let highest = (0..5).rev().find(|&r| aside[r] > 0).unwrap_or(0);
                ensure(from == Resource::ALL[highest], || "the discard order is out of step".into())?;
                let mut q = self.current;
                while q != player {
                    let pre = self.hand_size(q) + self.set_aside[q.index()].iter().sum::<u8>();
                    let discarded = pre > 7 && self.set_aside[q.index()].iter().sum::<u8>() == pre / 2;
                    ensure(discarded || pre <= 7, || format!("{q:?} skipped their discard"))?;
                    q = self.next(q);
                }
            }
            Phase::RoadBuilding { roads, .. } => {
                ensure((1..=2).contains(&roads), || format!("road building with {roads} roads"))?;
                ensure(
                    !self.road_spots(self.current).is_empty(),
                    || "road building with nowhere to build".into(),
                )?;
            }
            Phase::YearOfPlenty { picks, .. } => {
                ensure((1..=2).contains(&picks), || format!("year of plenty with {picks} picks"))?;
                ensure(
                    Resource::ALL.iter().any(|&r| self.bank(r) > 0),
                    || "year of plenty with an empty bank".into(),
                )?;
            }
            Phase::ChooseVictim { .. } => {
                ensure(
                    self.others(self.current).any(|v| self.can_rob(v)),
                    || "a victim to choose with no one to rob".into(),
                )?;
            }
            Phase::Steal { victim, .. } => {
                ensure(self.can_rob(victim), || "a steal from someone who cannot be robbed".into())?;
            }
            Phase::Draw => ensure(self.deck_size() > 0, || "a draw from an empty deck".into())?,
            Phase::Respond { deal, accepted, counters, .. } | Phase::Confirm { deal, accepted, counters } => {
                let known = |id: u8| usize::from(id) < DEALS;
                ensure(known(deal) && counters.iter().flatten().all(|&c| known(c)), || "a deal not on the menu".into())?;
                let offered = Deal::from_id(deal);
                ensure(self.can_pay(self.current, offered.give), || "an offer of cards the offerer does not hold".into())?;
                ensure(self.offers >= 1 && self.offers <= MAX_OFFERS, || "an offer beyond the limit".into())?;
                let asked = match self.phase {
                    Phase::Respond { responder, .. } => {
                        ensure(responder != self.current, || "the offerer answers their own offer".into())?;
                        let mut q = self.next(self.current);
                        let mut asked = 0u8;
                        while q != responder {
                            asked |= 1 << q.index();
                            q = self.next(q);
                        }
                        asked
                    }
                    _ => {
                        let any = accepted != 0 || counters.iter().any(Option::is_some);
                        ensure(any, || "no one to trade with".into())?;
                        self.others(self.current).fold(0u8, |bits, p| bits | 1 << p.index())
                    }
                };
                ensure(accepted & !asked == 0, || "an acceptance from a player not yet asked".into())?;
                for (p, counter) in counters.iter().enumerate() {
                    ensure(counter.is_none() || asked >> p & 1 == 1, || "a counter-offer from a player not yet asked".into())?;
                }
                for p in self.others(self.current) {
                    let i = p.index();
                    if accepted >> i & 1 == 1 {
                        ensure(counters[i].is_none(), || format!("{p:?} both accepted and countered"))?;
                        ensure(self.can_pay(p, offered.get), || format!("{p:?} accepted without the cards"))?;
                    }
                    if let Some(counter) = counters[i] {
                        let counter = Deal::from_id(counter);
                        ensure(counter != offered.reversed(), || format!("{p:?} countered with the same deal"))?;
                        ensure(self.can_pay(p, counter.give), || format!("{p:?} countered with cards they do not hold"))?;
                    }
                }
            }
            _ => {}
        }
        if !matches!(self.phase, Phase::Discard { .. }) {
            ensure(set_aside == 0, || "cards set aside outside the discards".into())?;
        }
        match self.phase {
            Phase::Over { winner } => {
                ensure(winner == self.current, || "a player won outside their turn".into())?;
                ensure(
                    self.victory_points(winner) >= WINNING_POINTS,
                    || "a winner without 10 points".into(),
                )?;
            }
            _ => {
                ensure(
                    self.victory_points(self.current) < WINNING_POINTS,
                    || format!("{:?} has won on their turn but play goes on", self.current),
                )?;
                for player in self.others(self.current) {
                    let points = self.victory_points(player);
                    let waiting = self.longest_road == Some(player) && points - 2 < WINNING_POINTS;
                    ensure(
                        points < WINNING_POINTS || waiting,
                        || format!("{player:?} has 10 points without having gained the last one off their turn"),
                    )?;
                }
            }
        }
        if let Some(Actor::Player(player)) = self.actor() {
            let options = self.legal_actions().len();
            ensure(
                options >= 2,
                || format!("{player:?} faces a forced decision ({options} options) in {:?}", self.phase),
            )?;
        }
        Ok(())
    }
}

fn leader(counts: &[u8]) -> bool {
    let best = counts.iter().copied().max().unwrap_or(0);
    counts.iter().filter(|&&c| c == best).count() == 1
}
