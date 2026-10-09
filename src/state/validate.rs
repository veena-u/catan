use std::fmt;

use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{Set, VertexKind};
use crate::player::Player;
use crate::resource::{Resource, RESOURCE_CARDS};
use crate::topology::topology;

use super::rules::{award, longest_road};
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
        self.validate_board()?;
        self.validate_cards()?;
        self.validate_turn()
    }

    fn validate_board(&self) -> Result<(), Violation> {
        let t = topology();
        let all: [Set<VertexKind>; 4] =
            [self.settlements[0], self.settlements[1], self.cities[0], self.cities[1]];
        for i in 0..4 {
            for j in i + 1..4 {
                ensure((all[i] & all[j]).is_empty(), || "a vertex holds two buildings".into())?;
            }
        }
        ensure((self.roads[0] & self.roads[1]).is_empty(), || "an edge holds two roads".into())?;
        let occupied = all.iter().fold(Set::EMPTY, |a, &b| a | b);
        for v in occupied.iter() {
            ensure(
                (t.adjacent(v) & occupied).is_empty(),
                || format!("buildings on adjacent vertices at {v:?}"),
            )?;
        }
        for player in Player::BOTH {
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
                self.road_lengths[p] == longest_road(self.roads[p], self.buildings(player.other())),
                || format!("{player:?}'s road length is stale"),
            )?;
        }
        Ok(())
    }

    fn validate_cards(&self) -> Result<(), Violation> {
        for resource in Resource::ALL {
            let r = resource.index();
            let held = u32::from(self.hands[0][r]) + u32::from(self.hands[1][r]) + u32::from(self.set_aside[r]);
            ensure(held <= u32::from(RESOURCE_CARDS), || format!("more than 19 {resource:?} in play"))?;
        }
        for card in DevCard::ALL {
            let c = card.index();
            let held = u32::from(self.deck[c]) + u32::from(self.dev_cards[0][c]) + u32::from(self.dev_cards[1][c]);
            ensure(held <= u32::from(DEV_CARDS[c]), || format!("more than {} {card:?} cards", DEV_CARDS[c]))?;
            ensure(
                self.bought[c] <= self.dev_cards[self.current.index()][c],
                || format!("bought {card:?} not held"),
            )?;
        }
        ensure(
            self.played(DevCard::Knight) == self.knights[0] + self.knights[1],
            || "the knights played do not match the knights missing".into(),
        )?;
        ensure(self.played(DevCard::VictoryPoint) == 0, || "a victory point card was played".into())?;
        ensure(
            self.longest_road == award(self.longest_road, self.road_lengths, 5),
            || "the longest road is not where the lengths put it".into(),
        )?;
        ensure(
            self.longest_road.is_some() || self.road_lengths.iter().all(|&l| l < 5),
            || "an unheld road of 5".into(),
        )?;
        ensure(
            self.largest_army == award(self.largest_army, self.knights, 3),
            || "the largest army is not where the knights put it".into(),
        )?;
        ensure(
            self.largest_army.is_some() || self.knights.iter().all(|&k| k < 3),
            || "an unheld army of 3".into(),
        )?;
        Ok(())
    }

    fn validate_turn(&self) -> Result<(), Violation> {
        let placed = self.settlements[0].len() + self.settlements[1].len() + self.cities[0].len() + self.cities[1].len();
        let in_setup = matches!(self.phase, Phase::PlaceSettlement | Phase::PlaceRoad { .. });
        let fresh_turn = matches!(self.phase, Phase::Turn { rolled: false }) && !self.card_played;
        let set_aside: u8 = self.set_aside.iter().sum();
        let bought: u8 = self.bought.iter().sum();
        if in_setup {
            let roads = usize::from(self.roads[0].len() + self.roads[1].len());
            let placed = usize::from(placed);
            let (roads_expected, placing) = match self.phase {
                Phase::PlaceSettlement => (placed, placed),
                _ => (placed.saturating_sub(1), placed.saturating_sub(1)),
            };
            ensure(
                placed >= 1 || matches!(self.phase, Phase::PlaceSettlement),
                || "a setup road with no settlement".into(),
            )?;
            ensure(roads == roads_expected, || "setup roads out of step".into())?;
            let order = [Player::First, Player::Second, Player::Second, Player::First];
            ensure(order.get(placing) == Some(&self.current), || "the wrong player is placing".into())?;
            for player in Player::BOTH {
                let p = player.index();
                let cards: u8 = self.hands[p].iter().sum();
                let earned = if self.settlements[p].len() == 2 { 3 } else { 0 };
                ensure(cards <= earned, || format!("{player:?} holds {cards} cards in setup"))?;
            }
            ensure(
                self.dev_cards == [[0; 5]; 2] && !self.card_played && bought == 0,
                || "cards in setup".into(),
            )?;
        } else {
            for player in Player::BOTH {
                let p = player.index();
                ensure(
                    self.buildings(player).len() >= 2 && self.roads[p].len() >= 2,
                    || "setup unfinished".into(),
                )?;
            }
        }
        ensure(!fresh_turn || bought == 0, || "cards bought before the roll".into())?;
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
                if player == self.current {
                    let highest = (0..5).rev().find(|&r| self.set_aside[r] > 0).unwrap_or(0);
                    ensure(from == Resource::ALL[highest], || "the discard order is out of step".into())?;
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
            Phase::Steal { .. } => {
                ensure(self.hand_size(self.current.other()) > 0, || "a steal from an empty hand".into())?;
            }
            Phase::Draw => ensure(self.deck_size() > 0, || "a draw from an empty deck".into())?,
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
                for player in Player::BOTH {
                    ensure(
                        self.victory_points(player) < WINNING_POINTS,
                        || format!("{player:?} has won but play goes on"),
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
