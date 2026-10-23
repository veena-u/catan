use crate::player::{Player, MAX_PLAYERS};
use crate::resource::Resource;
use crate::rng::Rng;

use super::validate::Violation;
use super::{Phase, State};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Hidden {
    pub dev_cards: [[u8; 5]; MAX_PLAYERS],
    pub bought: [u8; 5],
    pub deck: [u8; 5],
    pub hands: [[u8; 5]; MAX_PLAYERS],
    pub set_aside: [[u8; 5]; MAX_PLAYERS],
}

impl State {
    pub fn hidden_from(&self, observer: Player) -> Hidden {
        let mut hidden = Hidden {
            dev_cards: [[0; 5]; MAX_PLAYERS],
            bought: if self.current != observer { self.bought } else { [0; 5] },
            deck: self.deck,
            hands: [[0; 5]; MAX_PLAYERS],
            set_aside: [[0; 5]; MAX_PLAYERS],
        };
        for other in self.others(observer) {
            let o = other.index();
            hidden.dev_cards[o] = self.dev_cards[o];
            hidden.hands[o] = self.hands[o];
            hidden.set_aside[o] = self.set_aside[o];
        }
        hidden
    }

    pub fn with_hidden(&self, observer: Player, hidden: Hidden) -> Result<State, Violation> {
        let mismatch = |what: &str| Err(Violation(format!("the hidden {what} change a public count")));
        let sum = |counts: [u8; 5]| counts.iter().map(|&n| u32::from(n)).sum::<u32>();
        let me = observer.index();
        if hidden.dev_cards[me] != [0; 5] || hidden.hands[me] != [0; 5] || hidden.set_aside[me] != [0; 5] {
            return mismatch("cards of the observer");
        }
        for c in 0..5 {
            let held = |cards: &[[u8; 5]; MAX_PLAYERS]| self.others(observer).map(|p| u32::from(cards[p.index()][c])).sum::<u32>();
            if u32::from(hidden.deck[c]) + held(&hidden.dev_cards) != u32::from(self.deck[c]) + held(&self.dev_cards) {
                return mismatch("development cards");
            }
            let before = |hands: &[[u8; 5]; MAX_PLAYERS], aside: &[[u8; 5]; MAX_PLAYERS]| {
                self.others(observer).map(|p| u32::from(hands[p.index()][c]) + u32::from(aside[p.index()][c])).sum::<u32>()
            };
            if before(&hidden.hands, &hidden.set_aside) != before(&self.hands, &self.set_aside) {
                return mismatch("hands");
            }
        }
        for other in self.others(observer) {
            let o = other.index();
            if sum(hidden.dev_cards[o]) != sum(self.dev_cards[o]) {
                return mismatch("development cards");
            }
            if sum(hidden.hands[o]) != sum(self.hands[o]) || sum(hidden.set_aside[o]) != sum(self.set_aside[o]) {
                return mismatch("hands");
            }
            if (0..5).any(|r| hidden.hands[o][r] + hidden.set_aside[o][r] < self.known[me][o][r]) {
                return mismatch("hands");
            }
            if !self.discards_hidden_from(observer, other) && hidden.set_aside[o] != self.set_aside[o] {
                return mismatch("discards");
            }
        }
        let bought_now = if self.current != observer { self.bought } else { [0; 5] };
        if sum(hidden.bought) != sum(bought_now) || (self.current == observer && hidden.bought != [0; 5]) {
            return mismatch("bought cards");
        }
        let mut world = *self;
        world.deck = hidden.deck;
        if self.current != observer {
            world.bought = hidden.bought;
        }
        for other in self.others(observer) {
            let o = other.index();
            world.dev_cards[o] = hidden.dev_cards[o];
            world.hands[o] = hidden.hands[o];
            world.set_aside[o] = hidden.set_aside[o];
        }
        if let Phase::Discard { player, owed, .. } = self.phase {
            if player != observer {
                let aside = world.set_aside[player.index()];
                let highest = (0..5).rev().find(|&r| aside[r] > 0).unwrap_or(0);
                world.phase = Phase::Discard { player, owed, from: Resource::ALL[highest] };
            }
        }
        world.validate()?;
        Ok(world)
    }

    pub fn sample_hidden(&self, observer: Player, rng: &mut Rng) -> State {
        self.with_hidden(observer, self.hidden_from(observer)).expect("a valid state");
        let me = observer.index();
        let others: Vec<Player> = self.others(observer).collect();
        let unseen_cards: [u8; 5] =
            std::array::from_fn(|c| self.deck[c] + others.iter().map(|p| self.dev_cards[p.index()][c]).sum::<u8>());
        let bought: u8 = if self.current != observer { self.bought.iter().sum() } else { 0 };
        let before = |p: Player| -> [u8; 5] { std::array::from_fn(|r| self.hands[p.index()][r] + self.set_aside[p.index()][r]) };
        let unknown_pool: [u8; 5] =
            std::array::from_fn(|r| others.iter().map(|&p| before(p)[r] - self.known[me][p.index()][r]).sum::<u8>());
        loop {
            let mut hidden = self.hidden_from(observer);
            let mut cards = unseen_cards;
            for &p in &others {
                let dealt = deal_from(cards, self.dev_card_count(p), rng);
                (0..5).for_each(|c| cards[c] -= dealt[c]);
                hidden.dev_cards[p.index()] = dealt;
            }
            hidden.deck = cards;
            if self.current != observer {
                hidden.bought = deal_from(hidden.dev_cards[self.current.index()], bought, rng);
            }
            let mut pool = unknown_pool;
            for &p in &others {
                let o = p.index();
                let size: u8 = before(p).iter().sum();
                let known = self.known[me][o];
                let unknown = deal_from(pool, size - known.iter().sum::<u8>(), rng);
                (0..5).for_each(|r| pool[r] -= unknown[r]);
                let whole: [u8; 5] = std::array::from_fn(|r| known[r] + unknown[r]);
                let aside = if self.discards_hidden_from(observer, p) {
                    deal_from(whole, self.set_aside[o].iter().sum(), rng)
                } else {
                    self.set_aside[o]
                };
                hidden.set_aside[o] = aside;
                hidden.hands[o] = std::array::from_fn(|r| whole[r] - aside[r]);
            }
            if let Ok(world) = self.with_hidden(observer, hidden) {
                return world;
            }
        }
    }

    pub fn deal_hidden(&self, observer: Player, deal: &[[u8; 10]], rng: &mut Rng) -> Result<State, Violation> {
        let refused = |why: &str| Err(Violation(format!("the deal {why}")));
        let mut hidden = self.hidden_from(observer);
        let others: Vec<Player> = self.others(observer).collect();
        if deal.len() != others.len() {
            return refused("names the wrong number of players");
        }
        let mut deck: [i32; 5] =
            std::array::from_fn(|c| i32::from(self.deck[c]) + others.iter().map(|p| i32::from(self.dev_cards[p.index()][c])).sum::<i32>());
        for (k, &other) in others.iter().enumerate() {
            let o = other.index();
            let whole: [u8; 5] = std::array::from_fn(|r| deal[k][r]);
            let cards: [u8; 5] = std::array::from_fn(|c| deal[k][5 + c]);
            (0..5).for_each(|c| deck[c] -= i32::from(cards[c]));
            hidden.dev_cards[o] = cards;
            let aside_count: u8 = self.set_aside[o].iter().sum();
            let aside = if self.discards_hidden_from(observer, other) {
                if whole.iter().sum::<u8>() < aside_count {
                    return refused("holds fewer cards than were set aside");
                }
                deal_from(whole, aside_count, rng)
            } else {
                self.set_aside[o]
            };
            if (0..5).any(|r| aside[r] > whole[r]) {
                return refused("sets aside cards the hand does not hold");
            }
            hidden.set_aside[o] = aside;
            hidden.hands[o] = std::array::from_fn(|r| whole[r] - aside[r]);
        }
        if deck.iter().any(|&n| n < 0) {
            return refused("uses more development cards than are unseen");
        }
        hidden.deck = deck.map(|n| n as u8);
        if self.current != observer {
            let bought: u8 = self.bought.iter().sum();
            let held = hidden.dev_cards[self.current.index()];
            if held.iter().sum::<u8>() < bought {
                return refused("leaves too few cards for those bought this turn");
            }
            hidden.bought = deal_from(held, bought, rng);
        }
        self.with_hidden(observer, hidden)
    }

    pub(super) fn discards_hidden_from(&self, observer: Player, player: Player) -> bool {
        matches!(self.phase, Phase::Discard { .. }) && player != observer
    }
}

fn deal_from(mut counts: [u8; 5], n: u8, rng: &mut Rng) -> [u8; 5] {
    let mut dealt = [0; 5];
    for _ in 0..n {
        let k = rng.weighted(&counts);
        counts[k] -= 1;
        dealt[k] += 1;
    }
    dealt
}
