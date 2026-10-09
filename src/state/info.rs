use crate::player::Player;
use crate::resource::Resource;
use crate::rng::Rng;

use super::validate::Violation;
use super::{Phase, State};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Hidden {
    pub dev_cards: [u8; 5],
    pub bought: [u8; 5],
    pub deck: [u8; 5],
    pub set_aside: [u8; 5],
}

impl State {
    pub fn hidden_from(&self, observer: Player) -> Hidden {
        let other = observer.other();
        Hidden {
            dev_cards: self.dev_cards[other.index()],
            bought: if self.current == other { self.bought } else { [0; 5] },
            deck: self.deck,
            set_aside: if self.discards_hidden_from(observer) { self.set_aside } else { [0; 5] },
        }
    }

    pub fn with_hidden(&self, observer: Player, hidden: Hidden) -> Result<State, Violation> {
        let other = observer.other().index();
        let mismatch = |what: &str| Err(Violation(format!("the hidden {what} change a public count")));
        for c in 0..5 {
            if hidden.deck[c] + hidden.dev_cards[c] != self.deck[c] + self.dev_cards[other][c] {
                return mismatch("development cards");
            }
        }
        let sum = |counts: [u8; 5]| counts.iter().map(|&n| u32::from(n)).sum::<u32>();
        if sum(hidden.dev_cards) != sum(self.dev_cards[other]) {
            return mismatch("development cards");
        }
        let bought_now = if self.current.index() == other { self.bought } else { [0; 5] };
        if sum(hidden.bought) != sum(bought_now) || (self.current.index() != other && hidden.bought != [0; 5]) {
            return mismatch("bought cards");
        }
        let hidden_discards = self.discards_hidden_from(observer);
        let discards_match = if hidden_discards {
            sum(hidden.set_aside) == sum(self.set_aside)
        } else {
            hidden.set_aside == [0; 5]
        };
        if !discards_match {
            return mismatch("discards");
        }
        let mut world = *self;
        world.dev_cards[other] = hidden.dev_cards;
        world.deck = hidden.deck;
        if self.current.index() == other {
            world.bought = hidden.bought;
        }
        if hidden_discards {
            for r in 0..5 {
                let before = self.hands[other][r] + self.set_aside[r];
                if hidden.set_aside[r] > before {
                    return mismatch("discards");
                }
                world.set_aside[r] = hidden.set_aside[r];
                world.hands[other][r] = before - hidden.set_aside[r];
            }
            if let Phase::Discard { player, owed, .. } = self.phase {
                if player == self.current {
                    let highest = (0..5).rev().find(|&r| hidden.set_aside[r] > 0).unwrap_or(0);
                    world.phase = Phase::Discard { player, owed, from: Resource::ALL[highest] };
                }
            }
        }
        world.validate()?;
        Ok(world)
    }

    pub fn sample_hidden(&self, observer: Player, rng: &mut Rng) -> State {
        self.with_hidden(observer, self.hidden_from(observer)).expect("a valid state");
        let other = observer.other().index();
        let unseen: [u8; 5] = std::array::from_fn(|c| self.deck[c] + self.dev_cards[other][c]);
        let held: u8 = self.dev_cards[other].iter().sum();
        let bought: u8 = if self.current.index() == other { self.bought.iter().sum() } else { 0 };
        let before: [u8; 5] = std::array::from_fn(|r| self.hands[other][r] + self.set_aside[r]);
        let discarded: u8 = self.set_aside.iter().sum();
        loop {
            let dev_cards = deal(unseen, held, rng);
            let hidden = Hidden {
                dev_cards,
                bought: deal(dev_cards, bought, rng),
                deck: std::array::from_fn(|c| unseen[c] - dev_cards[c]),
                set_aside: if self.discards_hidden_from(observer) { deal(before, discarded, rng) } else { [0; 5] },
            };
            if let Ok(world) = self.with_hidden(observer, hidden) {
                return world;
            }
        }
    }

    fn discards_hidden_from(&self, observer: Player) -> bool {
        matches!(self.phase, Phase::Discard { .. }) && self.current != observer
    }
}

fn deal(mut counts: [u8; 5], n: u8, rng: &mut Rng) -> [u8; 5] {
    let mut dealt = [0; 5];
    for _ in 0..n {
        let k = rng.weighted(&counts);
        counts[k] -= 1;
        dealt[k] += 1;
    }
    dealt
}

