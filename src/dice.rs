use crate::rng::Rng;
use crate::state::{Outcome, DICE};

const RESHUFFLE_BELOW: u8 = 6;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dice {
    deck: Option<[u8; 11]>,
}

impl Dice {
    pub fn new(balanced: bool) -> Dice {
        Dice { deck: balanced.then_some(DICE) }
    }

    pub fn roll(&mut self, rng: &mut Rng) -> Outcome {
        let index = rng.weighted(&self.deck.unwrap_or(DICE));
        if let Some(deck) = &mut self.deck {
            deck[index] -= 1;
            if deck.iter().sum::<u8>() < RESHUFFLE_BELOW {
                *deck = DICE;
            }
        }
        Outcome::Dice(index as u8 + 2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_dice_deal_each_combination_once_per_deck() {
        let mut dice = Dice::new(true);
        let mut rng = Rng::new(5);
        let mut counts = [0u8; 11];
        for _ in 0..31 {
            let Outcome::Dice(sum) = dice.roll(&mut rng) else { unreachable!() };
            counts[usize::from(sum - 2)] += 1;
        }
        assert!(counts.iter().zip(DICE).all(|(&c, full)| c <= full), "no combination comes up more often than the deck holds");
        assert_eq!(dice.deck, Some(DICE), "with five combinations left the deck is reshuffled");
    }

    #[test]
    fn plain_dice_have_no_memory() {
        let mut dice = Dice::new(false);
        let mut rng = Rng::new(5);
        for _ in 0..100 {
            dice.roll(&mut rng);
        }
        assert_eq!(dice, Dice::new(false));
    }
}
