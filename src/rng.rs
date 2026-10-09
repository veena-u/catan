#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub const fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    pub fn fork(&mut self) -> Rng {
        Rng::new(self.next_u64())
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "there is no number below 0");
        let n = n as u64;
        let rejected = n.wrapping_neg() % n;
        loop {
            let word = self.next_u64();
            if word >= rejected {
                return (word % n) as usize;
            }
        }
    }

    pub fn weighted(&mut self, weights: &[u8]) -> usize {
        let total: usize = weights.iter().map(|&w| usize::from(w)).sum();
        let mut draw = self.below(total);
        for (index, &weight) in weights.iter().enumerate() {
            if draw < usize::from(weight) {
                return index;
            }
            draw -= usize::from(weight);
        }
        unreachable!("the draw is below the total")
    }

    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            items.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn the_output_matches_the_definition() {
        let mut rng = Rng::new(1_234_567);
        let expected: [u64; 5] = [
            6_457_827_717_110_365_317,
            3_203_168_211_198_807_973,
            9_817_491_932_198_370_423,
            4_593_380_528_125_082_431,
            16_408_922_859_458_223_821,
        ];
        assert!(expected.iter().all(|&value| rng.next_u64() == value));
        assert_eq!(Rng::new(0).next_u64(), 16_294_208_416_658_607_535);
    }

    #[test]
    fn below_is_even_even_when_n_is_most_of_the_range() {
        let n = usize::MAX / 3 * 2;
        let mut rng = Rng::new(3);
        let low = (0..10_000).filter(|_| rng.below(n) < n / 2).count();
        assert!((4_700..=5_300).contains(&low), "{low}");
        assert!((0..100).all(|_| rng.below(1) == 0));
        assert!(std::panic::catch_unwind(|| Rng::new(0).below(0)).is_err());
    }

    #[test]
    fn weighted_follows_the_weights() {
        let mut rng = Rng::new(9);
        let mut counts = [0; 3];
        for _ in 0..60_000 {
            counts[rng.weighted(&[1, 0, 3])] += 1;
        }
        assert_eq!(counts[1], 0);
        assert!((14_400..=15_600).contains(&counts[0]), "{counts:?}");
    }

    #[test]
    fn shuffle_makes_every_order_equally_likely() {
        let mut rng = Rng::new(11);
        let mut counts: HashMap<[u8; 3], u32> = HashMap::new();
        for _ in 0..60_000 {
            let mut items = [0, 1, 2];
            rng.shuffle(&mut items);
            *counts.entry(items).or_default() += 1;
        }
        assert_eq!(counts.len(), 6);
        assert!(counts.values().all(|&c| (9_500..=10_500).contains(&c)), "{counts:?}");
    }
}
