use crate::buffer::{parallel, Buffer};
use crate::observation::{observe, OBSERVATION_BYTES};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::{Action, Actor, Invalid, State};

pub struct Worlds {
    states: Vec<State>,
    rngs: Vec<Rng>,
    redealt: usize,
}

impl Worlds {
    pub fn new(buffer: &Buffer, games: &[usize], actions: &[u16], samples: usize, seed: u64) -> Result<Worlds, Invalid> {
        Worlds::dealt(buffer, games, actions, samples, None, seed)
    }

    pub fn dealt(
        buffer: &Buffer,
        games: &[usize],
        actions: &[u16],
        samples: usize,
        deals: Option<&[[[u8; 10]; 3]]>,
        seed: u64,
    ) -> Result<Worlds, Invalid> {
        let mut seeds = Rng::new(seed);
        let (mut states, mut rngs, mut redealt) = (Vec::new(), Vec::new(), 0);
        for (pair, (&game, &action)) in games.iter().zip(actions).enumerate() {
            let root = buffer.state(game);
            let Some(Actor::Player(observer)) = root.actor() else { return Err(Invalid { code: action }) };
            if !Action::from_index(action).is_some_and(|a| root.legal_actions().contains(a)) {
                return Err(Invalid { code: action });
            }
            let others = root.players() - 1;
            for sample in 0..samples {
                let mut rng = seeds.fork();
                let guess = deals.map(|d| &d[pair * samples + sample][..others]);
                let mut world = match guess.map(|g| root.deal_hidden(observer, g, &mut rng)) {
                    Some(Ok(world)) => world,
                    other => {
                        redealt += usize::from(other.is_some());
                        root.sample_hidden(observer, &mut rng)
                    }
                };
                world.step(action).expect("an action legal before re-dealing hidden cards is legal after");
                settle(&mut world, &mut rng);
                states.push(world);
                rngs.push(rng);
            }
        }
        Ok(Worlds { states, rngs, redealt })
    }

    pub fn redealt(&self) -> usize {
        self.redealt
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn state(&self, world: usize) -> &State {
        &self.states[world]
    }

    pub fn observe(&self, observations: &mut [[u8; OBSERVATION_BYTES]], masks: &mut [[u64; 5]], players: &mut [Option<Player>]) {
        for (((world, observation), mask), player) in self.states.iter().zip(observations).zip(masks).zip(players) {
            *player = match world.actor() {
                Some(Actor::Player(p)) => {
                    observe(world, p, observation);
                    *mask = world.legal_actions().words();
                    Some(p)
                }
                _ => None,
            };
        }
    }

    pub fn step(&mut self, actions: &[u16]) -> Result<(), Invalid> {
        for (world, &action) in self.states.iter().zip(actions) {
            if world.actor().is_some() && !Action::from_index(action).is_some_and(|a| world.legal_actions().contains(a)) {
                return Err(Invalid { code: action });
            }
        }
        let mut pairs: Vec<(&mut State, &mut Rng)> = self.states.iter_mut().zip(self.rngs.iter_mut()).collect();
        parallel(&mut pairs, &mut vec![(); actions.len()], |pairs, _, start| {
            for (i, (world, rng)) in pairs.iter_mut().enumerate() {
                if world.actor().is_some() {
                    world.step(actions[start + i]).expect("checked above");
                    settle(world, rng);
                }
            }
        });
        Ok(())
    }
}

fn settle(state: &mut State, rng: &mut Rng) {
    while let Some(Actor::Chance) = state.actor() {
        let outcome = state.sample_outcome(rng);
        state.resolve(outcome);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Config;

    #[test]
    fn worlds_keep_what_the_searcher_sees_and_play_on_legally() {
        let mut buffer = Buffer::new(Config { games: 16, history: 64, max_decisions: 5_000, seed: 5, players: 4, paired: false });
        let mut rng = Rng::new(1);
        let mut endings = vec![None; buffer.len()];
        for _ in 0..80 {
            let actions: Vec<u16> = (0..buffer.len())
                .map(|g| {
                    let legal: Vec<Action> = buffer.mask(g).iter().collect();
                    legal[rng.below(legal.len())].index()
                })
                .collect();
            buffer.step(&actions, &mut endings).unwrap();
        }
        let games: Vec<usize> = (0..buffer.len()).collect();
        let actions: Vec<u16> = games.iter().map(|&g| buffer.mask(g).iter().next().unwrap().index()).collect();
        let mut worlds = Worlds::new(&buffer, &games, &actions, 4, 7).unwrap();
        assert_eq!(worlds.len(), 64);
        for (i, &game) in games.iter().enumerate() {
            let root = buffer.state(game);
            assert!(matches!(root.actor(), Some(Actor::Player(_))));
            for s in 0..4 {
                let world = worlds.state(i * 4 + s);
                world.validate().unwrap();
                assert_eq!(world.board(), root.board(), "the board is never re-dealt");
            }
        }
        let n = worlds.len();
        let (mut obs, mut masks, mut players) = (vec![[0; OBSERVATION_BYTES]; n], vec![[0; 5]; n], vec![None; n]);
        for _ in 0..30 {
            worlds.observe(&mut obs, &mut masks, &mut players);
            let actions: Vec<u16> = (0..n)
                .map(|w| match players[w] {
                    Some(_) => {
                        let legal: Vec<Action> = worlds.state(w).legal_actions().iter().collect();
                        legal[rng.below(legal.len())].index()
                    }
                    None => 0,
                })
                .collect();
            worlds.step(&actions).unwrap();
            for w in 0..n {
                worlds.state(w).validate().unwrap();
            }
        }
        assert!(Worlds::new(&buffer, &[0], &[u16::MAX], 1, 0).is_err(), "an illegal action is refused");
        let truth: Vec<[[u8; 10]; 3]> = games
            .iter()
            .map(|&g| {
                let root = buffer.state(g);
                let Some(Actor::Player(me)) = root.actor() else { unreachable!() };
                let mut deal = [[0; 10]; 3];
                for (k, other) in root.others(me).enumerate() {
                    for r in 0..5 {
                        deal[k][r] = root.hand(other)[r] + root.set_aside(other)[r];
                        deal[k][5 + r] = root.dev_cards(other, crate::dev_card::DevCard::ALL[r]);
                    }
                }
                deal
            })
            .collect();
        let dealt = Worlds::dealt(&buffer, &games, &actions, 1, Some(&truth), 3).unwrap();
        assert_eq!(dealt.redealt(), 0, "the true deal is always a possible one");
        let nonsense = vec![[[19; 10]; 3]; games.len()];
        let fallback = Worlds::dealt(&buffer, &games, &actions, 1, Some(&nonsense), 3).unwrap();
        assert_eq!(fallback.redealt(), games.len(), "an impossible deal falls back to a random one");
    }
}
