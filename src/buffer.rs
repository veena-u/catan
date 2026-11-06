use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::thread;

use crate::board::Board;
use crate::observation::{observe, OBSERVATION_BYTES};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::{Action, Actor, Mask, Outcome, Phase, Rules, State, Violation, MASK_WORDS};

pub const CHECKPOINT_EVERY: u64 = 16;
pub const PAD: u16 = u16::MAX;
const _: () = assert!(PAD as usize >= crate::state::ACTIONS);

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub games: usize,
    pub history: usize,
    pub max_decisions: u32,
    pub seed: u64,
    pub players: usize,
    pub paired: bool,
    pub rules: Rules,
    pub random_starts: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Finish {
    Won(Player),
    Truncated,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Ending {
    pub finish: Finish,
    pub time: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct IllegalAction {
    pub game: usize,
    pub action: u16,
}

impl fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "action {} is not legal in game {}", self.action, self.game)
    }
}

impl std::error::Error for IllegalAction {}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Unavailable {
    pub game: usize,
    pub time: u64,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "game {} has no decision to show at time {}", self.game, self.time)
    }
}

impl std::error::Error for Unavailable {}

pub struct Buffer {
    games: Vec<Game>,
    max_decisions: u32,
}

/// One outcome of the luck a move leads to (a roll, a steal or a card draw): its weight in the engine's odds, its
/// outcome code, and the game played on to the next decision or to its end.
#[derive(Clone, Copy, Debug)]
pub struct Branch {
    pub weight: u16,
    pub code: u16,
    pub state: State,
}

struct Game {
    state: State,
    rules: Rules,
    mask: Mask,
    rng: Rng,
    dice_rng: Rng,
    random_starts: u8,
    randomize: bool,
    deal: Option<u64>,
    round: u64,
    time: u64,
    decisions: u32,
    codes: Box<[u16]>,
    checkpoints: Box<[State]>,
    rolls: Vec<u8>,
    luck: Option<u16>,
}

impl Buffer {
    pub fn new(config: Config) -> Buffer {
        assert!(config.games > 0, "a buffer needs games");
        assert!(
            config.history > 0 && config.history as u64 % CHECKPOINT_EVERY == 0,
            "the history must be a positive multiple of {CHECKPOINT_EVERY}"
        );
        assert!(config.max_decisions > 0, "a game needs decisions");
        assert!((2..=crate::player::MAX_PLAYERS).contains(&config.players), "2 to 4 players");
        let mut seeds = Rng::new(config.seed);
        let games = (0..config.games)
            .map(|slot| {
                let rng = seeds.fork();
                let deal = config.paired.then(|| mix(config.seed, (slot / config.players) as u64));
                let mut game = Game {
                    state: State::with_rules(Board::random(&mut Rng::new(0)), config.players, config.rules),
                    rules: config.rules,
                    mask: Mask::EMPTY,
                    rng,
                    dice_rng: Rng::new(0),
                    random_starts: config.random_starts,
                    randomize: false,
                    deal,
                    round: 0,
                    time: 0,
                    decisions: 0,
                    codes: vec![PAD; config.history].into_boxed_slice(),
                    checkpoints: Vec::new().into_boxed_slice(),
                    rolls: Vec::new(),
                    luck: None,
                };
                game.state = game.fresh(config.players);
                game.checkpoints = vec![game.state; config.history / CHECKPOINT_EVERY as usize].into_boxed_slice();
                game.settle();
                game
            })
            .collect();
        Buffer { games, max_decisions: config.max_decisions }
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    pub fn state(&self, game: usize) -> &State {
        &self.games[game].state
    }

    pub fn time(&self, game: usize) -> u64 {
        self.games[game].time
    }

    pub fn mask(&self, game: usize) -> Mask {
        self.games[game].mask
    }

    pub fn rolls(&self, game: usize) -> &[u8] {
        &self.games[game].rolls
    }

    /// The outcome code of the first luck the last step's move met in this game, if it met any.
    pub fn luck(&self, game: usize) -> Option<u16> {
        self.games[game].luck
    }

    /// For each game, if `actions` would lead straight to luck, every outcome of it with its odds, each played on to
    /// the next decision (later luck drawn from `seed`, the same for every outcome, so a branch is a fixed function of
    /// its outcome); empty where the move leads to a decision. Training subtracts how lucky the real outcome was against
    /// this average, which leaves the expected result unchanged and removes most of the dice from it.
    pub fn branches(&self, actions: &[u16], seed: u64) -> Vec<Vec<Branch>> {
        assert_eq!(actions.len(), self.games.len(), "one action per game");
        let n = self.games.len();
        let mut out = vec![Vec::new(); n];
        let games = &self.games;
        parallel(&mut out, &mut vec![(); n], |out, _, start| {
            for (i, branches) in out.iter_mut().enumerate() {
                let game = start + i;
                let Some(action) = Action::from_index(actions[game]) else { continue };
                if matches!(action, Action::Offer | Action::Counter) || !games[game].mask.contains(action) {
                    continue;
                }
                let mut state = games[game].state;
                state.apply(action);
                if state.actor() != Some(Actor::Chance) {
                    continue;
                }
                let chances = state.chances();
                for (code, &weight) in chances.weights().iter().enumerate().filter(|&(_, &w)| w > 0) {
                    let mut branch = state;
                    branch.resolve(branch.outcome(code as u16).expect("an outcome of this node"));
                    let mut rng = Rng::new(mix(seed, game as u64));
                    while branch.actor() == Some(Actor::Chance) {
                        let outcome = branch.sample_outcome(&mut rng);
                        branch.resolve(outcome);
                    }
                    branches.push(Branch { weight, code: code as u16, state: branch });
                }
            }
        });
        out
    }

    pub fn step(&mut self, actions: &[u16], endings: &mut [Option<Ending>]) -> Result<(), IllegalAction> {
        self.step_with_deals(actions, None, endings)
    }

    pub fn step_with_deals(&mut self, actions: &[u16], deals: Option<&[[i8; 5]]>, endings: &mut [Option<Ending>]) -> Result<(), IllegalAction> {
        assert!(actions.len() == self.games.len() && endings.len() == self.games.len(), "one entry per game");
        assert!(deals.is_none_or(|d| d.len() == actions.len()), "one deal per game");
        for (game, (&action, g)) in actions.iter().zip(&self.games).enumerate() {
            if !Action::from_index(action).is_some_and(|a| g.mask.contains(a)) {
                return Err(IllegalAction { game, action });
            }
        }
        let max_decisions = self.max_decisions;
        parallel(&mut self.games, endings, |games, endings, start| {
            for (i, (game, ending)) in games.iter_mut().zip(endings).enumerate() {
                *ending = game.take(actions[start + i], deals.map(|d| d[start + i]), max_decisions);
            }
        });
        Ok(())
    }

    pub fn observe(
        &self,
        observations: &mut [[u8; OBSERVATION_BYTES]],
        masks: &mut [[u64; MASK_WORDS]],
        players: &mut [Player],
    ) {
        let n = self.games.len();
        assert!(observations.len() == n && masks.len() == n && players.len() == n, "one entry per game");
        let mut outputs: Vec<_> = observations.iter_mut().zip(masks.iter_mut()).zip(players.iter_mut()).collect();
        let games = &self.games;
        parallel(&mut outputs, &mut vec![(); n], |outputs, _, start| {
            for (i, ((observation, mask), player)) in outputs.iter_mut().enumerate() {
                let game = &games[start + i];
                **player = to_act(&game.state);
                observe(&game.state, **player, observation);
                **mask = game.mask.words();
            }
        });
    }

    pub fn state_at(&self, game: usize, time: u64) -> Result<State, Unavailable> {
        let unavailable = Err(Unavailable { game, time });
        let g = &self.games[game];
        if time == g.time {
            return Ok(g.state);
        }
        if time > g.time {
            return unavailable;
        }
        let (block, latest) = (time / CHECKPOINT_EVERY, (g.time - 1) / CHECKPOINT_EVERY);
        let slots = g.checkpoints.len() as u64;
        if latest - block >= slots {
            return unavailable;
        }
        let mut state = g.checkpoints[(block % slots) as usize];
        for t in block * CHECKPOINT_EVERY..time {
            let code = g.codes[(t % g.codes.len() as u64) as usize];
            if state.step(code).is_err() {
                return unavailable;
            }
        }
        Ok(state)
    }

    pub fn observe_at(
        &self,
        games: &[usize],
        times: &[u64],
        observations: &mut [[u8; OBSERVATION_BYTES]],
        masks: &mut [[u64; MASK_WORDS]],
        players: &mut [Player],
    ) -> Result<(), Unavailable> {
        let n = games.len();
        assert!(
            times.len() == n && observations.len() == n && masks.len() == n && players.len() == n,
            "one entry per query"
        );
        let failure = Mutex::new(None);
        let mut outputs: Vec<_> = observations.iter_mut().zip(masks.iter_mut()).zip(players.iter_mut()).collect();
        parallel(&mut outputs, &mut vec![(); n], |outputs, _, start| {
            for (i, ((observation, mask), player)) in outputs.iter_mut().enumerate() {
                let (game, time) = (games[start + i], times[start + i]);
                match self.state_at(game, time) {
                    Ok(state) if matches!(state.actor(), Some(Actor::Player(_))) => {
                        **player = to_act(&state);
                        observe(&state, **player, observation);
                        **mask = state.legal_actions().words();
                    }
                    _ => {
                        failure.lock().expect("no thread panics holding it").get_or_insert(Unavailable { game, time });
                    }
                }
            }
        });
        match failure.into_inner().expect("no thread panics holding it") {
            Some(unavailable) => Err(unavailable),
            None => Ok(()),
        }
    }

    pub fn reset(&mut self, game: usize, state: State) -> Result<(), Violation> {
        state.validate()?;
        if state.actor().is_none() {
            return Err(Violation("the game is over".to_string()));
        }
        self.games[game].restart(state);
        Ok(())
    }
}

fn to_act(state: &State) -> Player {
    match state.actor() {
        Some(Actor::Player(player)) => player,
        _ => unreachable!("a buffer's games always wait at a decision"),
    }
}

impl Game {
    fn record(&mut self, code: u16) {
        if self.time % CHECKPOINT_EVERY == 0 {
            let slot = self.time / CHECKPOINT_EVERY % self.checkpoints.len() as u64;
            self.checkpoints[slot as usize] = self.state;
        }
        let at = self.time % self.codes.len() as u64;
        self.codes[at as usize] = code;
        self.time += 1;
    }

    fn fresh(&mut self, players: usize) -> State {
        let board = match self.deal {
            Some(deal) => {
                let seed = mix(deal, self.round);
                self.dice_rng = Rng::new(seed ^ 0xD1CE);
                Board::random(&mut Rng::new(seed))
            }
            None => {
                self.dice_rng = self.rng.fork();
                Board::random(&mut self.rng)
            }
        };
        self.round += 1;
        self.randomize = self.random_starts > 0 && self.rng.below(100) < usize::from(self.random_starts);
        State::with_rules(board, players, self.rules)
    }

    fn settle(&mut self) {
        if self.randomize && !matches!(self.state.phase(), Phase::PlaceSettlement | Phase::PlaceRoad { .. }) {
            self.randomize = false;
            self.state.randomize_start(&mut self.rng);
        }
        self.rolls.clear();
        self.luck = None;
        while let Some(Actor::Chance) = self.state.actor() {
            let outcome = if matches!(self.state.phase(), Phase::Dice) {
                self.state.sample_outcome(&mut self.dice_rng)
            } else {
                self.state.sample_outcome(&mut self.rng)
            };
            if let Outcome::Dice(sum) = outcome {
                self.rolls.push(sum);
            }
            self.luck = self.luck.or(Some(outcome.index()));
            self.record(outcome.index());
            self.state.resolve(outcome);
        }
        self.mask = self.state.legal_actions();
    }

    fn take(&mut self, action: u16, deal: Option<[i8; 5]>, max_decisions: u32) -> Option<Ending> {
        self.record(action);
        let chosen = Action::from_index(action).expect("checked by the caller");
        self.state.apply(chosen);
        if let (Some(counts), Action::Offer | Action::Counter) = (deal, chosen) {
            for click in self.state.finish_deal(counts) {
                self.record(click.index());
            }
        }
        self.decisions += 1;
        self.settle();
        let finish = match self.state.winner() {
            Some(winner) => Finish::Won(winner),
            None if self.decisions >= max_decisions => Finish::Truncated,
            None => return None,
        };
        let ending = Ending { finish, time: self.time };
        let players = self.state.players();
        let state = self.fresh(players);
        let (rolls, luck) = (std::mem::take(&mut self.rolls), self.luck);
        self.restart(state);
        self.rolls = rolls;
        self.luck = luck;
        Some(ending)
    }

    fn restart(&mut self, state: State) {
        loop {
            self.record(PAD);
            if self.time % CHECKPOINT_EVERY == 0 {
                break;
            }
        }
        self.state = state;
        self.decisions = 0;
        self.settle();
    }
}

fn mix(seed: u64, k: u64) -> u64 {
    let mut z = seed ^ k.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(crate) fn parallel<A: Send, B: Send>(a: &mut [A], b: &mut [B], work: impl Fn(&mut [A], &mut [B], usize) + Sync) {
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let chunk = a.len().div_ceil(threads).max(256);
    if chunk >= a.len() {
        work(a, b, 0);
        return;
    }
    thread::scope(|scope| {
        for (i, (a, b)) in a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate() {
            let work = &work;
            scope.spawn(move || work(a, b, i * chunk));
        }
    });
}

#[cfg(test)]
mod tests;
