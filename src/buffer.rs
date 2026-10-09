use std::fmt;
use std::num::NonZeroUsize;
use std::sync::Mutex;
use std::thread;

use crate::board::Board;
use crate::observation::{observe, OBSERVATION_BYTES};
use crate::player::Player;
use crate::rng::Rng;
use crate::state::{Action, Actor, Mask, State, Violation};

pub const CHECKPOINT_EVERY: u64 = 16;
pub const PAD: u16 = u16::MAX;
const _: () = assert!(PAD as usize >= crate::state::ACTIONS);

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub games: usize, // how many games to run in parallel
    pub history: usize, // how many recent steps each game remembers
    pub max_decisions: u32, // max decisions before a game is truncated
    pub seed: u64,
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
pub struct IllegalAction { // move is not legal
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
pub struct Unavailable { // past position cannot be built
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

struct Game {
    state: State,
    mask: Mask,
    rng: Rng,
    time: u64,
    decisions: u32,
    codes: Box<[u16]>,
    checkpoints: Box<[State]>,
}

impl Buffer {
    pub fn new(config: Config) -> Buffer {
        assert!(config.games > 0, "a buffer needs games");
        assert!(
            config.history > 0 && config.history as u64 % CHECKPOINT_EVERY == 0,
            "the history must be a positive multiple of {CHECKPOINT_EVERY}"
        );
        assert!(config.max_decisions > 0, "a game needs decisions");
        let mut seeds = Rng::new(config.seed);
        let games = (0..config.games)
            .map(|_| {
                let mut rng = seeds.fork();
                let state = State::new(Board::random(&mut rng));
                let mut game = Game {
                    state,
                    mask: Mask::EMPTY,
                    rng,
                    time: 0,
                    decisions: 0,
                    codes: vec![PAD; config.history].into_boxed_slice(),
                    checkpoints: vec![state; config.history / CHECKPOINT_EVERY as usize].into_boxed_slice(),
                };
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

    pub fn step(&mut self, actions: &[u16], endings: &mut [Option<Ending>]) -> Result<(), IllegalAction> {
        assert!(actions.len() == self.games.len() && endings.len() == self.games.len(), "one entry per game");
        for (game, (&action, g)) in actions.iter().zip(&self.games).enumerate() {
            if !Action::from_index(action).is_some_and(|a| g.mask.contains(a)) {
                return Err(IllegalAction { game, action });
            }
        }
        let max_decisions = self.max_decisions;
        parallel(&mut self.games, endings, |games, endings, start| {
            for (i, (game, ending)) in games.iter_mut().zip(endings).enumerate() {
                *ending = game.take(actions[start + i], max_decisions);
            }
        });
        Ok(())
    }

    pub fn observe(
        &self,
        observations: &mut [[u8; OBSERVATION_BYTES]],
        masks: &mut [[u64; 5]],
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
        masks: &mut [[u64; 5]],
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

    fn settle(&mut self) {
        while let Some(Actor::Chance) = self.state.actor() {
            let outcome = self.state.sample_outcome(&mut self.rng);
            self.record(outcome.index());
            self.state.resolve(outcome);
        }
        self.mask = self.state.legal_actions();
    }

    fn take(&mut self, action: u16, max_decisions: u32) -> Option<Ending> {
        self.record(action);
        self.state.apply(Action::from_index(action).expect("checked by the caller"));
        self.decisions += 1;
        self.settle();
        let finish = match self.state.winner() {
            Some(winner) => Finish::Won(winner),
            None if self.decisions >= max_decisions => Finish::Truncated,
            None => return None,
        };
        let ending = Ending { finish, time: self.time };
        let board = Board::random(&mut self.rng);
        self.restart(State::new(board));
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

fn parallel<A: Send, B: Send>(a: &mut [A], b: &mut [B], work: impl Fn(&mut [A], &mut [B], usize) + Sync) {
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
