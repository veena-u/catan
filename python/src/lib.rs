use catan::board::Board;
use catan::buffer::{self, CHECKPOINT_EVERY, Config, Ending, Finish};
use catan::dev_card::DevCard;
use catan::ids::{Edge, Hex, Vertex};
use catan::symmetry::{Symmetry, SYMMETRIES};
use catan::observation::{OBSERVATION_BYTES, PHASE_AT, TRADE_AT, observe};
use catan::player::{Player, MAX_PLAYERS};
use catan::rng::Rng;
use catan::search;
use catan::state::{ACTIONS, Action, Actor, MASK_WORDS, MAX_PROPOSED, Phase, Rules, State};
use catan::topology::topology as board_topology;
use catan::worlds;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const HIDDEN_BYTES: usize = (MAX_PLAYERS - 1) * 10;

/// Every other player's cards and development cards, as the player to act would need to see them; zeros if no one is.
fn hidden_of(state: &State, out: &mut [u8]) {
    let Some(Actor::Player(me)) = state.actor() else { return };
    for (k, other) in state.others(me).enumerate() {
        let (hand, aside) = (state.hand(other), state.set_aside(other));
        for r in 0..5 {
            out[k * 10 + r] = hand[r] + aside[r];
            out[k * 10 + 5 + r] = state.dev_cards(other, DevCard::ALL[r]);
        }
    }
}

type Batch<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<u8>>);

#[pyclass(module = "catan")]
struct Buffer {
    inner: buffer::Buffer,
    endings: Vec<Option<Ending>>,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (games, history = 1024, max_decisions = 1000, seed = 0, players = 4, paired = false, rules = None, random_starts = 0))]
    fn new(games: usize, history: usize, max_decisions: u32, seed: u64, players: usize, paired: bool, rules: Option<&GameRules>, random_starts: u8) -> PyResult<Self> {
        if games == 0 || max_decisions == 0 || history == 0 || history as u64 % CHECKPOINT_EVERY != 0 {
            return Err(PyValueError::new_err(format!(
                "games and max_decisions must be positive, and history a positive multiple of {CHECKPOINT_EVERY}"
            )));
        }
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(PyValueError::new_err(format!("2 to {MAX_PLAYERS} players")));
        }
        let rules = rules.map_or(Rules::STANDARD, |r| r.inner);
        let inner = buffer::Buffer::new(Config { games, history, max_decisions, seed, players, paired, rules, random_starts: random_starts.min(100) });
        Ok(Buffer { inner, endings: vec![None; games] })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (games, actions, samples, seed = 0, deals = None))]
    fn worlds(
        &self,
        py: Python<'_>,
        games: PyArrayLike1<'_, u64, AllowTypeChange>,
        actions: PyArrayLike1<'_, u16, AllowTypeChange>,
        samples: usize,
        seed: u64,
        deals: Option<PyArrayLike1<'_, u8, AllowTypeChange>>,
    ) -> PyResult<Worlds> {
        let games: Vec<usize> = games.as_slice()?.iter().map(|&g| g as usize).collect();
        let actions = actions.as_slice()?;
        if games.len() != actions.len() || games.iter().any(|&g| g >= self.inner.len()) {
            return Err(PyValueError::new_err("one action per game, and only games that exist"));
        }
        let deals: Option<Vec<[[u8; 10]; 3]>> = match &deals {
            None => None,
            Some(flat) => {
                let flat = flat.as_slice()?;
                if flat.len() != games.len() * samples * 30 {
                    return Err(PyValueError::new_err("deals hold 3 seats of 10 counts for every world"));
                }
                Some(flat.chunks(30).map(|w| std::array::from_fn(|k| std::array::from_fn(|i| w[k * 10 + i]))).collect())
            }
        };
        let inner = &self.inner;
        let worlds = py
            .detach(|| worlds::Worlds::dealt(inner, &games, actions, samples, deals.as_deref(), seed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Worlds { inner: worlds })
    }

    fn points<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let rows: Vec<Vec<u8>> = (0..self.inner.len())
            .map(|game| {
                let state = self.inner.state(game);
                (0..MAX_PLAYERS).map(|s| if s < state.players() { state.victory_points(Player::new(s)) } else { 0 }).collect()
            })
            .collect();
        PyArray2::from_vec2(py, &rows).expect("rows of MAX_PLAYERS")
    }

    /// The dice rolled so far in each game: one per turn.
    fn rolls<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u16>> {
        PyArray1::from_vec(py, (0..self.inner.len()).map(|game| self.inner.state(game).dice().rolls()).collect())
    }

    fn hidden<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArrayDyn<u8>> {
        let n = self.inner.len();
        let mut out = vec![0u8; n * HIDDEN_BYTES];
        for game in 0..n {
            hidden_of(self.inner.state(game), &mut out[game * HIDDEN_BYTES..][..HIDDEN_BYTES]);
        }
        PyArray1::from_vec(py, out).reshape(vec![n, MAX_PLAYERS - 1, 10]).expect("sizes match")
    }

    /// The outcome code of the first luck each game's last move met, or -1.
    fn luck<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i16>> {
        PyArray1::from_iter(py, (0..self.inner.len()).map(|g| self.inner.luck(g).map_or(-1, |c| c as i16)))
    }

    /// Every outcome of the luck these moves would lead to, one row per outcome: its game, outcome code and weight,
    /// and the position it leads to as the player then to act sees it (with every hand, for the critic that sees
    /// them), or the winner where the game ends there.
    #[allow(clippy::type_complexity)]
    fn branches<'py>(
        &self,
        py: Python<'py>,
        actions: PyArrayLike1<'_, u16, AllowTypeChange>,
        seed: u64,
    ) -> PyResult<(
        Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<i16>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i8>>,
        Bound<'py, PyArray1<i8>>, Bound<'py, PyArray2<u8>>, Bound<'py, numpy::PyArrayDyn<u8>>,
    )> {
        let actions = actions.as_slice()?;
        if actions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!("{} actions for {} games", actions.len(), self.inner.len())));
        }
        let all = self.inner.branches(actions, seed);
        let m: usize = all.iter().map(Vec::len).sum();
        let (mut games, mut codes, mut weights, mut actors, mut winners) = (Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m));
        let (mut views, mut hidden) = (vec![0u8; m * OBSERVATION_BYTES], vec![0u8; m * HIDDEN_BYTES]);
        let mut row = 0;
        for (game, branches) in all.iter().enumerate() {
            for b in branches {
                games.push(game as u32);
                codes.push(b.code as i16);
                weights.push(f32::from(b.weight));
                winners.push(b.state.winner().map_or(-1, |p| p.index() as i8));
                match b.state.actor() {
                    Some(Actor::Player(p)) => {
                        actors.push(p.index() as i8);
                        let out: &mut [u8; OBSERVATION_BYTES] = (&mut views[row * OBSERVATION_BYTES..][..OBSERVATION_BYTES]).try_into().expect("one view");
                        observe(&b.state, p, out);
                        hidden_of(&b.state, &mut hidden[row * HIDDEN_BYTES..][..HIDDEN_BYTES]);
                    }
                    _ => actors.push(-1),
                }
                row += 1;
            }
        }
        Ok((
            PyArray1::from_vec(py, games), PyArray1::from_vec(py, codes), PyArray1::from_vec(py, weights),
            PyArray1::from_vec(py, actors), PyArray1::from_vec(py, winners),
            PyArray1::from_vec(py, views).reshape([m, OBSERVATION_BYTES]).expect("sizes match"),
            PyArray1::from_vec(py, hidden).reshape(vec![m, MAX_PLAYERS - 1, 10]).expect("sizes match"),
        ))
    }

    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_iter(py, (0..self.inner.len()).map(|g| self.inner.time(g)))
    }

    fn observe<'py>(&self, py: Python<'py>) -> Batch<'py> {
        let n = self.inner.len();
        let observations = PyArray2::zeros(py, [n, OBSERVATION_BYTES], false);
        let (mut words, mut players) = (vec![[0; MASK_WORDS]; n], vec![Player::First; n]);
        {
            let mut view = observations.readwrite();
            let out = view.as_slice_mut().expect("a new array is contiguous").as_chunks_mut().0;
            let inner = &self.inner;
            py.detach(|| inner.observe(out, &mut words, &mut players));
        }
        (observations, masks(py, &words), player_codes(py, &players))
    }

    /// Every game seen by the seat this many places after the player whose turn it is.
    fn observe_as<'py>(&self, py: Python<'py>, games: PyArrayLike1<'_, u64, AllowTypeChange>, offsets: PyArrayLike1<'_, u8, AllowTypeChange>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let (games, offsets) = (games.as_slice()?, offsets.as_slice()?);
        if games.len() != offsets.len() || games.iter().any(|&g| g as usize >= self.inner.len()) {
            return Err(PyValueError::new_err("one offset per game, and only games that exist"));
        }
        let m = games.len();
        let mut bytes = vec![0u8; m * OBSERVATION_BYTES];
        for k in 0..m {
            let state = self.inner.state(games[k] as usize);
            let seat = state.current().after(usize::from(offsets[k]), state.players());
            let out: &mut [u8; OBSERVATION_BYTES] = (&mut bytes[k * OBSERVATION_BYTES..(k + 1) * OBSERVATION_BYTES]).try_into().expect("one observation");
            observe(state, seat, out);
        }
        Ok(PyArray1::from_vec(py, bytes).reshape([m, OBSERVATION_BYTES]).expect("sizes match"))
    }

    /// The position after each deal with each partner, seen by the proposer, or with `as_partner` by the partner.
    #[pyo3(signature = (games, deals, partners, as_partner = false))]
    fn trade_views<'py>(
        &self,
        py: Python<'py>,
        games: PyArrayLike1<'_, u64, AllowTypeChange>,
        deals: PyArrayLike2<'_, i8, AllowTypeChange>,
        partners: PyArrayLike1<'_, u8, AllowTypeChange>,
        as_partner: bool,
    ) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>)> {
        let (games, partners, deals) = (games.as_slice()?, partners.as_slice()?, deals.as_array());
        let m = games.len();
        if partners.len() != m || deals.shape() != [m, 5] {
            return Err(PyValueError::new_err("one game, partner and five counts per query"));
        }
        let (mut bytes, mut ok) = (vec![0u8; m * OBSERVATION_BYTES], vec![false; m]);
        for k in 0..m {
            let state = self.inner.state(games[k] as usize);
            let counts: [i8; 5] = std::array::from_fn(|r| deals[[k, r]]);
            if let Some(after) = state.after_trade(partners[k], counts) {
                let out: &mut [u8; OBSERVATION_BYTES] = (&mut bytes[k * OBSERVATION_BYTES..(k + 1) * OBSERVATION_BYTES]).try_into().expect("one observation");
                let seer = if as_partner { after.current().after(usize::from(partners[k]), after.players()) } else { after.current() };
                observe(&after, seer, out);
                ok[k] = true;
            }
        }
        Ok((PyArray1::from_vec(py, bytes).reshape([m, OBSERVATION_BYTES]).expect("sizes match"), PyArray1::from_vec(py, ok)))
    }

    /// The position after each responder's counter-offer is taken: seen by the responder, and by the player whose turn it is.
    fn counter_views<'py>(&self, py: Python<'py>, games: PyArrayLike1<'_, u64, AllowTypeChange>, deals: PyArrayLike2<'_, i8, AllowTypeChange>) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>)> {
        let (games, deals) = (games.as_slice()?, deals.as_array());
        let m = games.len();
        if deals.shape() != [m, 5] || games.iter().any(|&g| g as usize >= self.inner.len()) {
            return Err(PyValueError::new_err("five counts per game, and only games that exist"));
        }
        let (mut theirs, mut current, mut ok) = (vec![0u8; m * OBSERVATION_BYTES], vec![0u8; m * OBSERVATION_BYTES], vec![false; m]);
        for k in 0..m {
            let counts: [i8; 5] = std::array::from_fn(|r| deals[[k, r]]);
            if let Some((after, responder)) = self.inner.state(games[k] as usize).after_counter(counts) {
                let at = k * OBSERVATION_BYTES..(k + 1) * OBSERVATION_BYTES;
                observe(&after, responder, (&mut theirs[at.clone()]).try_into().expect("one observation"));
                observe(&after, after.current(), (&mut current[at]).try_into().expect("one observation"));
                ok[k] = true;
            }
        }
        let shape = [m, OBSERVATION_BYTES];
        Ok((PyArray1::from_vec(py, theirs).reshape(shape).expect("sizes match"), PyArray1::from_vec(py, current).reshape(shape).expect("sizes match"), PyArray1::from_vec(py, ok)))
    }

    /// Put a single game's position into one of the buffer's slots, so batch tools can work on it.
    fn load(&mut self, slot: usize, game: &Game) -> PyResult<()> {
        if slot >= self.inner.len() {
            return Err(PyValueError::new_err(format!("there is no slot {slot}")));
        }
        self.inner.reset(slot, game.state).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn answer_views<'py>(&self, py: Python<'py>, games: PyArrayLike1<'_, u64, AllowTypeChange>) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>)> {
        let games = games.as_slice()?;
        let m = games.len();
        let (mut bytes, mut ok) = (vec![0u8; m * OBSERVATION_BYTES], vec![false; m]);
        for k in 0..m {
            if let Some((after, responder)) = self.inner.state(games[k] as usize).after_answer() {
                let out: &mut [u8; OBSERVATION_BYTES] = (&mut bytes[k * OBSERVATION_BYTES..(k + 1) * OBSERVATION_BYTES]).try_into().expect("one observation");
                observe(&after, responder, out);
                ok[k] = true;
            }
        }
        Ok((PyArray1::from_vec(py, bytes).reshape([m, OBSERVATION_BYTES]).expect("sizes match"), PyArray1::from_vec(py, ok)))
    }

    #[pyo3(signature = (games, simulations, worlds = 4, considered = 16, seed = 0, noise = true, value_scale = 0.1, rescale = true, exact = false))]
    #[allow(clippy::too_many_arguments)]
    fn search(&self, py: Python<'_>, games: PyArrayLike1<'_, u64, AllowTypeChange>, simulations: u32, worlds: usize, considered: usize, seed: u64, noise: bool, value_scale: f32, rescale: bool, exact: bool) -> PyResult<Search> {
        let games = games.as_slice()?;
        if let Some(&game) = games.iter().find(|&&g| g as usize >= self.inner.len()) {
            return Err(PyValueError::new_err(format!("there is no game {game}")));
        }
        let roots: Vec<State> = games.iter().map(|&g| *self.inner.state(g as usize)).collect();
        let config = search::Config { simulations, worlds, considered: considered.max(1), noise, value_scale, rescale, exact };
        Ok(Search { inner: py.detach(|| search::Search::new(&roots, config, seed)) })
    }

    fn forbidden<'py>(&self, py: Python<'py>) -> (Bound<'py, numpy::PyArrayDyn<i8>>, Bound<'py, PyArray1<u8>>) {
        let n = self.inner.len();
        let (mut deals, mut counts) = (vec![0i8; n * MAX_PROPOSED * 5], vec![0u8; n]);
        for game in 0..n {
            let forbidden = self.inner.state(game).forbidden_deals();
            counts[game] = forbidden.len() as u8;
            for (k, deal) in forbidden.iter().enumerate() {
                deals[(game * MAX_PROPOSED + k) * 5..][..5].copy_from_slice(deal);
            }
        }
        (PyArray1::from_vec(py, deals).reshape(vec![n, MAX_PROPOSED, 5]).expect("sizes match"), PyArray1::from_vec(py, counts))
    }

    #[pyo3(signature = (actions, deals = None))]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyArrayLike1<'_, u16, AllowTypeChange>,
        deals: Option<PyArrayLike2<'_, i8, AllowTypeChange>>,
    ) -> PyResult<Vec<(usize, Option<u8>, u64)>> {
        let actions = actions.as_slice()?;
        if actions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!("{} actions for {} games", actions.len(), self.inner.len())));
        }
        let deals: Option<Vec<[i8; 5]>> = match deals {
            Some(d) => {
                let d = d.as_array();
                if d.shape() != [actions.len(), 5] {
                    return Err(PyValueError::new_err("deals must be one row of 5 counts per game"));
                }
                Some(d.rows().into_iter().map(|row| std::array::from_fn(|r| row[r])).collect())
            }
            None => None,
        };
        let (inner, endings) = (&mut self.inner, &mut self.endings);
        py.detach(|| inner.step_with_deals(actions, deals.as_deref(), endings)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(endings
            .iter()
            .enumerate()
            .filter_map(|(game, ending)| {
                ending.map(|e| {
                    let winner = match e.finish {
                        Finish::Won(p) => Some(p.index() as u8),
                        Finish::Truncated => None,
                    };
                    (game, winner, e.time)
                })
            })
            .collect())
    }

    fn observe_at<'py>(
        &self,
        py: Python<'py>,
        games: PyArrayLike1<'py, u64, AllowTypeChange>,
        times: PyArrayLike1<'py, u64, AllowTypeChange>,
    ) -> PyResult<Batch<'py>> {
        let games: Vec<usize> = games.as_slice()?.iter().map(|&g| g as usize).collect();
        let times = times.as_slice()?;
        if games.len() != times.len() {
            return Err(PyValueError::new_err(format!("{} games but {} times", games.len(), times.len())));
        }
        if let Some(&game) = games.iter().find(|&&g| g >= self.inner.len()) {
            return Err(PyValueError::new_err(format!("there is no game {game}")));
        }
        let n = games.len();
        let observations = PyArray2::zeros(py, [n, OBSERVATION_BYTES], false);
        let (mut words, mut players) = (vec![[0; MASK_WORDS]; n], vec![Player::First; n]);
        let result = {
            let mut view = observations.readwrite();
            let out = view.as_slice_mut().expect("a new array is contiguous").as_chunks_mut().0;
            let inner = &self.inner;
            py.detach(|| inner.observe_at(&games, times, out, &mut words, &mut players))
        };
        result.map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((observations, masks(py, &words), player_codes(py, &players)))
    }
}

#[pyclass(module = "catan")]
struct Worlds {
    inner: worlds::Worlds,
}

#[pymethods]
impl Worlds {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn observe<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<i8>>) {
        let n = self.inner.len();
        let observations = PyArray2::zeros(py, [n, OBSERVATION_BYTES], false);
        let (mut words, mut players) = (vec![[0; MASK_WORDS]; n], vec![None; n]);
        {
            let mut view = observations.readwrite();
            let out = view.as_slice_mut().expect("a new array is contiguous").as_chunks_mut().0;
            self.inner.observe(out, &mut words, &mut players);
        }
        let codes = players.iter().map(|p| p.map_or(-1, |p: Player| p.index() as i8));
        (observations, masks(py, &words), PyArray1::from_iter(py, codes))
    }

    fn step(&mut self, py: Python<'_>, actions: PyArrayLike1<'_, u16, AllowTypeChange>) -> PyResult<()> {
        let actions = actions.as_slice()?;
        if actions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!("{} actions for {} worlds", actions.len(), self.inner.len())));
        }
        let inner = &mut self.inner;
        py.detach(|| inner.step(actions)).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn redealt(&self) -> usize {
        self.inner.redealt()
    }

    fn winners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        let codes = (0..self.inner.len()).map(|w| self.inner.state(w).winner().map_or(-1, |p| p.index() as i8));
        PyArray1::from_iter(py, codes)
    }
}

#[pyclass(module = "catan")]
struct Search {
    inner: search::Search,
}

#[pymethods]
impl Search {
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// The positions that need the network next: observations, legal moves, every other seat's hidden cards
    /// (for a critic that sees them), and which root each belongs to. Empty once every root is done.
    fn leaves<'py>(&mut self, py: Python<'py>) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<bool>>, Bound<'py, numpy::PyArrayDyn<u8>>, Bound<'py, PyArray1<u64>>) {
        let inner = &mut self.inner;
        let leaves = py.detach(|| inner.leaves());
        let k = leaves.len();
        let mut bytes = Vec::with_capacity(k * OBSERVATION_BYTES);
        let mut hidden = Vec::with_capacity(k * (MAX_PLAYERS - 1) * 10);
        let mut words = Vec::with_capacity(k);
        for (_, leaf) in &leaves {
            bytes.extend_from_slice(&leaf.observation);
            leaf.hidden.iter().for_each(|h| hidden.extend_from_slice(h));
            words.push(leaf.mask);
        }
        let rows = PyArray1::from_iter(py, leaves.iter().map(|&(i, _)| i as u64));
        (
            PyArray1::from_vec(py, bytes).reshape([k, OBSERVATION_BYTES]).expect("sizes match"),
            masks(py, &words),
            PyArray1::from_vec(py, hidden).reshape(vec![k, MAX_PLAYERS - 1, 10]).expect("sizes match"),
            rows,
        )
    }

    /// The network's log-probabilities over every move and its values (the player to act first) for the leaves.
    fn expand(&mut self, py: Python<'_>, log_probs: PyArrayLike2<'_, f32, AllowTypeChange>, values: PyArrayLike2<'_, f32, AllowTypeChange>) -> PyResult<()> {
        let (log_probs, values) = (log_probs.as_array(), values.as_array());
        if log_probs.ncols() != ACTIONS || values.ncols() < MAX_PLAYERS || log_probs.nrows() != values.nrows() {
            return Err(PyValueError::new_err(format!("log-probabilities over {ACTIONS} moves and {MAX_PLAYERS} values per leaf")));
        }
        let rows: Vec<[f32; ACTIONS]> = log_probs.rows().into_iter().map(|r| std::array::from_fn(|i| r[i])).collect();
        let vals: Vec<[f32; MAX_PLAYERS]> = values.rows().into_iter().map(|r| std::array::from_fn(|i| r[i])).collect();
        let inner = &mut self.inner;
        py.detach(|| inner.expand(&rows, &vals));
        Ok(())
    }

    /// Per root: the improved policy over every move, the move the search picks (-1 where it did not search),
    /// and the value it found for the player to act.
    fn result<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f32>>) {
        let n = self.inner.len();
        let outcomes: Vec<search::Outcome> = (0..n).map(|r| self.inner.outcome(r)).collect();
        let policy: Vec<f32> = outcomes.iter().flat_map(|o| o.policy).collect();
        (
            PyArray1::from_vec(py, policy).reshape([n, ACTIONS]).expect("sizes match"),
            PyArray1::from_iter(py, outcomes.iter().map(|o| o.action.map_or(-1, i32::from))),
            PyArray1::from_iter(py, outcomes.iter().map(|o| o.value)),
        )
    }
}

#[pyclass(module = "catan")]
struct Game {
    seed: u64,
    state: State,
    rng: Rng,
    codes: Vec<u16>,
    history: Vec<(i8, u8, u16)>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (seed = 0, players = 4, rules = None))]
    fn new(seed: u64, players: usize, rules: Option<&GameRules>) -> PyResult<Self> {
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(PyValueError::new_err(format!("2 to {MAX_PLAYERS} players")));
        }
        let rules = rules.map_or(Rules::STANDARD, |r| r.inner);
        let state = State::with_rules(Board::random(&mut Rng::new(seed)), players, rules);
        Ok(Game { seed, state, rng: Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15), codes: Vec::new(), history: Vec::new() })
    }

    fn players(&self) -> usize {
        self.state.players()
    }

    fn to_act(&self) -> Option<u8> {
        match self.state.actor() {
            Some(Actor::Player(p)) => Some(p.index() as u8),
            _ => None,
        }
    }

    fn winner(&self) -> Option<u8> {
        self.state.winner().map(|p| p.index() as u8)
    }

    fn steps(&self) -> usize {
        self.codes.len()
    }

    fn observe<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<bool>>)> {
        let Some(Actor::Player(player)) = self.state.actor() else {
            return Err(PyValueError::new_err("no player is to act"));
        };
        let mut bytes = [0; OBSERVATION_BYTES];
        observe(&self.state, player, &mut bytes);
        let words = self.state.legal_actions().words();
        let legal = (0..ACTIONS).map(|i| words[i / 64] >> (i % 64) & 1 == 1);
        Ok((PyArray1::from_slice(py, &bytes), PyArray1::from_iter(py, legal)))
    }

    #[pyo3(signature = (code, deal = None))]
    fn step(&mut self, code: u16, deal: Option<[i8; 5]>) -> PyResult<()> {
        let actor = match self.state.actor() {
            Some(Actor::Player(p)) => p.index() as i8,
            _ => -1,
        };
        let mut seen = [0; OBSERVATION_BYTES];
        observe(&self.state, Player::First, &mut seen);
        self.state.step(code).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.codes.push(code);
        self.history.push((actor, seen[PHASE_AT], code));
        if let (Some(counts), Some(Action::Offer | Action::Counter)) = (deal, Action::from_index(code)) {
            let mut draft = self.state;
            let mut phase = [0; OBSERVATION_BYTES];
            for click in draft.finish_deal(counts) {
                observe(&self.state, Player::First, &mut phase);
                self.state.step(click.index()).map_err(|e| PyValueError::new_err(e.to_string()))?;
                self.codes.push(click.index());
                self.history.push((actor, phase[PHASE_AT], click.index()));
            }
        }
        Ok(())
    }

    fn draft(&self) -> Option<(u8, [u8; 5], [u8; 5], bool)> {
        match self.state.phase() {
            Phase::Draft { deal } => Some((1, deal.give, deal.get, false)),
            Phase::Respond { draft: Some(d), .. } => Some((2, d.give, d.get, false)),
            Phase::Respond { closed: true, .. } => Some((0, [0; 5], [0; 5], true)),
            _ => None,
        }
    }

    fn forbidden(&self) -> Vec<[i8; 5]> {
        self.state.forbidden_deals()
    }

    fn history(&self) -> Vec<(i8, u8, u16)> {
        self.history.clone()
    }

    fn observe_as<'py>(&self, py: Python<'py>, player: usize) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if player >= self.state.players() {
            return Err(PyValueError::new_err(format!("there is no seat {player}")));
        }
        let mut bytes = [0; OBSERVATION_BYTES];
        observe(&self.state, Player::new(player), &mut bytes);
        Ok(PyArray1::from_slice(py, &bytes))
    }

    fn legal(&self) -> Vec<u16> {
        self.state.legal_actions().iter().map(Action::index).collect()
    }

    #[pyo3(signature = (samples, seed = 0))]
    fn imagine(&self, samples: usize, seed: u64) -> PyResult<Worlds> {
        let Some(Actor::Player(observer)) = self.state.actor() else {
            return Err(PyValueError::new_err("only a player's turn can be imagined from"));
        };
        Ok(Worlds { inner: worlds::Worlds::imagine(&self.state, observer, samples, seed) })
    }

    fn answer_views<'py>(&self, py: Python<'py>, _games: Option<PyArrayLike1<'_, u64, AllowTypeChange>>) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<bool>>) {
        let mut bytes = [0; OBSERVATION_BYTES];
        let ok = match self.state.after_answer() {
            Some((after, responder)) => {
                observe(&after, responder, &mut bytes);
                true
            }
            None => false,
        };
        (PyArray1::from_slice(py, &bytes).reshape([1, OBSERVATION_BYTES]).expect("sizes match"), PyArray1::from_slice(py, &[ok]))
    }

    #[pyo3(signature = (simulations, worlds = 4, considered = 16, seed = 0, noise = true, value_scale = 0.1, rescale = true, exact = false))]
    #[allow(clippy::too_many_arguments)]
    fn search(&self, simulations: u32, worlds: usize, considered: usize, seed: u64, noise: bool, value_scale: f32, rescale: bool, exact: bool) -> PyResult<Search> {
        if !matches!(self.state.actor(), Some(Actor::Player(_))) {
            return Err(PyValueError::new_err("only a player's decision can be searched"));
        }
        let config = search::Config { simulations, worlds, considered: considered.max(1), noise, value_scale, rescale, exact };
        Ok(Search { inner: search::Search::new(&[self.state], config, seed) })
    }

    fn settle(&mut self) {
        while let Some(Actor::Chance) = self.state.actor() {
            let code = self.state.sample_outcome(&mut self.rng).index();
            self.step(code, None).expect("a sampled outcome can happen");
        }
    }

    fn observe_transformed<'py>(&self, py: Python<'py>, symmetry: usize) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<bool>>)> {
        if symmetry >= SYMMETRIES {
            return Err(PyValueError::new_err(format!("{SYMMETRIES} symmetries")));
        }
        let state = self.state.transformed(Symmetry::new(symmetry));
        let Some(Actor::Player(player)) = state.actor() else {
            return Err(PyValueError::new_err("no player is to act"));
        };
        let mut bytes = [0; OBSERVATION_BYTES];
        observe(&state, player, &mut bytes);
        let words = state.legal_actions().words();
        let legal = (0..ACTIONS).map(|i| words[i / 64] >> (i % 64) & 1 == 1);
        Ok((PyArray1::from_slice(py, &bytes), PyArray1::from_iter(py, legal)))
    }

    fn record(&self) -> String {
        let codes: Vec<String> = self.codes.iter().map(u16::to_string).collect();
        format!("{};{};{}", self.state.players(), self.seed, codes.join(","))
    }
}

fn masks<'py>(py: Python<'py>, words: &[[u64; MASK_WORDS]]) -> Bound<'py, PyArray2<bool>> {
    let array = PyArray2::zeros(py, [words.len(), ACTIONS], false);
    {
        let mut view = array.readwrite();
        let rows = view.as_slice_mut().expect("a new array is contiguous").chunks_mut(ACTIONS);
        for (row, words) in rows.zip(words) {
            for (i, legal) in row.iter_mut().enumerate() {
                *legal = words[i / 64] >> (i % 64) & 1 == 1;
            }
        }
    }
    array
}

#[pyfunction]
fn symmetries<'py>(py: Python<'py>) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u16>>) {
    let table = |f: &dyn Fn(Symmetry) -> Vec<u8>| -> Vec<Vec<u8>> { Symmetry::all().map(f).collect() };
    let hexes = table(&|s| Hex::all().map(|h| s.hex(h).index() as u8).collect());
    let vertices = table(&|s| Vertex::all().map(|v| s.vertex(v).index() as u8).collect());
    let edges = table(&|s| Edge::all().map(|e| s.edge(e).index() as u8).collect());
    let actions: Vec<Vec<u16>> = Symmetry::all()
        .map(|s| (0..ACTIONS as u16).map(|i| Action::from_index(i).expect("an action").transformed(s).index()).collect())
        .collect();
    (
        PyArray2::from_vec2(py, &hexes).expect("rows"),
        PyArray2::from_vec2(py, &vertices).expect("rows"),
        PyArray2::from_vec2(py, &edges).expect("rows"),
        PyArray2::from_vec2(py, &actions).expect("rows"),
    )
}

#[pyfunction]
fn geometry<'py>(py: Python<'py>) -> (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<i32>>, Bound<'py, PyArray1<u8>>) {
    let t = board_topology();
    let hexes: Vec<Vec<i32>> = Hex::all().map(|h| { let (x, y) = t.hex_point(h); vec![x, y] }).collect();
    let vertices: Vec<Vec<i32>> = Vertex::all().map(|v| { let (x, y) = t.vertex_point(v); vec![x, y] }).collect();
    let coast = t.coast().iter().map(|e| e.index() as u8);
    (
        PyArray2::from_vec2(py, &hexes).expect("rows of 2"),
        PyArray2::from_vec2(py, &vertices).expect("rows of 2"),
        PyArray1::from_iter(py, coast),
    )
}

#[pyfunction]
fn topology<'py>(py: Python<'py>) -> (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<u8>>) {
    let t = board_topology();
    let hex_corners: Vec<Vec<u8>> = Hex::all().map(|h| t.corners(h).map(|v| v.index() as u8).to_vec()).collect();
    let edge_ends: Vec<Vec<u8>> = Edge::all().map(|e| t.ends(e).map(|v| v.index() as u8).to_vec()).collect();
    (
        PyArray2::from_vec2(py, &hex_corners).expect("rows of 6"),
        PyArray2::from_vec2(py, &edge_ends).expect("rows of 2"),
    )
}

fn player_codes<'py>(py: Python<'py>, players: &[Player]) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_iter(py, players.iter().map(|p| p.index() as u8))
}

#[pyclass(name = "Rules", module = "catan", frozen, skip_from_py_object)]
#[derive(Clone, Copy)]
struct GameRules {
    inner: Rules,
}

#[pymethods]
impl GameRules {
    #[new]
    #[pyo3(signature = (points = 10, discard_limit = 7, friendly_robber = false, balanced_dice = false, trading = true, turn_budget = 40))]
    fn new(points: u8, discard_limit: u8, friendly_robber: bool, balanced_dice: bool, trading: bool, turn_budget: u8) -> PyResult<Self> {
        let inner = Rules { points, discard_limit, friendly_robber, balanced_dice, trading, turn_budget };
        inner.check().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(GameRules { inner })
    }

    #[staticmethod]
    fn colonist_1v1() -> Self {
        GameRules { inner: Rules::COLONIST_1V1 }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[getter]
    fn points(&self) -> u8 {
        self.inner.points
    }

    #[getter]
    fn discard_limit(&self) -> u8 {
        self.inner.discard_limit
    }

    #[getter]
    fn friendly_robber(&self) -> bool {
        self.inner.friendly_robber
    }

    #[getter]
    fn balanced_dice(&self) -> bool {
        self.inner.balanced_dice
    }

    #[getter]
    fn trading(&self) -> bool {
        self.inner.trading
    }

    #[getter]
    fn turn_budget(&self) -> u8 {
        self.inner.turn_budget
    }
}

#[pymodule]
#[pyo3(name = "catan")]
fn catan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_class::<Game>()?;
    m.add_class::<Search>()?;
    m.add_function(wrap_pyfunction!(topology, m)?)?;
    m.add_function(wrap_pyfunction!(geometry, m)?)?;
    m.add_function(wrap_pyfunction!(symmetries, m)?)?;
    m.add("OBSERVATION_BYTES", OBSERVATION_BYTES)?;
    m.add("ACTIONS", ACTIONS)?;
    m.add("TRADE_AT", TRADE_AT)?;
    m.add_class::<GameRules>()?;
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    Ok(())
}
