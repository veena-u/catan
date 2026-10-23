use catan::board::Board;
use catan::buffer::{self, CHECKPOINT_EVERY, Config, Ending, Finish};
use catan::dev_card::DevCard;
use catan::ids::{Edge, Hex, Vertex};
use catan::symmetry::{Symmetry, SYMMETRIES};
use catan::observation::{OBSERVATION_BYTES, observe};
use catan::player::{Player, MAX_PLAYERS};
use catan::rng::Rng;
use catan::state::{ACTIONS, Action, Actor, State};
use catan::topology::topology as board_topology;
use catan::worlds;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const OUTCOMES: usize = 11;

type Dice<'py> = (
    Bound<'py, PyArray1<bool>>,
    Bound<'py, numpy::PyArrayDyn<u8>>,
    Bound<'py, numpy::PyArrayDyn<u8>>,
    Bound<'py, numpy::PyArrayDyn<f32>>,
);

type Batch<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<u8>>);

#[pyclass(module = "catan")]
struct Buffer {
    inner: buffer::Buffer,
    endings: Vec<Option<Ending>>,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (games, history = 1024, max_decisions = 1000, seed = 0, players = 4, paired = false))]
    fn new(games: usize, history: usize, max_decisions: u32, seed: u64, players: usize, paired: bool) -> PyResult<Self> {
        if games == 0 || max_decisions == 0 || history == 0 || history as u64 % CHECKPOINT_EVERY != 0 {
            return Err(PyValueError::new_err(format!(
                "games and max_decisions must be positive, and history a positive multiple of {CHECKPOINT_EVERY}"
            )));
        }
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(PyValueError::new_err(format!("2 to {MAX_PLAYERS} players")));
        }
        let inner = buffer::Buffer::new(Config { games, history, max_decisions, seed, players, paired });
        Ok(Buffer { inner, endings: vec![None; games] })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[pyo3(signature = (games, actions, samples, seed = 0))]
    fn worlds(
        &self,
        py: Python<'_>,
        games: PyArrayLike1<'_, u64, AllowTypeChange>,
        actions: PyArrayLike1<'_, u16, AllowTypeChange>,
        samples: usize,
        seed: u64,
    ) -> PyResult<Worlds> {
        let games: Vec<usize> = games.as_slice()?.iter().map(|&g| g as usize).collect();
        let actions = actions.as_slice()?;
        if games.len() != actions.len() || games.iter().any(|&g| g >= self.inner.len()) {
            return Err(PyValueError::new_err("one action per game, and only games that exist"));
        }
        let inner = &self.inner;
        let worlds = py
            .detach(|| worlds::Worlds::new(inner, &games, actions, samples, seed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Worlds { inner: worlds })
    }

    fn rolls<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let n = self.inner.len();
        let mut counts = vec![vec![0u8; 11]; n];
        for (game, row) in counts.iter_mut().enumerate() {
            for &sum in self.inner.rolls(game) {
                row[usize::from(sum - 2)] += 1;
            }
        }
        PyArray2::from_vec2(py, &counts).expect("rows of 11")
    }

    fn hidden<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArrayDyn<u8>> {
        let n = self.inner.len();
        let mut out = vec![0u8; n * (MAX_PLAYERS - 1) * 10];
        for game in 0..n {
            let state = self.inner.state(game);
            let Some(Actor::Player(me)) = state.actor() else { continue };
            for (k, other) in state.others(me).enumerate() {
                let at = (game * (MAX_PLAYERS - 1) + k) * 10;
                let (hand, aside) = (state.hand(other), state.set_aside(other));
                for r in 0..5 {
                    out[at + r] = hand[r] + aside[r];
                    out[at + 5 + r] = state.dev_cards(other, DevCard::ALL[r]);
                }
            }
        }
        PyArray1::from_vec(py, out).reshape(vec![n, MAX_PLAYERS - 1, 10]).expect("sizes match")
    }

    fn track_dice(&mut self, on: bool) {
        self.inner.track_dice(on);
    }

    fn dice<'py>(&self, py: Python<'py>) -> Dice<'py> {
        let n = self.inner.len();
        let (mut has, mut weights) = (vec![false; n], vec![0f32; n * OUTCOMES]);
        let (mut players, mut bytes) = (vec![0u8; n * OUTCOMES], vec![0u8; n * OUTCOMES * OBSERVATION_BYTES]);
        for game in 0..n {
            let children = self.inner.dice(game);
            has[game] = !children.is_empty();
            let total: f32 = children.iter().map(|&(w, _)| f32::from(w)).sum();
            for (k, (weight, child)) in children.iter().enumerate() {
                let Some(Actor::Player(player)) = child.actor() else { continue };
                let at = game * OUTCOMES + k;
                weights[at] = f32::from(*weight) / total;
                players[at] = player.index() as u8;
                let out: &mut [u8; OBSERVATION_BYTES] =
                    (&mut bytes[at * OBSERVATION_BYTES..(at + 1) * OBSERVATION_BYTES]).try_into().expect("one observation");
                observe(child, player, out);
            }
        }
        let shape = |v: Vec<u8>, dims: &[usize]| PyArray1::from_vec(py, v).reshape(dims.to_vec()).expect("sizes match");
        (
            PyArray1::from_vec(py, has),
            shape(bytes, &[n, OUTCOMES, OBSERVATION_BYTES]),
            shape(players, &[n, OUTCOMES]),
            PyArray1::from_vec(py, weights).reshape(vec![n, OUTCOMES]).expect("sizes match"),
        )
    }

    fn times<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        PyArray1::from_iter(py, (0..self.inner.len()).map(|g| self.inner.time(g)))
    }

    fn observe<'py>(&self, py: Python<'py>) -> Batch<'py> {
        let n = self.inner.len();
        let observations = PyArray2::zeros(py, [n, OBSERVATION_BYTES], false);
        let (mut words, mut players) = (vec![[0; 5]; n], vec![Player::First; n]);
        {
            let mut view = observations.readwrite();
            let out = view.as_slice_mut().expect("a new array is contiguous").as_chunks_mut().0;
            let inner = &self.inner;
            py.detach(|| inner.observe(out, &mut words, &mut players));
        }
        (observations, masks(py, &words), player_codes(py, &players))
    }

    fn step(&mut self, py: Python<'_>, actions: PyArrayLike1<'_, u16, AllowTypeChange>) -> PyResult<Vec<(usize, Option<u8>, u64)>> {
        let actions = actions.as_slice()?;
        if actions.len() != self.inner.len() {
            return Err(PyValueError::new_err(format!("{} actions for {} games", actions.len(), self.inner.len())));
        }
        let (inner, endings) = (&mut self.inner, &mut self.endings);
        py.detach(|| inner.step(actions, endings)).map_err(|e| PyValueError::new_err(e.to_string()))?;
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
        let (mut words, mut players) = (vec![[0; 5]; n], vec![Player::First; n]);
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
        let (mut words, mut players) = (vec![[0; 5]; n], vec![None; n]);
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

    fn winners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i8>> {
        let codes = (0..self.inner.len()).map(|w| self.inner.state(w).winner().map_or(-1, |p| p.index() as i8));
        PyArray1::from_iter(py, codes)
    }
}

#[pyclass(module = "catan")]
struct Game {
    seed: u64,
    state: State,
    rng: Rng,
    codes: Vec<u16>,
}

#[pymethods]
impl Game {
    #[new]
    #[pyo3(signature = (seed = 0, players = 4))]
    fn new(seed: u64, players: usize) -> PyResult<Self> {
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(PyValueError::new_err(format!("2 to {MAX_PLAYERS} players")));
        }
        let state = State::new(Board::random(&mut Rng::new(seed)), players);
        Ok(Game { seed, state, rng: Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15), codes: Vec::new() })
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

    fn step(&mut self, code: u16) -> PyResult<()> {
        self.state.step(code).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.codes.push(code);
        Ok(())
    }

    fn settle(&mut self) {
        while let Some(Actor::Chance) = self.state.actor() {
            let code = self.state.sample_outcome(&mut self.rng).index();
            self.step(code).expect("a sampled outcome can happen");
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

fn masks<'py>(py: Python<'py>, words: &[[u64; 5]]) -> Bound<'py, PyArray2<bool>> {
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

#[pymodule]
#[pyo3(name = "catan")]
fn catan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add_class::<Game>()?;
    m.add_function(wrap_pyfunction!(topology, m)?)?;
    m.add_function(wrap_pyfunction!(symmetries, m)?)?;
    m.add("OBSERVATION_BYTES", OBSERVATION_BYTES)?;
    m.add("ACTIONS", ACTIONS)?;
    m.add("MAX_PLAYERS", MAX_PLAYERS)?;
    Ok(())
}
