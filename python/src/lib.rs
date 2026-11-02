use catan::board::Board;
use catan::buffer::{self, CHECKPOINT_EVERY, Config, Ending, Finish};
use catan::dev_card::DevCard;
use catan::dice::Dice;
use catan::ids::{Edge, Hex, Vertex};
use catan::symmetry::{Symmetry, SYMMETRIES};
use catan::observation::{OBSERVATION_BYTES, PHASE_AT, TRADE_AT, observe};
use catan::player::{Player, MAX_PLAYERS};
use catan::rng::Rng;
use catan::state::{ACTIONS, Action, Actor, MASK_WORDS, MAX_PROPOSED, Phase, Rules, State};
use catan::topology::topology as board_topology;
use catan::worlds;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Batch<'py> = (Bound<'py, PyArray2<u8>>, Bound<'py, PyArray2<bool>>, Bound<'py, PyArray1<u8>>);

#[pyclass(module = "catan")]
struct Buffer {
    inner: buffer::Buffer,
    endings: Vec<Option<Ending>>,
}

#[pymethods]
impl Buffer {
    #[new]
    #[pyo3(signature = (games, history = 1024, max_decisions = 1000, seed = 0, players = 4, paired = false, rules = None))]
    fn new(games: usize, history: usize, max_decisions: u32, seed: u64, players: usize, paired: bool, rules: Option<&GameRules>) -> PyResult<Self> {
        if games == 0 || max_decisions == 0 || history == 0 || history as u64 % CHECKPOINT_EVERY != 0 {
            return Err(PyValueError::new_err(format!(
                "games and max_decisions must be positive, and history a positive multiple of {CHECKPOINT_EVERY}"
            )));
        }
        if !(2..=MAX_PLAYERS).contains(&players) {
            return Err(PyValueError::new_err(format!("2 to {MAX_PLAYERS} players")));
        }
        let rules = rules.map_or(Rules::STANDARD, |r| r.inner);
        let inner = buffer::Buffer::new(Config { games, history, max_decisions, seed, players, paired, rules });
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
struct Game {
    seed: u64,
    state: State,
    rng: Rng,
    dice: Dice,
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
        let dice = Dice::new(rules.balanced_dice);
        Ok(Game { seed, state, rng: Rng::new(seed ^ 0x9E37_79B9_7F4A_7C15), dice, codes: Vec::new(), history: Vec::new() })
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

    fn settle(&mut self) {
        while let Some(Actor::Chance) = self.state.actor() {
            let code = if matches!(self.state.phase(), Phase::Dice) {
                self.dice.roll(&mut self.rng).index()
            } else {
                self.state.sample_outcome(&mut self.rng).index()
            };
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
