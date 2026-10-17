use catan::buffer::{self, CHECKPOINT_EVERY, Config, Ending, Finish};
use catan::observation::OBSERVATION_BYTES;
use catan::player::Player;
use catan::state::ACTIONS;
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayMethods};
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
    #[pyo3(signature = (games, history = 1024, max_decisions = 1000, seed = 0))]
    fn new(games: usize, history: usize, max_decisions: u32, seed: u64) -> PyResult<Self> {
        if games == 0 || max_decisions == 0 || history == 0 || history as u64 % CHECKPOINT_EVERY != 0 {
            return Err(PyValueError::new_err(format!(
                "games and max_decisions must be positive, and history a positive multiple of {CHECKPOINT_EVERY}"
            )));
        }
        let inner = buffer::Buffer::new(Config { games, history, max_decisions, seed });
        Ok(Buffer { inner, endings: vec![None; games] })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
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

fn player_codes<'py>(py: Python<'py>, players: &[Player]) -> Bound<'py, PyArray1<u8>> {
    PyArray1::from_iter(py, players.iter().map(|p| p.index() as u8))
}

#[pymodule]
#[pyo3(name = "catan")]
fn catan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Buffer>()?;
    m.add("OBSERVATION_BYTES", OBSERVATION_BYTES)?;
    m.add("ACTIONS", ACTIONS)?;
    Ok(())
}
