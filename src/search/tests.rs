use super::*;
use crate::buffer::{Buffer, Config as BufferConfig};
use crate::state::{Mask, Rules};

fn played(players: usize, games: usize, moves: usize, seed: u64) -> Buffer {
    let config = BufferConfig { games, history: 64, max_decisions: 5_000, seed, players, paired: false, rules: Rules::STANDARD, random_starts: 0 };
    let mut buffer = Buffer::new(config);
    let mut rng = Rng::new(seed);
    let mut endings = vec![None; buffer.len()];
    for _ in 0..moves {
        let actions: Vec<u16> = (0..buffer.len())
            .map(|g| {
                let legal: Vec<Action> = buffer.mask(g).iter().filter(|a| *a != Action::Offer).collect();
                legal[rng.below(legal.len())].index()
            })
            .collect();
        buffer.step(&actions, &mut endings).unwrap();
    }
    buffer
}

fn legal_of(words: [u64; MASK_WORDS]) -> Vec<usize> {
    (0..ACTIONS).filter(|&i| words[i / 64] >> (i % 64) & 1 == 1).collect()
}

/// Every legal move equally likely, every position a draw: the lookahead alone has to find anything.
fn run_uniform(search: &mut Search) -> u32 {
    let mut rounds = 0;
    loop {
        let leaves = search.leaves();
        if leaves.is_empty() {
            return rounds;
        }
        rounds += 1;
        let log_probs: Vec<[f32; ACTIONS]> = leaves
            .iter()
            .map(|(_, leaf)| {
                let legal = legal_of(leaf.mask);
                let mut row = [-1e9; ACTIONS];
                legal.iter().for_each(|&i| row[i] = -(legal.len() as f32).ln());
                row
            })
            .collect();
        search.expand(&log_probs, &vec![[0.0; MAX_PLAYERS]; leaves.len()]);
    }
}

#[test]
fn sequential_halving_visits_the_survivors_more() {
    assert_eq!(considered_visits(4, 8), vec![0, 0, 0, 0, 1, 1, 2, 2]);
    assert_eq!(considered_visits(1, 3), vec![0, 1, 2]);
    assert_eq!(considered_visits(16, 32).len(), 32);
    assert_eq!(&considered_visits(16, 32)[..16], &[0; 16]);
}

#[test]
fn every_searchable_root_spends_its_whole_budget() {
    for players in [2, 4] {
        let buffer = played(players, 24, 150, players as u64);
        let roots: Vec<State> = (0..buffer.len()).map(|g| *buffer.state(g)).collect();
        let config = Config { simulations: 12, worlds: 3, considered: 8, noise: true, value_scale: 0.1, rescale: true };
        let mut search = Search::new(&roots, config, 5);
        let rounds = run_uniform(&mut search);
        assert!(rounds <= 1 + config.simulations, "{rounds} rounds");
        for (root, state) in roots.iter().enumerate() {
            let outcome = search.outcome(root);
            let legal = state.legal_actions();
            let total: f32 = outcome.policy.iter().sum();
            assert!((total - 1.0).abs() < 1e-3, "the policy sums to {total}");
            for (i, &p) in outcome.policy.iter().enumerate() {
                assert!(p == 0.0 || legal.contains(Action::from_index(i as u16).unwrap()), "mass on an illegal move");
            }
            let searchable = legal.iter().filter(|&a| a != Action::Offer).count() >= 2
                && !matches!(state.phase(), Phase::Draft { .. } | Phase::Respond { .. } | Phase::Confirm { .. });
            if searchable {
                assert_eq!(search.visits(root), config.simulations);
                let action = Action::from_index(outcome.action.expect("a searched root picks a move")).unwrap();
                assert!(legal.contains(action) && action != Action::Offer);
            } else {
                assert!(outcome.action.is_none());
            }
        }
    }
}

#[test]
fn the_lookahead_finds_a_winning_move() {
    let mut found = 0;
    for seed in 0..40 {
        let buffer = played(2, 64, 400, 100 + seed);
        for game in 0..buffer.len() {
            let state = *buffer.state(game);
            if !matches!(state.phase(), Phase::Turn { rolled: true }) {
                continue;
            }
            let legal: Vec<Action> = state.legal_actions().iter().filter(|&a| a != Action::Offer).collect();
            let wins = |a: &Action| {
                let mut after = state;
                after.apply(*a);
                after.winner().is_some()
            };
            if legal.len() < 2 || !legal.iter().any(wins) || legal.iter().all(wins) {
                continue;
            }
            let config = Config { simulations: 2 * legal.len() as u32, worlds: 1, considered: legal.len(), noise: false, value_scale: 0.1, rescale: true };
            let mut search = Search::new(&[state], config, seed);
            run_uniform(&mut search);
            let chosen = Action::from_index(search.outcome(0).action.unwrap()).unwrap();
            assert!(wins(&chosen), "picked {chosen:?} over a win");
            found += 1;
        }
    }
    assert!(found > 0, "no position with a winning move turned up");
}

#[test]
fn trade_answers_are_left_to_the_policy() {
    let buffer = played(4, 64, 200, 3);
    let mut answered = 0;
    for game in 0..buffer.len() {
        let mut state = *buffer.state(game);
        if !state.legal_actions().contains(Action::Offer) {
            continue;
        }
        let hand = state.hand(state.current());
        let Some(give) = (0..5).find(|&r| hand[r] > 0) else { continue };
        let mut counts = [0i8; 5];
        counts[give] = -1;
        counts[(give + 1) % 5] = 1;
        state.apply(Action::Offer);
        state.finish_deal(counts);
        if !matches!(state.phase(), Phase::Respond { .. }) {
            continue;
        }
        let mut search = Search::new(&[state], Config { simulations: 8, worlds: 2, considered: 4, noise: true, value_scale: 0.1, rescale: true }, 0);
        run_uniform(&mut search);
        assert!(search.outcome(0).action.is_none(), "an answer is not searched");
        answered += 1;
    }
    assert!(answered > 0);
    let _ = Mask::EMPTY;
}
