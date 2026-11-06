use super::*;

fn decisions(buffer: &mut Buffer, rounds: usize, rng: &mut Rng) -> Vec<Vec<(u64, State)>> {
    let n = buffer.len();
    let mut seen = vec![Vec::new(); n];
    let mut endings = vec![None; n];
    for _ in 0..rounds {
        let actions: Vec<u16> = (0..n)
            .map(|g| {
                seen[g].push((buffer.time(g), *buffer.state(g)));
                let legal: Vec<Action> = buffer.mask(g).iter().collect();
                legal[rng.below(legal.len())].index()
            })
            .collect();
        buffer.step(&actions, &mut endings).unwrap();
    }
    seen
}

#[test]
fn recent_decisions_are_rebuilt_exactly_and_old_ones_refused() {
    let history = 256;
    let mut buffer = Buffer::new(Config { games: 300, history, max_decisions: 5_000, seed: 7, players: 4, paired: false, rules: crate::state::Rules::STANDARD, random_starts: 0 });
    let seen = decisions(&mut buffer, 300, &mut Rng::new(1));
    let window = history as u64 - CHECKPOINT_EVERY;
    let (mut rebuilt, mut refused) = (0, 0);
    for (g, states) in seen.iter().enumerate() {
        for &(time, state) in states {
            let age = buffer.time(g) - time;
            match buffer.state_at(g, time) {
                Ok(found) => {
                    assert_eq!(found, state, "game {g} at {time}");
                    rebuilt += 1;
                }
                Err(_) => {
                    assert!(age > window, "game {g} at {time}, {age} steps ago, is within the window");
                    refused += 1;
                }
            }
            if age > history as u64 {
                assert!(buffer.state_at(g, time).is_err(), "older than the history");
            }
        }
        assert!(buffer.state_at(g, buffer.time(g) + 1).is_err(), "the future");
    }
    assert!(rebuilt > 300 * 100 && refused > 300 * 10, "{rebuilt} rebuilt, {refused} refused");
}

#[test]
fn paired_games_share_their_board_and_their_rolls_whatever_is_played() {
    let players = 4;
    let mut buffer = Buffer::new(Config { games: 8, history: 64, max_decisions: 5_000, seed: 11, players, paired: true, rules: crate::state::Rules::STANDARD, random_starts: 0 });
    let boards: Vec<_> = (0..8).map(|g| *buffer.state(g).board()).collect();
    for g in 0..8 {
        assert_eq!(boards[g] == boards[0], g < players, "game {g}");
    }
    let mut rng = Rng::new(4);
    let mut endings = vec![None; buffer.len()];
    let mut rolls = vec![Vec::new(); 8];
    for _ in 0..600 {
        let actions: Vec<u16> = (0..8)
            .map(|g| {
                let legal: Vec<Action> = buffer.mask(g).iter().collect();
                legal[rng.below(legal.len())].index()
            })
            .collect();
        buffer.step(&actions, &mut endings).unwrap();
        for g in 0..8 {
            if rolls[g].len() < 40 && endings[g].is_none() {
                rolls[g].extend_from_slice(buffer.rolls(g));
            }
        }
    }
    for g in 1..players {
        let n = rolls[0].len().min(rolls[g].len()).min(20);
        assert!(n >= 10, "enough rolls to compare");
        assert_eq!(rolls[0][..n], rolls[g][..n], "the k-th roll is the same in every game of a group");
    }
    assert_ne!(rolls[0][..10], rolls[players][..10], "another group rolls differently");
}

#[test]
fn a_move_into_luck_branches_on_every_outcome_and_the_real_one_is_among_them() {
    let rules = crate::state::Rules::COLONIST_1V1;
    let mut buffer = Buffer::new(Config { games: 64, history: 64, max_decisions: 5_000, seed: 3, players: 2, paired: false, rules, random_starts: 0 });
    let mut rng = Rng::new(4);
    let (mut lucky, mut matched) = (0, 0);
    let mut endings = vec![None; 64];
    for round in 0..400 {
        let actions: Vec<u16> = (0..64)
            .map(|g| {
                let legal: Vec<Action> = buffer.mask(g).iter().collect();
                legal[rng.below(legal.len())].index()
            })
            .collect();
        let branches = buffer.branches(&actions, round);
        for (g, b) in branches.iter().enumerate() {
            if b.is_empty() {
                continue;
            }
            let mut after = *buffer.state(g);
            after.apply(Action::from_index(actions[g]).unwrap());
            let odds = after.chances();
            assert_eq!(b.iter().map(|x| u32::from(x.weight)).sum::<u32>(), odds.total(), "the branches carry the whole odds");
            assert!(b.iter().all(|x| x.state.actor() != Some(Actor::Chance)), "each branch is played on past its luck");
        }
        buffer.step(&actions, &mut endings).unwrap();
        for (g, b) in branches.iter().enumerate() {
            match (b.is_empty(), buffer.luck(g)) {
                (true, _) => {}
                (false, None) => panic!("game {g} branched but met no luck"),
                (false, Some(code)) => {
                    lucky += 1;
                    matched += usize::from(b.iter().any(|x| x.code == code));
                }
            }
        }
    }
    assert!(lucky > 1000, "{lucky} moves met luck");
    assert_eq!(matched, lucky, "the outcome that happened is always one of the branches");
}
