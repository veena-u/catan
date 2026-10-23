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
    let mut buffer = Buffer::new(Config { games: 300, history, max_decisions: 5_000, seed: 7, players: 4 });
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
fn a_roll_after_a_decision_is_expanded_into_its_eleven_outcomes() {
    let mut buffer = Buffer::new(Config { games: 64, history: 64, max_decisions: 5_000, seed: 3, players: 3 });
    buffer.track_dice(true);
    let mut rng = Rng::new(9);
    let mut endings = vec![None; buffer.len()];
    let (mut expanded, mut matched) = (0, 0);
    for _ in 0..400 {
        let before: Vec<u64> = (0..buffer.len()).map(|g| buffer.time(g)).collect();
        let actions: Vec<u16> = (0..buffer.len())
            .map(|g| {
                let legal: Vec<Action> = buffer.mask(g).iter().collect();
                legal[rng.below(legal.len())].index()
            })
            .collect();
        buffer.step(&actions, &mut endings).unwrap();
        for g in 0..buffer.len() {
            let children = buffer.dice(g);
            if children.is_empty() || endings[g].is_some() {
                continue;
            }
            expanded += 1;
            let weights: Vec<u8> = children.iter().map(|&(w, _)| w).collect();
            assert_eq!(weights, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]);
            for (_, child) in children {
                child.validate().unwrap();
                assert!(matches!(child.actor(), Some(Actor::Player(_))), "every outcome leads to a decision");
            }
            let roll = buffer.state_at(g, before[g] + 1).unwrap();
            assert_eq!(roll.phase(), Phase::Dice, "the step after the action is the roll");
            let now = *buffer.state(g);
            matched += children.iter().any(|&(_, child)| child == now) as u32;
        }
    }
    assert!(expanded > 1_000, "{expanded}");
    assert!(matched > expanded / 2, "the outcome that happened is usually among the children: {matched} of {expanded}");
}
