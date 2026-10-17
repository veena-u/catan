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
    let mut buffer = Buffer::new(Config { games: 300, history, max_decisions: 5_000, seed: 7 });
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
