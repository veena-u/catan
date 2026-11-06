// How big is a turn? Replays recorded games ("players;seed;codes" per line, as the viewer and Game.record() write them)
// and, at the first decision after each roll, counts the distinct positions the player can reach before ending the
// turn: every order of builds, bank trades and cards, with draws and steals expanded over their outcomes.

use std::collections::HashSet;
use std::io::{self, BufRead};

use catan::board::Board;
use catan::rng::Rng;
use catan::state::{Action, Actor, Phase, Rules, State};

const CAP: usize = 200_000;

fn reachable(start: &State) -> (usize, usize) {
    let me = start.current();
    let mut seen = HashSet::new();
    let mut stack = vec![*start];
    let mut ends = 0;
    while let Some(state) = stack.pop() {
        if !seen.insert(state) || seen.len() > CAP {
            continue;
        }
        match state.actor() {
            Some(Actor::Chance) => {
                for (i, &w) in state.chances().weights().iter().enumerate() {
                    if w > 0 {
                        let mut next = state;
                        next.resolve(state.outcome(i as u16).expect("an outcome"));
                        stack.push(next);
                    }
                }
            }
            Some(Actor::Player(p)) if p == me && state.current() == me => {
                if state.legal_actions().contains(Action::EndTurn) {
                    ends += 1;
                }
                for action in state.legal_actions().iter().filter(|&a| !matches!(a, Action::EndTurn)) {
                    let mut next = state;
                    next.apply(action);
                    stack.push(next);
                }
            }
            _ => ends += 1,
        }
    }
    (ends, seen.len())
}

fn main() {
    let rules = Rules { points: 15, discard_limit: 9, friendly_robber: true, balanced_dice: true, trading: false, turn_budget: 15 };
    let (mut ends, mut nodes) = (Vec::new(), Vec::new());
    for line in io::stdin().lock().lines() {
        let line = line.expect("a line");
        let mut parts = line.trim().split(';');
        let (Some(players), Some(seed), Some(codes)) = (parts.next(), parts.next(), parts.next()) else { continue };
        let seed: u64 = seed.parse().expect("a seed");
        let mut state = State::with_rules(Board::random(&mut Rng::new(seed)), players.parse().expect("players"), rules);
        let mut after_roll = false;
        for code in codes.split(',').filter(|c| !c.is_empty()).map(|c| c.parse::<u16>().expect("a code")) {
            let rolling = matches!(state.phase(), Phase::Dice);
            if after_roll && matches!(state.actor(), Some(Actor::Player(p)) if p == state.current())
                && matches!(state.phase(), Phase::Turn { rolled: true })
            {
                let (e, n) = reachable(&state);
                ends.push(e);
                nodes.push(n);
                after_roll = false;
            }
            if state.step(code).is_err() {
                break;
            }
            after_roll |= rolling;
        }
    }
    let show = |name: &str, xs: &mut Vec<usize>| {
        xs.sort_unstable();
        let q = |f: f64| xs[((xs.len() - 1) as f64 * f) as usize];
        println!("{name}: median {}  90% {}  99% {}  max {}  (over {} turns)", q(0.5), q(0.9), q(0.99), xs[xs.len() - 1], xs.len());
    };
    show("distinct end-of-turn positions", &mut ends);
    show("distinct positions searched", &mut nodes);
}
