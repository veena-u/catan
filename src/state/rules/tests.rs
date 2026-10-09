use super::*;
use crate::board::Board;
use crate::ids::{Hex, Set};
use crate::dev_card::DEV_CARDS;
use crate::resource::RESOURCE_CARDS;
use crate::state::Hidden;
use std::collections::{BTreeMap, HashSet};

fn new_game(seed: u64) -> State {
    State::new(Board::random(&mut Rng::new(seed)))
}

fn random_step(state: &mut State, rng: &mut Rng) {
    match state.actor() {
        Some(Actor::Chance) => {
            let outcome = state.sample_outcome(rng);
            state.resolve(outcome);
        }
        Some(Actor::Player(_)) => {
            let legal: Vec<Action> = state.legal_actions().iter().collect();
            state.apply(legal[rng.below(legal.len())]);
        }
        None => panic!("the game is over"),
    }
}

fn after_setup(seed: u64) -> State {
    let mut rng = Rng::new(seed);
    let mut state = new_game(seed);
    while matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad { .. }) {
        random_step(&mut state, &mut rng);
    }
    state
}

fn scenario(seed: u64) -> State {
    let mut state = after_setup(seed);
    state.hands = [[0; 5]; 2];
    state.current = Player::First;
    state.phase = Phase::Turn { rolled: true };
    state
}

#[test]
fn every_action_number_names_one_action() {
    let mut seen = HashSet::new();
    for i in 0..ACTIONS as u16 {
        let action = Action::from_index(i).expect("every number below ACTIONS is an action");
        assert_eq!(action.index(), i);
        assert!(seen.insert(action));
    }
    assert_eq!(Action::from_index(ACTIONS as u16), None);
    let trades = seen.iter().filter(|a| matches!(a, Action::Trade { .. })).count();
    assert_eq!(trades, 3 * 5 * 4, "three rates, five resources to give, four to get");
    assert!(seen.iter().all(|a| !matches!(a, Action::Trade { give, get, .. } if give == get)));
}

#[test]
fn a_mask_is_a_set_of_actions() {
    let mut rng = Rng::new(1);
    for _ in 0..200 {
        let mut mask = Mask::EMPTY;
        let mut model = std::collections::BTreeSet::new();
        for _ in 0..rng.below(40) {
            let action = Action::from_index(rng.below(ACTIONS) as u16).unwrap();
            mask.insert(action);
            model.insert(action.index());
        }
        assert_eq!(mask.len() as usize, model.len());
        assert_eq!(mask.iter().map(Action::index).collect::<Vec<_>>(), model.iter().copied().collect::<Vec<_>>());
        assert!((0..ACTIONS as u16).all(|i| mask.contains(Action::from_index(i).unwrap()) == model.contains(&i)));
    }
    for (offset, count, make) in [
        (SETTLEMENT, VERTICES, (|i| Action::Settlement(Vertex::new(i))) as fn(usize) -> Action),
        (CITY, VERTICES, |i| Action::City(Vertex::new(i))),
        (ROAD, EDGES, |i| Action::Road(Edge::new(i))),
    ] {
        let mut block = Mask::EMPTY;
        block.insert_bits(offset, u128::MAX >> (128 - count));
        let mut one_by_one = Mask::EMPTY;
        (0..count).for_each(|i| one_by_one.insert(make(i)));
        assert_eq!(block, one_by_one);
    }
}

#[test]
fn setup_goes_first_second_second_first_and_the_second_settlement_earns() {
    let t = topology();
    let mut state = new_game(3);
    assert_eq!(state.legal_actions().len(), 54);
    let mut placed = Vec::new();
    for (round, placer) in [Player::First, Player::Second, Player::Second, Player::First].into_iter().enumerate() {
        assert_eq!((state.phase, state.current), (Phase::PlaceSettlement, placer), "round {round}");
        let free = state.legal_actions().len();
        let blocked: HashSet<Vertex> =
            placed.iter().flat_map(|&v: &Vertex| t.adjacent(v).iter().chain([v])).collect();
        assert_eq!(free as usize, 54 - blocked.len(), "the distance rule alone limits setup settlements");
        let vertex = state.legal_actions().iter().last().map(|a| match a {
            Action::Settlement(v) => v,
            _ => unreachable!(),
        });
        let vertex = vertex.unwrap();
        let hand_before = state.hands[placer.index()];
        state.apply(Action::Settlement(vertex));
        let earned: [u8; 5] = std::array::from_fn(|r| state.hands[placer.index()][r] - hand_before[r]);
        let producing = t.hexes_at(vertex).iter().filter_map(|h| state.board.terrain(h).resource());
        let mut expected = [0; 5];
        if round >= 2 {
            producing.for_each(|r| expected[r.index()] += 1);
        }
        assert_eq!(earned, expected, "round {round}");
        assert_eq!(state.phase, Phase::PlaceRoad { settlement: vertex });
        let roads: Vec<Action> = state.legal_actions().iter().collect();
        let expected_roads: Vec<Action> = t.edges_at(vertex).iter().map(Action::Road).collect();
        assert_eq!(roads, expected_roads, "a setup road touches the new settlement");
        state.apply(roads[0]);
        placed.push(vertex);
    }
    assert_eq!((state.phase, state.current), (Phase::Dice, Player::First));
    state.validate().unwrap();
}

fn production_scenario() -> (State, Hex, Resource) {
    let mut state = scenario(4);
    let hex = Hex::all().find(|&h| state.board.token(h) == Some(12)).unwrap();
    let corners = topology().corners(hex);
    state.settlements = [Set::EMPTY, Set::single(corners[3])];
    state.cities = [Set::single(corners[0]), Set::EMPTY];
    state.robber = state.board.desert();
    (state, hex, state.board.terrain(hex).resource().unwrap())
}

fn set_bank(state: &mut State, resource: Resource, bank: u8, holder: Player) {
    state.hands[holder.index()][resource.index()] = RESOURCE_CARDS - bank;
}

#[test]
fn production_pays_a_settlement_one_and_a_city_two_unless_the_bank_is_short() {
    let (state, hex, resource) = production_scenario();
    let _ = hex;
    let r = resource.index();

    let mut plenty = state;
    plenty.produce(12);
    assert_eq!((plenty.hands[0][r], plenty.hands[1][r]), (2, 1));
    assert_eq!(plenty.hands.iter().flatten().map(|&n| u32::from(n)).sum::<u32>(), 3, "only that resource");

    for (bank, paid) in [(2, false), (3, true)] {
        let mut s = state;
        s.set_aside[r] = RESOURCE_CARDS - bank;
        s.produce(12);
        assert_eq!((s.hands[0][r], s.hands[1][r]), if paid { (2, 1) } else { (0, 0) }, "bank {bank}");
    }

    let mut lone = state;
    lone.settlements[1] = Set::EMPTY;
    set_bank(&mut lone, resource, 1, Player::Second);
    lone.produce(12);
    assert_eq!(lone.hands[0][r], 1);

    let mut robbed = state;
    robbed.robber = hex;
    robbed.produce(12);
    assert_eq!(robbed.hands, [[0; 5]; 2]);
}

fn discard_paths(state: State, player: Player, start: [u8; 5], paths: &mut Vec<[u8; 5]>) {
    match state.phase {
        Phase::Discard { player: p, .. } if p == player => {
            let legal: Vec<Action> = state.legal_actions().iter().collect();
            assert!(!legal.is_empty(), "a discard never gets stuck");
            for action in legal {
                let mut next = state;
                next.transition(action);
                discard_paths(next, player, start, paths);
            }
        }
        _ => paths.push(std::array::from_fn(|r| start[r] - state.hands[player.index()][r])),
    }
}

#[test]
fn each_choice_of_discards_has_exactly_one_path() {
    for hand in [[3, 0, 2, 0, 3], [7, 0, 0, 0, 1], [2, 2, 2, 2, 2], [0, 1, 0, 0, 9], [4, 4, 0, 1, 0]] {
        let mut state = scenario(5);
        state.hands[0] = hand;
        let owed = hand.iter().sum::<u8>() / 2;
        state.phase = Phase::Discard { player: Player::First, owed, from: Resource::Brick };
        let mut paths = Vec::new();
        discard_paths(state, Player::First, hand, &mut paths);
        let mut choices = vec![[0u8; 5]];
        for r in 0..5 {
            choices = choices
                .into_iter()
                .flat_map(|c| (0..=hand[r]).map(move |n| { let mut c = c; c[r] = n; c }))
                .collect();
        }
        choices.retain(|c| c.iter().sum::<u8>() == owed);
        let distinct: HashSet<[u8; 5]> = paths.iter().copied().collect();
        assert_eq!(distinct.len(), paths.len(), "{hand:?}: a choice reached twice");
        assert_eq!(distinct, choices.into_iter().collect(), "{hand:?}");
    }
}

#[test]
fn both_players_discard_and_the_first_players_cards_stay_face_down_until_the_end() {
    let mut state = scenario(6);
    state.hands = [[2, 2, 2, 2, 0], [3, 3, 3, 0, 0]];
    state.phase = Phase::Dice;
    state.happen(Outcome::Dice(7));
    assert_eq!(state.phase, Phase::Discard { player: Player::First, owed: 4, from: Resource::Brick });
    let bank_before = Resource::ALL.map(|r| state.bank(r));
    for resource in [Resource::Brick, Resource::Lumber, Resource::Lumber, Resource::Grain] {
        assert!(state.legal_actions().contains(Action::Resource(resource)));
        state.transition(Action::Resource(resource));
    }
    let Phase::Discard { player: Player::Second, owed: 4, .. } = state.phase else {
        panic!("the second player discards next, not {:?}", state.phase)
    };
    assert_eq!(state.set_aside.iter().sum::<u8>(), 4);
    assert_eq!(Resource::ALL.map(|r| state.bank(r)), bank_before, "set aside, not yet in the bank");
    let hidden = state.hidden_from(Player::Second);
    assert_eq!(hidden.set_aside.iter().sum::<u8>(), 4);
    for _ in 0..4 {
        let action = state.legal_actions().iter().next().unwrap();
        state.transition(action);
    }
    assert_eq!(state.phase, Phase::MoveRobber { rolled: true });
    assert_eq!(state.set_aside, [0; 5]);
    assert_eq!(Resource::ALL.iter().map(|&r| state.bank(r)).sum::<u8>(), bank_before.iter().sum::<u8>() + 4 + 4);
}

#[test]
fn the_robber_steals_only_from_a_building_on_its_hex_with_cards_in_hand() {
    let state = scenario(7);
    let victim = state.buildings(Player::Second).iter().next().unwrap();
    let exposed = topology().hexes_at(victim).iter().find(|&h| h != state.robber).unwrap();
    let safe = Hex::all()
        .find(|&h| h != state.robber && (topology().corner_set(h) & state.buildings(Player::Second)).is_empty())
        .unwrap();
    for (hex, cards, steals) in [(exposed, 1, true), (exposed, 0, false), (safe, 3, false)] {
        let mut s = state;
        s.hands[1][0] = cards;
        s.phase = Phase::MoveRobber { rolled: true };
        s.transition(Action::Robber(hex));
        assert_eq!(s.phase, if steals { Phase::Steal { rolled: true } } else { Phase::Turn { rolled: true } });
        assert_eq!(s.robber, hex);
    }
    let mut s = state;
    s.phase = Phase::MoveRobber { rolled: true };
    assert!(!s.legal_actions().contains(Action::Robber(s.robber)), "the robber must move");
    assert_eq!(s.legal_actions().len(), 18);
    s.hands[1] = [0, 0, 2, 0, 1];
    s.phase = Phase::Steal { rolled: true };
    assert_eq!(s.chances().weights(), &[0, 0, 2, 0, 1]);
    s.happen(Outcome::Steal(Resource::Wool));
    assert_eq!((s.hands[0][2], s.hands[1][2]), (1, 1));
}

#[test]
fn a_card_is_not_played_the_turn_it_is_bought_nor_two_in_a_turn() {
    let mut state = scenario(8);
    state.deck = [0, 0, 0, 0, 1];
    state.hands[0] = CARD_COST;
    state.transition(Action::BuyCard);
    assert_eq!(state.phase, Phase::Draw);
    state.happen(Outcome::Draw(DevCard::Monopoly));
    assert_eq!((state.dev_cards[0][4], state.bought[4]), (1, 1));
    assert!(!state.legal_actions().contains(Action::PlayMonopoly), "bought this turn");
    state.dev_cards[0][0] = 1;
    assert!(state.legal_actions().contains(Action::PlayKnight), "an older card");
    state.transition(Action::EndTurn);
    assert_eq!(state.bought, [0; 5]);
    state.transition(Action::Roll);
    state.current = Player::First;
    state.phase = Phase::Turn { rolled: false };
    state.transition(Action::PlayKnight);
    assert!(state.card_played);
    state.phase = Phase::Turn { rolled: false };
    assert!(!state.legal_actions().contains(Action::PlayMonopoly), "one card a turn");
}

#[test]
fn victory_point_cards_count_as_bought_and_win_when_drawn() {
    let mut state = scenario(9);
    let public = state.public_victory_points(Player::First);
    state.dev_cards[0][DevCard::VictoryPoint.index()] = WINNING_POINTS - 1 - public;
    state.deck = [0, 1, 0, 0, 0];
    state.phase = Phase::Draw;
    state.happen(Outcome::Draw(DevCard::VictoryPoint));
    assert_eq!(state.bought[DevCard::VictoryPoint.index()], 1, "public count of cards bought");
    assert_eq!(state.phase, Phase::Over { winner: Player::First });
}

#[test]
fn bank_trades_offer_every_rate_the_harbors_give() {
    let mut state = scenario(10);
    let board = state.board;
    let generic = board.harbor_vertices(crate::board::Harbor::Generic).iter().next().unwrap();
    let brick = board.harbor_vertices(crate::board::Harbor::Special(Resource::Brick)).iter().next().unwrap();
    let rates = |s: &State, give: Resource| -> Vec<u8> {
        let mut rates: Vec<u8> = s
            .legal_actions()
            .iter()
            .filter_map(|a| match a {
                Action::Trade { give: g, get: Resource::Ore, rate } if g == give => Some(rate),
                _ => None,
            })
            .collect();
        rates.sort();
        rates
    };
    state.settlements = [Set::EMPTY, state.settlements[1] | state.cities[1]];
    state.cities = [Set::EMPTY; 2];
    state.hands[0] = [4, 4, 0, 0, 0];
    assert_eq!(rates(&state, Resource::Brick), [4]);
    state.settlements[0] = Set::single(generic);
    assert_eq!(rates(&state, Resource::Brick), [3, 4]);
    state.settlements[0].insert(brick);
    assert_eq!(rates(&state, Resource::Brick), [2, 3, 4]);
    assert_eq!(rates(&state, Resource::Lumber), [3, 4], "the brick harbor is for brick only");
    state.hands[0] = [3, 0, 0, 0, 0];
    assert_eq!(rates(&state, Resource::Brick), [2, 3], "cannot pay 4");
    state.transition(Action::Trade { give: Resource::Brick, get: Resource::Ore, rate: 3 });
    assert_eq!(state.hands[0], [0, 0, 0, 0, 1]);
    state.hands[0] = [4, 0, 0, 0, 0];
    state.hands[1][4] = RESOURCE_CARDS;
    assert_eq!(rates(&state, Resource::Brick), Vec::<u8>::new(), "the bank has no ore");
}

#[test]
fn awards_stay_on_ties_and_go_to_a_strict_leader_with_the_minimum() {
    use Player::{First, Second};
    let cases = [
        (None, [5, 4], Some(First)),
        (None, [4, 4], None),
        (None, [5, 5], None),
        (Some(First), [5, 5], Some(First)),
        (Some(First), [5, 6], Some(Second)),
        (Some(First), [4, 5], Some(Second)),
        (Some(First), [4, 4], None),
        (Some(First), [4, 3], None),
        (Some(Second), [9, 9], Some(Second)),
    ];
    for (holder, counts, expected) in cases {
        assert_eq!(award(holder, counts, 5), expected, "{holder:?} {counts:?}");
    }
    assert_eq!(award(None, [3, 2], 3), Some(First));
    assert_eq!(award(None, [2, 0], 3), None);
}

fn longest_road_by_euler(roads: &[Edge], blocked: Set<VertexKind>) -> u8 {
    let t = topology();
    let mut best = 0;
    for subset in 1u32..1 << roads.len() {
        let edges: Vec<Edge> = (0..roads.len()).filter(|&i| subset >> i & 1 == 1).map(|i| roads[i]).collect();
        let mut degree: BTreeMap<Vertex, u8> = BTreeMap::new();
        for &e in &edges {
            for v in t.ends(e) {
                *degree.entry(v).or_default() += 1;
            }
        }
        let mut component: Set<VertexKind> = Set::single(t.ends(edges[0])[0]);
        loop {
            let grown: Set<VertexKind> =
                edges.iter().filter(|e| t.ends(**e).iter().any(|&v| component.contains(v))).flat_map(|&e| t.ends(e)).collect();
            if grown == component {
                break;
            }
            component = grown;
        }
        let connected = degree.keys().all(|&v| component.contains(v));
        let odd = degree.values().filter(|&&d| d % 2 == 1).count();
        let blocked_degrees: Vec<u8> = degree.iter().filter(|(v, _)| blocked.contains(**v)).map(|(_, &d)| d).collect();
        let ok = match odd {
            2 => blocked_degrees.iter().all(|&d| d == 1),
            0 => blocked_degrees.is_empty() || blocked_degrees == [2],
            _ => false,
        };
        if connected && ok {
            best = best.max(edges.len() as u8);
        }
    }
    best
}

#[test]
fn longest_road_matches_eulers_theorem() {
    let t = topology();
    let mut rng = Rng::new(11);
    let mut lengths_seen = HashSet::new();
    for case in 0..600 {
        let mut roads: Vec<Edge> = Vec::new();
        let size = 1 + rng.below(11);
        let mut at = Vertex::new(rng.below(54));
        while roads.len() < size {
            if rng.below(6) == 0 {
                at = Vertex::new(rng.below(54));
            }
            let options: Vec<Edge> = t.edges_at(at).iter().collect();
            let edge = options[rng.below(options.len())];
            if !roads.contains(&edge) {
                roads.push(edge);
            }
            at = t.ends(edge).into_iter().find(|&v| v != at).unwrap();
        }
        let touched: Vec<Vertex> = roads.iter().flat_map(|&e| t.ends(e)).collect();
        let blocked: Set<VertexKind> =
            (0..rng.below(3)).map(|_| touched[rng.below(touched.len())]).collect();
        let set: Set<EdgeKind> = roads.iter().copied().collect();
        let expected = longest_road_by_euler(&roads, blocked);
        assert_eq!(longest_road(set, blocked), expected, "case {case}: {roads:?} blocked {blocked:?}");
        lengths_seen.insert(expected);
    }
    assert!(lengths_seen.len() >= 9, "a spread of lengths: {lengths_seen:?}");
    let ring: Set<EdgeKind> = {
        let c = t.corners(Hex::new(9));
        (0..6).map(|i| t.edge_between(c[i], c[(i + 1) % 6]).unwrap()).collect()
    };
    let c = t.corners(Hex::new(9));
    assert_eq!(longest_road(ring, Set::EMPTY), 6);
    assert_eq!(longest_road(ring, Set::single(c[0])), 6);
    assert_eq!(longest_road(ring, [c[0], c[3]].into_iter().collect()), 3);
}

#[test]
fn a_settlement_that_cuts_the_longest_road_moves_or_sets_aside_the_title() {
    let t = topology();
    let mut state = scenario(12);
    let start = state.buildings(Player::Second).iter().next().unwrap();
    state.roads = [Set::EMPTY; 2];
    let others = state.buildings(Player::First);
    let mut path = vec![start];
    'search: for _ in 0..1000 {
        path.truncate(1);
        let mut rng = Rng::new(path.len() as u64 + 99);
        for _ in 0..5 {
            let at = *path.last().unwrap();
            let next: Vec<Vertex> = t.adjacent(at).iter().filter(|v| !path.contains(v) && !others.contains(*v)).collect();
            if next.is_empty() {
                continue 'search;
            }
            path.push(next[rng.below(next.len())]);
        }
        break;
    }
    assert_eq!(path.len(), 6);
    state.roads[1] = path.windows(2).map(|w| t.edge_between(w[0], w[1]).unwrap()).collect();
    state.road_lengths = [state.find_road_length(Player::First), state.find_road_length(Player::Second)];
    assert_eq!(state.road_lengths[1], 5);
    state.longest_road = Some(Player::Second);

    let mut cut = state;
    cut.place_settlement(path[2]);
    assert_eq!(cut.road_lengths[1], 3);
    assert_eq!(cut.longest_road, None);

    let mut taken = state;
    taken.road_lengths[0] = 5;
    taken.place_settlement(path[2]);
    assert_eq!(taken.longest_road, Some(Player::First));
}

#[test]
fn road_building_and_year_of_plenty_stop_early_when_nothing_is_left() {
    let mut state = scenario(13);
    state.dev_cards[0] = [0, 0, 1, 1, 0];
    let t = topology();
    let mut roads = state.roads[0];
    let mut frontier: Vec<Vertex> = state.buildings(Player::First).iter().collect();
    while roads.len() < 14 {
        let at = frontier.remove(0);
        for e in (t.edges_at(at) - state.roads[1]).iter() {
            if roads.len() < 14 && !roads.contains(e) {
                roads.insert(e);
                frontier.extend(t.ends(e));
            }
        }
    }
    state.roads[0] = roads;
    state.transition(Action::PlayRoadBuilding);
    assert_eq!(state.phase, Phase::RoadBuilding { rolled: true, roads: 2 });
    let road = state.legal_actions().iter().next().unwrap();
    state.transition(road);
    assert_eq!(state.roads[0].len(), 15);
    assert_eq!(state.phase, Phase::Turn { rolled: true }, "no road pieces left");

    let mut s = scenario(13);
    s.dev_cards[0] = [0, 0, 0, 1, 0];
    s.hands[1] = [19, 19, 19, 19, 18];
    s.transition(Action::PlayYearOfPlenty);
    assert_eq!(s.legal_actions().iter().collect::<Vec<_>>(), [Action::Resource(Resource::Ore)]);
    s.transition(Action::Resource(Resource::Ore));
    assert_eq!(s.phase, Phase::Turn { rolled: true });
    assert_eq!(s.hands[0][4], 1);
}

#[test]
fn monopoly_takes_every_card_of_the_resource_named() {
    let mut state = scenario(14);
    state.hands = [[1, 0, 0, 0, 0], [3, 2, 0, 0, 0]];
    state.dev_cards[0][DevCard::Monopoly.index()] = 1;
    state.transition(Action::PlayMonopoly);
    state.transition(Action::Resource(Resource::Brick));
    assert_eq!(state.hands, [[4, 0, 0, 0, 0], [0, 2, 0, 0, 0]]);
    assert_eq!(state.phase, Phase::Turn { rolled: true });
}

#[test]
fn forced_actions_are_taken_and_the_turn_passes_to_the_next_roll() {
    let mut state = scenario(15);
    state.phase = Phase::Dice;
    let silent = (2..=12)
        .filter(|&n| n != 7)
        .find(|&n| {
            let mut s = state;
            s.produce(n);
            s.hands == [[0; 5]; 2]
        })
        .unwrap();
    state.resolve(Outcome::Dice(silent));
    assert_eq!((state.phase, state.current), (Phase::Dice, Player::Second));
    let mut s = scenario(15);
    s.dev_cards[1][0] = 1;
    s.phase = Phase::Dice;
    s.resolve(Outcome::Dice(silent));
    assert_eq!((s.phase, s.current), (Phase::Turn { rolled: false }, Player::Second));
    assert_eq!(s.legal_actions().len(), 2);
}

#[test]
fn step_refuses_what_cannot_happen() {
    let mut state = new_game(16);
    assert_eq!(state.step(Action::EndTurn.index()), Err(Invalid { code: Action::EndTurn.index() }));
    assert!(state.step(ACTIONS as u16).is_err());
    state.phase = Phase::Steal { rolled: true };
    state.hands[1] = [0, 1, 0, 0, 0];
    assert!(state.step(0).is_err(), "no brick to steal");
    assert!(state.step(5).is_err(), "not an outcome");
    state.phase = Phase::Over { winner: Player::First };
    assert!(state.step(0).is_err());
}

#[test]
fn the_unseen_cards_are_the_deck_and_the_other_hand() {
    let mut state = scenario(17);
    state.deck = [10, 4, 1, 2, 2];
    state.dev_cards = [[1, 0, 1, 0, 0], [2, 1, 0, 0, 0]];
    state.knights = [1, 0];
    let unseen: [u8; 5] = std::array::from_fn(|c| DEV_CARDS[c] - state.dev_cards[0][c] - state.played(DevCard::ALL[c]));
    assert_eq!(unseen, [12, 5, 1, 2, 2]);
    assert_eq!(unseen, std::array::from_fn(|c| state.deck[c] + state.dev_cards[1][c]));
    let hidden = state.hidden_from(Player::First);
    assert_eq!(hidden, Hidden { dev_cards: [2, 1, 0, 0, 0], bought: [0; 5], deck: state.deck, set_aside: [0; 5] });
}

#[test]
fn only_a_hand_of_more_than_seven_discards_and_it_discards_half_rounded_down() {
    for (hands, expected) in [
        ([[2, 2, 2, 1, 0], [1, 1, 1, 1, 3]], Phase::MoveRobber { rolled: true }),
        ([[2, 2, 2, 2, 0], [0; 5]], Phase::Discard { player: Player::First, owed: 4, from: Resource::Brick }),
        ([[1; 5], [3, 3, 3, 0, 0]], Phase::Discard { player: Player::Second, owed: 4, from: Resource::Brick }),
        ([[1; 5], [3, 3, 1, 0, 0]], Phase::MoveRobber { rolled: true }),
    ] {
        let mut state = scenario(18);
        state.hands = hands;
        state.phase = Phase::Dice;
        state.happen(Outcome::Dice(7));
        assert_eq!(state.phase, expected, "{hands:?}");
    }
    for (second, expected_second) in [([3, 3, 1, 0, 0], false), ([3, 3, 2, 0, 0], true)] {
        let mut state = scenario(18);
        state.hands = [[8, 0, 0, 0, 1], second];
        state.phase = Phase::Discard { player: Player::First, owed: 1, from: Resource::Brick };
        state.transition(Action::Resource(Resource::Brick));
        assert_eq!(matches!(state.phase, Phase::Discard { player: Player::Second, owed: 4, .. }), expected_second);
    }
}

#[test]
fn a_road_does_not_extend_past_the_other_players_building() {
    let t = topology();
    let mut state = scenario(19);
    let home = state.buildings(Player::First).iter().next().unwrap();
    let road = t.edges_at(home).iter().next().unwrap();
    let far = t.ends(road).into_iter().find(|&v| v != home).unwrap();
    state.roads = [Set::single(road), Set::EMPTY];
    state.settlements = [Set::single(home), Set::single(far)];
    state.cities = [Set::EMPTY; 2];
    let spots = state.road_spots(Player::First);
    assert_eq!(spots, t.edges_at(home) - Set::single(road), "only from home, not past {far:?}");
    state.settlements[1] = Set::EMPTY;
    assert_eq!(state.road_spots(Player::First), (t.edges_at(home) | t.edges_at(far)) - Set::single(road));
}
