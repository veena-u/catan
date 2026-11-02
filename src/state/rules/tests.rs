use super::*;
use crate::board::Board;
use crate::ids::{Hex, Set};
use crate::dev_card::DEV_CARDS;
use crate::resource::RESOURCE_CARDS;
use crate::state::Hidden;
use crate::observation::{observe, OBSERVATION_BYTES, TRADE_AT};
use crate::state::Rules;
use std::collections::{BTreeMap, HashSet};

fn new_game(seed: u64) -> State {
    game_of(seed, 2)
}

fn game_of(seed: u64, players: usize) -> State {
    State::new(Board::random(&mut Rng::new(seed)), players)
}

fn seats<T: Copy + Default>(items: &[T]) -> [T; MAX_PLAYERS] {
    std::array::from_fn(|i| items.get(i).copied().unwrap_or_default())
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

fn after_setup_of(seed: u64, players: usize) -> State {
    let mut rng = Rng::new(seed);
    let mut state = game_of(seed, players);
    while matches!(state.phase, Phase::PlaceSettlement | Phase::PlaceRoad { .. }) {
        random_step(&mut state, &mut rng);
    }
    state
}

fn scenario(seed: u64) -> State {
    scenario_of(seed, 2)
}

fn scenario_of(seed: u64, players: usize) -> State {
    let mut state = after_setup_of(seed, players);
    state.hands = [[0; 5]; MAX_PLAYERS];
    state.known = [[[0; 5]; MAX_PLAYERS]; MAX_PLAYERS];
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
    assert_eq!(trades, 5 * 4, "five resources to give, four to get; the rate is the player's best");
    assert!(seen.iter().all(|a| !matches!(a, Action::Trade { give, get } if give == get)));
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
fn setup_snakes_out_and_back_and_the_second_settlement_earns() {
    for players in 2..=MAX_PLAYERS {
        setup_snakes_for(players);
    }
}

fn setup_snakes_for(players: usize) {
    let t = topology();
    let mut state = game_of(3, players);
    assert_eq!(state.legal_actions().len(), 54);
    let mut placed = Vec::new();
    let order: Vec<Player> = (0..players).chain((0..players).rev()).map(Player::new).collect();
    for (round, placer) in order.into_iter().enumerate() {
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
        if round >= players {
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
    state.settlements = seats(&[Set::EMPTY, Set::single(corners[3])]);
    state.cities = seats(&[Set::single(corners[0]), Set::EMPTY]);
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
        s.set_aside[0][r] = RESOURCE_CARDS - bank;
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
    assert_eq!(robbed.hands, [[0; 5]; MAX_PLAYERS]);
}

#[test]
fn with_several_players_short_of_the_bank_no_one_is_paid_and_a_lone_claim_takes_what_is_left() {
    let mut state = scenario_of(4, 3);
    let hex = Hex::all().find(|&h| state.board.token(h) == Some(12)).unwrap();
    let c = topology().corners(hex);
    state.settlements = seats(&[Set::single(c[0]), Set::single(c[2]), Set::single(c[4])]);
    state.cities = [Set::EMPTY; MAX_PLAYERS];
    state.robber = state.board.desert();
    let r = state.board.terrain(hex).resource().unwrap().index();
    let mut short = state;
    short.set_aside[0][r] = RESOURCE_CARDS - 2;
    short.produce(12);
    assert_eq!((short.hands[0][r], short.hands[1][r], short.hands[2][r]), (0, 0, 0));
    let mut plenty = state;
    plenty.produce(12);
    assert_eq!((plenty.hands[0][r], plenty.hands[1][r], plenty.hands[2][r]), (1, 1, 1));
    for observer in [0, 1, 2] {
        assert_eq!(plenty.known(Player::new(observer), Player::Third)[r], 1, "production is public");
    }
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
    state.hands = seats(&[[2, 2, 2, 2, 0], [3, 3, 3, 0, 0]]);
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
    assert_eq!(state.set_aside[0].iter().sum::<u8>(), 4);
    assert_eq!(Resource::ALL.map(|r| state.bank(r)), bank_before, "set aside, not yet in the bank");
    let hidden = state.hidden_from(Player::Second);
    assert_eq!(hidden.set_aside[0].iter().sum::<u8>(), 4);
    for _ in 0..4 {
        let action = state.legal_actions().iter().next().unwrap();
        state.transition(action);
    }
    assert_eq!(state.phase, Phase::MoveRobber { rolled: true });
    assert_eq!(state.set_aside, [[0; 5]; MAX_PLAYERS]);
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
        assert_eq!(s.phase, if steals { Phase::ChooseVictim { rolled: true } } else { Phase::Turn { rolled: true } });
        assert_eq!(s.robber, hex);
        if steals {
            assert_eq!(s.legal_actions().iter().collect::<Vec<_>>(), [Action::Rob(1)], "one victim, so the choice is forced");
        }
    }
    let mut s = state;
    s.phase = Phase::MoveRobber { rolled: true };
    assert!(!s.legal_actions().contains(Action::Robber(s.robber)), "the robber must move");
    assert_eq!(s.legal_actions().len(), 18);
    s.hands[1] = [0, 0, 2, 0, 1];
    s.phase = Phase::Steal { rolled: true, victim: Player::Second };
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
    state.dev_cards[0][DevCard::VictoryPoint.index()] = state.rules.points - 1 - public;
    state.deck = [0, 1, 0, 0, 0];
    state.phase = Phase::Draw;
    state.happen(Outcome::Draw(DevCard::VictoryPoint));
    assert_eq!(state.bought[DevCard::VictoryPoint.index()], 1, "public count of cards bought");
    assert_eq!(state.phase, Phase::Over { winner: Player::First });
}

#[test]
fn bank_trades_go_at_the_best_rate_the_harbors_give() {
    let mut state = scenario(10);
    let board = state.board;
    let generic = board.harbor_vertices(crate::board::Harbor::Generic).iter().next().unwrap();
    let brick = board.harbor_vertices(crate::board::Harbor::Special(Resource::Brick)).iter().next().unwrap();
    let ore_for = |s: &State, give: Resource| s.legal_actions().contains(Action::Trade { give, get: Resource::Ore });
    state.settlements = seats(&[Set::EMPTY, state.settlements[1] | state.cities[1]]);
    state.cities = [Set::EMPTY; MAX_PLAYERS];
    state.hands[0] = [4, 4, 0, 0, 0];
    assert_eq!(state.rate(Player::First, Resource::Brick), 4);
    state.settlements[0] = Set::single(generic);
    assert_eq!(state.rate(Player::First, Resource::Brick), 3);
    state.settlements[0].insert(brick);
    assert_eq!(state.rate(Player::First, Resource::Brick), 2);
    assert_eq!(state.rate(Player::First, Resource::Lumber), 3, "the brick harbor is for brick only");
    state.hands[0] = [2, 2, 0, 0, 0];
    assert!(ore_for(&state, Resource::Brick) && !ore_for(&state, Resource::Lumber), "two brick pay, two lumber do not");
    state.transition(Action::Trade { give: Resource::Brick, get: Resource::Ore });
    assert_eq!(state.hands[0], [0, 2, 0, 0, 1], "two brick for one ore");
    state.hands[0] = [4, 0, 0, 0, 0];
    state.hands[1][4] = RESOURCE_CARDS;
    assert!(!ore_for(&state, Resource::Brick), "the bank has no ore");
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
        assert_eq!(award(holder, &counts, 5), expected, "{holder:?} {counts:?}");
    }
    assert_eq!(award(None, &[3, 2], 3), Some(First));
    assert_eq!(award(None, &[2, 0], 3), None);
    use Player::Third;
    assert_eq!(award(None, &[5, 6, 6], 5), None, "a tie for the lead");
    assert_eq!(award(Some(First), &[4, 6, 6], 5), None, "the holder falls behind a tie");
    assert_eq!(award(Some(First), &[6, 6, 6], 5), Some(First), "the holder keeps it on a tie");
    assert_eq!(award(Some(Second), &[3, 4, 7], 5), Some(Third));
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
    state.roads = [Set::EMPTY; MAX_PLAYERS];
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
    state.road_lengths = seats(&[state.find_road_length(Player::First), state.find_road_length(Player::Second)]);
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
    state.hands = seats(&[[1, 0, 0, 0, 0], [3, 2, 0, 0, 0]]);
    state.dev_cards[0][DevCard::Monopoly.index()] = 1;
    state.transition(Action::PlayMonopoly);
    state.transition(Action::Resource(Resource::Brick));
    assert_eq!(state.hands, seats(&[[4, 0, 0, 0, 0], [0, 2, 0, 0, 0]]));
    assert_eq!(state.phase, Phase::Turn { rolled: true });

    let mut four = scenario_of(14, 4);
    four.hands = seats(&[[0; 5], [2, 0, 0, 0, 0], [0, 0, 0, 0, 0], [5, 1, 0, 0, 0]]);
    four.dev_cards[0][DevCard::Monopoly.index()] = 1;
    four.transition(Action::PlayMonopoly);
    four.transition(Action::Resource(Resource::Brick));
    assert_eq!(four.hands, seats(&[[7, 0, 0, 0, 0], [0; 5], [0; 5], [0, 1, 0, 0, 0]]));
    assert_eq!(four.known(Player::Third, Player::First), [7, 0, 0, 0, 0], "a monopoly is public");
}

#[test]
fn forced_actions_are_taken_and_the_turn_passes_to_the_next_roll() {
    let mut state = scenario(15);
    state.rules = Rules { trading: false, ..Rules::STANDARD };
    state.phase = Phase::Dice;
    let silent = (2..=12)
        .filter(|&n| n != 7)
        .find(|&n| {
            let mut s = state;
            s.produce(n);
            s.hands == [[0; 5]; MAX_PLAYERS]
        })
        .unwrap();
    state.resolve(Outcome::Dice(silent));
    assert_eq!((state.phase, state.current), (Phase::Dice, Player::Second));
    let mut s = scenario(15);
    s.rules = state.rules;
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
    state.phase = Phase::Steal { rolled: true, victim: Player::Second };
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
    state.dev_cards = seats(&[[1, 0, 1, 0, 0], [2, 1, 0, 0, 0]]);
    state.knights = seats(&[1, 0]);
    let unseen: [u8; 5] = std::array::from_fn(|c| DEV_CARDS[c] - state.dev_cards[0][c] - state.played(DevCard::ALL[c]));
    assert_eq!(unseen, [12, 5, 1, 2, 2]);
    assert_eq!(unseen, std::array::from_fn(|c| state.deck[c] + state.dev_cards[1][c]));
    let hidden = state.hidden_from(Player::First);
    let expected = Hidden {
        dev_cards: seats(&[[0; 5], [2, 1, 0, 0, 0]]),
        bought: [0; 5],
        deck: state.deck,
        hands: seats(&[[0; 5], state.hands[1]]),
        set_aside: [[0; 5]; MAX_PLAYERS],
    };
    assert_eq!(hidden, expected);
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
        state.hands = seats(&hands);
        state.phase = Phase::Dice;
        state.happen(Outcome::Dice(7));
        assert_eq!(state.phase, expected, "{hands:?}");
    }
    for (second, expected_second) in [([3, 3, 1, 0, 0], false), ([3, 3, 2, 0, 0], true)] {
        let mut state = scenario(18);
        state.hands = seats(&[[8, 0, 0, 0, 1], second]);
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
    state.roads = seats(&[Set::single(road), Set::EMPTY]);
    state.settlements = seats(&[Set::single(home), Set::single(far)]);
    state.cities = [Set::EMPTY; MAX_PLAYERS];
    let spots = state.road_spots(Player::First);
    assert_eq!(spots, t.edges_at(home) - Set::single(road), "only from home, not past {far:?}");
    state.settlements[1] = Set::EMPTY;
    assert_eq!(state.road_spots(Player::First), (t.edges_at(home) | t.edges_at(far)) - Set::single(road));
}

#[test]
fn the_robber_may_rob_any_exposed_player_with_cards() {
    let mut state = scenario_of(20, 4);
    let t = topology();
    let hex = Hex::all().find(|&h| h != state.robber && state.board.token(h).is_some()).unwrap();
    let c = t.corners(hex);
    state.settlements = seats(&[Set::EMPTY, Set::single(c[0]), Set::single(c[2]), Set::single(c[4])]);
    state.cities = [Set::EMPTY; MAX_PLAYERS];
    state.hands = seats(&[[0; 5], [1, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0; 5]]);
    state.phase = Phase::MoveRobber { rolled: true };
    state.transition(Action::Robber(hex));
    assert_eq!(state.phase, Phase::ChooseVictim { rolled: true });
    assert_eq!(state.legal_actions().iter().collect::<Vec<_>>(), [Action::Rob(1), Action::Rob(2)], "the fourth player has no cards");
    state.transition(Action::Rob(2));
    assert_eq!(state.phase, Phase::Steal { rolled: true, victim: Player::Third });
    assert_eq!(state.chances().weights(), &[0, 2, 0, 0, 0]);
}

#[test]
fn a_steal_is_seen_by_its_two_players_and_blurs_what_the_others_know() {
    let mut state = scenario_of(21, 3);
    state.hands = seats(&[[0; 5], [2, 1, 0, 0, 0], [0; 5]]);
    for o in 0..3 {
        for p in 0..3 {
            if o != p {
                state.known[o][p] = state.hands[p];
            }
        }
    }
    state.steal(Player::First, Player::Second, Resource::Brick.index());
    assert_eq!(state.known(Player::First, Player::Second), [1, 1, 0, 0, 0], "the thief knows what it took");
    assert_eq!(state.known(Player::Second, Player::First), [1, 0, 0, 0, 0], "the victim knows what it lost");
    assert_eq!(state.known(Player::Third, Player::Second), [1, 0, 0, 0, 0], "a bystander only knows a card left");
    assert_eq!(state.known(Player::Third, Player::First), [0; 5], "and that the thief gained one it cannot name");
}

#[test]
fn random_games_of_every_size_follow_the_rules_and_hide_what_they_should() {
    let mut rng = Rng::new(22);
    for rules in [Rules::STANDARD, Rules::COLONIST_1V1] {
        for players in 2..=MAX_PLAYERS {
            random_games(rules, players, &mut rng);
        }
    }
}

fn random_games(rules: Rules, players: usize, rng: &mut Rng) {
    let mut won = 0;
    for game in 0..40 {
        let mut state = State::with_rules(Board::random(&mut Rng::new(1000 + game)), players, rules);
        for step in 0..60_000 {
            state.validate().unwrap_or_else(|v| panic!("{players} players, game {game}, step {step}: {v}"));
            let Some(actor) = state.actor() else {
                won += 1;
                break;
            };
            if let Actor::Player(_) = actor {
                assert!(!state.legal_actions().is_empty(), "a player is stuck in {:?}", state.phase);
            }
            if step % 9 == 0 {
                if let Actor::Player(observer) = actor {
                    let mut seen = [0; OBSERVATION_BYTES];
                    observe(&state, observer, &mut seen);
                    let mut again = [0; OBSERVATION_BYTES];
                    observe(&state.sample_hidden(observer, rng), observer, &mut again);
                    assert_eq!(seen, again, "{players} players: re-dealing hidden cards changed what {observer:?} sees");
                }
            }
            random_step(&mut state, rng);
        }
    }
    let enough = if rules == Rules::STANDARD { 30 } else { 10 };
    assert!(won >= enough, "{rules:?}, {players} players: only {won} of 40 random games finished");
}

#[test]
fn a_player_given_the_longest_road_off_their_turn_wins_when_their_turn_comes() {
    let mut state = scenario_of(23, 3);
    let points = state.public_victory_points(Player::Third);
    state.longest_road = Some(Player::Third);
    state.dev_cards[2][DevCard::VictoryPoint.index()] = state.rules.points - points - 2;
    state.current = Player::Second;
    state.phase = Phase::Turn { rolled: true };
    assert!(state.victory_points(Player::Third) >= state.rules.points);
    state.transition(Action::EndTurn);
    assert_eq!(state.phase, Phase::Over { winner: Player::Third });
}

fn act(state: &mut State, action: Action) {
    assert!(state.legal_actions().contains(action), "{action:?} is not legal in {:?}", state.phase);
    state.transition(action);
}

fn draft(state: &mut State, give: [u8; 5], get: [u8; 5]) {
    for resource in Resource::ALL {
        (0..give[resource.index()]).for_each(|_| act(state, Action::Give(resource)));
        (0..get[resource.index()]).for_each(|_| act(state, Action::Ask(resource)));
    }
}

fn known_hands(state: &mut State, hands: &[[u8; 5]]) {
    state.hands = seats(hands);
    for o in 0..hands.len() {
        for p in (0..hands.len()).filter(|&p| p != o) {
            state.known[o][p] = state.hands[p];
        }
    }
}

fn offer(state: &mut State, give: [u8; 5], get: [u8; 5]) {
    act(state, Action::Offer);
    draft(state, give, get);
    act(state, Action::Send);
}

#[test]
fn an_offer_is_drafted_card_by_card_with_any_number_of_cards() {
    let mut state = scenario_of(27, 3);
    state.hands = seats(&[[3, 0, 1, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 2]]);
    act(&mut state, Action::Offer);
    assert_eq!(state.phase, Phase::Draft { deal: Deal::EMPTY });
    let legal = state.legal_actions();
    assert!(legal.contains(Action::Give(Resource::Brick)) && legal.contains(Action::Give(Resource::Wool)));
    assert!(!legal.contains(Action::Give(Resource::Ore)), "only cards in hand can be given");
    assert!(Resource::ALL.iter().all(|&r| legal.contains(Action::Ask(r))), "anything can be asked for");
    assert!(!legal.contains(Action::Send), "an empty draft can't be sent");
    draft(&mut state, [3, 0, 0, 0, 0], [0, 0, 0, 0, 4]);
    let legal = state.legal_actions();
    assert!(!legal.contains(Action::Give(Resource::Brick)), "all three brick are on the offer");
    assert!(!legal.contains(Action::Ask(Resource::Brick)) && !legal.contains(Action::Give(Resource::Ore)), "never both sides");
    assert!(legal.contains(Action::Send) && legal.contains(Action::Abandon));
    state.validate().unwrap();
    act(&mut state, Action::Send);
    assert!(matches!(state.phase, Phase::Respond { responder: Player::Second, .. }));
    assert_eq!(state.known(Player::Third, Player::First)[0], 3, "offering shows you hold the brick");
    assert!(!state.legal_actions().contains(Action::Accept), "the second player holds no ore");
    state.validate().unwrap();
}

#[test]
fn the_offerer_trades_with_an_accepter_or_kills_the_deal_but_cannot_make_it_again() {
    let mut state = scenario_of(25, 3);
    state.hands = seats(&[[2, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]);
    offer(&mut state, [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]);
    act(&mut state, Action::Accept);
    assert_eq!(state.known(Player::Third, Player::Second)[4], 1, "accepting shows you hold the ore");
    act(&mut state, Action::Accept);
    let options: Vec<Action> = state.legal_actions().iter().collect();
    assert_eq!(options, [Action::TradeWith(1), Action::TradeWith(2), Action::Cancel], "as in Colonist, the deal can still be killed");
    state.validate().unwrap();
    act(&mut state, Action::Cancel);
    assert_eq!(state.hands, seats(&[[2, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]), "a killed deal moves no cards");
    assert_eq!(state.last_offer().map(|l| (l.accepted, l.traded)), Some((0b110, false)));
    act(&mut state, Action::Offer);
    draft(&mut state, [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]);
    assert!(!state.legal_actions().contains(Action::Send), "the same offer can't be made again this turn");
    act(&mut state, Action::Give(Resource::Brick));
    act(&mut state, Action::Send);
    act(&mut state, Action::Accept);
    act(&mut state, Action::Decline);
    act(&mut state, Action::TradeWith(1));
    assert_eq!(state.hands, seats(&[[0, 0, 0, 0, 1], [2, 0, 0, 0, 0], [0, 0, 0, 0, 1]]));
    assert_eq!(state.known(Player::Third, Player::Second)[0], 2, "a trade is public");
    state.validate().unwrap();
    act(&mut state, Action::EndTurn);
    assert_eq!((state.proposed, state.moves, state.last_offer()), (0, 0, None), "a new turn starts with a clean slate");
}

#[test]
fn there_is_no_cap_on_offers_in_a_turn() {
    let mut state = scenario_of(26, 2);
    known_hands(&mut state, &[[5, 0, 0, 0, 0], [0; 5]]);
    for k in 1..=5 {
        offer(&mut state, [k, 0, 0, 0, 0], [0, 0, 0, 0, 1]);
        let options: Vec<Action> = state.legal_actions().iter().collect();
        assert_eq!(options, [Action::Decline], "an empty hand can neither accept nor counter");
        act(&mut state, Action::Decline);
        assert_eq!(state.phase, Phase::Turn { rolled: true });
    }
    assert_eq!(state.offers(), 5);
    state.validate().unwrap();
}

#[test]
fn an_open_offer_can_only_be_answered_with_a_counter() {
    let mut state = scenario_of(26, 2);
    known_hands(&mut state, &[[0, 0, 2, 0, 0], [1, 1, 0, 0, 0]]);
    offer(&mut state, [0, 0, 1, 0, 0], [0; 5]);
    let options: Vec<Action> = state.legal_actions().iter().collect();
    assert_eq!(options, [Action::Decline, Action::Counter], "one wool for anything names nothing to accept");
    act(&mut state, Action::Counter);
    act(&mut state, Action::Give(Resource::Brick));
    assert!(!state.legal_actions().contains(Action::Send), "a counter names both sides");
    act(&mut state, Action::Ask(Resource::Wool));
    state.validate().unwrap();
    act(&mut state, Action::Send);
    assert_eq!(state.known(Player::First, Player::Second)[0], 1, "countering shows you hold what you give");
    let options: Vec<Action> = state.legal_actions().iter().collect();
    assert_eq!(options, [Action::TakeCounter(1), Action::Cancel]);
    act(&mut state, Action::TakeCounter(1));
    assert_eq!(state.hands, seats(&[[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]]));
    state.validate().unwrap();
}

#[test]
fn a_counter_that_only_accepts_is_not_sent_and_a_dropped_counter_is_not_redrafted() {
    let mut state = scenario_of(26, 2);
    known_hands(&mut state, &[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]);
    offer(&mut state, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]);
    act(&mut state, Action::Counter);
    draft(&mut state, [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]);
    assert!(!state.legal_actions().contains(Action::Send), "that is the offer itself: accept it instead");
    act(&mut state, Action::Abandon);
    let options: Vec<Action> = state.legal_actions().iter().collect();
    assert_eq!(options, [Action::Accept, Action::Decline]);
    act(&mut state, Action::Accept);
    assert!(matches!(state.phase, Phase::Confirm { .. }));
}

#[test]
fn the_turn_budget_ends_a_turn_like_a_timer() {
    let mut state = scenario_of(26, 2);
    state.rules = Rules { turn_budget: 4, ..Rules::STANDARD };
    known_hands(&mut state, &[[3, 0, 0, 0, 0], [0; 5]]);
    state.apply(Action::Offer);
    assert_eq!(state.finish_deal([-1, 0, 0, 0, 1]), [Action::Give(Resource::Brick), Action::Ask(Resource::Ore), Action::Send]);
    assert_eq!((state.phase, state.moves), (Phase::Turn { rolled: true }, 1), "declined at once; the clicks cost nothing");
    state.moves = 3;
    state.apply(Action::Offer);
    state.finish_deal([-2, 0, 0, 0, 1]);
    assert_eq!(state.current, Player::Second, "the second offer used up the budget, so the turn ended");
    assert_eq!(state.moves, 0);
    state.validate().unwrap();
}

#[test]
fn a_whole_offer_is_made_in_one_step_and_a_repeat_is_dropped() {
    let mut state = scenario_of(25, 3);
    state.hands = seats(&[[2, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]);
    state.apply(Action::Offer);
    state.finish_deal([-1, 0, 0, 0, 1]);
    assert!(matches!(state.phase, Phase::Respond { responder: Player::Second, .. }));
    act(&mut state, Action::Decline);
    act(&mut state, Action::Decline);
    assert_eq!(state.forbidden_deals(), [[-1, 0, 0, 0, 1]]);
    state.apply(Action::Offer);
    assert_eq!(state.finish_deal([-1, 0, 0, 0, 1]).last(), Some(&Action::Abandon), "an offer already made is dropped");
    assert_eq!(state.phase, Phase::Turn { rolled: true });
    state.apply(Action::Offer);
    assert_eq!(state.finish_deal([-3, 0, 0, 0, 0]).last(), Some(&Action::Abandon), "more brick than in hand stops the offer");
    state.apply(Action::Offer);
    state.finish_deal([-2, 0, 0, 0, 0]);
    assert!(matches!(state.phase, Phase::Respond { .. }), "an open offer: two brick for anything");
    assert_eq!(state.forbidden_deals(), [[2, 0, 0, 0, 0]], "a counter can't simply accept");
    act(&mut state, Action::Counter);
    assert_eq!(state.finish_deal([2, 0, 0, 0, -1]).last(), Some(&Action::Send), "two brick for the ore");
    state.validate().unwrap();
}

#[test]
fn a_friendly_robber_spares_players_showing_two_points_or_fewer() {
    let mut state = scenario_of(20, 3);
    state.rules = Rules { friendly_robber: true, ..Rules::STANDARD };
    let t = topology();
    let hex = Hex::all().find(|&h| h != state.robber && state.board.token(h).is_some()).unwrap();
    let c = t.corners(hex);
    let far = Vertex::all().find(|&v| !t.corner_set(hex).contains(v) && (t.adjacent(v) & t.corner_set(hex)).is_empty()).unwrap();
    state.settlements = seats(&[Set::EMPTY, Set::single(c[0]), Set::single(far)]);
    state.cities = seats(&[Set::EMPTY, Set::EMPTY, Set::single(c[2])]);
    state.hands = seats(&[[0; 5], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]]);
    assert!(state.protected(Player::Second) && !state.protected(Player::Third), "one point against three");
    state.phase = Phase::MoveRobber { rolled: true };
    let legal: Vec<Hex> = state.legal_actions().iter().filter_map(|a| if let Action::Robber(h) = a { Some(h) } else { None }).collect();
    assert!(!legal.is_empty() && legal.iter().all(|&h| !state.touches(Player::Second, h)), "no hex of the protected player");
    state.robber = hex;
    assert!(!state.can_rob(Player::Second) && state.can_rob(Player::Third));
    state.rules = Rules::STANDARD;
    assert!(state.can_rob(Player::Second), "without the friendly robber anyone exposed can be robbed");
}

#[test]
fn the_rules_set_the_discard_limit_and_the_points_to_win() {
    let colonist = Rules { friendly_robber: false, ..Rules::COLONIST_1V1 };
    for (hand, expected) in [
        ([2, 2, 2, 2, 1], Phase::MoveRobber { rolled: true }),
        ([2, 2, 2, 2, 2], Phase::Discard { player: Player::First, owed: 5, from: Resource::Brick }),
    ] {
        let mut state = scenario(18);
        state.rules = colonist;
        state.hands = seats(&[hand, [0; 5]]);
        state.phase = Phase::Dice;
        state.happen(Outcome::Dice(7));
        assert_eq!(state.phase, expected, "nine cards are safe, ten discard half");
    }
    let mut state = scenario(9);
    state.rules = colonist;
    let public = state.public_victory_points(Player::First);
    state.dev_cards[0][DevCard::VictoryPoint.index()] = 14 - public;
    state.finish_if_won();
    assert_eq!(state.phase, Phase::Turn { rolled: true }, "14 points is not enough");
    state.dev_cards[0][DevCard::VictoryPoint.index()] = 15 - public;
    state.finish_if_won();
    assert_eq!(state.phase, Phase::Over { winner: Player::First });
}

#[test]
fn everyone_sees_the_offer_and_each_answer_but_no_one_sees_a_draft() {
    let mut state = scenario_of(26, 3);
    state.hands = seats(&[[2, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, 1]]);
    act(&mut state, Action::Offer);
    act(&mut state, Action::Give(Resource::Brick));
    let mut mine = [0; OBSERVATION_BYTES];
    let mut theirs = [0; OBSERVATION_BYTES];
    observe(&state, Player::First, &mut mine);
    observe(&state, Player::Second, &mut theirs);
    act(&mut state, Action::Give(Resource::Brick));
    let mut again = [0; OBSERVATION_BYTES];
    observe(&state, Player::First, &mut again);
    assert_eq!(mine, again, "a draft shows nowhere until it is sent");
    observe(&state, Player::Second, &mut again);
    assert_eq!(theirs, again);
    act(&mut state, Action::Abandon);
    act(&mut state, Action::Offer);
    act(&mut state, Action::Give(Resource::Brick));
    act(&mut state, Action::Ask(Resource::Ore));
    act(&mut state, Action::Send);
    act(&mut state, Action::Accept);
    observe(&state, Player::First, &mut mine);
    assert_eq!(&mine[TRADE_AT..TRADE_AT + 10], &[1, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
    let status = |seat: usize| mine[TRADE_AT + 10 + 11 * (seat - 1)];
    assert_eq!((status(1), status(2), status(3)), (2, 1, 0), "the next seat accepted, the one after is still to answer");
}
