use crate::board::{Harbor, Terrain};
use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{Hex, Vertex};
use crate::player::{Player, MAX_PLAYERS};
use crate::resource::Resource;
use crate::state::{Deal, Phase, State};

pub const OBSERVATION_BYTES: usize = 357;

pub const PHASES: [&str; 15] = [
    "PlaceSettlement",
    "PlaceRoad",
    "Turn",
    "Dice",
    "Discard",
    "MoveRobber",
    "Steal",
    "Draw",
    "RoadBuilding",
    "YearOfPlenty",
    "Monopoly",
    "Over",
    "ChooseVictim",
    "Respond",
    "Confirm",
];

pub fn observe(state: &State, observer: Player, out: &mut [u8; OBSERVATION_BYTES]) {
    let n = state.players();
    let seat = |s: usize| (s < n).then(|| observer.after(s, n));
    let relative = |player: Player| player.offset_from(observer, n) as u8;
    let board = state.board();
    let mut at = 0;
    let mut put = |bytes: &[u8]| {
        out[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };

    for s in 0..MAX_PLAYERS {
        match seat(s) {
            Some(p) => {
                put(&state.settlements(p).bits().to_le_bytes());
                put(&state.cities(p).bits().to_le_bytes());
                put(&state.roads(p).bits().to_le_bytes());
            }
            None => put(&[0; 32]),
        }
    }
    put(&Hex::all().map(|h| terrain_code(board.terrain(h))).collect::<Vec<_>>());
    put(&Hex::all().map(|h| board.token(h).unwrap_or(0)).collect::<Vec<_>>());
    put(&Vertex::all().map(|v| harbor_code(board.harbor_at(v))).collect::<Vec<_>>());
    put(&[state.robber().index() as u8, n as u8]);

    put(&state.hand(observer));
    put(&state.set_aside(observer));
    for s in 1..MAX_PLAYERS {
        match seat(s) {
            Some(p) => {
                let aside: u8 = state.set_aside(p).iter().sum();
                put(&state.known(observer, p));
                put(&[state.hand_size(p) + aside, aside]);
            }
            None => put(&[0; 7]),
        }
    }
    put(&Resource::ALL.map(|r| state.bank(r)));

    let my_turn = state.current() == observer;
    put(&DevCard::ALL.map(|c| state.dev_cards(observer, c)));
    put(&DevCard::ALL.map(|c| if my_turn { state.bought(c) } else { 0 }));
    for s in 1..MAX_PLAYERS {
        match seat(s) {
            Some(p) => {
                let bought: u8 = if state.current() == p { DevCard::ALL.iter().map(|&c| state.bought(c)).sum() } else { 0 };
                put(&[state.dev_card_count(p), bought]);
            }
            None => put(&[0; 2]),
        }
    }
    put(&[state.deck_size()]);
    put(&DevCard::ALL.map(|c| DEV_CARDS[c.index()] - state.dev_cards(observer, c) - state.played(c)));
    let per_seat = |f: &dyn Fn(Player) -> u8| -> [u8; MAX_PLAYERS] { std::array::from_fn(|s| seat(s).map_or(0, f)) };
    put(&per_seat(&|p| state.knights(p)));
    put(&per_seat(&|p| state.road_length(p)));
    put(&per_seat(&|p| if p == observer { state.victory_points(p) } else { state.public_victory_points(p) }));
    let holder = |h: Option<Player>| h.map_or(0, |p| 1 + relative(p));
    put(&[holder(state.longest_road()), holder(state.largest_army())]);
    put(&[relative(state.current()), u8::from(state.card_played())]);

    let none = u8::MAX;
    let (kind, rolled, count, player, vertex, from) = match state.phase() {
        Phase::PlaceSettlement => (0, false, 0, none, none, none),
        Phase::PlaceRoad { settlement } => (1, false, 0, none, settlement.index() as u8, none),
        Phase::Turn { rolled } => (2, rolled, 0, none, none, none),
        Phase::Dice => (3, false, 0, none, none, none),
        Phase::Discard { player, owed, from } => {
            let shown = if player == observer { from.index() as u8 } else { none };
            (4, true, owed, relative(player), none, shown)
        }
        Phase::MoveRobber { rolled } => (5, rolled, 0, none, none, none),
        Phase::Steal { rolled, victim } => (6, rolled, 0, relative(victim), none, none),
        Phase::Draw => (7, true, 0, none, none, none),
        Phase::RoadBuilding { rolled, roads } => (8, rolled, roads, none, none, none),
        Phase::YearOfPlenty { rolled, picks } => (9, rolled, picks, none, none, none),
        Phase::Monopoly { rolled } => (10, rolled, 0, none, none, none),
        Phase::Over { winner } => (11, false, 0, relative(winner), none, none),
        Phase::ChooseVictim { rolled } => (12, rolled, 0, none, none, none),
        Phase::Respond { responder, .. } => (13, true, 0, relative(responder), none, none),
        Phase::Confirm { .. } => (14, true, 0, none, none, none),
    };
    put(&[kind, u8::from(rolled), count, player, vertex, from]);

    let (deal, accepted, counters) = match state.phase() {
        Phase::Respond { deal, accepted, counters, .. } | Phase::Confirm { deal, accepted, counters } => (Some(deal), accepted, counters),
        _ => (None, 0, [None; MAX_PLAYERS]),
    };
    let sides = |deal: Option<u8>| deal.map_or([[0; 5]; 2], |id| [Deal::from_id(id).give, Deal::from_id(id).get]);
    let accepted_relative = (0..n).filter(|&p| accepted >> p & 1 == 1).fold(0u8, |bits, p| bits | 1 << relative(Player::new(p)));
    put(&sides(deal).concat());
    put(&[accepted_relative, state.offers()]);
    for s in 1..MAX_PLAYERS {
        put(&sides(seat(s).and_then(|p| counters[p.index()])).concat());
    }
    let last = state.last_offer();
    let relative_bits = |bits: u8| (0..n).filter(|&p| bits >> p & 1 == 1).fold(0u8, |acc, p| acc | 1 << relative(Player::new(p)));
    put(&sides(last.map(|l| l.deal)).concat());
    put(&[
        last.map_or(0, |l| relative_bits(l.accepted)),
        last.map_or(0, |l| relative_bits(l.countered)),
        last.map_or(0, |l| u8::from(l.traded)),
    ]);
    debug_assert_eq!(at, OBSERVATION_BYTES);
}

fn terrain_code(terrain: Terrain) -> u8 {
    match terrain {
        Terrain::Hills => 0,
        Terrain::Forest => 1,
        Terrain::Pasture => 2,
        Terrain::Fields => 3,
        Terrain::Mountains => 4,
        Terrain::Desert => 5,
    }
}

fn harbor_code(harbor: Option<Harbor>) -> u8 {
    match harbor {
        None => 0,
        Some(Harbor::Generic) => 1,
        Some(Harbor::Special(resource)) => 2 + resource.index() as u8,
    }
}
