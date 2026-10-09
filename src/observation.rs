use crate::board::{Harbor, Terrain};
use crate::dev_card::{DevCard, DEV_CARDS};
use crate::ids::{Hex, Vertex};
use crate::player::Player;
use crate::resource::Resource;
use crate::state::{Phase, State};

pub const OBSERVATION_BYTES: usize = 211;

pub const PHASES: [&str; 12] = [
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
];

pub fn observe(state: &State, observer: Player, out: &mut [u8; OBSERVATION_BYTES]) {
    let (me, them) = (observer, observer.other());
    let board = state.board();
    let mut at = 0;
    let mut put = |bytes: &[u8]| {
        out[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };

    for player in [me, them] {
        put(&state.settlements(player).bits().to_le_bytes());
    }
    for player in [me, them] {
        put(&state.cities(player).bits().to_le_bytes());
    }
    for player in [me, them] {
        put(&state.roads(player).bits().to_le_bytes());
    }
    put(&Hex::all().map(|h| terrain_code(board.terrain(h))).collect::<Vec<_>>());
    put(&Hex::all().map(|h| board.token(h).unwrap_or(0)).collect::<Vec<_>>());
    put(&Vertex::all().map(|v| harbor_code(board.harbor_at(v))).collect::<Vec<_>>());
    put(&[state.robber().index() as u8]);

    let aside = |r: Resource| state.set_aside(r);
    let hidden_aside = |r: Resource| if state.current() == them { aside(r) } else { 0 };
    put(&Resource::ALL.map(|r| state.resources(me, r)));
    put(&Resource::ALL.map(|r| state.resources(them, r) + hidden_aside(r)));
    put(&Resource::ALL.map(|r| if state.current() == me { aside(r) } else { 0 }));
    put(&Resource::ALL.map(|r| state.bank(r)));

    let my_turn = state.current() == me;
    put(&DevCard::ALL.map(|c| state.dev_cards(me, c)));
    put(&DevCard::ALL.map(|c| if my_turn { state.bought(c) } else { 0 }));
    let their_bought: u8 = if my_turn { 0 } else { DevCard::ALL.iter().map(|&c| state.bought(c)).sum() };
    put(&[state.dev_card_count(them), their_bought, state.deck_size()]);
    put(&DevCard::ALL.map(|c| DEV_CARDS[c.index()] - state.dev_cards(me, c) - state.played(c)));
    put(&[state.knights(me), state.knights(them)]);
    put(&[state.road_length(me), state.road_length(them)]);
    put(&[state.victory_points(me), state.public_victory_points(them)]);
    put(&[holder_code(state.longest_road(), me), holder_code(state.largest_army(), me)]);
    put(&[u8::from(my_turn), u8::from(state.card_played())]);

    let relative = |player: Player| u8::from(player != me);
    let none = u8::MAX;
    let (kind, rolled, count, player, vertex, from) = match state.phase() {
        Phase::PlaceSettlement => (0, false, 0, none, none, none),
        Phase::PlaceRoad { settlement } => (1, false, 0, none, settlement.index() as u8, none),
        Phase::Turn { rolled } => (2, rolled, 0, none, none, none),
        Phase::Dice => (3, false, 0, none, none, none),
        Phase::Discard { player, owed, from } => {
            let shown = if player == state.current() && player != me { none } else { from.index() as u8 };
            (4, true, owed, relative(player), none, shown)
        }
        Phase::MoveRobber { rolled } => (5, rolled, 0, none, none, none),
        Phase::Steal { rolled } => (6, rolled, 0, none, none, none),
        Phase::Draw => (7, true, 0, none, none, none),
        Phase::RoadBuilding { rolled, roads } => (8, rolled, roads, none, none, none),
        Phase::YearOfPlenty { rolled, picks } => (9, rolled, picks, none, none, none),
        Phase::Monopoly { rolled } => (10, rolled, 0, none, none, none),
        Phase::Over { winner } => (11, false, 0, relative(winner), none, none),
    };
    put(&[kind, u8::from(rolled), count, player, vertex, from]);
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

fn holder_code(holder: Option<Player>, me: Player) -> u8 {
    match holder {
        None => 0,
        Some(player) if player == me => 1,
        Some(_) => 2,
    }
}
