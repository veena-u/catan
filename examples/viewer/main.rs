use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use eframe::egui::{self, Align2, Color32, FontId, Pos2, RichText, Sense, Shape, Stroke, Vec2};

use catan::board::{Board, Harbor, Terrain};
use catan::dev_card::DevCard;
use catan::ids::{Edge, Hex, Vertex};
use catan::observation::{observe, OBSERVATION_BYTES, PHASES};
use catan::player::{Player, MAX_PLAYERS};
use catan::resource::Resource;
use catan::rng::Rng;
use catan::state::{Action, Actor, Deal, Outcome, Phase, State, ACTIONS};
use catan::topology::topology;

type View = [u8; OBSERVATION_BYTES];

const LEAK_SAMPLES: usize = 16;
const PAUSE: Duration = Duration::from_millis(400);
const PLAYERS: [Color32; MAX_PLAYERS] = [
    Color32::from_rgb(37, 99, 235),
    Color32::from_rgb(234, 88, 12),
    Color32::from_rgb(22, 163, 74),
    Color32::from_rgb(219, 39, 119),
];
const SEATS: [&str; MAX_PLAYERS] = ["me", "+1", "+2", "+3"];
const ACCENT: Color32 = Color32::from_rgb(139, 92, 246);
const GOOD: Color32 = Color32::from_rgb(34, 160, 80);
const BAD: Color32 = Color32::from_rgb(220, 38, 38);
const SEA: Color32 = Color32::from_rgb(28, 45, 54);
const CHANGED: Color32 = Color32::from_rgba_premultiplied(120, 95, 10, 90);
const TERRAIN: [Terrain; 6] =
    [Terrain::Hills, Terrain::Forest, Terrain::Pasture, Terrain::Fields, Terrain::Mountains, Terrain::Desert];
const BLOCKS: [(&str, u16, u16); 10] = [
    ("settlements", 0, 54),
    ("cities", 54, 108),
    ("roads", 108, 180),
    ("robber", 180, 199),
    ("resources", 199, 204),
    ("trades", 204, 264),
    ("other", 264, 271),
    ("rob a player", 271, 274),
    ("trade offers", 274, 504),
    ("answers", 504, ACTIONS as u16),
];

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let number = |name: &str, default: usize| flag(name).and_then(|v| v.parse().ok()).unwrap_or(default);
    let players = number("--players", 4).clamp(2, MAX_PLAYERS);
    let mut viewer = Viewer::new(number("--seed", 1) as u64, players);
    if let Some(checkpoint) = flag("--play").or_else(|| flag("--think")) {
        let python = flag("--python").unwrap_or_else(|| "python".into());
        let (greedy, search) = (args.iter().any(|a| a == "--greedy"), args.iter().any(|a| a == "--search"));
        match Brain::spawn(&python, &checkpoint, greedy, search) {
            Ok(brain) => viewer.brain = Some(brain),
            Err(e) => {
                eprintln!("could not start the network: {e}");
                std::process::exit(1);
            }
        }
        if flag("--play").is_some() {
            let human = Player::new(number("--seat", 0).min(players - 1));
            viewer.play = Some(Play { human, peek: false, last: Instant::now() });
        }
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native("Catan rules viewer", options, Box::new(|_| Ok(Box::new(viewer))))
}

/// The network, answering in a Python process: a move to play, or what it thinks of a position.
struct Brain {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    search: bool,
}

struct Play {
    human: Player,
    peek: bool,
    last: Instant,
}

/// What a player tends to do over their next turns in imagined futures, in words and as ghost pieces on the board.
struct Forecast {
    player: Player,
    lines: Vec<String>,
    settlements: Vec<(Vertex, f32)>,
    cities: Vec<(Vertex, f32)>,
    roads: Vec<(Edge, f32)>,
}

/// What the network thinks at one moment: each player's chance of winning, and its favourite moves for whoever acts.
#[derive(Clone)]
struct Thoughts {
    chances: [f32; MAX_PLAYERS],
    top: Vec<(u16, f32)>,
    cards: [f32; 5],
}

impl Brain {
    fn spawn(python: &str, checkpoint: &str, greedy: bool, search: bool) -> Result<Brain, String> {
        let mut child = Command::new(python)
            .args(["-u", "-m", "train.serve", checkpoint])
            .args(greedy.then_some("--greedy"))
            .args(search.then_some("--search"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{python}: {e}"))?;
        let input = child.stdin.take().expect("piped");
        let mut output = BufReader::new(child.stdout.take().expect("piped"));
        let mut line = String::new();
        output.read_line(&mut line).map_err(|e| e.to_string())?;
        if line.trim() != "ready" {
            return Err("train.serve did not start; is the checkpoint path right?".into());
        }
        Ok(Brain { child, input, output, search })
    }

    fn send(&mut self, prefix: &str, seen: &View, legal: &[u16]) -> Option<String> {
        let join = |values: Vec<String>| values.join(",");
        let line = format!(
            "{prefix}{};{}\n",
            join(seen.iter().map(u8::to_string).collect()),
            join(legal.iter().map(u16::to_string).collect())
        );
        self.input.write_all(line.as_bytes()).ok()?;
        let mut answer = String::new();
        self.output.read_line(&mut answer).ok()?;
        Some(answer.trim().to_string())
    }

    fn ask(&mut self, seen: &View, legal: &[u16]) -> Option<u16> {
        self.send("", seen, legal)?.parse().ok()
    }

    fn ask_with_search(&mut self, record: &str) -> Option<u16> {
        self.input.write_all(format!("*{record}\n").as_bytes()).ok()?;
        let mut answer = String::new();
        self.output.read_line(&mut answer).ok()?;
        answer.trim().parse().ok()
    }

    fn think(&mut self, state: &State) -> Option<Thoughts> {
        let Some(Actor::Player(actor)) = state.actor() else { return None };
        let legal: Vec<u16> = state.legal_actions().iter().map(Action::index).collect();
        let answer = self.send("?", &view(state, actor), &legal)?;
        let mut parts = answer.split('|');
        let (values, moves, cards) = (parts.next()?, parts.next()?, parts.next()?);
        let values: Vec<f32> = values.split(',').filter_map(|v| v.parse().ok()).collect();
        let n = state.players();
        let mut chances = [0.0; MAX_PLAYERS];
        for (k, &v) in values.iter().take(n).enumerate() {
            chances[actor.after(k, n).index()] = ((3.0 * v + 1.0) / 4.0).max(0.0);
        }
        let total: f32 = chances.iter().sum();
        chances.iter_mut().for_each(|c| *c /= total.max(1e-6));
        let top = moves
            .split(',')
            .filter_map(|m| m.split_once(':'))
            .filter_map(|(c, p)| Some((c.parse().ok()?, p.parse().ok()?)))
            .collect();
        let mut worth = [0.0; 5];
        for (slot, v) in worth.iter_mut().zip(cards.split(',').filter_map(|v| v.parse().ok())) {
            *slot = v;
        }
        Some(Thoughts { chances, top, cards: worth })
    }

    fn forecast(&mut self, record: &str, turns: u8, player: Player) -> Option<Forecast> {
        self.input.write_all(format!("!{turns}|{record}\n").as_bytes()).ok()?;
        let mut answer = String::new();
        self.output.read_line(&mut answer).ok()?;
        let (words, spots) = answer.trim_end().split_once("##")?;
        let mut forecast = Forecast {
            player,
            lines: words.split('\t').map(str::to_string).collect(),
            settlements: Vec::new(),
            cities: Vec::new(),
            roads: Vec::new(),
        };
        for part in spots.split(';') {
            let Some((kind, list)) = part.split_once(':') else { continue };
            let pairs = list.split(',').filter_map(|p| p.split_once('=')).filter_map(|(i, s)| Some((i.parse::<usize>().ok()?, s.parse::<f32>().ok()?)));
            for (i, share) in pairs {
                match kind {
                    "s" => forecast.settlements.push((Vertex::new(i), share)),
                    "c" => forecast.cities.push((Vertex::new(i), share)),
                    "r" => forecast.roads.push((Edge::new(i), share)),
                    _ => {}
                }
            }
        }
        Some(forecast)
    }
}

impl Drop for Brain {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Views,
    Mask,
    Hidden,
}

struct Viewer {
    seed: u64,
    players: usize,
    steps: Vec<u16>,
    states: Vec<State>,
    at: usize,
    rng: Rng,
    seed_text: String,
    game_text: String,
    show_ids: bool,
    tab: Tab,
    before: Option<Vec<View>>,
    violation: Option<(usize, String)>,
    leak_free: Vec<Option<bool>>,
    message: Option<String>,
    play: Option<Play>,
    brain: Option<Brain>,
    thoughts: HashMap<usize, Option<Thoughts>>,
    forecasts: HashMap<(usize, u8), Forecast>,
    plan_turns: u8,
    show_plan: bool,
    timeline: bool,
    draft: [[u8; 5]; 2],
    discard: [u8; 5],
}

impl Viewer {
    fn new(seed: u64, players: usize) -> Viewer {
        let mut viewer = Viewer {
            seed,
            players,
            steps: Vec::new(),
            states: vec![State::new(Board::random(&mut Rng::new(seed)), players)],
            at: 0,
            rng: Rng::new(seed),
            seed_text: seed.to_string(),
            game_text: String::new(),
            show_ids: false,
            tab: Tab::Views,
            before: None,
            violation: None,
            leak_free: Vec::new(),
            message: None,
            play: None,
            brain: None,
            thoughts: HashMap::new(),
            forecasts: HashMap::new(),
            plan_turns: 3,
            show_plan: true,
            timeline: false,
            draft: [[0; 5]; 2],
            discard: [0; 5],
        };
        viewer.refresh();
        viewer
    }

    fn state(&self) -> State {
        self.states[self.at]
    }

    fn views(&self) -> Vec<View> {
        let state = self.state();
        state.seats().map(|p| view(&state, p)).collect()
    }

    fn load(&mut self, seed: u64, players: usize, codes: &[u16]) {
        let (show_ids, tab, mut play, brain) = (self.show_ids, self.tab, self.play.take(), self.brain.take());
        if let Some(play) = play.as_mut() {
            play.human = Player::new(play.human.index().min(players - 1));
        }
        *self = Viewer::new(seed, players);
        (self.show_ids, self.tab, self.play, self.brain) = (show_ids, tab, play, brain);
        for (i, &code) in codes.iter().enumerate() {
            if !self.push(code) {
                self.message = Some(format!("step {i} (code {code}) is not legal there; stopped before it"));
                break;
            }
        }
        self.at = self.steps.len();
        self.refresh();
    }

    fn push(&mut self, code: u16) -> bool {
        let mut next = *self.states.last().expect("a state");
        let legal = next.step(code).is_ok();
        if legal {
            self.steps.push(code);
            self.states.push(next);
        }
        legal
    }

    fn seek(&mut self, at: usize) {
        let at = at.min(self.steps.len());
        if at != self.at {
            self.before = Some(self.views());
            self.at = at;
            self.refresh();
        }
    }

    fn branch(&mut self) {
        self.thoughts.retain(|&at, _| at <= self.at);
        self.forecasts.retain(|&(at, _), _| at <= self.at);
        self.before = Some(self.views());
        self.steps.truncate(self.at);
        self.states.truncate(self.at + 1);
        self.message = None;
    }

    fn take(&mut self, code: u16) {
        self.branch();
        self.push(code);
        self.at = self.steps.len();
        self.refresh();
    }

    fn random(&mut self, count: usize) {
        self.branch();
        for _ in 0..count {
            let state = *self.states.last().expect("a state");
            let code = match state.actor() {
                None => break,
                Some(Actor::Chance) => state.sample_outcome(&mut self.rng).index(),
                Some(Actor::Player(_)) => {
                    let legal: Vec<Action> = state.legal_actions().iter().collect();
                    legal[self.rng.below(legal.len())].index()
                }
            };
            self.push(code);
        }
        self.at = self.steps.len();
        self.refresh();
    }

    fn hiding(&self) -> Option<Player> {
        self.play.as_ref().filter(|p| !p.peek).map(|p| p.human)
    }

    fn may_move(&self) -> bool {
        self.play.as_ref().is_none_or(|p| self.state().actor() == Some(Actor::Player(p.human)))
    }

    fn advance(&mut self, ctx: &egui::Context) {
        if self.play.is_none() || self.at != self.steps.len() {
            return;
        }
        let state = self.state();
        let human = self.play.as_ref().expect("playing").human;
        let code = match state.actor() {
            Some(Actor::Chance) => state.sample_outcome(&mut self.rng).index(),
            Some(Actor::Player(p)) if p != human => {
                let waited = self.play.as_ref().expect("playing").last.elapsed();
                if waited < PAUSE {
                    ctx.request_repaint_after(PAUSE - waited);
                    return;
                }
                let legal: Vec<u16> = state.legal_actions().iter().map(Action::index).collect();
                let codes: Vec<String> = self.steps.iter().map(u16::to_string).collect();
                let record = format!("{};{};{}", self.players, self.seed, codes.join(","));
                let answer = self.brain.as_mut().and_then(|brain| match brain.search {
                    true => brain.ask_with_search(&record),
                    false => brain.ask(&view(&state, p), &legal),
                });
                match answer {
                    Some(code) if legal.contains(&code) => code,
                    _ => {
                        let code = legal[self.rng.below(legal.len())];
                        self.take(code);
                        self.message = Some("the network did not answer with a legal move; played a random one".into());
                        return;
                    }
                }
            }
            _ => return,
        };
        self.take(code);
        if let Some(play) = self.play.as_mut() {
            play.last = Instant::now();
        }
        ctx.request_repaint();
    }

    fn refresh(&mut self) {
        self.violation = self.states.iter().enumerate().find_map(|(i, s)| s.validate().err().map(|v| (i, v.0)));
        let state = self.state();
        let rng = &mut self.rng;
        self.leak_free = state.seats().map(|p| leak_free(&state, p, rng)).collect();
        let codes: Vec<String> = self.steps.iter().map(u16::to_string).collect();
        self.game_text = format!("{};{};{}", self.players, self.seed, codes.join(","));
    }

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            ui.strong("Catan rules viewer");
            ui.separator();
            ui.label("seed");
            let seed = ui.add(egui::TextEdit::singleline(&mut self.seed_text).desired_width(60.0));
            let entered = seed.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("New board").clicked() || entered {
                if let Ok(seed) = self.seed_text.trim().parse() {
                    self.load(seed, self.players, &[]);
                }
            }
            ui.label("players");
            for n in 2..=MAX_PLAYERS {
                if ui.selectable_label(self.players == n, n.to_string()).clicked() && self.players != n {
                    self.load(self.seed, n, &[]);
                }
            }
            ui.separator();
            if ui.add_enabled(self.at > 0, egui::Button::new("←")).clicked() {
                self.seek(self.at - 1);
            }
            if ui.add_enabled(self.at < self.steps.len(), egui::Button::new("→")).clicked() {
                self.seek(self.at + 1);
            }
            ui.monospace(format!("step {} / {}", self.at, self.steps.len()));
            ui.separator();
            if let Some(play) = self.play.as_mut() {
                ui.colored_label(PLAYERS[play.human.index()], format!("you are {:?}", play.human));
                if self.brain.as_ref().is_some_and(|b| b.search) {
                    ui.weak("opponents think ahead");
                }
                ui.checkbox(&mut play.peek, "peek");
            } else {
                ui.label("random");
                for (label, count) in [("1", 1), ("10", 10), ("100", 100), ("to end", 100_000)] {
                    if ui.button(label).clicked() {
                        self.random(count);
                    }
                }
            }
            ui.separator();
            ui.checkbox(&mut self.show_ids, "ids");
            ui.separator();
            ui.label("game");
            ui.add(egui::TextEdit::singleline(&mut self.game_text).desired_width(140.0));
            if ui.button("Copy").clicked() {
                ui.ctx().copy_text(self.game_text.clone());
            }
            if ui.button("Load").clicked() {
                let parts: Vec<&str> = self.game_text.split(';').collect();
                let (players, seed, codes) = match parts[..] {
                    [players, seed, codes] => (players.trim().parse().ok(), seed, codes),
                    [seed, codes] => (Some(2), seed, codes),
                    _ => (None, "", ""),
                };
                let codes: Vec<u16> = codes.split(',').filter_map(|c| c.trim().parse().ok()).collect();
                match (players.filter(|n| (2..=MAX_PLAYERS).contains(n)), seed.trim().parse()) {
                    (Some(players), Ok(seed)) => self.load(seed, players, &codes),
                    _ => self.message = Some("a game is written players;seed;code,code,...".into()),
                }
            }
            ui.separator();
            match &self.violation {
                None => ui.colored_label(GOOD, format!("all {} states valid", self.states.len())),
                Some((i, m)) => ui.colored_label(BAD, format!("rule violation at step {i}: {m}")),
            };
        });
    }

    fn side(&mut self, ui: &mut egui::Ui) {
        let state = self.state();
        heading(ui, "Now");
        egui::Grid::new("now").num_columns(2).show(ui, |ui| {
            ui.label("phase");
            ui.label(phase_label(&state));
            ui.end_row();
            ui.label("to act");
            ui.colored_label(actor_color(state.actor()), actor_name(state.actor()));
            ui.end_row();
            ui.label("whose turn");
            ui.colored_label(PLAYERS[state.current().index()], format!("{:?}", state.current()));
            ui.end_row();
            ui.label("this state");
            match state.validate() {
                Ok(()) => ui.colored_label(GOOD, "valid"),
                Err(v) => ui.colored_label(BAD, v.0),
            };
            ui.end_row();
        });
        if let Some(message) = &self.message {
            ui.colored_label(BAD, message);
        }
        let thinking = self.play.as_ref().filter(|_| !self.may_move() && state.actor().is_some()).map(|_| state.actor());
        match thinking {
            Some(actor) => {
                heading(ui, "Moves");
                ui.colored_label(actor_color(actor), format!("{} is thinking…", actor_name(actor)));
            }
            None if matches!(state.phase(), Phase::Discard { .. }) => self.discard_picker(ui, &state),
            None => {
                if let Some(code) = moves(ui, &state) {
                    self.take(code);
                }
                if state.legal_actions().iter().any(|a| matches!(a, Action::Deal(_))) {
                    self.offer_builder(ui, &state);
                }
            }
        }
        self.thoughts_panel(ui);
        players(ui, &state, self.hiding());
        if let Some(at) = self.history(ui) {
            self.seek(at);
        }
    }

    /// Whether the network's view at step `at` may be shown: always, unless playing without peeking and it is not your move.
    fn may_think(&self, at: usize) -> bool {
        match self.hiding() {
            None => true,
            Some(human) => self.states[at].actor() == Some(Actor::Player(human)),
        }
    }

    fn thoughts_at(&mut self, at: usize) -> Option<Thoughts> {
        if !self.thoughts.contains_key(&at) {
            let state = self.states[at];
            let thought = self.brain.as_mut().and_then(|brain| brain.think(&state));
            self.thoughts.insert(at, thought);
        }
        self.thoughts[&at].clone()
    }

    fn thoughts_panel(&mut self, ui: &mut egui::Ui) {
        if self.brain.is_none() {
            return;
        }
        heading(ui, "What the network thinks");
        let state = self.state();
        match self.may_think(self.at).then(|| self.thoughts_at(self.at)).flatten() {
            None => {
                ui.weak("Shown on your moves while playing (tick peek for every move), and not for dice or other chance events.");
            }
            Some(thought) => {
                for p in state.seats() {
                    ui.horizontal(|ui| {
                        ui.colored_label(PLAYERS[p.index()], format!("{p:?}"));
                        ui.add(egui::ProgressBar::new(thought.chances[p.index()]).text(format!("{:.0}% to win", 100.0 * thought.chances[p.index()])));
                    });
                }
                ui.label(format!("{} would choose among:", actor_name(state.actor())));
                for &(code, p) in &thought.top {
                    ui.horizontal(|ui| {
                        ui.add_sized([60.0, 16.0], egui::ProgressBar::new(p));
                        ui.label(format!("{:.0}%  {}", 100.0 * p, step_label(&state, code)));
                    });
                }
                ui.label(format!("What one more card would be worth to {}:", actor_name(state.actor())));
                ui.horizontal_wrapped(|ui| {
                    for (r, worth) in Resource::ALL.into_iter().zip(thought.cards) {
                        ui.colored_label(resource_color(r), format!("{} {:+.1}%", resource_name(r), worth));
                    }
                });
                heading(ui, "Its plan");
                ui.horizontal(|ui| {
                    ui.add(egui::Slider::new(&mut self.plan_turns, 1..=5).text("turns ahead"));
                    ui.checkbox(&mut self.show_plan, "draw on board");
                });
                let key = (self.at, self.plan_turns);
                match self.forecasts.get(&key) {
                    Some(forecast) => forecast.lines.iter().for_each(|line| {
                        ui.label(line);
                    }),
                    None => {
                        let label = format!("Imagine 64 futures: what {} does in the next {} turns", actor_name(state.actor()), self.plan_turns);
                        if let (true, Some(Actor::Player(player))) = (ui.button(label).clicked(), state.actor()) {
                            let codes: Vec<String> = self.steps[..self.at].iter().map(u16::to_string).collect();
                            let record = format!("{};{};{}", self.players, self.seed, codes.join(","));
                            let turns = self.plan_turns;
                            if let Some(forecast) = self.brain.as_mut().and_then(|brain| brain.forecast(&record, turns, player)) {
                                self.forecasts.insert(key, forecast);
                            }
                        }
                    }
                }
                ui.weak("Ghost pieces on the board: the fainter, the fewer imagined futures in which it builds there.");
            }
        }
        ui.checkbox(&mut self.timeline, "chart everyone's chances over the game");
        if self.timeline {
            self.chart(ui);
        }
    }

    fn chart(&mut self, ui: &mut egui::Ui) {
        let shown: Vec<usize> = (0..=self.steps.len()).filter(|&at| self.may_think(at)).collect();
        let points: Vec<(usize, [f32; MAX_PLAYERS])> =
            shown.into_iter().filter_map(|at| self.thoughts_at(at).map(|t| (at, t.chances))).collect();
        let (response, painter) = ui.allocate_painter(Vec2::new(ui.available_width(), 140.0), Sense::click());
        let rect = response.rect;
        painter.rect_filled(rect, 4.0, ui.visuals().extreme_bg_color);
        let last = self.steps.len().max(1) as f32;
        let at = |step: usize, chance: f32| Pos2::new(rect.left() + rect.width() * step as f32 / last, rect.bottom() - rect.height() * chance);
        painter.line_segment([at(0, 0.25), at(self.steps.len(), 0.25)], Stroke::new(1.0, Color32::GRAY));
        for p in self.state().seats() {
            let line: Vec<Pos2> = points.iter().map(|&(step, c)| at(step, c[p.index()])).collect();
            painter.add(Shape::line(line, Stroke::new(1.5, PLAYERS[p.index()])));
        }
        let x = at(self.at, 0.0).x;
        painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0, ACCENT));
        if let Some(pointer) = response.interact_pointer_pos().filter(|_| response.clicked()) {
            let step = ((pointer.x - rect.left()) / rect.width() * last).round() as usize;
            self.seek(step);
        }
        ui.weak("Each player's chance to win as the network sees it at each move; the grey line is an even 25%. Click to jump there.");
    }

    fn discard_picker(&mut self, ui: &mut egui::Ui, state: &State) {
        let Phase::Discard { player, owed, .. } = state.phase() else { return };
        let hand = state.hand(player);
        heading(ui, &format!("{player:?} discards {owed} more"));
        egui::Grid::new("discard").num_columns(3).show(ui, |ui| {
            for resource in Resource::ALL {
                let r = resource.index();
                self.discard[r] = self.discard[r].min(hand[r]);
                ui.colored_label(resource_color(resource), resource_name(resource));
                ui.label(format!("hold {}", hand[r]));
                ui.horizontal(|ui| {
                    if ui.small_button("−").clicked() {
                        self.discard[r] = self.discard[r].saturating_sub(1);
                    }
                    ui.monospace(self.discard[r].to_string());
                    if ui.add_enabled(self.discard[r] < hand[r], egui::Button::new("+").small()).clicked() {
                        self.discard[r] += 1;
                    }
                });
                ui.end_row();
            }
        });
        let chosen: u8 = self.discard.iter().sum();
        if ui.add_enabled(chosen == owed, egui::Button::new(format!("Discard {chosen} of {owed}"))).clicked() {
            let codes: Vec<u16> = Resource::ALL
                .iter()
                .flat_map(|&r| std::iter::repeat_n(Action::Resource(r).index(), usize::from(self.discard[r.index()])))
                .collect();
            let mut trial = *state;
            if codes.iter().all(|&c| trial.step(c).is_ok()) {
                self.branch();
                codes.iter().for_each(|&c| {
                    self.push(c);
                });
                self.at = self.steps.len();
                self.discard = [0; 5];
                self.refresh();
            } else {
                self.message = Some("those cards cannot be discarded".into());
            }
        }
        ui.weak("Pick any cards; they are discarded in resource order, which is how the game records them.");
    }

    fn offer_builder(&mut self, ui: &mut egui::Ui, state: &State) {
        let countering = matches!(state.phase(), Phase::Respond { .. });
        heading(ui, if countering { "Or counter with your own terms" } else { "Offer a trade" });
        if let Phase::Respond { deal, .. } = state.phase() {
            let offered = Deal::from_id(deal);
            ui.label(format!(
                "{:?} offered: you give {}, you get {}. Your counter below is from your side.",
                state.current(),
                bundle(offered.get),
                bundle(offered.give)
            ));
        }
        egui::Grid::new("offer").num_columns(3).show(ui, |ui| {
            ui.label("");
            ui.strong("you give");
            ui.strong("you get");
            ui.end_row();
            for resource in Resource::ALL {
                let r = resource.index();
                ui.colored_label(resource_color(resource), resource_name(resource));
                for side in 0..2 {
                    ui.horizontal(|ui| {
                        if ui.small_button("−").clicked() {
                            self.draft[side][r] = self.draft[side][r].saturating_sub(1);
                        }
                        ui.monospace(self.draft[side][r].to_string());
                        if ui.small_button("+").clicked() {
                            self.draft[side][r] += 1;
                        }
                    });
                }
                ui.end_row();
            }
        });
        let [give, get] = self.draft;
        let legal = Deal { give, get }.id().map(Action::Deal).filter(|&a| state.legal_actions().contains(a));
        if ui.button(if countering { "Counter" } else { "Propose" }).clicked() {
            match legal {
                Some(action) => {
                    self.draft = [[0; 5]; 2];
                    self.take(action.index());
                }
                None => self.message = Some("that trade is not allowed; see the rules below".into()),
            }
        }
        ui.weak(
            "One or two cards on each side, never the same resource on both. You must hold what you give. \
             You can only ask for cards someone might hold, judging from everything public: \
             what they collected, built, traded and showed. \
             A counter must give some of what was asked for, and cannot simply reverse the offer: that is accepting it.",
        );
    }

    fn history(&self, ui: &mut egui::Ui) -> Option<usize> {
        let hiding = self.hiding();
        heading(ui, "History");
        let mut chosen = None;
        let row = ui.text_style_height(&egui::TextStyle::Monospace) + 4.0;
        egui::ScrollArea::vertical().id_salt("history").max_height(320.0).stick_to_bottom(true).show_rows(
            ui,
            row,
            self.steps.len() + 1,
            |ui, rows| {
                for i in rows {
                    let (text, color) = match i {
                        0 => ("   0  start".to_string(), ui.visuals().text_color()),
                        _ => {
                            let (state, code) = (&self.states[i - 1], self.steps[i - 1]);
                            let actor = state.actor();
                            let label = match hiding {
                                Some(human) if !public(state, human) => "chance (hidden)".to_string(),
                                _ => step_label(state, code),
                            };
                            (format!("{i:>4}  {:<7} {label}  #{code}", actor_name(actor)), actor_color(actor))
                        }
                    };
                    let text = RichText::new(text).monospace().color(if i > self.at { color.gamma_multiply(0.4) } else { color });
                    if ui.selectable_label(i == self.at, text).clicked() {
                        chosen = Some(i);
                    }
                }
            },
        );
        chosen
    }

    fn board(&mut self, ui: &mut egui::Ui) {
        let state = self.state();
        let t = topology();
        let side = ui.available_width().min(780.0);
        let (response, painter) = ui.allocate_painter(Vec2::splat(side), Sense::click());
        let center = response.rect.center();
        let s = side / 11.0;
        let at = |(x, y): (i32, i32)| center + Vec2::new(x as f32 * 0.866 * s, y as f32 * 0.5 * s);
        let vertex = |v: Vertex| at(t.vertex_point(v));
        let hex_points = |h: Hex| t.corners(h).map(vertex).to_vec();
        let background = ui.visuals().panel_fill;
        let ink = Color32::from_rgb(34, 30, 25);
        let small = FontId::proportional(0.2 * s);

        painter.rect_filled(response.rect, 12.0, SEA);
        let board = state.board();
        for h in Hex::all() {
            let c = at(t.hex_point(h));
            painter.add(Shape::convex_polygon(hex_points(h), terrain_color(board.terrain(h)), Stroke::new(3.0, background)));
            if let Some(token) = board.token(h) {
                let red = token == 6 || token == 8;
                let color = if red { Color32::from_rgb(185, 28, 28) } else { ink };
                painter.circle_filled(c, 0.3 * s, Color32::from_rgb(248, 241, 220));
                painter.text(c - Vec2::new(0.0, 0.04 * s), Align2::CENTER_CENTER, token.to_string(), FontId::proportional(0.28 * s), color);
                let pips = 6 - (7 - i32::from(token)).abs();
                for k in 0..pips {
                    let x = (k as f32 - (pips - 1) as f32 / 2.0) * 0.07 * s;
                    painter.circle_filled(c + Vec2::new(x, 0.18 * s), 0.025 * s, color);
                }
            }
            if state.robber() == h {
                painter.circle_filled(c + Vec2::new(0.42 * s, -0.25 * s), 0.14 * s, ink);
                painter.rect_filled(egui::Rect::from_center_size(c + Vec2::new(0.42 * s, -0.08 * s), Vec2::splat(0.22 * s)), 4.0, ink);
            }
            if self.show_ids {
                painter.text(c - Vec2::new(0.0, 0.45 * s), Align2::CENTER_CENTER, format!("h{}", h.index()), small.clone(), ink);
            }
        }
        for (slot, harbor) in board.harbor_slots().into_iter().enumerate() {
            let Some(harbor) = harbor else { continue };
            let [a, b] = t.ends(t.coast()[slot]).map(vertex);
            let middle = a + (b - a) / 2.0;
            let tag = middle + (middle - center).normalized() * 0.6 * s;
            let fill = match harbor {
                Harbor::Generic => Color32::from_rgb(235, 235, 230),
                Harbor::Special(r) => resource_color(r),
            };
            painter.line_segment([a, tag], Stroke::new(1.5, Color32::GRAY));
            painter.line_segment([b, tag], Stroke::new(1.5, Color32::GRAY));
            painter.circle(tag, 0.3 * s, fill, Stroke::new(1.0, Color32::GRAY));
            let label = if harbor == Harbor::Generic { "3:1" } else { "2:1" };
            painter.text(tag, Align2::CENTER_CENTER, label, FontId::proportional(0.2 * s), ink);
        }
        for p in state.seats() {
            for e in state.roads(p).iter() {
                let [a, b] = t.ends(e).map(vertex);
                painter.line_segment([a, b], Stroke::new(0.2 * s, background));
                painter.line_segment([a, b], Stroke::new(0.13 * s, PLAYERS[p.index()]));
            }
        }
        for p in state.seats() {
            let color = PLAYERS[p.index()];
            for v in state.settlements(p).iter() {
                painter.circle(vertex(v), 0.16 * s, color, Stroke::new(2.5, background));
            }
            for v in state.cities(p).iter() {
                let rect = egui::Rect::from_center_size(vertex(v), Vec2::splat(0.38 * s));
                painter.rect(rect, 3.0, color, Stroke::new(2.5, background), egui::StrokeKind::Outside);
            }
        }
        if self.show_ids {
            for v in Vertex::all() {
                painter.text(vertex(v) + Vec2::new(0.12 * s, -0.12 * s), Align2::LEFT_BOTTOM, format!("v{}", v.index()), small.clone(), Color32::WHITE);
            }
            for e in Edge::all() {
                let [a, b] = t.ends(e).map(vertex);
                painter.text(a + (b - a) / 2.0, Align2::CENTER_CENTER, format!("e{}", e.index()), small.clone(), Color32::BLACK);
            }
        }

        if let Some(forecast) = self.forecasts.get(&(self.at, self.plan_turns)).filter(|_| self.show_plan) {
            let ghost = |share: f32| {
                let c = PLAYERS[forecast.player.index()];
                Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), (40.0 + 215.0 * share.clamp(0.0, 1.0)) as u8)
            };
            for &(e, share) in &forecast.roads {
                let [a, b] = t.ends(e).map(vertex);
                painter.line_segment([a, b], Stroke::new(0.12 * s, ghost(share)));
            }
            for &(v, share) in &forecast.settlements {
                painter.circle(vertex(v), 0.17 * s, ghost(share), Stroke::new(2.0, Color32::WHITE.gamma_multiply(share)));
            }
            for &(v, share) in &forecast.cities {
                let rect = egui::Rect::from_center_size(vertex(v), Vec2::splat(0.42 * s));
                painter.rect(rect, 3.0, ghost(share), Stroke::new(2.0, Color32::WHITE.gamma_multiply(share)), egui::StrokeKind::Outside);
            }
            for (spots, kind) in [(&forecast.settlements, "settlement"), (&forecast.cities, "city")] {
                for &(v, share) in spots.iter().filter(|(_, share)| *share >= 0.15) {
                    painter.text(vertex(v) + Vec2::new(0.0, -0.32 * s), Align2::CENTER_BOTTOM, format!("{kind} {:.0}%", 100.0 * share), FontId::proportional(0.18 * s), Color32::WHITE);
                }
            }
        }
        let legal: Vec<Action> = if self.may_move() { state.legal_actions().iter().collect() } else { Vec::new() };
        let highlight = Color32::from_rgba_unmultiplied(139, 92, 246, 70);
        for &action in &legal {
            match action {
                Action::Settlement(v) | Action::City(v) => {
                    painter.circle(vertex(v), 0.2 * s, highlight, Stroke::new(2.0, ACCENT));
                }
                Action::Road(e) => {
                    let [a, b] = t.ends(e).map(vertex);
                    painter.line_segment([a, b], Stroke::new(0.22 * s, highlight));
                }
                Action::Robber(h) => {
                    painter.add(Shape::convex_polygon(hex_points(h), highlight, Stroke::new(3.0, ACCENT)));
                }
                _ => {}
            }
        }

        let pointed = response.hover_pos().and_then(|pointer| {
            let distance = |action: &Action| match *action {
                Action::Settlement(v) | Action::City(v) => Some((vertex(v).distance(pointer), 0.25 * s, 0)),
                Action::Road(e) => {
                    let [a, b] = t.ends(e).map(vertex);
                    Some((segment_distance(pointer, a, b), 0.16 * s, 1))
                }
                Action::Robber(h) => Some((at(t.hex_point(h)).distance(pointer), 0.75 * s, 2)),
                _ => None,
            };
            legal
                .iter()
                .filter_map(|a| distance(a).filter(|&(d, reach, _)| d < reach).map(|(d, _, rank)| (rank, d, *a)))
                .min_by(|x, y| (x.0, x.1).partial_cmp(&(y.0, y.1)).expect("finite distances"))
                .map(|(_, _, action)| action)
        });
        if let Some(action) = pointed {
            if response.clicked() {
                self.take(action.index());
            }
            response.on_hover_text_at_pointer(format!("{}  #{}", action_label(action), action.index()));
        }
    }

    fn tabs(&mut self, ui: &mut egui::Ui) {
        if self.hiding().is_some() {
            ui.weak("The views, action mask and hidden cards are hidden while you play; tick peek in the toolbar to see them.");
            return;
        }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Views, "What each player sees");
            ui.selectable_value(&mut self.tab, Tab::Mask, "Action mask");
            ui.selectable_value(&mut self.tab, Tab::Hidden, "Hidden information");
        });
        ui.add_space(6.0);
        match self.tab {
            Tab::Views => self.views_tab(ui),
            Tab::Mask => {
                if let Some(code) = mask_tab(ui, &self.state()) {
                    self.take(code);
                }
            }
            Tab::Hidden => hidden_tab(ui, &self.state()),
        }
    }

    fn views_tab(&self, ui: &mut egui::Ui) {
        let views = self.views();
        ui.columns(views.len(), |columns| {
            for (k, ui) in columns.iter_mut().enumerate() {
                let player = Player::new(k);
                ui.colored_label(PLAYERS[k], RichText::new(format!("{player:?} sees")).strong());
                match self.leak_free.get(k).copied().flatten() {
                    None => ui.weak("hidden-info check skipped"),
                    Some(true) => ui.colored_label(GOOD, format!("hidden cards re-dealt {LEAK_SAMPLES}×: bytes unchanged")),
                    Some(false) => ui.colored_label(BAD, "bytes change with hidden cards"),
                };
                egui::Grid::new(("view", k)).striped(true).num_columns(3).show(ui, |ui| {
                    for (name, offset, len, value) in decode(&views[k]) {
                        let changed = self.before.as_ref().is_some_and(|b| b.get(k).is_some_and(|b| b[offset..offset + len] != views[k][offset..offset + len]));
                        let value = RichText::new(value);
                        ui.monospace(offset.to_string());
                        ui.label(name);
                        ui.add(egui::Label::new(if changed { value.background_color(CHANGED) } else { value }).wrap());
                        ui.end_row();
                    }
                });
            }
        });
        ui.weak(format!("Values are decoded from the {OBSERVATION_BYTES} bytes, not from the state. Seats are relative: +1 is the next player. Highlighted values changed since the last state shown."));
    }
}

impl eframe::App for Viewer {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.advance(ui.ctx());
        if !ui.ctx().egui_wants_keyboard_input() {
            if ui.input(|i| i.key_pressed(egui::Key::ArrowLeft)) {
                self.seek(self.at.saturating_sub(1));
            }
            if ui.input(|i| i.key_pressed(egui::Key::ArrowRight)) {
                self.seek(self.at + 1);
            }
        }
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui));
        egui::Panel::right("side").resizable(true).default_size(440.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.side(ui));
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.board(ui);
                ui.separator();
                self.tabs(ui);
            });
        });
    }
}

fn moves(ui: &mut egui::Ui, state: &State) -> Option<u16> {
    let mut chosen = None;
    match state.actor() {
        Some(Actor::Chance) => {
            heading(ui, "Chance outcomes");
            let weights = state.chances().weights().to_vec();
            let total: f32 = weights.iter().map(|&w| f32::from(w)).sum();
            for (i, &weight) in weights.iter().enumerate().filter(|&(_, &w)| w > 0) {
                ui.horizontal(|ui| {
                    let outcome = state.outcome(i as u16).expect("an outcome");
                    if ui.add_sized([110.0, 20.0], egui::Button::new(outcome_label(outcome))).clicked() {
                        chosen = Some(i as u16);
                    }
                    let share = f32::from(weight) / total;
                    ui.add(egui::ProgressBar::new(share).text(format!("{:.1}%", share * 100.0)));
                });
            }
        }
        Some(Actor::Player(player)) => {
            let legal: Vec<Action> = state.legal_actions().iter().collect();
            if let Phase::Confirm { accepted, counters, .. } = state.phase() {
                answers(ui, state, accepted, counters);
            }
            heading(ui, &format!("Moves for {player:?} · {} legal", legal.len()));
            let (board, other): (Vec<Action>, Vec<Action>) =
                legal.into_iter().filter(|a| !matches!(a, Action::Deal(_))).partition(|a| on_board(*a));
            let mut buttons = |ui: &mut egui::Ui, actions: &[Action]| {
                ui.horizontal_wrapped(|ui| {
                    for &a in actions {
                        if ui.button(format!("{}  #{}", step_label(state, a.index()), a.index())).clicked() {
                            chosen = Some(a.index());
                        }
                    }
                });
            };
            buttons(ui, &other);
            if !board.is_empty() {
                ui.weak(format!("{} board moves: click a highlighted spot", board.len()));
                ui.collapsing("list them", |ui| buttons(ui, &board));
            }
        }
        None => {
            heading(ui, "Game over");
            if let Some(winner) = state.winner() {
                ui.colored_label(PLAYERS[winner.index()], format!("{winner:?} wins"));
            }
        }
    }
    chosen
}

fn public(state: &State, human: Player) -> bool {
    match state.phase() {
        Phase::Steal { victim, .. } => state.current() == human || victim == human,
        Phase::Draw => state.current() == human,
        _ => true,
    }
}

fn answers(ui: &mut egui::Ui, state: &State, accepted: u8, counters: [Option<u8>; MAX_PLAYERS]) {
    heading(ui, "Answers to the offer");
    let proposer = state.current();
    for other in state.others(proposer) {
        let answer = match counters[other.index()] {
            Some(id) => {
                let deal = Deal::from_id(id);
                let affordable = if state.hand(proposer).iter().zip(deal.get).all(|(&held, n)| held >= n) {
                    ""
                } else {
                    " (you do not hold that)"
                };
                format!("counters: you would give {}, you would get {}{affordable}", bundle(deal.get), bundle(deal.give))
            }
            None if accepted >> other.index() & 1 == 1 => "accepts".into(),
            None => "declines".into(),
        };
        ui.colored_label(PLAYERS[other.index()], format!("{other:?} {answer}"));
    }
}

fn players(ui: &mut egui::Ui, state: &State, hiding: Option<Player>) {
    let secret = |p: Player| hiding.is_some_and(|human| human != p);
    heading(ui, "Players");
    let resources = |ui: &mut egui::Ui, counts: [u8; 5]| {
        ui.horizontal(|ui| {
            for (r, n) in Resource::ALL.into_iter().zip(counts) {
                ui.colored_label(resource_color(r), "■").on_hover_text(resource_name(r));
                ui.label(n.to_string());
            }
        });
    };
    let cards = |counts: [u8; 5]| {
        let short = ["Kn", "VP", "RB", "YP", "Mo"];
        (0..5).map(|i| format!("{} {}", short[i], counts[i])).collect::<Vec<_>>().join("  ")
    };
    let titles = |p: Player| {
        let held = [(state.longest_road(), "longest road"), (state.largest_army(), "largest army")];
        let names: Vec<&str> = held.iter().filter(|(h, _)| *h == Some(p)).map(|(_, n)| *n).collect();
        if names.is_empty() { "—".to_string() } else { names.join(", ") }
    };
    egui::Grid::new("players").striped(true).num_columns(state.players() + 1).show(ui, |ui| {
        let row = |ui: &mut egui::Ui, label: &str, cell: &dyn Fn(&mut egui::Ui, Player)| {
            ui.label(label);
            for p in state.seats() {
                cell(ui, p);
            }
            ui.end_row();
        };
        row(ui, "", &|ui, p| {
            ui.colored_label(PLAYERS[p.index()], RichText::new(format!("{p:?}")).strong());
        });
        row(ui, "points", &|ui, p| {
            ui.label(match secret(p) {
                true => format!("{} public", state.public_victory_points(p)),
                false => format!("{} (public {})", state.victory_points(p), state.public_victory_points(p)),
            });
        });
        row(ui, "hand", &|ui, p| match secret(p) {
            true => {
                ui.label(format!("{} cards", state.hand(p).iter().sum::<u8>()));
            }
            false => resources(ui, state.hand(p)),
        });
        row(ui, "set aside", &|ui, p| match secret(p) {
            true => {
                ui.label(format!("{} cards", state.set_aside(p).iter().sum::<u8>()));
            }
            false => resources(ui, state.set_aside(p)),
        });
        row(ui, "dev cards", &|ui, p| {
            ui.label(match secret(p) {
                true => format!("{} cards", state.dev_card_count(p)),
                false => cards(DevCard::ALL.map(|c| state.dev_cards(p, c))),
            });
        });
        row(ui, "knights played", &|ui, p| {
            ui.label(state.knights(p).to_string());
        });
        row(ui, "road length", &|ui, p| {
            ui.label(state.road_length(p).to_string());
        });
        row(ui, "titles", &|ui, p| {
            ui.label(titles(p));
        });
        row(ui, "pieces left", &|ui, p| {
            let (s, c, r) = (state.settlements(p).len(), state.cities(p).len(), state.roads(p).len());
            ui.label(format!("{} settl · {} cities · {} roads", 5 - s, 4 - c, 15 - r));
        });
    });
    ui.add_space(6.0);
    egui::Grid::new("table").striped(true).num_columns(2).show(ui, |ui| {
        ui.label("bank");
        resources(ui, Resource::ALL.map(|r| state.bank(r)));
        ui.end_row();
        for (label, value) in [
            ("deck", format!("{} cards", state.deck_size())),
            ("played", cards(DevCard::ALL.map(|c| state.played(c)))),
            ("bought this turn", cards(DevCard::ALL.map(|c| state.bought(c)))),
            ("card played this turn", state.card_played().to_string()),
            ("offers this turn", state.offers().to_string()),
        ] {
            ui.label(label);
            ui.label(value);
            ui.end_row();
        }
    });
}

fn mask_tab(ui: &mut egui::Ui, state: &State) -> Option<u16> {
    let legal = state.legal_actions();
    let mut chosen = None;
    for (name, low, high) in BLOCKS {
        ui.weak(format!("{name} · {low}–{}", high - 1));
        ui.horizontal_wrapped(|ui| {
            ui.spacing_mut().item_spacing = Vec2::splat(2.0);
            for i in low..high {
                let action = Action::from_index(i).expect("an action");
                let is_legal = legal.contains(action);
                let (rect, response) = ui.allocate_exact_size(Vec2::splat(13.0), Sense::click());
                ui.painter().rect_filled(rect, 2.0, if is_legal { ACCENT } else { ui.visuals().faint_bg_color });
                if response.on_hover_text(format!("{i}: {}", action_label(action))).clicked() && is_legal {
                    chosen = Some(i);
                }
            }
        });
    }
    ui.weak(format!("{} of {ACTIONS} actions legal. Hover a cell for its action; click a legal one to take it.", legal.len()));
    chosen
}

fn hidden_tab(ui: &mut egui::Ui, state: &State) {
    let names = |counts: [u8; 5], labels: [&str; 5]| {
        (0..5).filter(|&i| counts[i] > 0).map(|i| format!("{} {}", labels[i], counts[i])).collect::<Vec<_>>().join(" · ")
    };
    let cards = DevCard::ALL.map(card_name);
    let resources = Resource::ALL.map(resource_name);
    ui.columns(state.players(), |columns| {
        for (k, ui) in columns.iter_mut().enumerate() {
            let observer = Player::new(k);
            let hidden = state.hidden_from(observer);
            ui.colored_label(PLAYERS[k], RichText::new(format!("Hidden from {observer:?}")).strong());
            egui::Grid::new(("hidden", k)).striped(true).num_columns(2).show(ui, |ui| {
                for other in state.others(observer) {
                    let o = other.index();
                    let known = state.known(observer, other);
                    ui.colored_label(PLAYERS[o], format!("{other:?}"));
                    ui.add(
                        egui::Label::new(format!(
                            "hand {} (sure of {}) · dev {} · aside {}",
                            names(hidden.hands[o], resources),
                            names(known, resources),
                            names(hidden.dev_cards[o], cards),
                            names(hidden.set_aside[o], resources)
                        ))
                        .wrap(),
                    );
                    ui.end_row();
                }
                ui.label("bought this turn");
                ui.label(names(hidden.bought, cards));
                ui.end_row();
                ui.label("deck");
                ui.label(names(hidden.deck, cards));
                ui.end_row();
            });
        }
    });
    ui.weak("What hidden_from returns for each player, and the lower bound each is sure of for every other hand.");
}

fn heading(ui: &mut egui::Ui, text: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(text.to_uppercase()).small().strong().weak());
    ui.separator();
}

fn view(state: &State, observer: Player) -> View {
    let mut bytes = [0; OBSERVATION_BYTES];
    observe(state, observer, &mut bytes);
    bytes
}

fn leak_free(state: &State, observer: Player, rng: &mut Rng) -> Option<bool> {
    if state.validate().is_err() || state.actor().is_none() {
        return None;
    }
    let seen = view(state, observer);
    Some((0..LEAK_SAMPLES).all(|_| view(&state.sample_hidden(observer, rng), observer) == seen))
}

fn segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let along = ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0);
    p.distance(a + ab * along)
}

fn on_board(action: Action) -> bool {
    matches!(action, Action::Settlement(_) | Action::City(_) | Action::Road(_) | Action::Robber(_))
}

fn decode(b: &View) -> Vec<(String, usize, usize, String)> {
    let u64_at = |at: usize| u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"));
    let u128_at = |at: usize| u128::from_le_bytes(b[at..at + 16].try_into().expect("16 bytes"));
    let bits = |word: u128, prefix: &str| {
        let on: Vec<String> = (0..128).filter(|i| word >> i & 1 == 1).map(|i| format!("{prefix}{i}")).collect();
        if on.is_empty() { "none".into() } else { on.join(" ") }
    };
    let named = |at: usize, name: &dyn Fn(usize) -> &'static str| {
        (0..5).map(|i| format!("{} {}", name(i), b[at + i])).collect::<Vec<_>>().join(" · ")
    };
    let resources = |at: usize| named(at, &|i| resource_name(Resource::ALL[i]));
    let cards = |at: usize| named(at, &|i| card_name(DevCard::ALL[i]));
    let seat = |code: u8| SEATS.get(usize::from(code)).map_or("—".to_string(), |s| s.to_string());
    let holder = |code: u8| if code == 0 { "no one".to_string() } else { seat(code - 1) };
    let flag = |code: u8| (code == 1).to_string();
    let terrain = |code: u8| TERRAIN.get(usize::from(code)).map_or("?", |&t| terrain_name(t));
    let harbor = |code: u8| match code {
        1 => "3:1".to_string(),
        c => format!("2:1 {}", resource_name(Resource::ALL[usize::from(c - 2)])),
    };
    let join = |items: Vec<String>| items.join(" ");
    let per_seat = |at: usize| join((0..usize::from(b[221])).map(|s| format!("{} {}", SEATS[s], b[at + s])).collect());
    let players = usize::from(b[221]);

    let mut rows: Vec<(String, usize, usize, String)> = Vec::new();
    for s in 0..players {
        let at = 32 * s;
        rows.push((format!("{} settlements", SEATS[s]), at, 8, bits(u128::from(u64_at(at)), "v")));
        rows.push((format!("{} cities", SEATS[s]), at + 8, 8, bits(u128::from(u64_at(at + 8)), "v")));
        rows.push((format!("{} roads", SEATS[s]), at + 16, 16, bits(u128_at(at + 16), "e")));
    }
    let mut row = |name: &str, at: usize, len: usize, value: String| rows.push((name.to_string(), at, len, value));
    row("terrain by hex", 128, 19, join(b[128..147].iter().map(|&t| terrain(t).to_string()).collect()));
    row("token by hex", 147, 19, join(b[147..166].iter().map(u8::to_string).collect()));
    row("harbors by vertex", 166, 54, join((0..54).filter(|&v| b[166 + v] != 0).map(|v| format!("v{v} {}", harbor(b[166 + v]))).collect()));
    row("robber", 220, 1, format!("h{}", b[220]));
    row("players", 221, 1, b[221].to_string());
    row("my hand", 222, 5, resources(222));
    row("my set-aside discards", 227, 5, resources(227));
    for s in 1..players {
        let at = 232 + 7 * (s - 1);
        row(&format!("{} hand: sure of", SEATS[s]), at, 5, resources(at));
        row(&format!("{} hand: size, set aside", SEATS[s]), at + 5, 2, format!("{}, {}", b[at + 5], b[at + 6]));
    }
    row("bank", 253, 5, resources(253));
    row("my dev cards", 258, 5, cards(258));
    row("my cards bought this turn", 263, 5, cards(263));
    for s in 1..players {
        let at = 268 + 2 * (s - 1);
        row(&format!("{} dev cards, bought this turn", SEATS[s]), at, 2, format!("{}, {}", b[at], b[at + 1]));
    }
    row("deck size", 274, 1, b[274].to_string());
    row("unseen dev cards", 275, 5, cards(275));
    row("knights", 280, 4, per_seat(280));
    row("road length", 284, 4, per_seat(284));
    row("points (mine all, others public)", 288, 4, per_seat(288));
    row("longest road", 292, 1, holder(b[292]));
    row("largest army", 293, 1, holder(b[293]));
    row("whose turn", 294, 1, seat(b[294]));
    row("card played this turn", 295, 1, flag(b[295]));
    row("phase", 296, 1, PHASES.get(usize::from(b[296])).map_or("?".into(), |p| p.to_string()));
    row("rolled", 297, 1, flag(b[297]));
    row("phase count", 298, 1, b[298].to_string());
    row("phase player", 299, 1, seat(b[299]));
    row("phase vertex", 300, 1, match b[300] { u8::MAX => "—".into(), v => format!("v{v}") });
    row("next discard from", 301, 1, match b[301] { u8::MAX => "—".into(), r => resource_name(Resource::ALL[usize::from(r)]).into() });
    row("offer: gives", 302, 5, resources(302));
    row("offer: wants", 307, 5, resources(307));
    row("offer: accepted by", 312, 1, join((0..players).filter(|s| b[312] >> s & 1 == 1).map(|s| SEATS[s].to_string()).collect()));
    row("offers this turn", 313, 1, b[313].to_string());
    for s in 1..MAX_PLAYERS {
        let at = 314 + 10 * (s - 1);
        row(&format!("counter from {}", SEATS[s]), at, 10, format!("gives {}, gets {}", resources(at), resources(at + 5)));
    }
    let who = |bits: u8| join((0..players).filter(|s| bits >> s & 1 == 1).map(|s| SEATS[s].to_string()).collect());
    row("last offer this turn", 344, 10, format!("gives {}, gets {}", resources(344), resources(349)));
    row("last offer: accepted by", 354, 1, who(b[354]));
    row("last offer: countered by", 355, 1, who(b[355]));
    row("last offer: traded", 356, 1, flag(b[356]));
    rows
}

fn action_label(action: Action) -> String {
    match action {
        Action::Settlement(v) => format!("settlement at v{}", v.index()),
        Action::City(v) => format!("city at v{}", v.index()),
        Action::Road(e) => {
            let [a, b] = topology().ends(e);
            format!("road e{} (v{}–v{})", e.index(), a.index(), b.index())
        }
        Action::Robber(h) => format!("robber to h{}", h.index()),
        Action::Resource(r) => format!("name {}", resource_name(r)),
        Action::Trade { give, get, rate } => format!("trade {rate} {} → 1 {}", resource_name(give), resource_name(get)),
        Action::Roll => "roll".into(),
        Action::EndTurn => "end turn".into(),
        Action::BuyCard => "buy dev card".into(),
        Action::PlayKnight => "play knight".into(),
        Action::PlayRoadBuilding => "play road building".into(),
        Action::PlayYearOfPlenty => "play year of plenty".into(),
        Action::PlayMonopoly => "play monopoly".into(),
        Action::Rob(offset) => format!("rob player +{offset}"),
        Action::Deal(id) => {
            let deal = Deal::from_id(id);
            format!("offer: gives {}, wants {}", bundle(deal.give), bundle(deal.get))
        }
        Action::Accept => "accept the offer".into(),
        Action::Decline => "decline the offer".into(),
        Action::TradeWith(offset) => format!("trade with player +{offset}"),
        Action::TakeCounter(offset) => format!("take player +{offset}'s counter-offer"),
        Action::Cancel => "decline them all".into(),
    }
}

fn outcome_label(outcome: Outcome) -> String {
    match outcome {
        Outcome::Dice(sum) => format!("dice {sum}"),
        Outcome::Steal(r) => format!("steal {}", resource_name(r)),
        Outcome::Draw(c) => format!("draw {}", card_name(c)),
    }
}

fn step_label(state: &State, code: u16) -> String {
    match state.actor() {
        Some(Actor::Chance) => state.outcome(code).map_or("?".into(), outcome_label),
        _ => match (state.phase(), Action::from_index(code)) {
            (Phase::Respond { deal, responder, .. }, Some(Action::Accept)) => {
                let deal = Deal::from_id(deal);
                format!("accept: {responder:?} gives {}, gets {}", bundle(deal.get), bundle(deal.give))
            }
            (Phase::Respond { responder, .. }, Some(Action::Deal(id))) => {
                let deal = Deal::from_id(id);
                let proposer = state.current();
                format!("counter: {responder:?} gives {}, {proposer:?} gives {}", bundle(deal.give), bundle(deal.get))
            }
            (Phase::Confirm { counters, .. }, Some(Action::TakeCounter(offset))) => {
                let partner = state.current().after(usize::from(offset), state.players());
                let deal = Deal::from_id(counters[partner.index()].expect("a counter-offer"));
                format!("take {partner:?}'s counter: you give {}, you get {}", bundle(deal.get), bundle(deal.give))
            }
            (Phase::Confirm { deal, .. }, Some(Action::TradeWith(offset))) => {
                let partner = state.current().after(usize::from(offset), state.players());
                let deal = Deal::from_id(deal);
                format!("trade with {partner:?}: you give {}, you get {}", bundle(deal.give), bundle(deal.get))
            }
            (_, action) => action.map_or("?".into(), action_label),
        },
    }
}

fn phase_label(state: &State) -> String {
    let proposer = state.current();
    let rolled = |r: bool| if r { "after the roll" } else { "before the roll" };
    match state.phase() {
        Phase::PlaceSettlement => "setup: place a settlement".into(),
        Phase::PlaceRoad { settlement } => format!("setup: place a road touching v{}", settlement.index()),
        Phase::Turn { rolled: r } => format!("turn, {}", rolled(r)),
        Phase::Dice => "dice".into(),
        Phase::Discard { player, owed, from } => format!("{player:?} discards: {owed} owed, from {}", resource_name(from)),
        Phase::MoveRobber { rolled: r } => format!("move the robber, {}", rolled(r)),
        Phase::Steal { rolled: r, victim } => format!("steal from {victim:?}, {}", rolled(r)),
        Phase::ChooseVictim { rolled: r } => format!("choose whom to rob, {}", rolled(r)),
        Phase::Draw => "draw the bought card".into(),
        Phase::RoadBuilding { roads, .. } => format!("road building: {roads} left"),
        Phase::YearOfPlenty { picks, .. } => format!("year of plenty: {picks} left"),
        Phase::Monopoly { .. } => "monopoly: name a resource".into(),
        Phase::Over { winner } => format!("over: {winner:?} wins"),
        Phase::Respond { deal, responder, .. } => {
            let Deal { give, get } = Deal::from_id(deal);
            format!(
                "{proposer:?} offers to give {} for {}: {responder:?} accepts (gives {}, gets {}), declines, or counters",
                bundle(give),
                bundle(get),
                bundle(get),
                bundle(give)
            )
        }
        Phase::Confirm { deal, .. } => {
            let Deal { give, get } = Deal::from_id(deal);
            format!("{proposer:?} trades {} for {} with someone who accepted, takes a counter-offer, or cancels", bundle(give), bundle(get))
        }
    }
}

fn bundle(counts: [u8; 5]) -> String {
    let parts: Vec<String> =
        (0..5).filter(|&r| counts[r] > 0).map(|r| format!("{} {}", counts[r], resource_name(Resource::ALL[r]))).collect();
    if parts.is_empty() { "nothing".into() } else { parts.join(" + ") }
}

fn actor_name(actor: Option<Actor>) -> String {
    match actor {
        Some(Actor::Player(p)) => format!("{p:?}"),
        Some(Actor::Chance) => "Chance".into(),
        None => "nobody".into(),
    }
}

fn actor_color(actor: Option<Actor>) -> Color32 {
    match actor {
        Some(Actor::Player(p)) => PLAYERS[p.index()],
        _ => Color32::GRAY,
    }
}

fn resource_name(resource: Resource) -> &'static str {
    match resource {
        Resource::Brick => "brick",
        Resource::Lumber => "lumber",
        Resource::Wool => "wool",
        Resource::Grain => "grain",
        Resource::Ore => "ore",
    }
}

fn resource_color(resource: Resource) -> Color32 {
    match resource {
        Resource::Brick => Color32::from_rgb(181, 83, 45),
        Resource::Lumber => Color32::from_rgb(47, 107, 63),
        Resource::Wool => Color32::from_rgb(155, 209, 116),
        Resource::Grain => Color32::from_rgb(226, 185, 59),
        Resource::Ore => Color32::from_rgb(125, 135, 146),
    }
}

fn card_name(card: DevCard) -> &'static str {
    match card {
        DevCard::Knight => "knight",
        DevCard::VictoryPoint => "victory point",
        DevCard::RoadBuilding => "road building",
        DevCard::YearOfPlenty => "year of plenty",
        DevCard::Monopoly => "monopoly",
    }
}

fn terrain_name(terrain: Terrain) -> &'static str {
    match terrain {
        Terrain::Hills => "hills",
        Terrain::Forest => "forest",
        Terrain::Pasture => "pasture",
        Terrain::Fields => "fields",
        Terrain::Mountains => "mountains",
        Terrain::Desert => "desert",
    }
}

fn terrain_color(terrain: Terrain) -> Color32 {
    match terrain.resource() {
        Some(r) => resource_color(r),
        None => Color32::from_rgb(220, 204, 163),
    }
}
