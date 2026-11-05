//! Gumbel lookahead (Danihelka et al. 2022) over imagined deals of the hidden cards.
//!
//! Each root is the decision of one player. Its hidden cards are re-dealt into a few worlds, which share the
//! root's statistics; every simulation descends into one world's tree. Dice, steals and card draws are chance
//! nodes. Values are kept for every seat, and each player picks for itself (max-n), so 2 and 4 players work
//! alike. Offers never happen inside the lookahead: trading is left to the trade search.

use crate::buffer::parallel;
use crate::dev_card::DevCard;
use crate::observation::{observe, OBSERVATION_BYTES};
use crate::player::{Player, MAX_PLAYERS};
use crate::rng::Rng;
use crate::state::{Action, Actor, Phase, State, ACTIONS, MASK_WORDS};

const NONE: u32 = u32::MAX;
const OFFER: u16 = 234;
const VISIT_INIT: f32 = 50.0;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub simulations: u32,
    pub worlds: usize,
    pub considered: usize,
    pub noise: bool,
    /// How strongly a better Q-value outweighs the prior (mctx's 0.1).
    pub value_scale: f32,
    /// Stretch each node's Q-values to [0, 1] (mctx), or read them on the values' own scale of [-1, 1], so that
    /// moves of nearly equal value stay nearly equal.
    pub rescale: bool,
}

#[derive(Clone, Copy)]
pub struct Leaf {
    pub observation: [u8; OBSERVATION_BYTES],
    pub mask: [u64; MASK_WORDS],
    pub hidden: [[u8; 10]; MAX_PLAYERS - 1],
}

pub struct Outcome {
    pub policy: [f32; ACTIONS],
    pub action: Option<u16>,
    pub value: f32,
}

#[derive(Clone, Copy)]
enum Kind {
    Decision(Player),
    Chance,
    Over(Player),
}

struct Node {
    state: State,
    kind: Kind,
    visits: u32,
    total: [f32; MAX_PLAYERS],
    raw: [f32; MAX_PLAYERS],
    first: u32,
    count: u32,
    expanded: bool,
}

#[derive(Clone, Copy)]
struct Edge {
    code: u16,
    weight: f32,
    child: u32,
}

enum Pending {
    Root,
    Node(u32),
}

struct Tree {
    value_scale: f32,
    rescale: bool,
    simulations: u32,
    considered: usize,
    noise: bool,
    observer: Player,
    root: State,
    worlds: Vec<State>,
    searchable: bool,
    expanded: bool,
    actions: Vec<u16>,
    logits: Vec<f32>,
    gumbel: Vec<f32>,
    visits: Vec<u32>,
    total: Vec<[f32; MAX_PLAYERS]>,
    raw: [f32; MAX_PLAYERS],
    prior: Vec<(u16, f32)>,
    offer: f32,
    schedule: Vec<u32>,
    subtrees: Vec<u32>,
    world_visits: Vec<u32>,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    path: Vec<u32>,
    branch: (usize, usize),
    pending: Option<Pending>,
    done: u32,
    rng: Rng,
}

pub struct Search {
    trees: Vec<Tree>,
    config: Config,
    waiting: Vec<usize>,
}

impl Search {
    pub fn new(roots: &[State], config: Config, seed: u64) -> Search {
        let mut seeds = Rng::new(seed);
        let rngs: Vec<Rng> = roots.iter().map(|_| seeds.fork()).collect();
        let mut trees: Vec<Option<Tree>> = roots.iter().map(|_| None).collect();
        let mut work: Vec<(State, Rng)> = roots.iter().copied().zip(rngs).collect();
        parallel(&mut work, &mut trees, |work, trees, _| {
            for ((root, rng), tree) in work.iter_mut().zip(trees) {
                *tree = Some(Tree::new(*root, &config, rng.clone()));
            }
        });
        Search { trees: trees.into_iter().map(|t| t.expect("every root planted")).collect(), config, waiting: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// The next position of every root that needs the network, in root order.
    pub fn leaves(&mut self) -> Vec<(usize, Leaf)> {
        assert!(self.waiting.is_empty(), "the last leaves were not expanded");
        let config = self.config;
        let mut found: Vec<Option<Leaf>> = vec![None; self.trees.len()];
        parallel(&mut self.trees, &mut found, |trees, found, _| {
            for (tree, slot) in trees.iter_mut().zip(found) {
                *slot = tree.next_leaf(&config);
            }
        });
        let leaves: Vec<(usize, Leaf)> = found.into_iter().enumerate().filter_map(|(i, l)| l.map(|l| (i, l))).collect();
        self.waiting = leaves.iter().map(|&(i, _)| i).collect();
        leaves
    }

    /// The network's log-probabilities and values (seat 0 is the player to act) for the last leaves.
    pub fn expand(&mut self, log_probs: &[[f32; ACTIONS]], values: &[[f32; MAX_PLAYERS]]) {
        assert!(log_probs.len() == self.waiting.len() && values.len() == self.waiting.len(), "one evaluation per leaf");
        let waiting = std::mem::take(&mut self.waiting);
        let mut chosen: Vec<&mut Tree> = Vec::with_capacity(waiting.len());
        let mut next = waiting.iter().peekable();
        for (i, tree) in self.trees.iter_mut().enumerate() {
            if next.peek() == Some(&&i) {
                chosen.push(tree);
                next.next();
            }
        }
        let mut inputs: Vec<(&[f32; ACTIONS], &[f32; MAX_PLAYERS])> = log_probs.iter().zip(values).collect();
        parallel(&mut chosen, &mut inputs, |trees, inputs, _| {
            for (tree, (log_probs, values)) in trees.iter_mut().zip(inputs.iter()) {
                tree.expand(&log_probs[..], &values[..]);
            }
        });
    }

    pub fn outcome(&self, root: usize) -> Outcome {
        self.trees[root].outcome()
    }

    pub fn visits(&self, root: usize) -> u32 {
        self.trees[root].done
    }
}

impl Tree {
    fn new(root: State, config: &Config, mut rng: Rng) -> Tree {
        let Some(Actor::Player(observer)) = root.actor() else { panic!("a search starts at a player's decision") };
        let trading = matches!(root.phase(), Phase::Draft { .. } | Phase::Respond { .. } | Phase::Confirm { .. });
        let actions: Vec<u16> = root.legal_actions().iter().map(Action::index).filter(|&c| c != OFFER).collect();
        let searchable = !trading && actions.len() >= 2 && config.simulations > 0 && config.worlds > 0;
        let worlds = if searchable { (0..config.worlds).map(|_| root.sample_hidden(observer, &mut rng)).collect() } else { Vec::new() };
        Tree {
            value_scale: config.value_scale,
            rescale: config.rescale,
            simulations: config.simulations,
            considered: config.considered,
            noise: config.noise,
            observer,
            root,
            worlds,
            searchable,
            expanded: false,
            actions,
            logits: Vec::new(),
            gumbel: Vec::new(),
            visits: Vec::new(),
            total: Vec::new(),
            raw: [0.0; MAX_PLAYERS],
            prior: Vec::new(),
            offer: 0.0,
            schedule: Vec::new(),
            subtrees: Vec::new(),
            world_visits: Vec::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            path: Vec::new(),
            branch: (0, 0),
            pending: None,
            done: 0,
            rng,
        }
    }

    fn next_leaf(&mut self, config: &Config) -> Option<Leaf> {
        assert!(self.pending.is_none(), "a leaf is still waiting for the network");
        if !self.expanded {
            self.pending = Some(Pending::Root);
            return Some(leaf(&self.root, self.observer));
        }
        if !self.searchable {
            return None;
        }
        while self.done < config.simulations {
            if let Some(id) = self.simulate() {
                self.pending = Some(Pending::Node(id));
                let node = &self.nodes[id as usize];
                let Kind::Decision(actor) = node.kind else { unreachable!("only decisions wait for the network") };
                return Some(leaf(&node.state, actor));
            }
        }
        None
    }

    fn expand(&mut self, log_probs: &[f32], values: &[f32]) {
        match self.pending.take().expect("a leaf was waiting") {
            Pending::Root => self.expand_root(log_probs, values),
            Pending::Node(id) => self.expand_node(id, log_probs, values),
        }
    }

    fn players(&self) -> usize {
        self.root.players()
    }

    fn expand_root(&mut self, log_probs: &[f32], values: &[f32]) {
        self.raw = absolute(values, self.observer, self.players());
        self.prior = self.root.legal_actions().iter().map(|a| (a.index(), log_probs[usize::from(a.index())].exp())).collect();
        self.offer = self.prior.iter().find(|&&(c, _)| c == OFFER).map_or(0.0, |&(_, p)| p);
        self.logits = normalized(self.actions.iter().map(|&c| log_probs[usize::from(c)]).collect());
        let noise = self.noise;
        self.gumbel = (0..self.actions.len()).map(|_| if noise { gumbel(&mut self.rng) } else { 0.0 }).collect();
        let n = self.actions.len();
        self.visits = vec![0; n];
        self.total = vec![[0.0; MAX_PLAYERS]; n];
        self.subtrees = vec![NONE; n * self.worlds.len()];
        self.world_visits = vec![0; n * self.worlds.len()];
        self.expanded = true;
    }

    fn expand_node(&mut self, id: u32, log_probs: &[f32], values: &[f32]) {
        let (state, kind) = {
            let node = &self.nodes[id as usize];
            (node.state, node.kind)
        };
        let Kind::Decision(actor) = kind else { unreachable!("only decisions wait for the network") };
        let codes: Vec<u16> = state.legal_actions().iter().map(Action::index).filter(|&c| c != OFFER).collect();
        let logits = normalized(codes.iter().map(|&c| log_probs[usize::from(c)]).collect());
        let first = self.edges.len() as u32;
        self.edges.extend(codes.iter().zip(&logits).map(|(&code, &weight)| Edge { code, weight, child: NONE }));
        let raw = absolute(values, actor, self.players());
        let node = &mut self.nodes[id as usize];
        node.first = first;
        node.count = codes.len() as u32;
        node.raw = raw;
        node.expanded = true;
        self.backup(raw);
    }

    fn simulate(&mut self) -> Option<u32> {
        let a = self.choose_root();
        let k = self.worlds.len();
        let w = (0..k).min_by_key(|&w| self.world_visits[a * k + w]).expect("at least one world");
        self.branch = (a, w);
        self.path.clear();
        let slot = a * k + w;
        if self.subtrees[slot] == NONE {
            let mut state = self.worlds[w];
            state.apply(Action::from_index(self.actions[a]).expect("a legal action"));
            self.subtrees[slot] = self.add(state);
        }
        let mut id = self.subtrees[slot];
        loop {
            self.path.push(id);
            let node = &self.nodes[id as usize];
            match node.kind {
                Kind::Over(winner) => {
                    self.backup(terminal(winner, self.players()));
                    return None;
                }
                Kind::Decision(_) if !node.expanded => return Some(id),
                Kind::Decision(actor) => {
                    let e = self.interior(id, actor);
                    id = self.child(id, e);
                }
                Kind::Chance => {
                    let e = self.chance(id);
                    id = self.child(id, e);
                }
            }
        }
    }

    fn add(&mut self, state: State) -> u32 {
        let kind = match state.actor() {
            Some(Actor::Player(p)) => Kind::Decision(p),
            Some(Actor::Chance) => Kind::Chance,
            None => Kind::Over(state.winner().expect("a finished game has a winner")),
        };
        let first = self.edges.len() as u32;
        let mut count = 0;
        if let Kind::Chance = kind {
            let chances = state.chances();
            let total = chances.total() as f32;
            for (i, &w) in chances.weights().iter().enumerate().filter(|&(_, &w)| w > 0) {
                self.edges.push(Edge { code: i as u16, weight: f32::from(w) / total, child: NONE });
                count += 1;
            }
        }
        let expanded = !matches!(kind, Kind::Decision(_));
        self.nodes.push(Node { state, kind, visits: 0, total: [0.0; MAX_PLAYERS], raw: [0.0; MAX_PLAYERS], first, count, expanded });
        (self.nodes.len() - 1) as u32
    }

    fn child(&mut self, id: u32, e: usize) -> u32 {
        let edge = self.edges[e];
        if edge.child != NONE {
            return edge.child;
        }
        let node = &self.nodes[id as usize];
        let mut state = node.state;
        match node.kind {
            Kind::Chance => state.resolve(state.outcome(edge.code).expect("an outcome of this node")),
            _ => state.apply(Action::from_index(edge.code).expect("a legal action")),
        }
        let child = self.add(state);
        self.edges[e].child = child;
        child
    }

    fn edge_stats(&self, id: u32, seat: usize) -> (Vec<f32>, Vec<u32>, Vec<Option<f32>>) {
        let node = &self.nodes[id as usize];
        let edges = &self.edges[node.first as usize..(node.first + node.count) as usize];
        let weights = edges.iter().map(|e| e.weight).collect();
        let visits = edges.iter().map(|e| if e.child == NONE { 0 } else { self.nodes[e.child as usize].visits }).collect();
        let q = edges
            .iter()
            .map(|e| {
                let child = (e.child != NONE).then(|| &self.nodes[e.child as usize]);
                child.filter(|c| c.visits > 0).map(|c| c.total[seat] / c.visits as f32)
            })
            .collect();
        (weights, visits, q)
    }

    fn interior(&self, id: u32, actor: Player) -> usize {
        let seat = actor.index();
        let (logits, visits, q) = self.edge_stats(id, seat);
        let completed = self.completed(self.nodes[id as usize].raw[seat], &logits, &visits, &q);
        let probs = softmax(&logits.iter().zip(&completed).map(|(l, c)| l + c).collect::<Vec<_>>());
        let sum: u32 = visits.iter().sum();
        let best = argmax(probs.iter().zip(&visits).map(|(p, &n)| p - n as f32 / (1 + sum) as f32));
        self.nodes[id as usize].first as usize + best
    }

    fn chance(&self, id: u32) -> usize {
        let (probs, visits, _) = self.edge_stats(id, 0);
        let sum: u32 = visits.iter().sum();
        let best = argmax(probs.iter().zip(&visits).map(|(p, &n)| p - n as f32 / (1 + sum) as f32));
        self.nodes[id as usize].first as usize + best
    }

    fn root_q(&self) -> Vec<Option<f32>> {
        let seat = self.observer.index();
        self.visits.iter().zip(&self.total).map(|(&n, t)| (n > 0).then(|| t[seat] / n as f32)).collect()
    }

    fn choose_root(&mut self) -> usize {
        if self.schedule.is_empty() {
            let considered = self.considered.min(self.actions.len());
            self.schedule = considered_visits(considered, self.simulations);
        }
        let simulation: u32 = self.visits.iter().sum();
        let wanted = self.schedule[(simulation as usize).min(self.schedule.len() - 1)];
        let completed = self.completed(self.raw[self.observer.index()], &self.logits, &self.visits, &self.root_q());
        self.score(wanted, &completed)
    }

    fn score(&self, wanted: u32, completed: &[f32]) -> usize {
        let top = self.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        argmax((0..self.actions.len()).map(|i| {
            if self.visits[i] == wanted {
                (self.gumbel[i] + self.logits[i] - top + completed[i]).max(-1e9)
            } else {
                f32::NEG_INFINITY
            }
        }))
    }

    fn completed(&self, raw: f32, logits: &[f32], visits: &[u32], q: &[Option<f32>]) -> Vec<f32> {
        completed_q(raw, logits, visits, q, self.value_scale, self.rescale)
    }

    fn backup(&mut self, value: [f32; MAX_PLAYERS]) {
        for &id in &self.path {
            let node = &mut self.nodes[id as usize];
            node.visits += 1;
            (0..MAX_PLAYERS).for_each(|s| node.total[s] += value[s]);
        }
        let (a, w) = self.branch;
        self.visits[a] += 1;
        (0..MAX_PLAYERS).for_each(|s| self.total[a][s] += value[s]);
        self.world_visits[a * self.worlds.len() + w] += 1;
        self.done += 1;
    }

    fn outcome(&self) -> Outcome {
        let mut policy = [0.0; ACTIONS];
        if !self.expanded {
            return Outcome { policy, action: None, value: 0.0 };
        }
        let seat = self.observer.index();
        if !self.searchable || self.done == 0 {
            for &(code, p) in &self.prior {
                policy[usize::from(code)] = p;
            }
            return Outcome { policy, action: None, value: self.raw[seat] };
        }
        let q = self.root_q();
        let completed = self.completed(self.raw[seat], &self.logits, &self.visits, &q);
        let improved = softmax(&self.logits.iter().zip(&completed).map(|(l, c)| l + c).collect::<Vec<_>>());
        for (&code, p) in self.actions.iter().zip(&improved) {
            policy[usize::from(code)] = p * (1.0 - self.offer);
        }
        policy[usize::from(OFFER)] += self.offer;
        let most = self.visits.iter().copied().max().unwrap_or(0);
        let best = self.score(most, &completed);
        Outcome { policy, action: Some(self.actions[best]), value: q[best].unwrap_or(self.raw[seat]) }
    }
}

fn leaf(state: &State, actor: Player) -> Leaf {
    let mut observation = [0; OBSERVATION_BYTES];
    observe(state, actor, &mut observation);
    let mut hidden = [[0; 10]; MAX_PLAYERS - 1];
    for (k, other) in state.others(actor).enumerate() {
        let (hand, aside) = (state.hand(other), state.set_aside(other));
        for r in 0..5 {
            hidden[k][r] = hand[r] + aside[r];
            hidden[k][5 + r] = state.dev_cards(other, DevCard::ALL[r]);
        }
    }
    Leaf { observation, mask: state.legal_actions().words(), hidden }
}

fn absolute(relative: &[f32], actor: Player, players: usize) -> [f32; MAX_PLAYERS] {
    let mut value = [0.0; MAX_PLAYERS];
    for (s, v) in value.iter_mut().enumerate().take(players) {
        *v = relative[(s + players - actor.index()) % players];
    }
    value
}

fn terminal(winner: Player, players: usize) -> [f32; MAX_PLAYERS] {
    let mut value = [0.0; MAX_PLAYERS];
    for (s, v) in value.iter_mut().enumerate().take(players) {
        *v = if s == winner.index() { 1.0 } else { -1.0 / (players - 1) as f32 };
    }
    value
}

fn normalized(logits: Vec<f32>) -> Vec<f32> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total = top + logits.iter().map(|l| (l - top).exp()).sum::<f32>().ln();
    logits.into_iter().map(|l| l - total).collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - top).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

fn argmax(scores: impl Iterator<Item = f32>) -> usize {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, s) in scores.enumerate() {
        if s > best.1 {
            best = (i, s);
        }
    }
    best.0
}

fn gumbel(rng: &mut Rng) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    -(-u.ln()).ln() as f32
}

/// Q-values completed by the mixed value, rescaled to [0, 1] and weighted by visits (mctx's
/// `qtransform_completed_by_mix_value`).
fn completed_q(raw: f32, logits: &[f32], visits: &[u32], q: &[Option<f32>], value_scale: f32, rescale: bool) -> Vec<f32> {
    let priors = softmax(logits);
    let sum: u32 = visits.iter().sum();
    let mixed = if sum == 0 {
        raw
    } else {
        let seen = || priors.iter().zip(q).filter_map(|(&p, q)| q.map(|q| (p.max(f32::MIN_POSITIVE), q)));
        let mass: f32 = seen().map(|(p, _)| p).sum();
        let weighted: f32 = seen().map(|(p, q)| p * q / mass).sum();
        (raw + sum as f32 * weighted) / (sum as f32 + 1.0)
    };
    let completed: Vec<f32> = q.iter().map(|q| q.unwrap_or(mixed)).collect();
    let low = completed.iter().copied().fold(f32::INFINITY, f32::min);
    let high = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let most = visits.iter().copied().max().unwrap_or(0);
    let scale = (VISIT_INIT + most as f32) * value_scale;
    let unit = |c: f32| if rescale { (c - low) / (high - low).max(1e-8) } else { (c + 1.0) / 2.0 };
    completed.iter().map(|&c| unit(c) * scale).collect()
}

/// For each simulation, how many visits the root action it goes to must already have: sequential halving
/// (mctx's `get_sequence_of_considered_visits`).
pub fn considered_visits(considered: usize, simulations: u32) -> Vec<u32> {
    if considered <= 1 {
        return (0..simulations).collect();
    }
    let log2 = (considered as f64).log2().ceil() as u32;
    let (mut sequence, mut visits, mut kept) = (Vec::new(), vec![0u32; considered], considered);
    while sequence.len() < simulations as usize {
        let extra = (simulations / (log2 * kept as u32)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..kept]);
            visits[..kept].iter_mut().for_each(|v| *v += 1);
        }
        kept = (kept / 2).max(2);
    }
    sequence.truncate(simulations as usize);
    sequence
}

#[cfg(test)]
mod tests;
