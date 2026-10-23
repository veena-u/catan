# catan

a catan engine in rust for training by self-play: base catan for 2 to 4 players, with bank, harbor and player-to-player trading

```sh
cargo test                              
cargo run --release --example viewer   
```

the viewer lets you click through the game, force dice rolls, step back through history and see what the player / network sees. every move is 1 of 292 numbered actions so a full game is just a player count, a seed and a log of these numbers (`players;seed;codes`). Forced moves are played automatically. 

## python

```sh
conda create -n catan python=3.12 numpy maturin -c conda-forge
conda activate catan
cd python && maturin develop --release
```

## train

```sh
python -m train.evaluate    # sanity check the win-rate harness: random vs random should be ~1/players
python -m train.quiz        # can each network learn legal moves and dice production, given the answers?
python -m train.selfplay    # self-play PPO with a win/loss reward; logs win rates vs random, the bot, and itself
python -m train.ladder runs/selfplay   # round-robin of saved checkpoints, the bot and random, rated by Elo
python -m train.watch runs/selfplay/latest.pt --opponent bot   # play one game; paste the printed line into the viewer
python -m train.compare runs/anneal/latest.pt runs/dice/latest.pt   # challengers vs a baseline, head to head
python -m train.habits runs/anneal/latest.pt    # what it does per game vs the bot: builds, trades, cards, harbors
python -m train.search runs/anneal/latest.pt --bot   # does search (imagine hidden cards, play each candidate forward) beat the plain network?
python -m train.belief runs/base/latest.pt   # learn to guess other players' hidden cards; then search --belief runs/belief/belief.pt
```

every command takes `--players` (default 4). on a rented GPU, `bash overnight.sh` trains a 4-player baseline, then a belief network and search on top of it, then an expert-iteration run, then one run per idea (symmetry, future-dice baseline, privileged critic, dice averaging) and all of them together for twice as long, comparing each against `runs/base` as it goes and packing everything into `results-overnight.tar.gz`

useful `selfplay` flags: `--symmetry` turns each training sample by a random one of the board's 12 rotations/reflections; `--future-dice` scores each move against a baseline that also knows the dice rolled next (input-dependent baseline, ICLR 2019); `--privileged` scores moves with a critic that also sees every hidden hand and card (training only; the playing network never sees them); evaluations are paired: each group of games deals the same board and dice with seats rotated, so luck cancels; `--expert` (expert iteration, as in AlphaZero) searches a few positions every few rounds and also trains the policy toward the moves search prefers.

more flags: `--anneal-power 0.5` shrinks the learning rate and magnet weight over training; `--keep 0.3` trains the policy only on the 30% of moves with the largest advantage; `--average-dice` scores a move followed by a roll against the average over all 11 rolls, so dice luck isn't credited to the move

- `features.py` turns the 314 observation bytes into network inputs (one flat vector, or 146 tokens); seats are relative: me, then the next players in turn order
- `model.py` has the two networks: `Flat` (an MLP baseline) and `Tokens` (a transformer over hexes, corners, edges and one global token); both predict one value per seat
- `agents.py` has the random players and a wrapper that lets a network play
- `evaluate.py` plays two agents against each other and reports the win rate
- `selfplay.py` is the self-play training loop: PPO, reward +1 for the winner and -1/(players-1) for everyone else, a pull toward random-by-kind play to keep exploring
- `bot.py` is a hand-written opponent: builds the most valuable thing it can afford, trades only to afford something, accepts offers that let it build, robs the leader
- `ladder.py` plays saved checkpoints, the bot and random against each other and fits Elo ratings
- `watch.py` plays one game with a saved network and prints it in the viewer's `players;seed;codes` format
- `habits.py` plays a saved network against the bot and reports what it does per game: builds, trades by rate, cards, harbor use
- `search.py` thinks before moving: the network's top moves are each played forward in several random deals of the hidden cards, and the move with the best mix of prior and outcome is chosen; with `--belief` the hidden cards are dealt from the belief network
- `belief.py` trains a network to guess, from one player's view, the hand and development cards of every other player; it reports how surprised it is per unseen card next to a baseline that only counts which cards are left

## rules

- setup snakes out and back (1-2-3-4-4-3-2-1); the second settlement earns its resources
- a 7: everyone over 7 cards discards half, one card at a time in resource order, face down until all are done; then the robber moves and its mover picks whom to rob among exposed players with cards
- player trades are built card by card (asking only for cards the others could hold, by public counts): offer, add cards to give and to want (each in resource order, up to 4 a side), propose; every other player in turn accepts or declines; the offerer picks a partner among those who accepted, or cancels. at most 3 offers a turn
- hands are partly hidden with 3+ players: each player tracks a lower bound on every other hand (production, trades and monopolies are public; a steal between two others only shows that a card moved)
- a player who reaches 10 points off their turn (e.g. given the longest road by someone else's settlement) wins when their turn comes
