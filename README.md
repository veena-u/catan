# catan

a catan engine in rust for training by self-play, it is built for 2-player base catan with bank trades only (no trading between players)

```sh
cargo test                              
cargo run --release --example viewer   
```

the viewer lets you click through the game, force dice rolls, step back through history and see what the player / network sees. every move is 1 of 271 numbered actions so a full game is just a seed + a log of these numbers. Forced moves are played automatically. 

## python

```sh
conda create -n catan python=3.12 numpy maturin -c conda-forge
conda activate catan
cd python && maturin develop --release
```

## train

```sh
python -m train.evaluate    # sanity check the win-rate harness: random vs random should be ~50%
python -m train.quiz        # can each network learn legal moves and dice production, given the answers?
python -m train.selfplay    # self-play PPO with a win/loss reward; logs win rates vs random, the bot, and itself
python -m train.ladder runs/selfplay   # round-robin of saved checkpoints, the bot and random, rated by Elo
python -m train.watch runs/selfplay/latest.pt --opponent bot   # play one game; paste the printed line into the viewer
python -m train.compare runs/anneal/latest.pt runs/dice/latest.pt   # challengers vs a baseline, head to head
python -m train.habits runs/anneal/latest.pt    # what it does per game vs the bot: builds, trades, cards, harbors
```

on a rented GPU, `bash overnight.sh` runs the whole experiment queue (dice averaging, informative moves, a bigger network, combined, long), comparing each against `runs/anneal` as it goes and packing everything into `results-overnight.tar.gz`

useful `selfplay` flags: `--anneal-power 0.5` shrinks the learning rate and magnet weight over training; `--keep 0.3` trains the policy only on the 30% of moves with the largest advantage; `--average-dice` scores a move followed by a roll against the average over all 11 rolls, so dice luck isn't credited to the move

- `features.py` turns the 211 observation bytes into network inputs (one flat vector, or 146 tokens)
- `model.py` has the two networks: `Flat` (an MLP baseline) and `Tokens` (a transformer over hexes, corners, edges and one global token)
- `agents.py` has the random players and a wrapper that lets a network play
- `evaluate.py` plays two agents against each other and reports the win rate
- `selfplay.py` is the self-play training loop: PPO, win/loss reward only, a pull toward random-by-kind play to keep exploring
- `bot.py` is a hand-written opponent: builds the most valuable thing it can afford, trades only to afford something, robs the opponent's best hex 
- `ladder.py` plays saved checkpoints, the bot and random against each other and fits Elo ratings
- `watch.py` plays one game with a saved network and prints it in the viewer's `seed;codes` format
- `habits.py` plays a saved network against the bot and reports what it does per game: builds, trades by rate, cards, harbor use
