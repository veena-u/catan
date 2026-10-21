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
```

- `features.py` turns the 211 observation bytes into network inputs (one flat vector, or 146 tokens)
- `model.py` has the two networks: `Flat` (an MLP baseline) and `Tokens` (a transformer over hexes, corners, edges and one global token)
- `agents.py` has the random players and a wrapper that lets a network play
- `evaluate.py` plays two agents against each other and reports the win rate
- `selfplay.py` is the self-play training loop: PPO, win/loss reward only, a pull toward random-by-kind play to keep exploring
- `bot.py` is a hand-written opponent: builds the most valuable thing it can afford, trades only to afford something, robs the opponent's best hex 
- `ladder.py` plays saved checkpoints, the bot and random against each other and fits Elo ratings
