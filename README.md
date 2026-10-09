# catan

a catan engine in rust for training by self-play, it is built for 2-player base catan with bank trades only (no trading between players)

```sh
cargo test                              
cargo run --release --example viewer   
```

the viewer lets you click through the game, force dice rolls, step back through history and see what the player / network sees. every move is 1 of 271 numbered actions so a full game is just a seed + a log of these numbers. Forced moves are played automatically. 
