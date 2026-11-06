---
title: A Good Catan Player That Is Not Me!
category: RL
tags: [RL]
created_at: 2026-09-26
updated_at: 2026-11-04
language: en
---

# A Good Catan Player That Is Not Me!

The pure unbridled joy of winning a Catan game is not something a learned policy can understand, but through reward signals the mechanism can get pretty close! I set out Bear Grylls style to find a wild set of weights and tame it through RL into becoming a Catan prowess that can trigger all my friends mwahaha. But to get to a similar Catan bot vs human showdown as 2016's Lee Sedol vs AlphaGo, the journey was a bit like: two roads diverged into a yellow wood, and to know which to follow, I would try out both.

### What makes Catan hard to learn

Naturally one might ask why hasn't every board game been solved by RL by now? Given superhuman performances in games like Chess, Go and even StarCraft, it would seem that significant progress has been made against many types of game mechanics. One constraint is that RL aims to approximate optimal strategy, but does not provide any sort of guarantee of correctness. It learns to estimate from games it has played how good each move is, then shifts toward the moves that did better than expected, but its ability to learn comes down to how noisy those estimates are.

In chess and Go, the same moves from the same position always produce the same game, and both players see the whole board. Catan adds four sources of noise:
1. Dice rolls determine resource production
2. Dev cards are drawn at random from a shuffled deck, and player hands are hidden from opponents (and may be uncertain even to someone keeping track, after a robber steal)
3. Up to four players compete for the same resources and board space
4. Trades rely on another player agreeing to go through

On top of that, the reward is assigned at the end: +1 for the winner, −1/(n−1) for each of the other n−1 players. This means a player shares credit for the result across its several hundred game decisions.

So the project is mostly about two things: making these estimates less noisy and measuring whether game play improved.

### Setup

The version of Catan played is the one seen on Colonist. For 1v1: the first to 15 VPs wins, the discard limit is 9 when a 7 is rolled, the friendly robber is on (the robber can't be put on a hex touching a player with 2 or fewer visible points, and that player can't be robbed), and dice are balanced. Colonist 1v1 allows player-to-player trading, but the 2-player bot here is trained with it turned off: with only two players, any trade that helps one side hurts the other by exactly as much, so a sound opponent never accepts. For 4 players: the first to 10 VPs wins, the discard limit is 7, the friendly robber and balanced dice are on, and players can trade with each other, up to 4 cards on each side of a deal.

Balanced dice are worth a note. Colonist deals rolls from a shuffled deck of all 36 dice outcomes instead of rolling fresh each time, reshuffling once only five are left, so a streak of 8s makes another 8 less likely. The engine plays this way, but the deck is kept out of what the agent sees: the bot plays as if every roll were fresh.

RL needs three things from a game: what the agent sees, what it can do, and what it's rewarded for.

## 1. What the agent sees

At each decision point, the network receives a fixed-size, 369-byte snapshot of the game. The mover is always seat P0, so the snapshot means the same thing whoever's turn it is. The snapshot includes the board, every building (packed as bitmasks), the mover's hand and what it knows of everyone else's, the bank, development cards, scores, what phase the turn is in, the offer on the table with each player's answer and counter-offer, the previous offer, the mover's bank rates, and who the friendly robber protects. An example from a 2-player game is shown below.

<iframe src="/embeds/catan-state.html" title="Catan state" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 850px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

Raw bytes are effectively unlearnable, and bitmasks reside in a special pineapple at the bottom of the sea. The network's forward function addresses this limitation of our representation by using arithmetic (nothing learned) to unpack the 369-byte snapshot into six matrices arranged by board piece: $19 \times 29$ for the 19 hexes, $54 \times 33$ for the 54 corners, $72 \times 16$ for the 72 edges, $4 \times 48$ for the players, $5 \times 27$ for the five resources, and $1 \times 203$ for the remaining game information. An example for a four-player unpacked representation is shown below.

<iframe src="/embeds/catan-decode.html" title="Catan decode" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 760px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

The resource matrix is the newest. Each resource gets a row that says how many you hold, how many the bank has, your rate for it, how many each opponent is known to hold, how much of it is in the offer on the table, and how much of it each player's buildings produce. Trading is mostly a question about resources, so it helps to have a place where everything about one resource sits together.

However, these matrices have different column counts (29, 33, 16, 48, 27, 203). We want to combine them because the information that matters for a decision is spread across them:
- Whether a corner is a good settlement spot depends on its hexes
- Whose roads are closing in, and which player is ahead and has the resources to race you there
- Game information like the phase and an ongoing trade lives in the summary

They cannot be stacked into one grid yet. So, each matrix is passed through its own learned layer which turns each row into 128 numbers. For instance, a hex's 29 numbers (in the $19 \times 29$ matrix) will become 128 numbers through a $128 \times 29$ learned matrix plus 128 biases, and all 19 rows use that same matrix. Corners, edges, players, resources and the summary each have their own. Every row is now 128 wide, so they stack into one grid: summary (1), players (4), resources (5), hexes (19), corners (54), and edges (72), giving $155 \times 128$.

After the per-kind matrices, each row is 128 numbers describing what's at that spot. For example: an empty corner touching wheat 6 and ore 8, with nobody's road nearby. But the row does not store where it is on the board. This is an issue because later the transformer will treat the 155 rows as position agnostic. If two corners on opposite sides of the board have identical features, their rows are identical and the network will not be able to distinguish them.

To address this, each row gets a learned offset that encodes its position. These start small. The row mostly still describes what is there, nudged slightly by where it is, so a coastal corner's tag can come to mean "on the coast". The summary and the 145 board rows share a $146 \times 128$ position-tag matrix, and the players get their own $4 \times 128$ seat tags. Resources need no tag: each resource row already says which resource it is in its own features.

A small aside, why would we not just give each corner an ID to mark its position? We could append 54 extra columns to every corner's features: all 0, except a 1 in that corner's own slot. Each corner's input becomes 33 + 54 = 87 numbers and its learned matrix grows from $128 \times 33$ to $128 \times 87$.

But if you look at what those extra 54 columns do, in multiplication the single 1 gets multiplied by one column of the matrix, which really means one learned vector is added per corner. This is the same as the position tag (without multiplying/storing extra 0s).

The $155 \times 128$ grid then passes through 4 transformer blocks. Each returns a new $155 \times 128$ grid, and each has two steps. Before each step, every row is rescaled to a standard size (a layer norm), which keeps the numbers from drifting as they pass through the blocks.

The first is attention. Each row wants to collect information from some of the other 154 rows, the ones that matter to it. A corner cares about its hexes, not about an edge on the far side of the board. So each row needs to answer two questions:
- Which rows should I listen to?
- What should I take from them?

It could try to score different pairs by how similar the two rows are, then take the weighted average. But this fails because relevance may not be tied to similarity and you get the whole row, not just the useful part.

Instead, for each row the network computes three 128-long vectors using three learned matrices (stored as three $128 \times 128$ matrices stacked: a $128 \times 384$ grid): a query (what this row searches for), a key (how other rows find it), and a value (what it passes on when found). Each 128-long query, key and value is cut into 4 pieces of 32. Piece 1 of everything belongs to head 1, piece 2 to head 2, and so on. Each head has its own Q, K and V, each $155 \times 32$.

Within each head, every row's query is multiplied with every row's key and the products are added up (then divided by $\sqrt{32}$ to keep the numbers tame). If the two are large in the same places, the sum is large, so it's a good match. This gives each row 155 scores, one for every row, including itself. A learned bonus is added to each score depending on how the two rows are related: a hex and its corner, two neighbouring corners, an edge and its corner, two touching edges, a row and itself, the summary and anything, a player and a resource, and so on: 12 kinds of pair in all, counting "unrelated" for everything else. Each block and each head learns its own bonuses, and "unrelated" starts at −2, so far-off rows begin mostly ignored. A softmax then turns the 155 scores into 155 percentages that add up to 100%. The head's output is the sum of all 155 rows' values, each weighted by its percentage. For example, a corner might learn to put most of its weight on its own hexes.

The four heads run these searches in parallel, each with its own 32-number slice. Their four 32-number results are placed side by side to make 128 numbers, passed through one more learned $128 \times 128$ matrix, and added to the row. All 155 rows share the same matrices within a block, and each of the 4 blocks has its own. They all start random.

The second is a feed forward network. Each row passes through a small two layer network: its 128 numbers turn into 512, pass through a GELU (which squashes negative numbers to about zero and keeps positive ones) and turn back into 128. The result is added to the row. All 155 rows use the same two matrices.

Attention gives each row a weighted average of the values of the rows it listened to. The feed forward network then works on each row alone, making sense of what was just gathered. For example, one of the 512 columns of the first $128 \times 512$ matrix might weight "near ore" positively, "near grain" positively, and "opponent road nearby" negatively. Its number comes out high when the row has ore and grain around it and no opponent road, so it acts as a learned question. The 512 numbers are answers to 512 such questions. The GELU lets the strong yeses through and squashes the no's to about zero. The second matrix ($512 \times 128$) turns those answers into a delta added to the original row. If the "strong city spot" question fired, the row gets nudged in a direction that means "strong city spot", which later blocks can search for through attention.

## 2. What the agent can do

The engine numbers every possible move from 0 to 256: 54 settlement corners, 54 city corners, 72 road edges, 19 robber hexes, 5 resources to pick (for a discard, Year of Plenty or Monopoly), 20 bank trades, the turn buttons (roll, end the turn, buy a card, play each of the four card types), 3 players to rob, and 23 trading moves: offer; give or ask for one card of each resource, then send or abandon (the clicks a human uses to build an offer; the network builds its whole offer in one decision and the engine makes the clicks for it); accept, decline or counter; trade with one of 3 accepters, take one of 3 counter-offers, or cancel. A bank trade only names what you give and what you get; your harbors decide whether that costs 4, 3 or 2 cards. At every decision the engine also reports which of the 257 are legal right now, and the network can only choose among those. When exactly one move is legal, the engine plays it without asking.

Choosing directly among 257 moves would make "build a road" compete with 72 separate road entries. So, like OpenAI Five and AlphaStar, the network chooses in two steps. First it picks a kind of move (settlement, road, roll, offer, and so on: 25 kinds) from the summary row. Then, if the kind has a target, it points at one. Each kind turns the summary row into a question, and the question is scored against the rows the targets live in: corners for settlements and cities, edges for roads, hexes for the robber, resource rows for picks and bank trades, player rows for robbing. A learned gate per kind lets each kind look at different parts of those rows. Illegal choices are masked out at both steps.

Offers get a head of their own. Once the network picks "offer", it builds the whole deal card by card. At each step it either stops or adds one card: one of the five resources to give or one of the five to ask for, scored against the resource rows. A side holds at most 4 cards, you can only give what you hold, and the same deal can't be offered twice in a turn. When the deal is finished, the engine sends it as one offer. Each other player then accepts, declines or counters, and the proposer trades with an accepter, takes a counter-offer, or cancels. A turn has a budget of decisions after the roll, which keeps a player from offering forever: 15 in 1v1, and 8 in the current 4-player run, where every offer also draws an answer from each of the other three players.

<iframe src="/embeds/catan-actions.html" title="Catan actions" loading="lazy" scrolling="no" style="width: min(940px, 94vw); margin-left: calc(50% - min(470px, 47vw)); height: 760px; border: 0; border-radius: 8px; overflow: hidden;"></iframe>

## 3. What the agent is rewarded for

Winning, and nothing else. When a game ends, the winner gets +1 and every other player gets −1/(n−1), so the rewards add up to zero. A game that runs past 8,000 decisions is cut off and counts as a loss for everyone. There is no reward for victory points, roads, or good trades along the way. An earlier version paid a small reward for each point gained, but rewards like that can teach the bot to chase points instead of wins, so it was removed. In 4-player games there is one more twist: a win counts for slightly less the more decisions it takes (each decision multiplies it by 0.999), so a move that only stalls the game, like an offer nobody will take, has a small cost. In 4-player games there is one more twist: a win counts for slightly less the more decisions it takes (each decision multiplies it by 0.999), so a move that only stalls the game, like an offer nobody will take, has a small cost.

That leaves one number per player per game to learn from, which is the noise problem from the start. The network also learns to predict a few things that are not rewards, such as every player's final points and what cards opponents hold. Those predictions change what the network notices, not what it plays for. How the network learns from so little signal is the next section.

### Training

---

