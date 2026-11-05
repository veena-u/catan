import itertools

import numpy as np
import torch

RESOURCES = 5
SIDE = 4
MAX_CARDS = 2 * SIDE
STEPS = MAX_CARDS + 1
COUNT_OFFSET = 12
CHOICES = 1 + 2 * RESOURCES
STOP = 0
OFFER, COUNTER = 234, 249
BANK_CARDS = 19
UNLIKELY = -1e9
# Every deal of one or two cards for one or two: the candidates a searched offer always considers.
SMALL = np.array([c for c in itertools.product(range(-2, 3), repeat=RESOURCES)
                  if 1 <= -sum(min(x, 0) for x in c) <= 2 and 1 <= sum(max(x, 0) for x in c) <= 2], np.int8)

def apply(counts, choice):
    card = (choice - 1).clamp(min=0)
    sign = torch.where(card < RESOURCES, -1, 1)
    change = torch.zeros_like(counts).scatter_(1, (card % RESOURCES)[:, None], sign[:, None])
    return counts + torch.where(choice[:, None] > STOP, change, 0)

def step_mask(counts, hand, counter, forbidden, listed, cards):
    gave, asked = (counts < 0).any(1), (counts > 0).any(1)
    valid = torch.where(counter, gave & asked, gave | asked)
    same = (forbidden.long() == counts[:, None]).all(-1) & (torch.arange(forbidden.shape[1], device=counts.device)[None] < listed[:, None])
    valid &= ~same.any(1)
    gives, asks = (-counts).clamp(min=0).sum(1), counts.clamp(min=0).sum(1)
    give = (counts <= 0) & (-counts < hand) & (gives < SIDE)[:, None]
    ask = (counts >= 0) & (counts < BANK_CARDS - hand) & (asks < SIDE)[:, None]
    mask = torch.cat([valid[:, None], give, ask], 1)
    mask[~mask.any(1), STOP] = True
    return mask

def step_log_probs(stop, cards, mask):
    stop_lp = nn_logsigmoid(stop)
    go_lp = nn_logsigmoid(-stop)
    card_lp = cards.float().masked_fill(~mask[:, 1:], UNLIKELY).log_softmax(-1)
    can_go = mask[:, 1:].any(1)
    stop_lp = torch.where(mask[:, STOP] & can_go, stop_lp, torch.where(mask[:, STOP], 0.0, UNLIKELY))
    go_lp = torch.where(can_go & mask[:, STOP], go_lp, torch.where(can_go, 0.0, UNLIKELY))
    return torch.cat([stop_lp[:, None], go_lp[:, None] + card_lp], 1)

def nn_logsigmoid(x):
    return torch.nn.functional.logsigmoid(x.float())

def sample(logits_for, hand, counter, forbidden, listed, greedy=False, temperature=1.0, generator=None):
    n, device = len(hand), hand.device
    counts = torch.zeros(n, RESOURCES, dtype=torch.long, device=device)
    choices = torch.full((n, STEPS), -1, dtype=torch.long, device=device)
    masks = torch.ones(n, STEPS, CHOICES, dtype=torch.bool, device=device)
    open_rows = torch.ones(n, dtype=torch.bool, device=device)
    for k in range(STEPS):
        rows = torch.nonzero(open_rows).squeeze(1)
        if len(rows) == 0:
            break
        mask = step_mask(counts[rows], hand[rows], counter[rows], forbidden[rows], listed[rows], torch.full((len(rows),), k, device=device))
        stop, cards = logits_for(rows, counts[rows])
        log_probs = step_log_probs(stop, cards, mask)
        if greedy:
            pick = log_probs.argmax(1)
        else:
            pick = torch.multinomial((log_probs / temperature).softmax(1).cpu(), 1, generator=generator).squeeze(1).to(device)
        choices[rows, k], masks[rows, k] = pick, mask
        counts[rows] = apply(counts[rows], pick)
        open_rows[rows] = pick != STOP
    return counts, choices, masks

def walk(targets, hand, forbidden, listed, counter=False):
    n, device = len(targets), targets.device
    counts = torch.zeros(n, RESOURCES, dtype=torch.long, device=device)
    choices = torch.full((n, STEPS), -1, dtype=torch.long, device=device)
    masks = torch.ones(n, STEPS, CHOICES, dtype=torch.bool, device=device)
    open_rows = torch.ones(n, dtype=torch.bool, device=device)
    counter = torch.full((n,), counter, dtype=torch.bool, device=device)
    for k in range(STEPS):
        left = targets - counts
        give, ask = left < 0, left > 0
        pick = torch.where(give.any(1), 1 + give.long().argmax(1),
                           torch.where(ask.any(1), 1 + RESOURCES + ask.long().argmax(1), STOP))
        mask = step_mask(counts, hand, counter, forbidden, listed, None)
        choices[open_rows, k], masks[open_rows, k] = pick[open_rows], mask[open_rows]
        counts = torch.where(open_rows[:, None], apply(counts, pick), counts)
        open_rows &= pick != STOP
        if not open_rows.any():
            break
    return choices, masks

def replay(choices):
    n, steps = choices.shape
    counts = torch.zeros(n, steps, RESOURCES, dtype=torch.long, device=choices.device)
    running = torch.zeros(n, RESOURCES, dtype=torch.long, device=choices.device)
    for k in range(steps):
        counts[:, k] = running
        running = apply(running, choices[:, k].clamp(min=0))
    return counts

def needs_deal(actions):
    return np.isin(actions, (OFFER, COUNTER))

def choose(actions, obs, forbidden, listed, logits_for, device, **options):
    deals = np.zeros((len(actions), RESOURCES), np.int8)
    rows = np.flatnonzero(needs_deal(actions))
    if len(rows) == 0:
        return deals, rows, None
    tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    hand = tensor(obs[rows, 222:227]).long()
    counter = tensor(actions[rows] == COUNTER)
    counts, choices, masks = sample(lambda r, c: logits_for(rows[r.cpu().numpy()], c), hand, counter,
                                    tensor(forbidden[rows]), tensor(listed[rows]).long(), **options)
    deals[rows] = counts.cpu().numpy()
    return deals, rows, (choices, masks)
