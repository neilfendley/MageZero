# MageZero Architecture

## Overview

MageZero trains a neural network to play Magic: The Gathering using the AlphaZero
self-play framework. A neural network learns to evaluate board states and suggest
moves by watching itself play thousands of games, getting stronger each iteration.

## The Loop

```
ver0: MCTS with no model (uniform priors, heuristic eval) → generate games → train model
ver1: MCTS guided by ver0 model → generate games → train model (init from ver0)
ver2: MCTS guided by ver1 model → generate games → train model (init from ver1)
...
```

## How MCTS Uses the Network

At each decision point, the game engine enumerates all legal actions (cast a spell,
attack with a creature, pass, etc). MCTS then runs ~1000 iterations of:

```
select:       starting at root, pick the child with the best PUCT score
              PUCT = avg_winrate(child) + C * prior(child) * sqrt(parent_visits) / (1 + child_visits)
expand:       at the leaf, enumerate its legal actions as new child nodes
evaluate:     send the game state to the neural network, get back (policy, value)
              policy → sets priors on child nodes (which branches to explore first)
              value  → estimates who's winning without playing to the end
backpropagate: update visit counts and win rates along the path back to root
```

Children with high priors get explored sooner. Children with high win rates keep
getting revisited. After 1000 iterations, visit counts concentrate on the best action.

## What Gets Recorded

Each decision becomes one training sample:
- **State**: sparse binary vector (~1500 active features out of 2M) encoding the board
  (cards in play, life totals, mana available, zones, card properties, etc.)
- **Policy target**: MCTS visit counts across the 128-slot action space (not one-hot — 
  it's a soft distribution reflecting search confidence)
- **Value target**: eventual game outcome (-1 to +1)

## What the Network Learns

Two heads trained simultaneously:
- **Policy head**: given a board state, predict which actions deserve exploration
- **Value head**: given a board state, predict who's winning

The policy head makes MCTS faster (fewer iterations needed to find good moves).
The value head replaces random rollouts with a learned position evaluation.

## Action Encoding

Actions are hashed into 128 slots. Indices 0-6 are reserved (pass, tap for each
mana color). Indices 7-127 are hash-bucketed by ability text. Collisions mean some
distinct abilities share a prior — MCTS compensates by simulating both and letting
visit counts diverge based on actual outcomes.
