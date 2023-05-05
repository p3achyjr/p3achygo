# Gonna try and document my process here :)

## 2-07-2023

I'm starting with a simple supervised learning loop, training the network to
output a policy given board position + 5 previous moves. The network has no
value head at all yet. Since I am (extremely) compute-bound, the idea is to
bootstrap the network with SL first, before setting it off onto the RL loop.

Finally built the initial dataset, model definition, training loop, and board
utils to run a v0 model. Running on CPU with a batch size of 32 takes ~10 seconds
to train 5 mini-batches. With ~15 million training samples, this would take
a full 10 days just to train a single epoch. This was prohibitively expensive, so I
started looking into ways to get this to run on a GPU. Playing against the bot,
I judged it to be DDK (maybe 17-18k). The loss around this time seemed to plateau
at around 3.2 - 3.3.

## 2-10-2023

I ran the model on Google Colab, with a Nvidia T4 as GPU. With mixed precision,
The GPU can train 100 mini-batches in ~5 seconds. I tried batch sizes of 64,
128, and 256, eventually settling on 256. The GPU takes ~15 seconds to train
100 mini-batches of 256, for a runtime of 2.5 hours to train single epoch.

Colab eventually timed me out for inactivity, but after 20000 mini-batches,
the bot was a lot stronger (maybe 7-8k). Oddly enough, the bot would miss obvious
moves, like capturing big groups in atari. Presumably this is because this situation
never occurs in professional games, as professionals would just connect.

I wonder if, by bootstrapping the model, the model would ever learn to capture
groups in atari within the RL loop. Some ways to mitigate this would be to add
game score in the RL reward function, or to add groups in atari to the input vector.

## 5-05-2023

Lots of stuff has gone in since the last update. Gumbel-based MCTS is written,
and a (basic) inference engine also exists now in C++. Much of the work in the
last couple months has been around getting C++ tensorflow to play nice, and
porting code into C++.

Early testing of MCTS shows that the move that MCTS selects is often worse than
the raw policy net. Often, these moves will be completely unsensible. This is
because the value net is poorly trained from the bootstrapping step. Why?

We generate around ~40mil samples from ~15000 games. Each of these is an indepedent
datapoint for the policy net. However, every datapoint in a single game shares
the same value target, meaning out of all ~40mil samples, we only have ~15000
indepedent value datapoints. Naive bootstrapping leads to heavy overfitting.

I re-bootstrapped the model with a value loss coefficient of 0.01. This fixed the
overfitting issue, but did not produce a _good_ value net. I am considering using
this [Tygem](https://github.com/yenw/computer-go-dataset#1-tygem-dataset) dataset
to bootstrap the value net.

Some further directions to try:

- [Root position caching](https://arxiv.org/abs/2302.12359)
  - The idea behind this is to start MCTS from positions other than an empty board.
  - This will lead to greater position diversity and more middle/endgame samples.
- [Tensor MCTS](https://www.mdpi.com/2076-3417/13/3/1406)
  - I just found this paper, it is a fully vectorized MCTS implementation.
  - Should see if it works for Gumbel.
- [Attention Layers](https://en.wikipedia.org/wiki/Attention_(machine_learning))
  - One of the biggest weaknesses of current top-level bots is complex conflict.
  - I wonder if the forced locality relationships in CNNs hinders the flexibility
    of the models. Residual, Global Pooling, and Broadcast Blocks all have fixed
    relational structures.
  - Attention layers are more flexible based on my rudimentary knowledge. Maybe
    giving the model the ability to self-highlight certain parts of the board
    could lead to some gains, especially in conflict where explicit locality connections
    are not needed.

In term of direct work items, next up is:

- Implement a TFExample and SGF serializer.
- Figure out a storage method for selfplay games (GCS? Bigtable?).
- Create training and selfplay docker containers (evaluation too?).
- Create a GKE cluster to orchestrate the entire thing.

Lots to do :)
