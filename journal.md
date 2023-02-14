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
at around 3.2 - 3.3

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
