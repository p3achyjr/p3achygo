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

## 5-12-2023

This week I mainly worked on serializing things to disk. I also did lots of cleanup,
including changing `Board` to use status codes, a `Game` object, getting continuous
self-play to work, fixing thread registration/de-registration in `NNInterface`, and
adding ownership to score computation. I also finally ripped off the bandaid and
spun up my GPU instance. Building there is an absolute nightmare, but we finally have
a working build.

Next I will need to write a rsync process and a dataset shuffler. The rsync process
should be simple--it will just ping a directory every 1-2 minutes for new games,
and rsync new files when they pop up. The shuffler will be based off [this](https://www.moderndescartes.com/essays/shuffle_viz/).

Other things to do:

- Draw first `n` moves from raw policy net, where `n ~ [0, 30]`.
- Reduce visit count near endgame when one side is sure to lose. These visit counts
probably could be the minimum possible (`n=2, k=2`), since the policy net is pretty good.
- Discourage passing, and maybe randomly continue play sometimes when we see a pass?
The model is very happy to pass atm.
- Implement [root position caching](https://arxiv.org/abs/2302.12359).

## 5-19-2023

I spent the early part of the week debugging and improving the self-play pipeline. I've also tried
instrumenting some things to see what the best number of threads are. There was a bug where when `Pass` is the only legal move, we perform no visits. In this case, search returns a nullptr, and we segfault. At least we know pass-alive logic is working :). I've settled on 48 threads for a 8 thread N1 machine on Google Cloud.

I also implemented a timeout for NN inference. This is especially important when separate games start finishing. I'm not exactly sure what the cause of the slowdown is, but it could be any number of things between threads buffering their games, calculating pass-alive regions, thread contention, or less frequent signals as a result of a smaller number of active threads.

I am also writing my own, home-grown shuffler, finally. After perusing all options, it seems like a home-grown is still the best option (read: the only one I can afford). Not going to pay 500$ a month for bigtable :). The current approach is to have workers rsync to a master machine, and continually interleave reading samples from the stream of files. The shuffler is being written in Python, so it'll be interesting to do this in a single-threaded way.

Next Tasks:

- Finish shuffler
- Implement Docker Containers
- Write some minimal Gumbel tests
- Implement KL loss instead of one-hot for policy. I imagine this will be important for me, since early training samples are likely to make the policy net _worse_ as a result of how bad the value net is.

## 5-27-2023

This week has been such a blur that I don't even remember what I've done. I've written a bunch of python wrapper scripts for the RL loop, written a C++ shuffler, and set up GCS. I still don't know if I'll be able to directly scp files between docker containers or machines, so in case I can't, I'll just use GCS for IPC as well.

Next Tasks:

- Implement KL Divergence on completed q-values.
- Implement/benchmark caching for NN evaluations.
- Implement opening noise (0-30 moves of purely drawing from policy).
- Use TensorRT.

## 6-09-2023

Was traveling last week so forgot to update this.

I have rudimentary python wrappers around all the main processes now. I got sidetracked trying to figure out why the pass logit from SL models is very high--the tl;dr is that I still don't know, but using a FC layer for policy instead of the conv + gpool based architecture would be a huge waste of compute. The model has ~1.5mil parameters, and switching out the FC layer would cause the parameter count to jump to ~1.8mil. 300k parameters in a single layer seems excessive and a bad use of compute.

I ran an ablation disabling all pass examples in the SL dataset. The result was that the pass logit is still a bit higher than the logits for other moves (i.e. -7 for pass, -12 for corner moves). One theory is that policy gradients for board moves do not affect the FC layer from global pooling to the pass logit, so the only time that layer gets negative examples is if the model predicts pass, but the true policy is elsewhere. I haven't had time to look into it further.

I have TRT set up now. It's about ~25% faster for single inference. In the process, I had to deal with tensorflow model signatures and contemplate jumping off a bridge while doing it. I also found a bug (I think?) in tf.data where \<dataset>.map() is not thread safe. What can you do :)

I also discovered a serious bug wherein non-root search selects the _worst_ node to traverse instead of the best. The lesson here is to remember to flip your signs :)

I created a validation dataset from pro games, and can use that in RL training to see how the model is doing. I also wrote a rudimentary MCTS test that doesn't do much of anything, but at least it's there.

I also extract completed-Q values at root nodes now. We should be able to train using KL divergence now. This is still yet to be done.

I also implemented an NN cache, which ended up being a massive headache getting it to work with the current concurrency model. While beforehand, each thread would load and fetch inference in lockstep, with caching, one thread can fetch multiple results before loading a batch for inference. I ended up adding a few flags to address this. I haven't verified that deadlock is impossible or that the results are always correct, but it runs well and seems fine. At the moment there is honestly just too much to do to stop and verify everything.

I feel like I am constantly so, so close to kicking off the whole thing, but I've been there for a couple weeks now. Next steps, in order of importance, would be:

- Read Kubernetes docs and figure out what the capabilities are. As corollary, figure out how to sync local data to shuffler (GCS? Direct SCP?). GCS seems like the easiest, so I might just go with that. Also as corollary, figure out how to tell when we have played enough games within a generation.
- Rewrite shuffler to adapt to new filename scheme.
- Figure out how to flush pending games on self-play shutdown. Across several machines, we can potentially lose lots of games if we terminate without flushing our game buffer.
- Train using KL loss on RL examples.
- Implement evaluation. This should just play ~50 games using some high number of evaluations (16, 196) and somehow signal which model is better. Maybe we can keep a single golden in GCS, and on new models, pull the new model, have the models play each other, and swap the GCS golden if it's better. I actually haven't tried running two TF models at once, so hopefully it doesn't all crash and burn.
- Implement opening noise.
- Implement Gumbel K, N buckets. The idea is instead of a flat 8, 64, we can draw from a pool of (K, N) tuples, each of which has different characteristics for search. I.e. if we run (16, 192), we get higher quality policy samples, but if we run (2, 8), we play more games and can generate more independent samples for value training.
- Implement GoExploit buffer (LRU cache of positions to sample from).
- True random seeds for each separate process, to prevent processes starting at the same time from generating the exact same examples.
- Read EfficientZero [paper](https://arxiv.org/pdf/2111.00210.pdf) and see if there's anything there that might be useful.

At this point the most important thing to do is just to get this thing off the ground--so close :)

## 7-04-2023

I have a working local loop now. It _should_ work while distributed as well. I still haven't written my .yaml files, but I have a goal to get it up and running by the end of this week.

I implemented everything listed in the last checkpoint, except for reading EfficientZero. I am using GCS as a shared filestore. A friend of mine mentioned that I end up doing tons of disk IO, which is true. Somehow I had never considered setting up gRPC endpoints and just doing it that way instead.

I re-read the Gumbel paper and realized the key insight is that the PUCT formula in the original AlphaZero search does not actually treat the NN policy as a probability distribution--rather, it treats it as a prior mass, and (I think) will deterministically select the actions with the highest priors to look at first. In Gumbel, we do treat the priors as a distribution, and sample actions to evaluate, bumping the probability of actions we find to be good. Another misunderstanding I had was that Gumbel does not guarantee a policy improvement for each sample, it simply guarantees a policy improvement _in expectation_. It is totally possible for Gumbel to recommend a trash move on individual searches, but will statistically guarantee policy improvement.

Since we are sampling from our policy, I also realized that there isn't much of a reason to use high values of `k`. We can rely on sampling to give us a diverse set of actions to try, and use our visit budget to make deeper reads. In practice, it seems like using larger `k` values does produce more exploration, but we should remember that we are continuing training from a bootstrapped model. Since we already know the policy is good, using small values of `k` is akin to doing PPO/TRPO, iteratively refining our policy. If we were training from scratch, we could consider using higher `k`s. This might not hold once our model encounters off-policy data--we'll have to see how it fares there :)

I have implemented Gumbel bucketing, but have not observed how it affects training. It's possible that doing so may produce inconsistent policy updates, but maybe the diversity in positions and in value/policy training makes up for it.

I also spent some time doing performance improvements, the results of which are in cc/game/board_bench.md and cc/mcts/gumbel_bench.md. One surprise to me was that any code split between .h and .cc files cannot be inlined. The reason is because one translation unit is a cc file with all its #includes expanded, and header files usually do not contain any code, leaving nothing for the compiler to actually inline. Moving small functions into .h files produces a massive CPU speedup (~2x) for MCTS.

Other performance improvements are:

- Auto-assign group IDs. Instead of maintaining a stack of available group IDs, we can simply assign a new group the index at which its first stone was placed. After all, no two stones can share the same board location :). This saves ~15% CPU and 1kb of stack size per board.
- Pre-compute adjacent locations for each board point.
- Use int16_t for group_id/region_id types. This saves 724 bytes.
- SSE Softmax implementation. This saves about ~25% CPU for MCTS.

I _still_ need to get my stuff running on Kubernetes. Honestly I've been pushing this off since it just seems like the least interesting part. But by the end of this week, it should be there :)
