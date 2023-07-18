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

## 7-06-2023 (Run v0)
I am currently training on an HPC cluster from Lambda Labs, with 30 vCPUs and 1 A100 GPU. It takes about ~70 minutes to complete one generation, with 5000 games per generation. Gumbel N, K is a flat 32, 4. I am on generation 15 at the moment. The net learns steadily, with each generation improving by 50-100 Elo. I have yet to evaluate against other bots, or to check whether the Elo gain is transitive.

Some observations:

- The model prefers a moyo-based playstyle in early generationos. Maybe this is a result of the net not being able to do good enough reads to justify the tight, fighting-heavy style in professional play, or maybe the N, K parameters are just too low.
- The model is very bad at detecting atari, especially for larger groups. This might be because it's easy to create an atari circuit for single stones, but hard for chains of arbitrary shape and size. Our model is small, so it might be worth just adding this as a field, even if it means that we will not be able to tell if the model will eventually learn to recognize arbitrary atari. This may also be a result of low visit MCTS, but considering how compute bound I am I can't just simply raise the number of visits.
- Opening and midgame play looks good for the most part, but late game play is atrocious. For one, the model will pretty much never pass. I think this might be because we set win/loss to 1.5/-1.5, so the losing side will pretty much never pass. It might be better to just have this as 1.0/-1.0 + score_diff.
- Groups that should not die end up dying. This leads to wild swings in score across the game and huge captures at the end. The model is actually pretty good at predicting these large score differentials, but this could also indicate value overfitting. This might also be a result (again) of low visit counts, as we do not explore the search tree deep enough to see these groups die, or to have the effect of the dying group affect our search tree.
- The model does steadily improve, but the winrate of the model as white is far, far higher as white than black. Maybe it's because the komi parameter was not normalized (we pass in 7.5 instead of .5). Whoops :) I should start logging the average winrate of each color, overall and per candidate, during evaluation games.

## 7-09-2023 (Run v0)

- The last run seemed like it would not recover, as white was winning practically every game. I restarted the run with 64, 4 and normalized the komi, and so far it looks better. Funnily enough I forgot to update the training code to normalize the komi but oh well.
- I added support for resignation in eval, and low visits in selfplay after one side is pretty much guaranteed to lose. I only lower to visit count of the losing side, so in one game, the side that wins will actually produce more training examples. This is bad. Playout Cap Randomization should fix this. If I were to implement that, I would also just drop the visit counts of both players.
- I also realized that in my Go-Exploit buffer, I add many states per game, but pop the earliest state inserted. This could also be bad, since all those states will share the same result. Instead, I should pop a random value. Since buffer ordering does not matter, I can afford to do a swap-and-pop.
- Minigo implemented resignation in self-play partly to prevent overfitting to a large number of endgame states. I can emulate this by implementing some kind of weighting to moves past a certain move number or certain threshold. Maybe this could be min(first_down_bad, 300) or something.

## 7-10-2023 (Run v0)
- In a rerun with N=64, K=4, it looks like the model is more willing to fight. I would not be surprised if this is because N=32, K=4 does not give the model enough reading horizon to make accurate predictions, so a moyo-based approach is safer.


## 7-14-2023

I ended up just renting an A100 on Lambda Labs. Dollar for dollar it seems to be a much better use of resources.

I did some early runs and the results seemed to be promising--but it turns out that I was just doing evaluation incorrectly. Instead of playing against the current best model, I was just playing against the last model. Kind of sad, but maybe this is at least signal that the RL feedback loop works? I plan on testing the RL loop feedback loop on 9x9 games.

I am doing some testing on MCTS hyperparameters to see what the best values of N, K are. It seems like my initial hypothesis that keeping K low was wrong :) I will also run some experiments to check the position diversity that low visit counts generate, as well as tests on Elo gain relative to N, given the optimal K value for each N. Maybe there is a rough closed-form formula we can use?

Some other ideas:

- Learning Rate Growth: We know we have a good model to begin with, so we do not want to destroy it. Also, with SWA, we compute an average of $0.75w_{old} + 0.25w_{new}$. However, the beginning chunks that we train on are very small, and probably just serve to introduce noise instead of leading to convergence (these early chunks often will not even span 100 mini-batches). Thus, we can gradually increase the learning rate to its full value, maybe based on the size of the chunk we receive.
  - Alternatively, we can vary the SWA momentum based on the number of mini-batches. A simple way to do this is to scale it linearly based on the number of mini-batches in the chunk. We know that a full-sized chunk should contain 8000 mini-batches (in our formulation, this depends on the number of samples per game and could be between 6000 - 8000). We can scale the momentum via $m_{SWA_{new}} = 0.25 * max(1, \frac{\text{num\_batches}}{6000})$
  - (Probably better) write down checkpoints every 1000 batches, instead of after every chunk. We can use the chunk number as a generation counter. Additionally, we can do a "model_0 bootstrap", where model_0 plays a large number of games before the shuffler creates a chunk for it. This prevents us from overdrawing games from the first generation, and will make it so that the first chunk should, in expectation, contain at least 1000 batches.
    - To make this easier, we would probably have to rewrite the shuffler to output 1000 batches at a time. Maybe keeping it simple for now is fine (i.e. just do the model_0 bootstrap).
- Entropy-weighted policy selection: instead of using a fixed probability for selecting nodes for training, weight the probability of selecting a node for training based on the entropy of the prior distribution. We could weight samples based on value estimation as well, but this could just lead to overfitting (i.e. the model overfits to give sharp value estimates )
- Cache `N`, `v_mix` values for evaluated nodes, and use a weighted average of this estimate + the NN estimate for leaf nodes based on the number of visits used to calculate `v_mix`. This is kind of like subtree bias correction in Katago. I still need to flesh out how to accurately update these values throughout self-play. I am not sure if will introduce bias into training.
- Policy surprise weighting. Same as Katago.
- Some kind of quiescence search. Maybe better for test time than self-play training.
- Reset `n` on each search? If we keep `n` values at each search, we already have some pre-defined idea of how good each node is. It will be hard to override this value if `n` is already high (should verify that `v_mix` behaves this way). If we reset `n` values, we keep the `q` values already found for each note, but reset the _weight_ that the existing `q` values give. Thus, we depend solely on the value observations for the _current_ search to find our observed `q` for the current node.
  - The result of this would be to weight deeper nodes' value estimates higher than shallower nodes. Is this valid? I'm not sure. Deeper nodes lie closer to the end of the game, but may be noisier.
  - Another way to incorporate this is to have `q` values from previous searches decay by some factor.
- NN uncertainty estimation (estimate $\mu$, $\sigma$ for value targets). Still not sure how this would help, but it may increase position diversity. We could also have the MCTS planning change based on uncertainty, although this could also be detrimental (or maybe not? would the model just learn to avoid big fights and ladders?). With this, I'm not sure if caching `v_mix` would still work. Maybe we can cache $\mu_{v_{mix}}$, $\sigma_{v_{mix}}$.
  - I also need to flesh out how to combine these values via MCTS. If we just do a running statistical average, this assumes that all subtree Q estimates are i.i.d, which of course they are not. Granted--the original MCTS algorithm does the same with the regular Q-values, so maybe this would not be an issue in practice.
- NN Uncertainty Estimation and planning.
  - If we do not predict $\mu_q$, $\sigma_q$ from the model, we can also have an empirical estimate $\sigma_{q_{MCTS}}$ gathered from the search. We can use Welford's algorithm to calculate this online.
  - If we do predict $\mu_q$, $\sigma_q$, we can simply treat each value prediction from MCTS as a gaussian. Similar to AlphaZero, we will treat these as i.i.d reward estimates and keep the running average (of course they are not i.i.d but w/e).

Other tests:
- Check position diversity generated by low visit count MCTS.
  - Update 7-15: This is essentially negligible, all values give ~2.5% shared root positions across 180 games. These shared root positions are likely at the beginning of games.
- Check relative Elo for high visit counts, given the optimal K for each.
  - MCTS starts to go blind above 192 visits. We are probably brushing up against the limits of our value function. Maybe deeper search nodes give noisier estimates, or non-root planning gives compounding errors deeper in the search tree.
- Compare KL-div of completed Q-values at high visit counts compared to the true distribution (estimated by `Gumbel(10000, 64)`), and find the first-order peak. This is similar to https://github.com/leela-zero/leela-zero/issues/1416
- Check whether playing according to completed-Q values is stronger than playing according to $A_{n+1}$ from gumbel.
- Check Elo loss between base model and TRT converted model.
- Introspect the training data to see, for each training example:
  - The prior entropy of the position.
  - The KL Divergence of the MCTS move relative to the prior.

Other implementation
- GTP Commands
- Merge game-playing options

## 7-15-2023

I'm currently doing another run. The model seems to be improving, besting the old goldens every other model. I plan on letting this run to completion, which should take about a week. I did make a few mistakes, so jotting down some notes here:

- Don't add initial states for positions with under 5% winrate for the losing side.
- Implement the down-bad metric to be 5 consecutive turns across both sides.
- Implement "soft resign" visit count cap.
- Lift control knobs for selfplay into configs.
- Add timestamps to self-play training chunks.
- Only convert new goldens to TRT. Play eval games with unconverted model.
- Implement some kind of training window growth.
- Output score gaussian instead of score logits, to prevent the model from always chasing deterministic high scores in MCTS.
- Experiment with training configuration.
  - Try Cyclic LR per chunk.
  - Play more games in generation 1, to ensure that we have a critical mass of training examples in the first generation (20000, 8000 maybe?).
  - SWA momentum growth seems to work well. Maybe we can try replacing it with learning rate growth, but this is low-pri.
  - Only promote models as new goldens if we hit some confidence bound. The original AlphaGo Zero paper picked a 55% winrate across 400 games, which corresponds to a 95% CI. If I play 75 games, the corresponding winrate would be about 62%, but making this strict could end up overfitting to a single policy. Maybe one standard deviation is a good start.

I also went down a whole rabbit hole of model uncertainty. I was considering having the model predict its own error, but a quick glance at some UC Berkeley slides tabled that idea. The issue is that we need to measure our uncertainty about the model, not the model's uncertainty about our state space. In Go, the state space's uncertainty might well be 0, and the model is incentivized to output 0 if it is optimizing against a fixed dataset.

I also had an idea around online uncertainty calculation, which we can easily compute by using Welford's algorithm. This seems to be something that a lot of people have thought about. Pasting some resources here:

- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/pdf/1707.06887.pdf)
- [Deep Exploration via Randomized Value Functions](https://arxiv.org/pdf/1703.07608.pdf)
- [Randomized Prior Functions for DRL](https://arxiv.org/pdf/1806.03335.pdf)
- [Model Based Value Expansion for Efficient Model-free Reinforcement Learning](https://arxiv.org/pdf/1803.00101.pdf)
- [Online Robust Reinforcement Learning with Model Uncertainty](https://arxiv.org/pdf/2109.14523.pdf)

In general there seems to be a big parallel between model-based methods (where we are trying to learn and refine a dynamics model), and the way we calculate Q via MCTS. We can view Q as a dynamics function $p(q_{s'} | s, a)$, where instead of modeling a distribution over next states, we are modeling a distribution over our next q-value. I need to refine this line of thinking, but methods to deal with model-based RL should be directly applicable to our case, where we are planning through a noisy q-estimate on every timestep.

## 7-16-2023

- Found a bug where we are not viewing our root score estimate from the perspective of the current player...

## 7-17-2023

- Starting to implement a training sample window, and realized that we should train past the end of self-play. Samples that are drawn through their entire window are used an average of $k$ times (depending on how we pick our sample draw probability), but samples that lie at the end are drawn much less than that (i.e. the last generation is drawn $\frac{k}{gen_{window}}$ times). Therefore we should train past the end of self-play. I think a reasonable starting point is to train $gen_{window} - 10$ generations past the end of self-play.
