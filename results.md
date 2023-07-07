# Results

Some informal notes about our training runs

# v0
## 7-06-2023

I am currently training on an HPC cluster from Lambda Labs, with 30 vCPUs and 1 A100 GPU. It takes about ~70 minutes to complete one generation, with 5000 games per generation. Gumbel N, K is a flat 32, 4. I am on generation 15 at the moment. The net learns steadily, with each generation improving by 50-100 Elo. I have yet to evaluate against other bots, or to check whether the Elo gain is transitive.

Some observations:

- The model prefers a moyo-based playstyle in early generationos. Maybe this is a result of the net not being able to do good enough reads to justify the tight, fighting-heavy style in professional play, or maybe the N, K parameters are just too low.
- The model is very bad at detecting atari, especially for larger groups. This might be because it's easy to create an atari circuit for single stones, but hard for chains of arbitrary shape and size. Our model is small, so it might be worth just adding this as a field, even if it means that we will not be able to tell if the model will eventually learn to recognize arbitrary atari. This may also be a result of low visit MCTS, but considering how compute bound I am I can't just simply raise the number of visits.
- Opening and midgame play looks good for the most part, but late game play is atrocious. For one, the model will pretty much never pass. I think this might be because we set win/loss to 1.5/-1.5, so the losing side will pretty much never pass. It might be better to just have this as 1.0/-1.0 + score_diff.
- Groups that should not die end up dying. This leads to wild swings in score across the game and huge captures at the end. The model is actually pretty good at predicting these large score differentials, but this could also indicate value overfitting. This might also be a result (again) of low visit counts, as we do not explore the search tree deep enough to see these groups die, or to have the effect of the dying group affect our search tree.
