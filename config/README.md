# Configuration for a given training run.

## Keys
`num_generations`: Number of generations to play.
`games_per_gen`: Number of games to play per gen.
`batch_size`: Training batch size.
`lr | (min_lr, max_lr)`: Training learning rate. if `lr` is specified, use a flat learning rate at `lr`. Otherwise, use a one-cycle learning rate between `[min_lr, max_lr]` per generation.
`games_first_gen` (optional): Number of games to play in the first generation.
