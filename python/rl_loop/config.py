'''
Parse config from //config
'''
import json

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class RunConfig(object):
  from_existing_run: str
  model_config: str

  # Training Controls
  num_generations: int
  games_first_gen: int
  games_per_gen: int
  batch_size: int
  use_cyclic_lr: bool
  lr: float
  lr_schedule: List[Tuple[int, float]]
  min_lr: float
  max_lr: float
  extra_train_gens: int
  lr_growth_window: int
  optimizer: str  # 'sgd' or 'muon'

  # Gumbel Controls
  min_train_selected_k: int
  min_train_selected_n: int
  max_train_selected_k: int
  max_train_selected_n: int
  min_train_default_k: int
  min_train_default_n: int
  max_train_default_k: int
  max_train_default_n: int
  n_growth_window: int
  k_growth_window: int
  eval_k: int
  eval_n: int
  min_eval_n: int
  eval_n_growth_window: int
  lr_transition_window: int
  use_seen_state_prob: float


def parse(run_id: str) -> RunConfig:
  proj_root = Path(__file__).parent.parent.parent
  config_path = Path(proj_root, 'config', run_id + '.json')

  with open(config_path) as f:
    obj = json.loads(f.read())
    from_existing_run = obj.get('from_existing_run', '')
    model_config = obj.get('model_config', 'small')
    num_generations = obj.get('num_generations', 0)
    games_per_gen = obj.get('games_per_gen', 0)
    games_first_gen = obj.get('games_first_gen', games_per_gen)
    batch_size = obj.get('batch_size', 256)
    lr = obj.get('lr', 1e-2)
    lr_schedule = obj.get('lr_schedule', None)
    if lr_schedule is not None:
      lr_schedule = [(int(t[0]), float(t[1])) for t in lr_schedule]

    min_lr = obj.get('min_lr', lr)
    max_lr = obj.get('max_lr', lr)
    use_cyclic_lr = min_lr != max_lr
    extra_train_gens = obj.get('extra_train_gens', 0)
    lr_growth_window = obj.get('lr_growth_window', 0)
    optimizer = obj.get('optimizer', 'sgd')

    min_train_selected_k = obj.get('min_train_selected_k', 8)
    min_train_selected_n = obj.get('min_train_selected_n', 128)
    max_train_selected_k = obj.get('max_train_selected_k', 8)
    max_train_selected_n = obj.get('max_train_selected_n', 128)
    min_train_default_k = obj.get('min_train_default_k', 5)
    min_train_default_n = obj.get('min_train_default_n', 32)
    max_train_default_k = obj.get('max_train_default_k', 5)
    max_train_default_n = obj.get('max_train_default_n', 32)
    n_growth_window = obj.get('n_growth_window', num_generations)
    k_growth_window = obj.get('k_growth_window', num_generations)
    eval_k = obj.get('eval_k', 8)
    eval_n = obj.get('eval_n', 128)
    min_eval_n = obj.get('min_eval_n', eval_n)
    eval_n_growth_window = obj.get('eval_n_growth_window', 0)
    lr_transition_window = obj.get('lr_transition_window', 0)
    use_seen_state_prob = obj.get('use_seen_state_prob', 0.5)

    return RunConfig(
        from_existing_run, model_config, num_generations, games_first_gen,
        games_per_gen, batch_size, use_cyclic_lr, lr, lr_schedule, min_lr,
        max_lr, extra_train_gens, lr_growth_window, optimizer, min_train_selected_k,
        min_train_selected_n, max_train_selected_k, max_train_selected_n,
        min_train_default_k, min_train_default_n, max_train_default_k,
        max_train_default_n, n_growth_window, k_growth_window, eval_k, eval_n,
        min_eval_n, eval_n_growth_window, lr_transition_window,
        use_seen_state_prob)
