'''
Parse config from //config
'''
import json

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig(object):
  # Training Controls
  num_generations: int
  games_first_gen: int
  games_per_gen: int
  batch_size: int
  use_cyclic_lr: bool
  lr: float
  min_lr: float
  max_lr: float
  extra_train_gens: int

  # Gumbel Controls
  min_train_selected_k: int
  min_train_selected_n: int
  max_train_selected_k: int
  max_train_selected_n: int
  min_train_default_k: int
  min_train_default_n: int
  max_train_default_k: int
  max_train_default_n: int
  eval_k: int
  eval_n: int


def parse(run_id: str) -> RunConfig:
  proj_root = Path(__file__).parent.parent.parent
  config_path = Path(proj_root, 'config', run_id + '.json')

  with open(config_path) as f:
    obj = json.loads(f.read())
    games_first_gen = obj.get('games_first_gen', obj['lr'])
    min_lr = obj.get('min_lr', obj['lr'])
    max_lr = obj.get('max_lr', obj['lr'])
    use_cyclic_lr = min_lr != max_lr
    extra_train_gens = obj.get('extra_train_gens', 0)

    min_train_selected_k = obj.get('min_train_selected_k', 8)
    min_train_selected_n = obj.get('min_train_selected_n', 128)
    max_train_selected_k = obj.get('max_train_selected_k', 8)
    max_train_selected_n = obj.get('max_train_selected_n', 128)
    min_train_default_k = obj.get('min_train_default_k', 5)
    min_train_default_n = obj.get('min_train_default_n', 32)
    max_train_default_k = obj.get('max_train_default_k', 5)
    max_train_default_n = obj.get('max_train_default_n', 32)
    eval_k = obj.get('eval_k', 8)
    eval_n = obj.get('eval_n', 128)

    return RunConfig(obj['num_generations'], games_first_gen,
                     obj['games_per_gen'], obj['batch_size'], use_cyclic_lr,
                     obj['lr'], min_lr, max_lr, extra_train_gens,
                     min_train_selected_k, min_train_selected_n,
                     max_train_selected_k, max_train_selected_n,
                     min_train_default_k, min_train_default_n,
                     max_train_default_k, max_train_default_n, eval_k, eval_n)
