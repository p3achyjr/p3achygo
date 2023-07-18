'''
Parse config from //config
'''
import json

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig(object):
  num_generations: int
  games_first_gen: int
  games_per_gen: int
  batch_size: int
  use_cyclic_lr: bool
  lr: float
  min_lr: float
  max_lr: float


def parse(run_id: str) -> RunConfig:
  proj_root = Path(__file__).parent.parent.parent
  config_path = Path(proj_root, 'config', run_id + '.json')

  with open(config_path) as f:
    obj = json.loads(f.read())
    games_first_gen = obj[
        'games_first_gen'] if 'games_first_gen' in obj else obj['games_per_gen']
    min_lr = obj['min_lr'] if 'min_lr' in obj else obj['lr']
    max_lr = obj['max_lr'] if 'max_lr' in obj else obj['lr']
    use_cyclic_lr = True if min_lr != max_lr else False
    return RunConfig(obj['num_generations'], games_first_gen,
                     obj['games_per_gen'], obj['batch_size'], use_cyclic_lr,
                     obj['lr'], min_lr, max_lr)
