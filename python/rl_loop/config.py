'''
Parse config from //config
'''
import json

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunConfig(object):
  num_generations: int
  games_per_gen: int
  batch_size: int
  lr: float


def parse(run_id: str) -> RunConfig:
  proj_root = Path(__file__).parent.parent.parent
  config_path = Path(proj_root, 'config', run_id + '.json')

  with open(config_path) as f:
    obj = json.loads(f.read())
    return RunConfig(obj['num_generations'], obj['games_per_gen'],
                     obj['batch_size'], obj['lr'])
