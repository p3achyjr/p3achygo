import numpy as np
import tensorflow as tf
from model import P3achyGoModel
from model_config import ModelConfig

model = P3achyGoModel.create(ModelConfig.tiny(), 'test')
x = np.random.random((2, 19, 19, 7))
pi_logits, game_outcome, game_ownership, score_logits, gamma = model(x)

print('-------- Policy --------')
print(pi_logits.numpy())
print('-------- Outcome --------')
print(game_outcome.numpy())
print('-------- Ownership --------')
print(game_ownership.numpy())
print('-------- Score --------')
print(game_ownership.numpy())
