'''
Hyperparameters for training.
'''


class TrainingConfig:

  def __init__(self,
               init_learning_rate=1e-2,
               learning_rate_momentum=.9,
               batch_size=32,
               epochs=10,
               ds_shuffle_size=1000,
               gcs_checkpoint_path='',
               gcs_client=None):
    self.kInitLearningRate = init_learning_rate
    self.kLearningRateMomentum = learning_rate_momentum
    self.kBatchSize = batch_size
    self.kEpochs = epochs
    self.kDatasetShuffleSize = ds_shuffle_size
    self.kGcsCheckpointPath = gcs_checkpoint_path
    self.kGcsClient = gcs_client
