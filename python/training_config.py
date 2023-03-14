'''
Training Config.
'''

from __future__ import annotations


class TrainingConfig:

  def __init__(self,
               dataset_name='',
               batch_size=32,
               epochs=10,
               init_learning_rate=1e-2,
               learning_rate_interval=200000,
               learning_rate_cutoff=800000,
               learning_rate_momentum=.9,
               ds_shuffle_size=1000,
               log_interval=100,
               model_save_interval=5000,
               local_model_path=None,
               local_checkpoint_dir=None,
               local_checkpoint_path=None,
               uploading_to_gcs=False,
               gcs_model_path=None,
               gcs_checkpoint_path=None,
               gcs_client=None):
    # training related
    self.kDatasetName = dataset_name
    self.kBatchSize = batch_size
    self.kEpochs = epochs
    self.kInitLearningRate = init_learning_rate
    self.kLearningRateInterval = learning_rate_interval
    self.kLearningRateCutoff = learning_rate_cutoff
    self.kLearningRateMomentum = learning_rate_momentum
    self.kDatasetShuffleSize = ds_shuffle_size

    # debug related
    self.kLogInterval = log_interval

    # storage related
    self.kModelSaveInterval = model_save_interval
    self.kLocalModelPath = local_model_path
    self.kLocalCheckpointDir = local_checkpoint_dir
    self.kLocalCheckpointPath = local_checkpoint_path
    self.kUploadingToGcs = uploading_to_gcs
    self.kGcsModelPath = gcs_model_path
    self.kGcsCheckpointPath = gcs_checkpoint_path
    self.kGcsClient = gcs_client
