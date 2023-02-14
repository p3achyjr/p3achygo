'''
Configuration for model
'''


class ModelConfig:

  def __init__(self,
               blocks=16,
               conv_size=3,
               broadcast_interval=8,
               bottleneck_length=4,
               channels=128,
               bottleneck_channels=64,
               head_channels=32,
               c_val=64):
    self.kBlocks = blocks
    self.kConvSize = conv_size
    self.kBroadcastInterval = broadcast_interval
    self.kBottleneckLength = bottleneck_length
    self.kChannels = channels
    self.kBottleneckChannels = bottleneck_channels
    self.kHeadChannels = head_channels
    self.kCVal = c_val

  @staticmethod
  def tiny():
    return ModelConfig(blocks=6,
                       broadcast_interval=4,
                       bottleneck_length=3,
                       channels=16,
                       bottleneck_channels=8,
                       head_channels=8,
                       c_val=16)

  @staticmethod
  def small():
    return ModelConfig()
