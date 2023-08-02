from __future__ import annotations


class ModelConfig:
  """
  Configuration for model.
  """

  def __init__(self,
               blocks=16,
               conv_size=3,
               broadcast_interval=8,
               inner_bottleneck_layers=2,
               channels=128,
               bottleneck_channels=64,
               head_channels=32,
               c_val=64):
    self.kBlocks = blocks
    self.kConvSize = conv_size
    self.kBroadcastInterval = broadcast_interval
    self.kInnerBottleneckLayers = inner_bottleneck_layers
    self.kChannels = channels
    self.kBottleneckChannels = bottleneck_channels
    self.kHeadChannels = head_channels
    self.kCVal = c_val

  @staticmethod
  def tiny():
    return ModelConfig(blocks=6,
                       broadcast_interval=4,
                       inner_bottleneck_layers=1,
                       channels=16,
                       bottleneck_channels=8,
                       head_channels=8,
                       c_val=16)

  @staticmethod
  def b6c96():
    # Tries to mimic KataGo architecture. Need 8 blocks b/c model creates
    # `blocks-2` blocks in the trunk.
    return ModelConfig(blocks=8,
                       broadcast_interval=4,
                       inner_bottleneck_layers=1,
                       channels=96,
                       bottleneck_channels=48,
                       head_channels=32,
                       c_val=48)

  @staticmethod
  def small():
    return ModelConfig()

  @staticmethod
  def medium():
    return ModelConfig(blocks=24,
                       broadcast_interval=6,
                       inner_bottleneck_layers=3,
                       channels=192,
                       bottleneck_channels=96,
                       head_channels=32,
                       c_val=64)
