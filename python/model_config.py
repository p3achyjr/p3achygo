from __future__ import annotations

CONFIG_OPTIONS = [
    'tiny',
    'small',
    'b12c128btl3',
]


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
  def small():
    return ModelConfig()

  @staticmethod
  def b10c128btl3():
    return ModelConfig(blocks=10,
                       broadcast_interval=4,
                       inner_bottleneck_layers=3,
                       channels=128,
                       bottleneck_channels=64)

  @staticmethod
  def b15c192btl3():
    return ModelConfig(blocks=15,
                       broadcast_interval=5,
                       inner_bottleneck_layers=3,
                       channels=192,
                       bottleneck_channels=96)

  @staticmethod
  def b20c256btl3():
    return ModelConfig(blocks=20,
                       broadcast_interval=6,
                       inner_bottleneck_layers=3,
                       channels=256,
                       bottleneck_channels=128)

  # @staticmethod
  # def v2_test():
  #   return ModelConfig(
  #       blocks=16,
  #       broadcast_interval=4,
  #       channels=256,
  #       bottleneck_channels=128,
  #       pool_type=ModelConfig.POOL_TYPE_BROADCAST,
  #       # activation_order=ModelConfig.ACTIVATION_ORDER_PRE,)
  #   )

  @staticmethod
  def from_str(s: str):
    if s == 'tiny':
      return ModelConfig.tiny()
    elif s == 'small':
      return ModelConfig.small()
    elif s == 'b12c128btl3':
      return ModelConfig.b12c128btl3()

    raise Exception("Unknown Model Config")
