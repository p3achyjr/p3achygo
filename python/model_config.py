from __future__ import annotations

CONFIG_OPTIONS = [
    "tiny",
    "small",
    "b10c128btl3",
    "b12c256btl3",
    "b14c384btl3",
    "b15c192_classic",
    "b8c128nbt",
    "b12c256nbt",
    "b10c384nbt",
    "b14d96h3_transformer",
]

B14_D96_N3_TRANSFORMER = {
    "trunk": 14 * [["transformer", {"embed_dim": 96, "num_heads": 3}]]
}


class ModelConfig:
    """
    Configuration for model.
    """

    TRUNK_BLOCK_TYPES = [
        "classic",
        "btl",
        "nbt",
    ]

    def __init__(
        self,
        blocks=16,
        conv_size=3,
        broadcast_interval=8,
        inner_bottleneck_layers=2,
        channels=128,
        bottleneck_channels=64,
        head_channels=32,
        c_val=64,
        trunk_block_type="btl",
        generic_arch=None,
        is_transformer=False,
        c_l2=1e-4,
    ):
        self.kBlocks = blocks
        self.kConvSize = conv_size
        self.kBroadcastInterval = broadcast_interval
        self.kInnerBottleneckLayers = inner_bottleneck_layers
        self.kChannels = channels
        self.kBottleneckChannels = bottleneck_channels
        self.kHeadChannels = head_channels
        self.kCVal = c_val
        self.kTrunkBlockType = trunk_block_type

        self.generic_arch = generic_arch
        self.is_transformer = is_transformer
        self.c_l2 = c_l2

    @staticmethod
    def tiny():
        return ModelConfig(
            blocks=6,
            broadcast_interval=4,
            inner_bottleneck_layers=1,
            channels=16,
            bottleneck_channels=8,
            head_channels=8,
            c_val=16,
        )

    @staticmethod
    def small():
        return ModelConfig()

    @staticmethod
    def b10c128btl3():
        return ModelConfig(
            blocks=10,
            broadcast_interval=4,
            inner_bottleneck_layers=3,
            channels=128,
            bottleneck_channels=64,
        )

    @staticmethod
    def b5c256btl3():
        return ModelConfig(
            blocks=5,
            broadcast_interval=2,
            inner_bottleneck_layers=3,
            channels=256,
            bottleneck_channels=128,
        )

    @staticmethod
    def b12c256btl3():
        return ModelConfig(
            blocks=12,
            broadcast_interval=5,
            inner_bottleneck_layers=3,
            channels=256,
            bottleneck_channels=128,
        )

    @staticmethod
    def b14c384btl3():
        return ModelConfig(
            blocks=14,
            broadcast_interval=6,
            inner_bottleneck_layers=3,
            channels=384,
            bottleneck_channels=192,
            head_channels=32,
            c_val=80,
        )

    @staticmethod
    def b15c192_classic():
        return ModelConfig(
            blocks=15,
            broadcast_interval=6,
            channels=192,
            head_channels=32,
            c_val=80,
            trunk_block_type="classic",
        )

    @staticmethod
    def b8c128nbt():
        return ModelConfig(
            blocks=8,
            broadcast_interval=3,
            channels=128,
            bottleneck_channels=64,
            head_channels=32,
            trunk_block_type="nbt",
        )

    @staticmethod
    def b12c256nbt():
        return ModelConfig(
            blocks=12,
            broadcast_interval=3,
            channels=256,
            bottleneck_channels=128,
            head_channels=32,
            c_val=80,
            trunk_block_type="nbt",
        )

    @staticmethod
    def b10c384nbt():
        return ModelConfig(
            blocks=10,
            broadcast_interval=4,
            channels=384,
            bottleneck_channels=192,
            head_channels=32,
            c_val=80,
            trunk_block_type="nbt",
        )

    @staticmethod
    def b14d96h3_transformer():
        return ModelConfig(
            channels=96,
            generic_arch=B14_D96_N3_TRANSFORMER,
            is_transformer=True,
            c_l2=0.0,
        )

    @staticmethod
    def from_generic_arch(generic_arch: dict):
        return ModelConfig(generic_arch=generic_arch)

    @staticmethod
    def from_str(s: str):
        if s == "tiny":
            return ModelConfig.tiny()
        elif s == "small":
            return ModelConfig.small()
        elif s == "b10c128btl3":
            return ModelConfig.b10c128btl3()
        elif s == "b12c256btl3":
            return ModelConfig.b12c256btl3()
        elif s == "b14c384btl3":
            return ModelConfig.b14c384btl3()
        elif s == "b15c192_classic":
            return ModelConfig.b15c192_classic()
        elif s == "b8c128nbt":
            return ModelConfig.b8c128nbt()
        elif s == "b12c256nbt":
            return ModelConfig.b12c256nbt()
        elif s == "b10c384nbt":
            return ModelConfig.b10c384nbt()
        elif s == "b14d96h3_transformer":
            return ModelConfig.b14d96h3_transformer()

        raise Exception("Unknown Model Config")
