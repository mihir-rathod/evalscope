# Copyright (c) Alibaba, Inc. and its affiliates.
from evalscope.pruners.base import PruningStrategy, get_pruning_strategy, register_pruning_strategy
from evalscope.pruners.discriminative_stratified import DiscriminativeStratifiedPruner
from evalscope.pruners.encoder_stress import EncoderStressPruner

__all__ = [
    'PruningStrategy',
    'register_pruning_strategy',
    'get_pruning_strategy',
    'DiscriminativeStratifiedPruner',
    'EncoderStressPruner',
]
