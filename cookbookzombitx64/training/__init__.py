"""
โมดูลสำหรับการเทรนโมเดลด้วยวิธีการต่างๆ
"""

from . import epoch_based_training
from . import vanilla_gradient_descent
from . import stochastic_gradient_descent
from . import mini_batch_gradient_descent
from . import energy_efficient_scheduler

__all__ = [
    "epoch_based_training",
    "vanilla_gradient_descent",
    "stochastic_gradient_descent",
    "mini_batch_gradient_descent",
    "energy_efficient_scheduler"
] 