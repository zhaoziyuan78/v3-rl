import random
import copy

import numpy as np
import torch


class DDPWithMethods(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        # Redirect attribute access to the wrapped module if it exists there
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cosine_annealing_with_warmup(
    epoch, lr_anneal_epochs, lr_anneal_min_factor, warmup_epochs, warmup_factor
):
    """
    Cosine annealing with warmup learning rate schedule.
    """
    if epoch < warmup_epochs:
        return warmup_factor
    else:
        return lr_anneal_min_factor + 0.5 * (
            1.0 + np.cos(np.pi * (epoch - warmup_epochs) / lr_anneal_epochs)
        ) * (1.0 - lr_anneal_min_factor)


def exponential_decay_with_warmup(
    epoch,
    lr_decay_factor,
    lr_decay_epochs,
    lr_decay_min_factor,
    warmup_epochs,
    warmup_factor,
):
    """
    Exponential decay with warmup learning rate schedule.
    Decays by lr_decay_factor every lr_decay_epochs epochs.
    When lr smaller than lr_min_factor, lr stays at lr_min_factor.
    """
    if epoch < warmup_epochs:
        return warmup_factor
    else:
        return max(
            lr_decay_min_factor,
            lr_decay_factor ** ((epoch - warmup_epochs) / lr_decay_epochs),
        )