import torch
import os
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from fms.models.llama import LLaMABlock

from functools import partial

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, LLaMABlock)


def apply_fsdp_checkpointing(model, every_xth_item):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print(f"--> applying fdsp activation checkpointing...")

    def selective_checkpointing(submodule):
        selective_checkpointing.__dict__.setdefault("_count", 0)

        if isinstance(submodule, LLaMABlock):
            selective_checkpointing._count += 1
            if (
                    not every_xth_item
                    or selective_checkpointing._count % every_xth_item == 0
            ):
                return True
        return False

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=selective_checkpointing)
