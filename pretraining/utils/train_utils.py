import os
import torch.cuda.nccl as nccl
import torch.distributed as dist

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging

from pretraining.policies import *

from torch.distributed.fsdp import ShardingStrategy


def setup():
    dist.init_process_group("nccl")


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping and sharding strategy"""

    verify_bfloat_support = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and packaging.version.parse(torch.version.cuda).release >= (11, 0)
            and dist.is_nccl_available()
            and nccl.version() >= (2, 10)
    )

    # mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
    else:
        mixed_precision_policy = None

    # wrapping policy
    wrapping_policy = get_llama_wrapper()

    # sharding strategy
    if cfg.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif cfg.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    return mixed_precision_policy, wrapping_policy, sharding_strategy
