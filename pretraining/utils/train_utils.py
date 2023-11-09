import os
import time

from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist

try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging

from pretraining.policies import *

from torch.distributed.fsdp import ShardingStrategy


def train(
    cfg,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    profiler,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    start = time.time()
    loop_start = time.time()
    for batch_idx, (input, label) in enumerate(train_loader, start=1):
        input = input.to(local_rank)
        label = label.to(local_rank)

        optimizer.zero_grad()
        output = model(input)
        ce_loss = torch.nn.CrossEntropyLoss()
        loss = ce_loss(output.transpose(-1, -2), label)

        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if profiler:
            profiler.step()

        if batch_idx % cfg.report_interval == 0:
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            elapsed_tokens = batch_idx * world_size * cfg.batch_size * cfg.seq_length // cfg.tp_size
            if rank == 0:
                print("step:", batch_idx)
                print(f"speed for these {cfg.report_interval} steps:", (time.time() - start) / cfg.report_interval)
                print("overall speed:", elapsed_time / batch_idx)
                print("reserved memory:", torch.cuda.max_memory_reserved(device=torch.cuda.current_device()))
                print("active memory:", torch.cuda.max_memory_allocated(device=torch.cuda.current_device()))
                print("overall token per gpu per sec:", int(elapsed_tokens / world_size / elapsed_time))
                print("token per day:", int(elapsed_tokens / elapsed_time * 3600 * 24))
            start = time.time()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

    # consolidate final loss number - do not use .reduce here, requires global synch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        print(f"Loss: \t{train_accuracy:.4f}")
    return train_accuracy


def setup():
    dist.init_process_group(backend="nccl",timeout=datetime.timedelta(seconds=5400))


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
    elif cfg.sharding_strategy == "zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {cfg.sharding_strategy}")

    return mixed_precision_policy, wrapping_policy, sharding_strategy


def get_profiler(cfg):
    if cfg.use_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(skip_first=5, wait=5, warmup=5, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "profile_traces"
            ),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        )
    else:
        profiler = None
    return profiler
