import os
import time

import fire
import torch
import torch.optim as optim

from fms.models import llama
from fms.models.llama import LLaMAConfig, LLaMA
from torch import distributed as dist
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from transformers import LlamaForCausalLM, LlamaConfig

import config
import policies
from pretraining.utils.dataset_utils import get_train_loader
from pretraining.utils.config_utils import update_config
from pretraining.utils.train_utils import setup, setup_environ_flags, get_policies


# ----------  Training ----------------------------------------------------------
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
            elapsed_tokens = batch_idx * world_size * cfg.batch_size * cfg.seq_length
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


def main(**kwargs):

    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    mixed_precision_policy, wrapping_policy, sharding_strategy_policy = get_policies(cfg, rank)

    llama_config = LLaMAConfig(
        kvheads=20,
        src_vocab_size=32000,
        emb_dim=5120,
        norm_eps=1e-05,
        nheads=40,
        nlayers=40,
        hidden_grow_factor=3,
        multiple_of=1,  # this is set to 1 as it is encoded in the hidden dimension
        activation_fn="silu",
        max_expected_seq_len=2048,
    )

    if rank == 0:
        model = LLaMA(llama_config)
    else:
        with torch.device("meta"):
            model = LLaMA(llama_config)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params/1e6} Million params\n")

    # get data loader
    train_loader = get_train_loader(cfg, rank, world_size)

    # fsdp
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=True,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if rank != 0 else None,
        device_mesh=init_device_mesh("cuda", (world_size // cfg.sharding_group_size, cfg.sharding_group_size))
    )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        policies.apply_fsdp_checkpointing(model, cfg.selective_checkpointing)

    if cfg.use_torch_compile:
        print("compile not supported yet for llama")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.85)
    if rank == 0:
        print(f"Training for {cfg.num_epochs} epochs")

    # Profiler
    if cfg.use_profiler:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                "profile_traces"
            ),
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        )
    else:
        profiler = None

    for epoch in range(1, cfg.num_epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

        train(
            cfg,
            model,
            local_rank,
            rank,
            train_loader,
            optimizer,
            profiler,
        )
        scheduler.step()

    dist.barrier()
    dist.destroy_process_group()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    fire.Fire(main)
    
