import os
import time

import fire
import torch
import torch.optim as optim
from fms.models import llama
from torch import distributed as dist
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.optim.lr_scheduler import StepLR
from transformers import LlamaForCausalLM, LlamaConfig

import config
import policies
from pretraining.utils.config_utils import update_config
from pretraining.utils.dataset_utils import get_train_loader
from pretraining.utils.train_utils import setup, setup_environ_flags, get_policies, train, get_profiler


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

    # get hf model
    if rank == 0:
        t1 = time.time()
    if cfg.low_cpu_fsdp:
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(cfg.model_name, low_cpu_mem_usage=True)
        else:
            llama_config = LlamaConfig.from_pretrained(cfg.model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)
    else:
        model = LlamaForCausalLM.from_pretrained(cfg.model_name)
    if rank == 0:
        t2 = time.time()
        print("hf model loaded.", "time:", t2 - t1)

    # get fms model
    model = llama.convert_hf_llama(model)
    if rank == 0:
        t3 = time.time()
        print("fms model converted.", "time:", t3 - t2)

    if rank == 0:
        print(f"--> Training for {cfg.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {cfg.model_name} has {total_params / 1e6} Million params\n")

    # get data loader
    train_loader = get_train_loader(cfg, rank, world_size)

    # TP
    if cfg.tp_size > 1:
        assert enable_2d_with_fsdp()
        twod_mesh = init_device_mesh("cuda", (world_size // cfg.tp_size, cfg.tp_size))
        blocks = model.get_submodule("layers")
        for i, block in enumerate(blocks):
            if rank == 0:
                print("parallelization of block:", i)
            block = parallelize_module(
                module=block,
                device_mesh=twod_mesh,
                parallelize_plan={
                    "attn.query": ColwiseParallel(),
                    "attn.key": ColwiseParallel(),
                    "attn.value": ColwiseParallel(),
                    "attn.dense": RowwiseParallel(),
                    "ff_sub_layer.w1": ColwiseParallel(),
                    "ff_sub_layer.wg": ColwiseParallel(),
                    "ff_sub_layer.w2": RowwiseParallel(),
                },
                tp_mesh_dim=1,
            )
        fsdp_pg = twod_mesh.get_dim_groups()[0]
    else:
        fsdp_pg = None

    # FSDP
    model = FSDP(
        model,
        process_group=fsdp_pg,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if cfg.low_cpu_fsdp and rank != 0 else None,
        device_mesh=init_device_mesh("cuda", (world_size // cfg.sharding_group_size, cfg.sharding_group_size))
        if cfg.sharding_strategy in ("hsdp", "zero2") else None,
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
    profiler = get_profiler(cfg)

    if rank == 0:
        print(f"Training for {cfg.num_epochs} epochs")

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


if __name__ == "__main__":
    fire.Fire(main)
