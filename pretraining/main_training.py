import os
import time

import fire
import torch
import torch.optim as optim
from fms.datasets import instructions
from fms.models import llama
from fms.utils import tokenizers
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from transformers import LlamaForCausalLM, LlamaConfig

import config
import policies
from utils.config_utils import update_config
from utils.train_utils import setup, setup_environ_flags, get_policies


# ----------  Training ----------------------------------------------------------
# our train function, called per epoch
def train(
    cfg,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    profiler=None,
):
    model.train()
    ddp_loss = torch.zeros(2).to(local_rank)

    if sampler:
        sampler.set_epoch(epoch)
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

        if batch_idx % 200 == 0:
            elapsed_time = time.time() - loop_start
            world_size = int(os.environ["WORLD_SIZE"])
            elapsed_tokens = batch_idx * world_size * cfg.batch_size
            if rank == 0:
                print("step:", batch_idx)
                print("avg speed for these 200 steps:", (time.time() - start) / 200)
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
        print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
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
        print("rank:", rank, "hf model loaded.", "time:", t2-t1)

    # get fms model
    model = llama.convert_hf_llama(model)
    if rank == 0:
        t3 = time.time()
        print("rank:", rank, "fms model converted.", "time:", t3-t2)

    if rank == 0:
        print(f"--> Training for {cfg.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {cfg.model_name} has {total_params/1e6} Million params\n")

    # get tokenizer
    tokenizer = tokenizers.get_tokenizer(cfg.tokenizer)

    # ____________ create batch dataset

    bos_token = '<s>'
    eos_token = '</s>'
    bos_token_id = tokenizer.convert_tokens_to_ids([bos_token])[0]
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    train_dataset = instructions.JsonInstructions(
        cfg.data_file,
        tokenizer=tokenizer, max_len=4096, device="cpu",
        bos_tok_id=bos_token_id, eos_tok_id=eos_token_id
    )
    if 0 == os.getenv("RANK"):
        print(f"--> Training Set Len = {len(train_dataset)}")
        print(f"using dataset {cfg.data_file}")

    val_dataset = train_dataset
    if 0 == os.getenv("RANK"):
        print(f"--> Validation set len = {len(val_dataset)}")

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {"batch_size": cfg.batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": cfg.val_batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": cfg.num_workers_dataloader,
        "pin_memory": False,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    # Main FSDP call
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if cfg.low_cpu_fsdp and rank != 0 else None,
    )

    # if fsdp activation checkpointing:
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        policies.apply_fsdp_checkpointing(model, cfg.selective_checkpointing)

    if cfg.use_torch_compile:
        print("compile not supported yet for llama")

    lr = cfg.learning_rate
    gamma = 0.85
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs
    if rank == 0:
        print(f"Training for {epochs} epochs")

    # Profiler
    if cfg.use_profiler:
        torch_profiler = torch.profiler.profile(
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
        torch_profiler = None

    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

        train(
            model,
            local_rank,
            rank,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
            profiler=torch_profiler,
        )
        scheduler.step()

    dist.barrier()
    dist.destroy_process_group()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    fire.Fire(main)
    