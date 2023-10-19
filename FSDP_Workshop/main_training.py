# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import time
from collections import deque
from datetime import datetime

import torch
import torch.optim as optim
import tqdm
from fms.datasets import instructions
from fms.models import llama
from fms.utils import tokenizers
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    ShardingStrategy,
)
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
import config
import performance
import policies
import verify
from transformers import LlamaForCausalLM, LlamaConfig

# some globals
g_gigabyte = 1024**3

bf16_ready = verify.bf16_ready

def _is_rank_0():
    return 0 == os.getenv("RANK")


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch fsdp T5.11 Example")
    """parser.add_argument("--save-dir", default="/model_chkpt", type=str)
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    """

    args = parser.parse_args()
    return args


# ----------------   Main functions --------------------
def get_policies(cfg):

    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.use_mixed_precision:

        if bf16_ready:
            mixed_precision_policy = policies.bfSixteen
            print(f"BFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = policies.fpSixteen
            print(f"BFloat16 support not present. Switching to FP16 with dynamic scaling.")

    wrapping_policy = policies.get_llama_wrapper()

    return mixed_precision_policy, wrapping_policy


# distributed setup
def setup(rank, world_size, cfg):
    # initialize the process group
    dist.init_process_group("nccl")

# various debug settings (show C++ stack if crash, etc.)
def setup_environ_flags(cfg, rank):
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    if cfg.nccl_debug_handler:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if cfg.distributed_debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        if rank == 0:
            print(f"--> running with torch dist debug set to detail")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


def setup_tasks(rank, world_size, cfg):
    """keep the basic setup list here"""
    setup(rank, world_size, cfg)
    # clear_gpu_cache() - need to call torch set device first?
    # set_printing()
    setup_environ_flags(cfg, rank)


def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


# ----------  Training ----------------------------------------------------------
# our train function, called per epoch
def train(
    args,
    model,
    local_rank,
    rank,
    world_size,
    train_loader,
    optimizer,
    epoch,
    sampler=None,
    profiler=None,
    scaler=None,
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

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()  # adjust scaling for next minibatch
        else:
            loss.backward()
            optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += 1

        if profiler:
            profiler.step()

        if batch_idx % 200 == 0:
            elapsed_time = time.time() - loop_start
            if rank == 0:
                print("step:", batch_idx)
                print("avg speed for these 200 steps", (time.time() - start) / 200)
                print("overall speed: ", elapsed_time / batch_idx)
            start = time.time()

    # consolidate final loss number - do not use .reduce here, requires global synch
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_accuracy = ddp_loss[0] / ddp_loss[1]
    if rank == 0:
        print(f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}")
    return train_accuracy


# ---- Validation ---------------
# validation function for one epoch of validation

def validation(model, local_rank, rank, world_size, test_loader):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(local_rank)
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(test_loader)), colour="green", desc="Validation Epoch"
        )
    with torch.no_grad():
        for batch in test_loader:
            for key in batch.keys():
                batch[key] = batch[key].to(local_rank)
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"],
            )
            ddp_loss[0] += output["loss"].item()  # sum up batch loss
            ddp_loss[1] += 1

            if rank == 0:
                inner_pbar.update(1)

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    val_loss = ddp_loss[0] / ddp_loss[1]

    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss


# ---- fsdp main ------------------------------------------------------------


def fsdp_main(args):

    """main process, run within each individual GPU process"""

    cfg = config.train_config()  # loads from defaults

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these defaults {cfg}")
        time_of_run = get_date_of_run()

    setup_tasks(rank, world_size, cfg)

    # fsdp_unit_params = cfg.fsdp_unit_size  

    batch_size = cfg.batch_size
    if rank == 0:
        print(f"\n BatchSize = {batch_size}\n")

    val_batch_size = cfg.val_batch_size

    mp_policy, wrapping_policy = get_policies(cfg)
    
    scaler=None

    if cfg.use_mixed_precision and not bf16_ready:
        # we'll switch to fp16 for V100 etc where BFloat is not supported but user wants mixed precision
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        scaler = ShardedGradScaler()
        if rank==0:
            print(f"--> FP16 implemented for mixed precision support.")

    model_name = cfg.model_name
    if rank == 0:
        print(f"--> training for model {model_name}")

    printable_model_name = str.replace(model_name, "/", "=")

    tokenizer = tokenizers.get_tokenizer(cfg.tokenizer)

    if cfg.low_cpu_fsdp:
        if rank == 0:
            model = LlamaForCausalLM.from_pretrained(
                model_name, use_safetensors=True
            )
        else:
            llama_config = LlamaConfig.from_pretrained(model_name)
            with torch.device("meta"):
                model = LlamaForCausalLM(llama_config)
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_name, use_safetensors=True
        )
    model = llama.convert_hf_llama(model)

    if rank == 0:
        print(f"--> Training for {model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {model_name} has {total_params/1e6} Million params\n")

    # ____________ create batch dataset
    train_name = None
    if cfg.dataset_train:
        train_name = cfg.dataset_train

    bos_token = '<s>'  # for char tokenizer should be 2 and 3. TODO add to tokenizer?
    eos_token = '</s>'
    bos_token_id = tokenizer.convert_tokens_to_ids([bos_token])[0]
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    train_dataset = instructions.JsonInstructions(
        "/home/bvaughan/alpaca_data.json",
        tokenizer=tokenizer, max_len=4096, device="cpu",
        bos_tok_id=bos_token_id, eos_tok_id=eos_token_id
    )
    if 0 == os.getenv("RANK"):
        print(f"--> Training Set Len = {len(train_dataset)}")
        print(f"using dataset {train_name}")

    val_dataset = train_dataset
    if 0 == os.getenv("RANK"):
        print(f"--> Validation set len = {len(val_dataset)}")
        print(f"using dataset {cfg.dataset_test}")

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    print(f"batch size = {batch_size}")

    train_kwargs = {"batch_size": batch_size, "sampler": sampler1}
    test_kwargs = {"batch_size": val_batch_size, "sampler": sampler2}
    cuda_kwargs = {
        "num_workers": cfg.num_workers_dataloader,
        "pin_memory": False,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    torch.cuda.set_device(local_rank)
    
    clear_gpu_cache(local_rank)

    # --- sharding policy
    model_sharding_strategy = (
        cfg.sharding_strategy or ShardingStrategy.FULL_SHARD
    )  # use config, but default to normal if not available
    if rank == 0:
        print(f"Sharding strategy = {model_sharding_strategy}")
    
    # Main FSDP call
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mp_policy,
        sharding_strategy=model_sharding_strategy,
        use_orig_params=cfg.use_orig_params,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if cfg.low_cpu_fsdp and rank != 0 else None,
    )

    # if fsdp activation checkpointing:
    if cfg.FSDP_activation_checkpointing:
        import activation_checkpointing as ac
        ac.apply_fsdp_checkpointing(model, cfg.selective_checkpointing)

    if cfg.use_torch_compile:
        model = torch.compile(model)
        if rank == 0:
            print("######### Compile is being use ###############")
    
    #optional - you can print the sharding plan to see how FSDP has structured the layout.
    if rank == 0 and cfg.print_sharding_plan:
        print(f"model ")
        fn = printable_model_name + "-sharded_layout.txt"
        with open(fn, "w") as external_file:
            header_text = (
                f"model = {model_name}\n"
            )
            print(header_text, file=external_file)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            milli_params = total_params * 4 / 1e6
            print(
                f"\n--> {model_name} has {milli_params} Million params\n",
                file=external_file,
            )
            print(f"model wrapping = \n{model}\n\n", file=external_file)

            external_file.close()

    lr = cfg.learning_rate
    gamma = 0.85

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    if rank==0:
        print(f"--> AdamW whole model tuning with lr of {lr} ")

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    epochs = cfg.num_epochs
    if rank == 0:
        print(f"Training for {epochs} epochs")

    best_train_accuracy = float("-inf")
    best_val_loss = float("inf")
    curr_val_loss = float("inf")

    # will hold our performance timer
    memmax = None

    # --- main training loop 
    if rank == 0:
        memmax = performance.Memory_Maximizer()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        dq = deque(maxlen=cfg.checkpoint_max_save_count+1)
        training_start_time = time.time()

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

    if rank == 0 and cfg.track_memory:
        mem_alloc_tracker = []

    # -- Start Training -----
    if rank==0:
        memmax.start()

    for epoch in range(1, epochs + 1):
        if rank == 0:
            print(f"\n--> Starting Epoch {epoch}")

            t0 = time.time()
        train_accuracy = train(
            args,
            model,
            local_rank,
            rank,
            world_size,
            train_loader,
            optimizer,
            epoch,
            sampler=sampler1,
            profiler=torch_profiler,
            scaler=scaler,
        )
        if rank==0:
            memmax.update()

        if cfg.run_validation:
            curr_val_loss = validation(model, local_rank, rank, world_size, test_loader)

        scheduler.step()

        if rank == 0:
            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if cfg.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if cfg.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )


        # announce new val loss record:
        if rank == 0 and curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # init_end_event.record()
    if rank == 0:
        
        total_training_time = time.time() - training_start_time
        print(f"Total training time = {total_training_time:.2f}")
        print("Times per epoch:")
        for i, val in enumerate(dur):
            print(f"epoch {i}, time {val:.2f}")
        print()

        # training is done...show some training stats for memory use.
        # memory
        memmax.stop()

        if cfg.track_memory:
            print(f"total memory allocated: {mem_alloc_tracker}")

        print(f"Training accuracy: {train_acc_tracking}")
        if cfg.run_validation:
            print(f"Validation accuracy: {val_acc_tracking}") 
            print(f"\n Best Val accuracy: {best_val_loss}")

        # memory summary
        if cfg.memory_report and rank == 0:
            print(
                f"CUDA Memory Summary After Last training:\n {torch.cuda.memory_summary()}"
            )
        
    # all done, set barrier to ensure all GPU's complete, and then cleanup 
    dist.barrier()
    cleanup()


# ------------------ Main functions above ------------


if __name__ == "__main__":
    print(torch.__version__)
    args = parse_args()  # atm we don't use any args..available if needed.
    fsdp_main(args)

    