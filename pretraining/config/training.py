import os

from dataclasses import dataclass

@dataclass
class train_config:
    # seed
    seed: int = 2023

    # model
    model_name = "/lustre/llama_weights/hf/7B"
    tokenizer = "/lustre/llama_weights/tokenizer.model"

    # data and dataloader
    data_file = "/home/bvaughan/alpaca_data.json"
    num_workers_dataloader: int = 0

    # save models

    # compile
    use_torch_compile = False

    # profiler
    use_profiler = False

    ### fsdp policies
    mixed_precision: bool = True

    fsdp_activation_checkpointing: bool = True
    selective_checkpointing = 1

    sharding_strategy = "fsdp"
    low_cpu_fsdp: bool = True

    # training spec
    batch_size: int = 2
    num_epochs: int = 2
    learning_rate: float = 4e-8

    # validation
    run_validation: bool = False
    val_batch_size = 8
    block_for_validation: bool = False
