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
    seq_length = 4096

    # save models

    # compile
    use_torch_compile = False

    # profiler
    use_profiler = False

    # fsdp policies
    mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = True
    selective_checkpointing = 1
    sharding_strategy = "fsdp"
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 3e-4

    # reporting
    report_interval = 200
