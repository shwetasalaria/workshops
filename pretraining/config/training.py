import os

from dataclasses import dataclass


@dataclass
class train_config:
    # seed
    seed: int = 2023

    # model
    model_name: str = "/lustre/llama_weights/hf/7B"
    tokenizer: str = "/lustre/llama_weights/tokenizer.model"

    # data and dataloader
    data_file: str = "/home/bvaughan/alpaca_data.json"
    seq_length: int = 4096

    # save models

    # compile
    use_torch_compile: bool = False

    # profiler
    use_profiler: bool = False

    # fsdp policies
    mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = True
    selective_checkpointing: int = 1
    sharding_strategy: str = "hsdp"
    sharding_group_size: int = 8
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 3e-4

    # reporting
    report_interval: int = 200
