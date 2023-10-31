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
    data_path: str = "/home/bvaughan/"
    seq_length: int = 4096
    sep_token: int = 0
    datasets = [
        "dataset=commoncrawl",
        "dataset=wikimedia",
    ]
    weights = [
        .9,
        .1,
    ]
    logical_shards: int = 768

    # save models

    # compile
    use_torch_compile: bool = False

    # profiler
    use_profiler: bool = False

    # tp
    tp_size: int = 1

    # fsdp policies
    mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = True
    selective_checkpointing: int = 1
    sharding_strategy: str = "hsdp"
    sharding_group_size: int = 8
    low_cpu_fsdp: bool = False

    # training spec
    batch_size: int = 2
    num_epochs: int = 100
    num_steps: int = 1e3
    learning_rate: float = 3e-4

    # reporting
    report_interval: int = 200
