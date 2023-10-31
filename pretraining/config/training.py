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
    data_path: str = "/lustre/bluepile-processing/rel0_5/tokens_llama2/lang=en/"
    seq_length: int = 4096
    sep_token: int = 0
    datasets = [
        "dataset=commoncrawl",
        "dataset=webhose",
        "dataset=github_clean",
        "dataset=wikipedia/lang=de",
        "dataset=wikipedia/lang=es",
        "dataset=wikipedia/lang=fr",
        "dataset=wikipedia/lang=ja",
        "dataset=wikipedia/lang=pt",
        "dataset=wikimedia",
        "dataset=uspto",
        "dataset=pubmedcentral",
        "dataset=arxiv",
        "dataset=stackexchange",
        "dataset=PG19",
    ]
    weights = [
        7700,
        500,
        550,
        28,
        17,
        22,
        25,
        8,
        100,
        500,
        175,
        250,
        100,
        25
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
    num_epochs: int = 1
    num_steps: int = 250000
    learning_rate: float = 3e-4

    # reporting
    report_interval: int = 200
