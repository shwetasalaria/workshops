# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.
import os

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class train_config:
    # seed
    seed: int = 2023

    # model
    model_name = os.getenv("MODEL_NAME", "/lustre/llama_weights/7B")
    tokenizer = "/lustre/llama_weights/tokenizer.model"   # no need to adjust, tokenizer works for all model sizes

    # save models
    save_model: bool = False
    checkpoint_max_save_count: int = 2

    # compile
    use_torch_compile = os.getenv("USE_TORCH_COMPILE", "false").lower()
    if use_torch_compile == "true":
        use_torch_compile = True
        use_orig_params = True
    else:
        use_torch_compile = False
        use_orig_params = False

    # sharding policy
    sharding_strategy = os.getenv("SHARDING_STRATEGY", "fsdp").lower()
    if sharding_strategy == "fsdp":
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hsdp":
        sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD
    else:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies
    use_mixed_precision: bool = True

    FSDP_activation_checkpointing: bool = True
    selective_checkpointing = 7

    # datasets
    dataset_train = "datasets_grammar/gtrain_150K.csv"
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    num_epochs: int = 2

    # validation
    run_validation: bool = False
    val_batch_size = 8
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    learning_rate: float = 4e-8

    use_task_free: bool = True
    use_fisher_matrix: bool = False
    percent_F: float = 0.35
