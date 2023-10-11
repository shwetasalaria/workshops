# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.
import os

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch


@dataclass
class train_config:
    # seed
    seed: int = 2023

    # model
    model_name = os.getenv("MODEL_NAME", "/lustre/llama_weights/7B")
    tokenizer = "/lustre/llama_weights/tokenizer.model"   # no need to adjust, tokenizer works for all model sizes

    # save models
    save_model: bool = False
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # compile
    use_torch_compile = os.getenv("USE_TORCH_COMPILE", "false").lower()
    if use_torch_compile == "true":
        use_torch_compile = True
        use_orig_params = True
    else:
        use_torch_compile = False
        use_orig_params = False

    # sharding policy
    # sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  #FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    sharding_strategy = os.getenv("SHARDING_STRATEGY", "full").lower()
    if sharding_strategy == "full":
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "grad":
        sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "no":
        sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False

    # dataloaders
    num_workers_dataloader: int = 0

    # policies 
    # mixed precision this will default to BFloat16, but if no native support detected, will 
    # use FP16.  (note that FP16 is not recommended for larger models...)
    use_mixed_precision: bool = True

    FSDP_activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/gtrain_150K.csv"
    # dataset_train = "/workspace/data/lchu/gtrain_1M.csv"  # /workspace/data/lchu/gtrain_10M.csv, /workspace/data/lchu/gtrain_150K.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = int(os.getenv("BATCH_SIZE", "2"))
    num_epochs: int = 2

    # validation
    run_validation: bool = True
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
