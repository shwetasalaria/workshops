import torch
from torch.utils.data.distributed import DistributedSampler

from fms.utils import tokenizers
from fms.datasets import instructions


def get_train_loader(cfg, rank, world_size):

    tokenizer = tokenizers.get_tokenizer(cfg.tokenizer)

    bos_token = '<s>'
    eos_token = '</s>'
    bos_token_id = tokenizer.convert_tokens_to_ids([bos_token])[0]
    eos_token_id = tokenizer.convert_tokens_to_ids([eos_token])[0]

    train_dataset = instructions.JsonInstructions(
        cfg.data_file,
        tokenizer=tokenizer, max_len=cfg.seq_length, device="cpu",
        bos_tok_id=bos_token_id, eos_tok_id=eos_token_id
    )

    sampler = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )

    train_kwargs = {"batch_size": cfg.batch_size, "sampler": sampler}
    cuda_kwargs = {
        "pin_memory": False,
        "shuffle": False,
    }
    train_kwargs.update(cuda_kwargs)

    return torch.utils.data.DataLoader(train_dataset, **train_kwargs)