import torch
from torch.utils.data.distributed import DistributedSampler

from fms.utils import tokenizers
from fms.datasets import instructions
from fms.datasets import dataset as fmdata

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


def get_data_loader(cfg, rank, world_size):
    datasets, weights = parse_data_args(cfg.datasets, cfg.weights)

    def causal_lm(data_seq, prompt_len=1):
        """
        Perform causal language modeling by right-shifting the input sequence.
        Sets first prompt_len tokens to be ignored by the loss. Assumes inputs start with BOS.
        """
        data_seq = torch.IntTensor(data_seq)
        t = data_seq.clone()[1:]
        data_seq = data_seq[:-1]
        t[:prompt_len] = -100
        return data_seq, t

    base_scalable = fmdata.Scalable_Sampling_Dataset
    data = base_scalable(
        cfg.data_path,
        fmdata.Streaming_Doc_Dataset,
        rank,
        world_size,
        cfg.sep_token,
        trainsplit=1,
        is_val=False,
        min_length=3,
        datasets=datasets,
        weights=weights,
        seed=cfg.seed,
        verbose=(rank == 0),
        n_logical_shards=cfg.logical_shards,
    )
    data = fmdata.Buffer_Dataset(
        data,
        [cfg.seq_length + 1],
        bos_token=cfg.sep_token,
        pack_hard=True,
    )
    data = fmdata.Preload_Buffer_Dataset(data, 10000)
    data = fmdata.Preprocess_Dataset(data, causal_lm)

    return iter(torch.utils.data.DataLoader(data, num_workers=0, batch_size=cfg.batch_size))


def parse_data_args(datas, weights):
    def splitstrip(x):
        return [item.strip() for item in x.split(",")]
    datas = ["dataset="+x for x in splitstrip(datas)]
    weights = [float(x) for x in splitstrip(weights)]
    return datas, weights
