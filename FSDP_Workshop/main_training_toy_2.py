import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

# from: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#initialize-ddp-with-torch-distributed-run-torchrun

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 100000000)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100000000, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def demo_basic():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"Start running basic DDP example on rank {rank}.")

    # prepare all gather tensors and matrix
    unit_size = int(34 * 1000000000 / 60 / world_size)

    all_gather_tensor_list = [torch.zeros(unit_size, dtype=torch.bfloat16).to(local_rank) for _ in range(world_size)]
    all_gather_tensor = torch.randn(unit_size, dtype=torch.bfloat16).to(local_rank)

    scatter_reduce_tensor_list = [torch.randn(unit_size, dtype=torch.bfloat16).to(local_rank) for _ in range(world_size)]
    scatter_reduce_tensor = torch.zeros(unit_size, dtype=torch.bfloat16).to(local_rank)

    b = 1
    l = 4096
    h = 5120
    attn_scores = torch.randn(b, l, l, dtype=torch.bfloat16).to(local_rank)
    query = torch.randn(b, l, h, dtype=torch.bfloat16).to(local_rank)
    key = torch.randn(b, l, h, dtype=torch.bfloat16).to(local_rank)

    # run
    for _ in tqdm(range(1000000000)):
        torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2)
        )
        dist.all_gather(all_gather_tensor_list, all_gather_tensor, async_op=True)
        torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2)
        )
        dist.all_gather(all_gather_tensor_list, all_gather_tensor, async_op=True)
        dist.reduce_scatter(scatter_reduce_tensor, scatter_reduce_tensor_list, async_op=True)


if __name__ == "__main__":
    print(torch.__version__)
    demo_basic()
