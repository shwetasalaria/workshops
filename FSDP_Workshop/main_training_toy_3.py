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
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    print(f"\n--> model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6} Million params\n")

    for _ in tqdm(range(1)):
        output1 = model.net1(torch.randn(20, 10).to(device_id))
        print(rank, 'output1', output1[:10])
        output2 = model.relu(output1)
        print(rank, 'output2', output2[:10])
        output3 = model.net2(output2)
        print(rank, 'output3', output3[:10])
        dist.all_reduce(output3, op=dist.ReduceOp.SUM)
        print(rank, 'output3', output3[:10])
        dist.all_reduce(output2, op=dist.ReduceOp.SUM)
        print(rank, 'output2', output2[:10])
        dist.all_reduce(output1, op=dist.ReduceOp.SUM)
        print(rank, 'output1', output1[:10])


if __name__ == "__main__":
    print(torch.__version__)
    demo_basic()
