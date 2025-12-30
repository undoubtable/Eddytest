import os, time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class MatmulModel(nn.Module):
    def __init__(self, n: int, dtype=torch.float16):
        super().__init__()
        # 一个大参数矩阵，DDP 会对它做梯度 all-reduce
        self.w = nn.Parameter(torch.randn(n, n, dtype=dtype))

    def forward(self, x):
        return x @ self.w

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    N = int(os.environ.get("N", "8192"))
    WARMUP = int(os.environ.get("WARMUP", "10"))
    ITERS  = int(os.environ.get("ITERS", "50"))

    model = MatmulModel(N, dtype=torch.float16).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    x = torch.randn(N, N, device=device, dtype=torch.float16)

    # warmup
    for _ in range(WARMUP):
        z = model(x)
        loss = z.float().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # timed
    t0 = time.time()
    for _ in range(ITERS):
        z = model(x)
        loss = z.float().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()

    it_s = ITERS / (t1 - t0)
    t = torch.tensor([it_s], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    avg_it_s = t.item() / dist.get_world_size()

    if dist.get_rank() == 0:
        print(f"[DDP] world_size={dist.get_world_size()}  N={N}  ITERS={ITERS}")
        print(f"[DDP] avg it/s (across ranks): {avg_it_s:.3f}")
        print("Tip: 另开窗口 watch -n 1 nvidia-smi，应看到 4 个进程各占一张卡。")

    dist.destroy_process_group()

if __name__ == "__main__":
    # A100 上 TF32 通常更快（对 matmul）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main()
