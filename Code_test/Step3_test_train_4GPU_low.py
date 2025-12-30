import os
import time
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from deepproblog.dataset import Dataset, DataLoader
from deepproblog.query import Query
from problog.logic import Term, Constant
from deepproblog.network import Network
from deepproblog.model import Model
import deepproblog.engines as engines
import deepproblog.engines.prolog_engine.heuristics as heuristics


# =========================
# Config
# =========================
NPZ_PATH = "/ybz/ybz/Eddytest/data/pretain/components_train.npz"
WORK_DIR = "/ybz/ybz/Eddytest/Code_test"

MIN_AREA = 15
MAX_AREA = 400

BATCH_SIZE_PER_GPU = 128
EPOCHS = 20
LR = 3e-4
LOG_ITER = 20
CLIP_NORM = 5.0
USE_AMP = True

OUT_DIR = os.path.join(WORK_DIR, "Output_fasttrain_ddp_approx")
PL_PATH = os.path.join(WORK_DIR, "eddy_dp_areaok.pl")


# =========================
# Prolog program
# =========================
def write_ascii_pl(path):
    program = (
        "nn(seednet, [Img], Y, [0,1]) :: seed_present(Img, Y).\n"
        "keep_label(Img, 1, 1) :- seed_present(Img, 1).\n"
        "keep_label(Img, 1, 0) :- seed_present(Img, 0).\n"
        "keep_label(_Img, 0, 0).\n"
    )
    with open(path, "w") as f:
        f.write(program)


# =========================
# util
# =========================
def result_to_tensor(r, device):
    """
    Robustly extract a scalar probability from DeepProbLog result
    Works for:
      - torch.Tensor
      - float / int
      - ExactEngine Result
      - ApproximateEngine Result
    """
    # 1. already tensor
    if isinstance(r, torch.Tensor):
        return r.to(device, non_blocking=True)

    # 2. Approximate / Exact Result object
    # 常见属性顺序：value -> result
    for attr in ("value", "result"):
        if hasattr(r, attr):
            v = getattr(r, attr)

            # 有些版本 value/result 是 dict
            if isinstance(v, dict):
                v = next(iter(v.values()))

            if isinstance(v, torch.Tensor):
                return v.to(device, non_blocking=True)

            if isinstance(v, (float, int, np.floating, np.integer)):
                return torch.tensor(float(v), device=device)

    # 3. fallback：尝试转成 float
    try:
        return torch.tensor(float(r), device=device)
    except Exception:
        raise TypeError(
            f"Cannot convert DeepProbLog result to tensor. "
            f"type={type(r)}, dir={dir(r)}"
        )



# =========================
# Tensor source
# =========================
class ComponentTensorSource:
    def __init__(self, path):
        d = np.load(path)
        self.x = torch.from_numpy(d["patches"]).float().pin_memory()

    def __getitem__(self, i):
        if isinstance(i, (tuple, list)):
            i = i[0]
        if hasattr(i, "value"):
            i = i.value
        return self.x[int(i)]

    def __len__(self):
        return self.x.shape[0]


# =========================
# Dataset
# =========================
class ComponentQueries(Dataset):
    def __init__(self, path, subset="train"):
        d = np.load(path)
        self.areas = d["areas"]
        self.labels = d["labels"]
        self.subset = subset

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        area_ok = int(MIN_AREA <= self.areas[i] <= MAX_AREA)
        y = int(self.labels[i])
        sub = {Term("Img"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("keep_label", Term("Img"), Constant(area_ok), Constant(y)), sub)


class ShardedDataset(Dataset):
    def __init__(self, base, rank, world):
        n = len(base)
        per = (n + world - 1) // world
        self.start = rank * per
        self.end = min(n, (rank + 1) * per)
        self.base = base

    def __len__(self):
        return self.end - self.start

    def to_query(self, i):
        return self.base.to_query(self.start + i)


# =========================
# Net
# =========================
class SeedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.to(next(self.parameters()).device, non_blocking=True)
        return F.softmax(self.net(x), dim=-1).clamp(1e-6, 1 - 1e-6)


# =========================
# DDP init
# =========================
def ddp_init():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}")


# =========================
# 自动选择 ApproximateEngine
# =========================
def build_engine(model, k=10):
    # 找到所有 Heuristic 子类
    hs = []
    for name in dir(heuristics):
        obj = getattr(heuristics, name)
        if inspect.isclass(obj) and obj is not heuristics.Heuristic:
            try:
                h = obj()
                hs.append((name, h))
            except Exception:
                pass

    if hasattr(engines, "ApproximateEngine") and hs:
        # 优先用名字里带 prob / greedy 的
        hs.sort(key=lambda x: (
            0 if "prob" in x[0].lower() else
            1 if "greedy" in x[0].lower() else
            5
        ))
        name, h = hs[0]
        if dist.get_rank() == 0:
            print(f"[ENGINE] ApproximateEngine k={k} heuristic={name}")
        return engines.ApproximateEngine(model, k=k, heuristic=h)

    if dist.get_rank() == 0:
        print("[ENGINE] Fallback ExactEngine")
    return engines.ExactEngine(model)


# =========================
# main
# =========================
def main():
    rank, world, local_rank, device = ddp_init()

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        write_ascii_pl(PL_PATH)
        print("[WORLD]", world, "TOTAL_BATCH", world * BATCH_SIZE_PER_GPU)

    dist.barrier(device_ids=[local_rank])

    base_model = SeedNet().to(device)
    ddp_model = DDP(base_model, device_ids=[local_rank])

    net = Network(ddp_model, "seednet", batching=True)
    net.optimizer = torch.optim.Adam(base_model.parameters(), lr=LR)

    model = Model(PL_PATH, [net])
    model.set_engine(build_engine(model, k=10))
    model.add_tensor_source("train", ComponentTensorSource(NPZ_PATH))

    base_ds = ComponentQueries(NPZ_PATH)
    ds = ShardedDataset(base_ds, rank, world)
    loader = DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True)

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        for step, batch in enumerate(loader, 1):
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                probs = model.solve(batch)
                probs = torch.stack([result_to_tensor(p, device) for p in probs])
                loss = (-torch.log(probs)).mean()

            net.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(net.optimizer)
            scaler.update()

            if step % LOG_ITER == 0 and rank == 0:
                print(f"[E{epoch}] step {step} loss {loss.item():.6f}")

        if rank == 0:
            torch.save(base_model.state_dict(), f"{OUT_DIR}/seednet_e{epoch}.pt")
            print(f"[EPOCH {epoch}] done in {time.time()-t0:.1f}s")

        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("[DONE]")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
