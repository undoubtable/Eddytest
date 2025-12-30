import os
import time
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
from deepproblog.engines import ExactEngine


# =========================
# Config
# =========================
NPZ_PATH = r"/ybz/ybz/Eddytest/data/pretain/components_train.npz"
WORK_DIR = r"/ybz/ybz/Eddytest/Code_test"

MIN_AREA = 15
MAX_AREA = 400

# 每张卡的 batch；总吞吐约 = world_size * BATCH_SIZE_PER_GPU
BATCH_SIZE_PER_GPU = 128
EPOCHS = 20
LR = 3e-4
LOG_ITER = 20
CLIP_NORM = 5.0

USE_AMP = True

OUT_DIR = os.path.join(WORK_DIR, "Output_fasttrain_ddp")
PL_PATH = os.path.join(WORK_DIR, "eddy_dp_areaok.pl")


# =========================
# Prolog program writer
# =========================
def write_ascii_pl(pl_path: str):
    program = (
        "nn(seednet, [Img], Y, [0,1]) :: seed_present(Img, Y).\n"
        "\n"
        "keep_label(Img, 1, 1) :- seed_present(Img, 1).\n"
        "keep_label(Img, 1, 0) :- seed_present(Img, 0).\n"
        "keep_label(_Img, 0, 0).\n"
    )
    with open(pl_path, "w", encoding="ascii", errors="strict") as f:
        f.write(program)


# =========================
# Result -> Tensor
# =========================
def _maybe_pick_one(x):
    if isinstance(x, dict):
        return next(iter(x.values()))
    return x


def result_to_tensor(r, device: torch.device):
    if isinstance(r, torch.Tensor):
        return r.to(device, non_blocking=True)

    if hasattr(r, "value"):
        v = _maybe_pick_one(getattr(r, "value"))
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, (float, int, np.floating, np.integer)):
            return torch.tensor(float(v), dtype=torch.float32, device=device)

    if hasattr(r, "result"):
        v = _maybe_pick_one(getattr(r, "result"))
        if isinstance(v, torch.Tensor):
            return v.to(device, non_blocking=True)
        if isinstance(v, (float, int, np.floating, np.integer)):
            return torch.tensor(float(v), dtype=torch.float32, device=device)

    return torch.tensor(float(r), dtype=torch.float32, device=device)


# =========================
# TensorSource
# =========================
class ComponentTensorSource:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.x = torch.from_numpy(d["patches"]).float().pin_memory()

    def __len__(self):
        return self.x.shape[0]

    @staticmethod
    def _to_int(idx):
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        if hasattr(idx, "value"):
            return int(idx.value)
        return int(str(idx))

    def __getitem__(self, i):
        i = self._to_int(i)
        return self.x[i]


# =========================
# Base dataset
# =========================
class ComponentQueries(Dataset):
    def __init__(self, npz_path, subset_name="train"):
        d = np.load(npz_path)
        self.areas = d["areas"].tolist()
        self.labels = d["labels"].tolist()
        self.subset = subset_name

    def __len__(self):
        return len(self.labels)

    def to_query(self, i):
        area = int(self.areas[i])
        y = int(self.labels[i])
        area_ok = 1 if (MIN_AREA <= area <= MAX_AREA) else 0
        sub = {Term("Img"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("keep_label", Term("Img"), Constant(area_ok), Constant(y)), sub)


# =========================
# Manual sharded dataset wrapper (no sampler needed)
# =========================
class ShardedDataset(Dataset):
    """
    把原 dataset 按 rank 手动切分：rank 只看自己的样本。
    这里用 contiguous shard：每个 rank 一段连续区间。
    """
    def __init__(self, base_ds: Dataset, rank: int, world: int):
        self.base = base_ds
        n = len(base_ds)
        per = (n + world - 1) // world
        self.start = rank * per
        self.end = min(n, (rank + 1) * per)
        self._len = max(0, self.end - self.start)

    def __len__(self):
        return self._len

    def to_query(self, i):
        return self.base.to_query(self.start + i)


# =========================
# Neural net
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
        dev = next(self.parameters()).device
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        logits = self.net(x)
        return F.softmax(logits, dim=-1).clamp(1e-6, 1 - 1e-6)


# =========================
# DDP init
# =========================
def ddp_init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world, local_rank, device


def main():
    rank, world, local_rank, device = ddp_init()

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        write_ascii_pl(PL_PATH)
        print("[WORLD]", world)
        print("[BATCH_PER_GPU]", BATCH_SIZE_PER_GPU, "=> total", BATCH_SIZE_PER_GPU * world)
    # ✅ 明确用哪张卡做 barrier，消除 NCCL warning
    dist.barrier(device_ids=[local_rank])

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # ----- model -----
    base_module = SeedNet().to(device)
    ddp_module = DDP(base_module, device_ids=[local_rank], output_device=local_rank)

    net = Network(ddp_module, "seednet", batching=True)
    net.optimizer = torch.optim.Adam(base_module.parameters(), lr=LR)

    model = Model(PL_PATH, [net])
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("train", ComponentTensorSource(NPZ_PATH))

    # ----- data -----
    base_ds = ComponentQueries(NPZ_PATH, subset_name="train")
    ds = ShardedDataset(base_ds, rank=rank, world=world)

    loader = DataLoader(ds, batch_size=BATCH_SIZE_PER_GPU, shuffle=True)

    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    if rank == 0:
        d = np.load(NPZ_PATH)
        print("[DATA] patches:", d["patches"].shape, "labels_sum:", int(d["labels"].sum()))
        print("[TRAIN] start DDP manual-split loop")

    global_step = 0
    running_loss = 0.0
    running_count = 0
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        for batch in loader:
            with torch.amp.autocast("cuda", enabled=(USE_AMP and device.type == "cuda"), dtype=torch.float16):
                probs_out = model.solve(batch)

                if isinstance(probs_out, list):
                    probs = torch.stack([result_to_tensor(r, device) for r in probs_out])
                else:
                    probs = result_to_tensor(probs_out, device)

                probs = probs.clamp(1e-6, 1 - 1e-6)
                loss = (-torch.log(probs)).mean()

            net.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if CLIP_NORM and CLIP_NORM > 0:
                scaler.unscale_(net.optimizer)
                torch.nn.utils.clip_grad_norm_(base_module.parameters(), CLIP_NORM)

            scaler.step(net.optimizer)
            scaler.update()

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1

            if global_step % LOG_ITER == 0 and rank == 0:
                dt = time.time() - t0
                avg = running_loss / max(1, running_count)
                print(f"Iter {global_step:6d}  s/iter:{dt/LOG_ITER:.4f}  loss:{avg:.6f}")
                running_loss = 0.0
                running_count = 0
                t0 = time.time()

        if rank == 0:
            torch.save(base_module.state_dict(), os.path.join(OUT_DIR, f"seednet_epoch{epoch}.pt"))
            model.save_state(os.path.join(OUT_DIR, f"deepproblog_epoch{epoch}.mdl"))
            print(f"[EPOCH {epoch}] saved to {OUT_DIR}")

        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print("[DONE] training finished.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
