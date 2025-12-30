import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

BATCH_SIZE = 512       # A100 80GB: 可继续加到 1024/2048
EPOCHS = 20
LR = 3e-4
LOG_ITER = 20
CLIP_NORM = 5.0

USE_AMP = True

OUT_DIR = os.path.join(WORK_DIR, "Output_fasttrain")
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
# TensorSource + Dataset
# =========================
class ComponentTensorSource:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        # ✅ pinned memory: 配合 non_blocking H2D
        self.x = torch.from_numpy(d["patches"]).float().pin_memory()

    def __len__(self):
        return self.x.shape[0]

    @staticmethod
    def _to_int(idx):
        """
        DeepProbLog 传进来的索引有多种形态：
        - int
        - (int,)  / [int]
        - probLog Constant / Term with .value
        - 其他可 str/int 转换的对象
        """
        if isinstance(idx, (tuple, list)):
            idx = idx[0]
        if hasattr(idx, "value"):  # ProbLog Constant
            return int(idx.value)
        # 有些情况下 idx 可能是 numpy scalar / string
        return int(str(idx))

    def __getitem__(self, i):
        i = self._to_int(i)
        return self.x[i]  # CPU pinned tensor


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
# Train
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    write_ascii_pl(PL_PATH)

    # 性能开关
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[WORKDIR]", WORK_DIR)
    print("[DEVICE]", device)
    print("[NPZ]", NPZ_PATH)
    print("[PL ]", PL_PATH)
    print("[BATCH]", BATCH_SIZE, "AMP:", USE_AMP)

    d = np.load(NPZ_PATH)
    print("[DATA] patches:", d["patches"].shape, "labels_sum:", int(d["labels"].sum()))

    net_module = SeedNet().to(device)
    net = Network(net_module, "seednet", batching=True)
    net.optimizer = torch.optim.Adam(net_module.parameters(), lr=LR)

    model = Model(PL_PATH, [net])
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("train", ComponentTensorSource(NPZ_PATH))

    dataset = ComponentQueries(NPZ_PATH, subset_name="train")
    # ✅ DeepProbLog DataLoader 只用它支持的参数
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # ✅ 新 AMP API（torch 2.5）
    scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type == "cuda"))

    print("[TRAIN] start custom loop")

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
                torch.nn.utils.clip_grad_norm_(net_module.parameters(), CLIP_NORM)

            scaler.step(net.optimizer)
            scaler.update()

            global_step += 1
            running_loss += float(loss.detach().cpu())
            running_count += 1

            if global_step % LOG_ITER == 0:
                dt = time.time() - t0
                avg = running_loss / max(1, running_count)
                print(f"Iteration: {global_step:6d}  s/iter:{dt/LOG_ITER:.4f}  Average Loss: {avg:.6f}")
                running_loss = 0.0
                running_count = 0
                t0 = time.time()

        torch.save(net_module.state_dict(), os.path.join(OUT_DIR, f"seednet_epoch{epoch}.pt"))
        model.save_state(os.path.join(OUT_DIR, f"deepproblog_epoch{epoch}.mdl"))
        print(f"[EPOCH {epoch}] saved to {OUT_DIR}")

    print("[DONE] training finished.")
    print("[SAVE] last:", os.path.join(OUT_DIR, f"seednet_epoch{EPOCHS}.pt"))


if __name__ == "__main__":
    main()
