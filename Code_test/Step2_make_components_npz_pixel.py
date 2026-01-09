# Step2_make_components_npz_pixel.py
# 生成像素级训练集：x=(T,1,168,168) + y=(T,168,168) (0/1/2, land=ignore)
# 修正：当 ssh 是 (T,240,280) 而 mask/ocean 是 (T,168,168) 时自动裁剪 ssh

import os
import numpy as np

# =========================
# 可调参数（只要你 Step1 裁剪窗口一致就行）
# =========================
IGNORE_INDEX = 255     # 训练时忽略陆地像素
CLIP = 5.0
EPS = 1e-6

# ---- 裁剪参数：必须与 Step1 的裁剪一致 ----
Y0 = 25               # 常见：25
USE_RIGHT_CROP = True # True: x0 = W - target_w
X0_FIXED = None       # 如果你想固定 x0，比如 112，就设为 112；否则用右侧裁剪/居中裁剪


def normalize_frame(x2d, eps=EPS, clip=CLIP):
    """对单帧做标准化（均值方差）+ clip，风格与 Step2 patch 类似"""
    x2d = x2d.astype(np.float32)
    m = x2d.mean()
    s = x2d.std()
    if (not np.isfinite(s)) or s < eps:
        s = 1.0
    x2d = (x2d - m) / (s + eps)
    x2d = np.clip(x2d, -clip, clip)
    return x2d.astype(np.float32)


def crop_ssh_to_match(ssh, target_h, target_w, y0=Y0, use_right_crop=USE_RIGHT_CROP, x0_fixed=X0_FIXED):
    """
    ssh: (T,H,W) -> (T,target_h,target_w)
    默认：y0 固定，x0 = W - target_w（右侧裁剪）
    """
    T, H, W = ssh.shape
    if H < target_h or W < target_w:
        raise ValueError(f"SSH too small to crop: {ssh.shape} -> {(target_h, target_w)}")

    if x0_fixed is not None:
        x0 = int(x0_fixed)
    else:
        if use_right_crop:
            x0 = W - target_w
        else:
            x0 = (W - target_w) // 2

    y1 = y0 + target_h
    x1 = x0 + target_w

    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        raise ValueError(f"Bad crop window y[{y0}:{y1}] x[{x0}:{x1}] for SSH shape {ssh.shape}")

    ssh_crop = ssh[:, y0:y1, x0:x1].copy()
    return ssh_crop, (y0, y1, x0, x1)


def build_pixel_dataset(
    ssh_npy,
    ocean_npy,
    eddy_mask_npy,
    out_npz,
    use_float16=True,          # x 用 float16 大幅节省空间
    apply_ocean_ignore=True,   # ocean==0 的像素 label 设为 IGNORE_INDEX
):
    # ---- load ----
    ssh = np.load(ssh_npy).astype(np.float32)       # (T,H,W) 可能是 (4018,240,280)
    ocean = np.load(ocean_npy).astype(np.uint8)     # (T,168,168)
    y = np.load(eddy_mask_npy).astype(np.uint8)     # (T,168,168) values {0,1,2}

    # ---- clean nan/inf ----
    ssh = np.nan_to_num(ssh, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ---- crop ssh if needed ----
    if ssh.shape[1:] != y.shape[1:]:
        target_h, target_w = y.shape[1], y.shape[2]
        ssh, win = crop_ssh_to_match(ssh, target_h, target_w)
        print(f"[INFO] Cropped SSH to match mask: y[{win[0]}:{win[1]}], x[{win[2]}:{win[3]}] -> {ssh.shape}")

    # ---- shape check ----
    assert ssh.shape == y.shape == ocean.shape, f"shape mismatch: {ssh.shape}, {y.shape}, {ocean.shape}"

    T, H, W = ssh.shape
    print(f"[INFO] Using T={T}, H={H}, W={W}")

    # ---- normalize x ----
    x = np.empty((T, 1, H, W), dtype=np.float32)
    for t in range(T):
        x[t, 0] = normalize_frame(ssh[t])

    # ---- apply ignore on land ----
    if apply_ocean_ignore:
        y2 = y.copy()
        y2[ocean == 0] = IGNORE_INDEX
        y = y2

    # ---- stats (ocean only) ----
    valid = (y != IGNORE_INDEX)
    flat = y[valid].reshape(-1)
    counts = np.bincount(flat, minlength=3)
    total = counts.sum()
    print("[INFO] Pixel label counts (ocean only):", counts, "total:", int(total))
    if total > 0:
        print("[INFO] Pixel label ratio:", (counts / total))

    # ---- save ----
    x_save = x.astype(np.float16) if use_float16 else x.astype(np.float32)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        x=x_save,                           # (T,1,168,168)
        y=y.astype(np.uint8),               # (T,168,168) with IGNORE_INDEX on land
        ocean=ocean.astype(np.uint8),
        ignore_index=np.array([IGNORE_INDEX], dtype=np.uint16),
    )
    print("[OK] saved:", out_npz)
    print("[OK] x:", x_save.shape, x_save.dtype, "| y:", y.shape, y.dtype, "| ocean:", ocean.shape, ocean.dtype)


if __name__ == "__main__":
    # 按你的路径改
    build_pixel_dataset(
        ssh_npy=r"/ybz/ybz/Eddytest/data/trainAVISO-SSH_2000-2010.npy",
        ocean_npy=r"/ybz/ybz/Eddytest/data/facts/facts_ocean_2000-2010.npy",
        eddy_mask_npy=r"/ybz/ybz/Eddytest/data/eddy_mask_2000-2010.npy",
        out_npz=r"/ybz/ybz/Eddytest/data/pretrain/pixel_train_2000-2010.npz",
        use_float16=True,
        apply_ocean_ignore=True,
    )
