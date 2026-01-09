[![中关村学院 GitHub 组织](https://img.shields.io/badge/Linked%20to-bjzgcai%20Org-blue?logo=github)](https://github.com/bjzgcai)

# EddyNN-Problog framework
# Eddytest Code_test Pipeline

这套代码实现了一条从 **168×168 SSH 场** 出发，生成像素级 **facts/Truth**，再构建对象级训练样本，使用 **DeepProbLog + CNN** 学习“候选对象是否应该保留”，并最终回填得到像素级涡旋掩膜预测的完整流程。

---

## 0. 总览（一句话）

- **Step1（规则生成）**：SSH → SLA → seed/grow 阈值化 → 滞回连通域 → 面积筛选 → 输出像素级 facts + Truth `eddy_mask`
- **Step2（样本构建）**：把 grow 连通域当“候选对象”，裁 32×32 patch，用 `eddy_mask` 给对象打 keep/drop 标签 → 输出训练集 `.npz`
- **Step3（DeepProbLog 训练）**：CNN 输出“概率事实”，ProbLog 规则加入硬约束（如面积）→ 学习对象级 `keep_label`
- **Step4（推理回填）**：在测试数据上对候选对象预测 keep 概率，并涂回像素区域 → 得到像素级预测 `eddy_mask_pred` + 指标/可视化

---

## 1. 数据与文件约定

### 1.1 SSH 输入
- `ssh`: `(T, 168, 168)`，连续实数（海表高度）

### 1.2 像素级 facts（Step1 输出）
语义上都是二值 mask（0/1 或 True/False），保存时可能是 `float32` 但 `unique` 仍是 `[0., 1.]`：

- `facts_pos_seed_*.npy`：`SLA >= +T_seed`
- `facts_pos_grow_*.npy`：`SLA >= +T_grow`
- `facts_neg_seed_*.npy`：`SLA <= -T_seed`
- `facts_neg_grow_*.npy`：`SLA <= -T_grow`
- `facts_ocean_*.npy`：海洋掩膜（海=1，陆/无效=0）

形状均为 `(T, 168, 168)`。

### 1.3 Truth 掩膜（Step1 输出）
- `eddy_mask_*.npy`：多类掩膜（不是二值）
  - `0`：非涡
  - `1`：负异常涡（neg）
  - `2`：正异常涡（pos）

形状 `(T, 168, 168)`。

---

## 2. Step1：生成像素级 facts 与 Truth
**脚本：** `Code_test/Step1_compute_eddy_mask.py`

### 2.1 做了什么
1) （可选）裁剪原始数据到 `(T,168,168)`，尽量减少陆地影响  
2) 计算 SLA：
   - 使用 **加权高斯滤波** 估计大尺度背景（只在海洋区域有效）
   - `SLA = SSH - background`
3) seed/grow 双阈值化生成像素级事实：
   - `pos_seed / pos_grow / neg_seed / neg_grow`
4) **滞回筛选（hysteresis）**：
   - 对 grow 做连通域，仅保留 **包含 seed 像素** 的连通域
5) 面积筛选：仅保留面积在 `[MIN_AREA, MAX_AREA]` 的区域
6) 合成最终 Truth：
   - `eddy_mask`（0/1/2）

### 2.2 输出
保存到指定 `OUT_DIR` 下：
- `facts_pos_seed_*.npy`
- `facts_pos_grow_*.npy`
- `facts_neg_seed_*.npy`
- `facts_neg_grow_*.npy`
- `facts_ocean_*.npy`
- `eddy_mask_*.npy`

---

## 3. Step2：构建对象级训练样本（最关键的“桥梁”）
**脚本：** `Code_test/Step2_make_components_npz.py`

> Step2 的作用：**把像素级 grow/eddy_mask 变成“对象级样本集”**，每个样本对应一个候选对象（连通域），包含 patch + area + keep/drop 标签。

### 3.1 输入
- `ssh (T,168,168)`
- `pos_grow/neg_grow/ocean (T,168,168)`
- `eddy_mask (T,168,168)`（0/1/2）

### 3.2 做了什么（逐步）
对每个时间步 `t`、每个极性（pos/neg）分别：

1) **候选对象提取（连通域）**
   - 对 `(grow & ocean)` 做连通域标记（通常 8 邻接）
   - 每个连通域 = 一个候选对象（component）

2) **计算对象面积**
   - `area = 候选对象像素数`

3) **用 Truth 给对象打监督标签（keep/drop）**
   - 计算重叠率：  
     `overlap = (region ∩ (eddy_mask==gt_val)).sum / region.sum`
   - 若 `overlap >= OVERLAP_THR` → `label=1(keep)`
   - 否则 → `label=0(drop)`

4) **过滤“逻辑矛盾”的正样本**
   - 若 `label=1` 但 `area` 不在 `[MIN_AREA, MAX_AREA]`，直接丢弃  
   - 原因：后续 ProbLog 规则会把 `area_ok` 当硬约束，否则训练会出现 `log(0)` 导致 `inf/nan`

5) **裁剪 32×32 patch（从 SSH 场中取局部图像）**
   - 以候选对象 bbox 中心为参考，从 `ssh[t]` 裁 `PATCH×PATCH`（默认 32×32）
   - 超出边界则 padding

6) **patch 归一化**
   - 均值方差标准化（可 clip）

### 3.3 输出
一个压缩的对象级数据集：
- `components_train.npz`，包含：
  - `patches`: `(N,1,32,32)`
  - `areas`: `(N,)`
  - `labels`: `(N,)`（0/1：drop/keep）

---

## 4. Step3：DeepProbLog + CNN 训练
**脚本：**
- `Code_test/Step3_test_train_1GPU.py`（单卡）
- `Code_test/Step3_test_train_4GPU.py`（多卡 DDP）
- `Code_test/Step3_test_train_4GPU_low.py`（多卡 + 近似推理加速）
- `Code_test/Step3_train_eddyproblog_version0.py`（更朴素的版本）

### 4.1 做了什么
- 定义 CNN（SeedNet）：输入 `patch(1,32,32)` → 输出 `P(seed_present=0/1)`
- ProbLog 规则将 `seed_present`（概率事实）与 `area_ok`（硬规则）组合，得到查询 `keep_label`
- 训练目标：最大化 `P(keep_label=y)`（等价于最小化 `-log(prob)`）

### 4.2 规则文件
- `eddy_dp.pl`：在 problog 内做数值比较得到 `area_ok`
- `eddy_dp_areaok.pl`：提前把 `area_ok` 当作 0/1 输入（更稳更快，推荐）

### 4.3 输出
- `seednet_epoch*.pt`：CNN 权重
- `deepproblog_epoch*.mdl`：DeepProbLog 模型文件（若脚本保存）

---

## 5. Step4：预测 + 回填像素级掩膜
**脚本：** `Code_test/Step4_predict_eddy_mask_seednet.py`

### 5.1 做了什么
1) 在测试集读取 `ssh + grow + ocean + (gt eddy_mask)`
2) 对某个时间步：
   - 从 `(grow & ocean)` 提取候选对象（连通域）
   - 面积过滤
   - 裁 patch → seednet 输出对象 `p_keep`
3) 将 `p_keep` “涂回”对象区域，得到像素级概率图
4) 阈值化生成像素级预测 `eddy_mask_pred`
5) 计算指标（P/R/F1/IoU 等）并可视化

> 注意：当前框架是“候选生成靠 Step1 规则（grow），候选筛选靠 seednet + ProbLog”。

---

## 6. 快速运行（示例）
> 实际运行以你脚本内的路径/参数为准（建议后续用 argparse/config 统一管理）。

```bash
# Step1: 生成 facts & eddy_mask
python Code_test/Step1_compute_eddy_mask.py

# Step2: 构建对象级训练数据
python Code_test/Step2_make_components_npz.py

# Step3: 单卡训练
python Code_test/Step3_test_train_1GPU.py

# Step3：多卡训练
python Code_test/Step3_test_train_4GPU.py

# Step4: 推理与可视化
python Code_test/Step4_predict_eddy_mask_seednet.py
