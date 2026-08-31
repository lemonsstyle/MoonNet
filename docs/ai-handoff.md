# MoonNet 研究项目交接说明

请继续 MoonNet 项目的研究、代码验证和实验准备。开始前先阅读工作区的 `AGENTS.md`，遵守其中的权限、Python 环境、验证和 Git 规则。

## 一、仓库与分支

本地仓库：

`/Users/lemonstyle/Documents/ali_sync_mac/玩耍/三维研究/MoonNet`

远程仓库：

https://github.com/lemonsstyle/MoonNet

当前工作分支：

`flex`

当前提交：

`1087ad74a95581b6334d404a283ae1ed858a97f3`

提交信息：

`Add reliability-gated monocular candidate injection`

远程关系：

- `origin`：`git@github.com:lemonsstyle/MoonNet.git`
- `upstream`：`https://github.com/JianfeiJ/MonoMVSNet.git`

开始工作前请确认：

```bash
git status --short --branch
git branch -vv
git remote -v
git log -3 --oneline --decorate
```

不要覆盖或回退现有 `flex` 分支的提交。

## 二、研究背景

MoonNet 基于 ICCV 2025 的 MonoMVSNet：

https://github.com/JianfeiJ/MonoMVSNet

原 MonoMVSNet 使用 Depth Anything V2 的 reference-view 单目特征和相对深度帮助多视图重建。

代码审读发现，MonoMVSNet 的单目深度引导并不是整体重设深度搜索中心或深度范围。实际逻辑是：

1. 根据上一阶段的 MVS depth 和 photometric confidence，将归一化单目深度线性对齐到 metric inverse depth；
2. 通过 TEED 边缘阈值选择待引导像素；
3. 在这些像素上，找到当前 inverse-depth hypotheses 中最接近单目深度的候选；
4. 使用对齐后的单目深度直接替换该候选。

现有方法只进行边缘、有效范围和 confidence alignment 判断，没有显式判断这个单目候选是否真的比原 MVS candidate 更可靠。

## 三、核心研究问题

研究问题刻意保持单一：

> MonoMVSNet 在向深度假设中注入对齐后的单目候选时，什么时候应该相信该候选？

方法定位：

> Reliability-Aware Monocular Candidate Injection  
> 可靠性感知的单目候选注入

暂定论文标题：

> MoonNet: Learning to Trust Monocular Candidate Injection for Multi-View Stereo

不要把研究扩大成：

- 新的通用 MVS backbone；
- pose 与 geometry 联合优化；
- depth-range-free reconstruction；
- 3DGS、NeRF 或 SDF；
- 动态场景；
- 主动补拍；
- 大场景分块；
- 新的点云融合方法。

当前论文只研究单目候选注入可靠性。

## 四、核心方法

对于原始 MVS inverse-depth candidate `h`、对齐后的单目 candidate `m` 和预测的 trust score `r`：

\[
h_{\text{new}} = h + r(m-h), \qquad r\in[0,1]
\]

含义：

- `r = 1`：退化为原 MonoMVSNet 的无条件候选替换；
- `r = 0`：保留原 MVS candidate；
- `0 < r < 1`：进行软候选注入。

Gate 使用四个输入：

1. 对齐单目 inverse depth 与上一阶段 MVS inverse depth 的相对分歧；
2. 上一阶段 photometric confidence；
3. TEED reference edge response；
4. 单目 candidate 相对当前 inverse-depth interval center 的归一化偏移。

Gate 是轻量 2D CNN：

```text
4 → 16 → 16 → 1 → sigmoid
```

最终层使用零权重和正 bias 初始化，使初始 trust 约为 `0.9`，让训练初期接近原 MonoMVSNet。

第一个 cascade stage 保留原始 MonoMVSNet 逻辑，因为此时还没有上一阶段 metric MVS 证据。Gate 只应用于第 2～4 阶段。

## 五、Trust 监督

训练时根据“单目 candidate 和原 candidate 谁更接近 GT”构造 soft oracle target：

\[
y =
\sigma\left(
\frac{|h-d^*|-|m-d^*|}
{T\Delta d+\epsilon}
\right)
\]

其中：

- `d*`：GT depth；
- `Δd`：当前阶段 candidate interval；
- `T`：temperature。

总损失：

\[
L =
L_{\text{MonoMVSNet}}
+
\lambda_{\text{trust}}L_{\text{trust}}
\]

推理时不使用 GT，只使用 gate 的可观测输入。

## 六、已经完成的代码

### 1. 研究方案

`docs/research-plan.md`

包含：

- 研究问题；
- 方法边界；
- 数学定义；
- 代码落点；
- 最小实验矩阵；
- Go/No-Go 条件；
- 论文定位。

请先完整阅读该文件。

### 2. Trust gate

`models/trust.py`

包含：

- `PriorTrustGate`
- `prior_trust_loss`
- 输入 resize/channel helper

### 3. 候选软注入

`models/module.py`

修改了：

`schedule_inverse_range_mono`

实现：

- `trust=None` 保留原 MonoMVSNet candidate replacement；
- `trust=1` 应与原始候选替换一致；
- `trust=0` 应保留标准 MVS hypotheses；
- 使用向量化 `scatter`，避免逐 batch 原地写入；
- 返回 trust supervision 所需 metadata：
  - `trust_score`
  - `trust_valid_mask`
  - `mono_candidate_depth`
  - `base_candidate_depth`

### 4. 主网络接入

`models/monomvsnet.py`

新增：

- `trust_mode='always'`
- `trust_mode='learned'`
- `trust_initial`
- 第 2～4 阶段 gate 接入；
- DTU 和 BlendedMVS trust loss；
- trust metadata 输出。

### 5. 训练入口

已修改：

- `train_dtu.py`
- `train_bld.py`

新增参数：

```text
--trust_mode always|learned
--trust_loss_weight
--trust_temperature
--disable_mono_sampling
```

实验含义：

```text
--disable_mono_sampling --trust_mode always
```

表示 MVS-only sampling ablation。

```text
--trust_mode always
```

表示原 MonoMVSNet 无条件候选注入。

```text
--trust_mode learned
```

表示 MoonNet 学习式可靠性门控。

### 6. 推理入口

已修改：

- `test_dtu_dypcd.py`
- `test_dtu_pcd.py`
- `test_dypcd_tnt_inter.py`
- `test_dypcd_tnt_adv.py`

均支持 `--trust_mode` 和 `--disable_mono_sampling`。

### 7. 测试

`tests/test_trust.py`

已经编写以下测试：

- gate 初始输出；
- gate 梯度；
- trust loss 有限梯度；
- `trust=1` 与原 MonoMVSNet 替换路径一致；
- `trust=0` 与标准 MVS hypotheses 一致；
- metadata shape 和 mask。

### 8. README

`README.md` 已完整改写为 MoonNet 研究说明，包含：

- 方法介绍；
- 安装；
- 权重准备；
- 训练与测试；
- 消融参数；
- 验证状态；
- 尚未完成的 benchmark；
- 原 MonoMVSNet 引用。

不要把 README 中引用的 MonoMVSNet 原论文数字写成 MoonNet 已复现或已超越的结果。

## 七、已完成的自检

以下检查已经运行并通过：

```bash
python -m compileall -q models train_dtu.py train_bld.py \
  test_dtu_dypcd.py test_dtu_pcd.py \
  test_dypcd_tnt_inter.py test_dypcd_tnt_adv.py tests
```

结果：通过。

以下检查已经运行并通过：

```bash
bash -n scripts/train_dtu.sh scripts/train_bld.sh \
  scripts/test_dtu_dypcd.sh scripts/test_dtu_pcd.sh \
  scripts/test_tnt_inter.sh scripts/test_tnt_adv.sh
```

结果：通过。

以下检查已经运行并通过：

```bash
git diff --check
```

结果：通过。

另外修复了原仓库两个 DTU 测试脚本中的变量错误：

```text
$DTU_LOG_DIRm
```

已改为：

```text
$DTU_LOG_DIR
```

## 八、当前未完成和已知限制

动态 PyTorch 单元测试实际运行过，但失败于环境依赖：

```text
ModuleNotFoundError: No module named 'torch'
```

用户指定的 Python 环境是：

```bash
conda activate tb
```

但该环境当前没有安装 PyTorch。

不要未经用户明确批准安装依赖。`AGENTS.md` 要求安装包前必须获得用户许可。

上游 `requirements.txt` 还存在环境问题：

- `torch==1.13.1+cu116`
- `torchvision==0.14.1+cu116`
- `torchaudio==0.13.1+rocm5.2`

CUDA PyTorch 与 ROCm torchaudio 混用，而且不适用于当前 macOS 环境。不要直接盲目执行整个 requirements 安装。

当前尚未验证：

- PyTorch tensor shape；
- trust loss 反向传播；
- `scatter` 注入的真实梯度；
- 官方 checkpoint 兼容加载；
- DTU 数据加载；
- 完整 DTU 训练；
- BlendedMVS fine-tune；
- DTU 点云指标；
- Tanks and Temples 指标；
- 运行时间和显存；
- MoonNet 是否真的优于 MonoMVSNet。

不得宣称这些项目已经通过。

## 九、下一步任务，按优先级执行

### P0：确认环境和获得安装许可

先检查：

```bash
conda activate tb
python --version
conda list | rg 'torch|torchvision|torchaudio'
nvidia-smi
```

如果没有 NVIDIA GPU，不要安装 CUDA wheel。

在安装任何包前，先向用户说明：

- 当前硬件；
- 推荐的 PyTorch 版本；
- 为什么不能直接使用上游 requirements；
- 准备安装哪些包。

获得明确许可后再安装。

### P1：运行现有单元测试

环境准备完成后运行：

```bash
conda activate tb
python -m unittest tests.test_trust -v
```

必须确认：

1. `trust=1` 与原替换路径数值一致；
2. `trust=0` 与未替换 hypotheses 数值一致；
3. gate 参数存在有限、非零梯度；
4. trust loss 存在有限、非零梯度；
5. metadata shape 正确。

如果测试失败，先修复，不要开始训练。

### P2：做一次最小 synthetic forward/backward

不要立刻跑完整 DTU。

先构造小尺寸 synthetic tensors，只运行：

- `PriorTrustGate`
- `schedule_inverse_range_mono`
- `prior_trust_loss`
- `loss.backward()`

检查：

```text
torch.isfinite(loss)
torch.isfinite(grad)
trust_score range
hypothesis ordering
candidate depth positivity
```

尤其检查软注入后 inverse-depth hypotheses 是否仍满足后续网络的排序假设。

### P3：检查 checkpoint 兼容性

下载或由用户提供官方 MonoMVSNet checkpoint 和：

- Depth Anything V2 ViT-S 权重；
- TEED 权重。

验证：

1. `trust_mode=always` 严格加载官方 MonoMVSNet checkpoint；
2. `trust_mode=learned` 非严格加载官方 checkpoint；
3. 非严格加载时，missing keys 只能属于 `prior_trust_gate`；
4. 不允许静默忽略其他 unexpected/missing keys。

如果当前 loader 只是打印 `_IncompatibleKeys`，建议改成显式白名单检查。

### P4：先做 Oracle 分析，不要直接完整训练

这是最重要的研究 Go/No-Go 实验。

在 DTU validation 上比较：

- 原始 candidate；
- aligned monocular candidate；
- GT oracle 根据哪个 candidate 更接近 GT 进行选择。

必须计算：

- 原 candidate error；
- mono candidate error；
- always-injection error；
- oracle-selection error；
- 单目优于原 candidate 的像素比例；
- 按 edge、confidence、texture 分组的比例。

判断标准：

- 如果 oracle 相比 always-injection 没有明显优势，说明可靠性选择没有研究空间，应停止扩大工程；
- 如果 oracle 明显更好，再训练 learned gate。

### P5：训练最小 Gate

第一轮不要全网络联合训练。

建议：

1. 加载 MonoMVSNet checkpoint；
2. 冻结 MonoMVSNet 主干；
3. 只训练 `PriorTrustGate`；
4. 验证 learned gate 能否接近 oracle；
5. 再决定是否解冻 cascade fine stages。

记录：

- trust loss；
- gate 平均值和分布；
- 正负 oracle 样本比例；
- ROC-AUC；
- gate calibration；
- candidate quantization error；
- GT candidate coverage。

### P6：最小对照实验

只做三个主设置：

| 设置 | 参数 |
|---|---|
| MVS-only | `--disable_mono_sampling --trust_mode always` |
| MonoMVSNet | `--trust_mode always` |
| MoonNet | `--trust_mode learned` |

必须保证：

- 相同 views；
- 相同输入分辨率；
- 相同 depth candidates；
- 相同 checkpoint 初始化；
- 相同训练 epochs；
- 相同 fusion；
- 相同置信度阈值。

### P7：标准实验

研究假设通过后再运行：

1. DTU standard evaluation；
2. BlendedMVS fine-tune；
3. Tanks and Temples；
4. controlled prior corruption；
5. 速度、参数量和显存。

不要一开始加入 ETH3D、3DGS 或新的 backbone。

## 十、需要重点审查的潜在代码风险

请优先审查：

1. `schedule_inverse_range_mono` 的 `scatter` 是否保持正确的 candidate ordering；
2. soft injection 是否可能使相邻 inverse-depth hypotheses 交叉；
3. `mono_candidate_depth` 是否在 edge mask 外产生极大值，虽然 loss mask 应排除这些位置；
4. `candidate_interval` 是否应该使用注入前 interval，而不是注入后的前两个 hypotheses；
5. `prior_trust_loss` 的 soft target 是否存在严重类别不平衡；
6. `trust_score` resize 是否与 candidate 注入的实际低分辨率位置一致；
7. BatchNorm 在 gate 只训练、小 batch 时是否稳定，是否应换成 GroupNorm；
8. 官方 checkpoint 加载是否只忽略 gate keys；
9. 第一个 stage 保留无条件注入，是否会掩盖后续 gate 的收益；
10. `trust=1` 是否真的逐元素复现原代码，而不只是大致相近。

发现问题后先写失败测试，再修改实现。

## 十一、Git 要求

每次修改后至少运行：

```bash
git diff --check
python -m compileall -q models tests train_dtu.py train_bld.py
python -m unittest tests.test_trust -v
```

如果更改 shell 脚本，再运行：

```bash
bash -n scripts/*.sh
```

不要修改 `main`。继续在 `flex` 分支工作。

提交和推送前先向用户说明准备提交的内容。除非用户已经在当前任务中明确授权，否则不要自行 commit 或 push。

最终报告必须区分：

- 已静态检查；
- 已动态测试；
- 已使用真实数据验证；
- 尚未验证。

禁止把计划结果写成已取得结果。

## 十二、给下一位 AI 的首要任务

请先不要扩展研究范围，也不要直接开始完整训练。第一目标是配置可用的 PyTorch 环境、跑通现有单元测试，并完成 DTU validation 上的 oracle candidate 分析。

Oracle 没有明显上限，就及时停止这个方向；Oracle 成立后才训练 learned gate。

交接过程中始终把“事实”和“计划”分开：

- 已完成：代码、文档、静态检查、Git 推送；
- 未完成：动态测试、权重兼容、真实数据实验；
- 下一步判断点：Oracle 是否证明可靠性选择有价值。

不要误以为当前代码已经跑通或已经取得论文结果。
