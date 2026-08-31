# MoonNet / TrustMVS 研究与实现计划

## 1. 研究定位

MoonNet 研究一个刻意收敛的问题：**MonoMVSNet 在向深度假设中注入单目基础模型先验时，何时应该相信该先验？**

MonoMVSNet 已经证明，Depth Anything V2 的 reference-view 特征与相对深度能够改善低纹理、反射和边缘区域的多视图重建。代码审读显示，它的深度引导并不是整体替换搜索区间，而是：

1. 使用上一阶段 MVS 深度与 photometric confidence，将归一化单目深度线性对齐到 metric inverse depth；
2. 由 TEED 边缘阈值选择待引导像素；
3. 在这些像素处，找到当前 inverse-depth hypotheses 中最接近单目深度的候选；
4. 用对齐后的单目深度直接替换该候选。

现有决策依赖固定边缘阈值和范围合法性判断。只要一个单目候选落在当前区间并位于检测到的边缘，它就会被无条件注入；系统没有判断该先验是否比原候选更接近真实几何。

本研究提出 **Reliability-Aware Monocular Candidate Injection（可靠性感知的单目候选注入）**：学习一个轻量的逐像素 trust gate，连续控制候选从原 MVS hypothesis 向对齐单目 hypothesis 移动的幅度。

## 2. 核心假设

给定原始 inverse-depth candidate \(h\)、对齐后的单目候选 \(m\) 和可靠性 \(r\in[0,1]\)，MoonNet 使用：

\[
h' = h + r(m-h).
\]

- \(r=1\)：严格退化为原 MonoMVSNet 的候选替换；
- \(r=0\)：保留原始 MVS candidate；
- \(0<r<1\)：进行软注入，降低错误先验的破坏性。

研究假设是：**单目—多视图分歧、MVS photometric confidence、边缘响应和候选区间位置，足以预测一次单目候选注入是否有利。**

## 3. 唯一方法贡献

论文只主张一个方法贡献：

> 一个轻量、可训练、保持原 MonoMVSNet 为特例的 prior trust gate，用于控制 cascade 后续阶段的单目候选注入强度。

以下内容不作为本研究目标：

- 不重新设计 cost volume 或 regularization network；
- 不处理未知相机位姿或联合 bundle adjustment；
- 不声称 depth-range-free；
- 不引入 NeRF、SDF、3DGS 或主动补拍；
- 不修改标准深度融合方法；
- 不专项解决动态、透明或镜面场景。

## 4. Gate 输入与结构

对 cascade 的第 2–4 阶段，gate 使用四个单通道输入：

1. 对齐单目 inverse depth 与上一阶段 MVS inverse depth 的相对分歧；
2. 上一阶段 photometric confidence；
3. 当前 reference-view edge response；
4. 对齐单目 inverse depth 相对当前 inverse-depth interval center 的归一化偏移。

网络采用三层 2D convolution：`4 -> 16 -> 16 -> 1`，最终 sigmoid 输出 trust map。最后一层使用零权重和正偏置初始化，使模型初始行为接近原 MonoMVSNet，避免刚开始训练时随机破坏候选。

第 1 阶段仍保持原 MonoMVSNet 行为，因为此时尚没有上一阶段 metric MVS 证据可用于判断可靠性。

## 5. Trust 监督

对于将被替换的原 candidate \(h\)、单目 candidate \(m\) 与真值深度 \(d^*\)，构造软 oracle target：

\[
y = \sigma\left(\frac{|h-d^*|-|m-d^*|}{T\Delta d+\epsilon}\right),
\]

其中 \(\Delta d\) 是该阶段候选间隔，\(T\) 是 temperature。

- 单目候选明显更接近真值时，\(y\to1\)；
- 原 MVS 候选更接近真值时，\(y\to0\)；
- 二者相近时，target 接近 0.5。

训练目标为：

\[
L=L_{\text{MonoMVSNet}}+\lambda_{trust}L_{trust}.
\]

该监督只使用训练集 GT；推理时 gate 仅依赖可观测输入。

## 6. 代码落点

- `models/trust.py`
  - `PriorTrustGate`：预测 trust map；
  - `prior_trust_loss`：生成软 oracle target 并计算 BCE；
- `models/module.py`
  - 扩展 `schedule_inverse_range_mono`，支持软候选注入并返回监督元数据；
- `models/monomvsnet.py`
  - 在第 2–4 阶段构造 trust 输入；
  - 输出 trust map、候选与有效 mask；
  - 在 DTU/BlendedMVS loss 中加入 trust loss；
- `train_dtu.py`、`train_bld.py`
  - 增加 `--trust_mode`、`--trust_loss_weight`、`--trust_temperature`；
  - 记录 trust loss；
- 推理脚本
  - 增加 `--trust_mode`，保证 checkpoint 结构与模型配置一致。

## 7. 最小实验矩阵

### 7.1 必做对照

| 设置 | `mono_sampling` | `trust_mode` | 含义 |
|---|---:|---|---|
| MVS only | false | always | 不注入单目候选 |
| MonoMVSNet | true | always | 原始无条件候选注入 |
| MoonNet | true | learned | 学习式可靠性软注入 |

训练分析时还应离线计算 oracle gate 的上限，但不把 GT oracle 暴露到推理路径。

### 7.2 数据集

- DTU：训练、消融、标准点云评测；
- BlendedMVS：按原协议 fine-tune；
- Tanks and Temples：现实场景泛化；
- ETH3D：在核心结论成立后补充，不阻塞第一轮研究。

### 7.3 直接指标

除 DTU Accuracy / Completeness / Overall 和 T&T F-score 外，必须报告：

- candidate quantization error；
- 注入前后 GT coverage；
- trust score 与“单目候选优于原候选”事件的 ROC-AUC / calibration；
- 额外参数、峰值显存和推理延迟。

### 7.4 受控 prior failure

对单目 prior 施加可控扰动：

- scale / shift drift；
- 局部平滑；
- 边界膨胀；
- 局部错误 depth block。

这些实验应明确标为 controlled corruption，不冒充自然真实场景结果。目标是比较随 prior error 增强时，`always` 与 `learned` 的性能退化曲线。

## 8. Go / No-Go 检查点

继续完整训练前必须满足：

1. 原 MonoMVSNet checkpoint 能在本分支以 `trust_mode=always` 兼容加载；
2. `trust=1` 的采样结果与原函数数值一致；
3. `trust=0` 时被选候选保持不变；
4. trust loss 能对 gate 参数产生有限、非零梯度；
5. oracle 分析显示条件选择单目候选确有高于 always-trust 的上限。

如果第 5 项不成立，应停止扩大工程规模并重新评估研究假设。

## 9. 论文定位与边界

建议标题：

> **MoonNet: Learning to Trust Monocular Candidate Injection for Multi-View Stereo**

建议核心表述：

> Monocular foundation priors improve modern MVS, but existing candidate injection treats every valid edge prior as trustworthy. MoonNet introduces a lightweight reliability gate that preserves the original MonoMVSNet behavior as a special case while learning to suppress harmful candidate replacements.

这不是一篇“全新 MVS backbone”论文，也不以重新定义所有 SOTA 为目标。论文是否成立主要取决于：在标准设置不明显退步的同时，能否在自然域偏移与受控先验错误下表现出稳定且可解释的鲁棒性收益。

## 10. 当前可验证范围

代码阶段可以验证模块形状、退化行为、梯度、语法和 CLI；完整 DTU/T&T 数字仍需要数据集、预训练权重和 GPU 训练。任何尚未运行的 benchmark 均不得写成已取得结果。
