# AOFE / AOFE_ratio 与模型形状实验分析报告

## 1. 报告目的

本报告基于仓库中现有代码与结果文件，对你的核心猜想做一次完整的证据梳理：

1. 在固定参数量下，模型形状变化带来的 loss 差异，是否可以由 superposition 强度解释。
2. 你提出的 `AOFE = AGOP 的副对角线能量` 与 `AOFE_ratio = AOFE / ||AGOP||_F^2`，哪个更适合作为跨形状、跨任务的 loss 代理变量。
3. 是否已经得到类似 Chinchilla 的“shape law”：给定参数预算 `N` 和数据量 `D`，存在一个稳定的最优深宽比 `alpha*`。

分析对象覆盖：

- `data_scaling.py` 对应的 bottleneck data-scaling 实验
- 四个固定参数量 shape sweep
  - `experiments/cnn_shape_sweep_cifar10_agop.py`
  - `experiments/mlp_shape_sweep_supervised_pde_agop.py`
  - `experiments/rnn_shape_sweep_mackeyglass_superposition_agop.py`
  - `experiments/transformer_shape_agop.py`
- 两个多预算 Transformer shape sweep
  - `experiments/transformer_scaling_shape_sweep.py`
  - `experiments/transformer_ntp_shape_sweep.py`

## 2. 最重要的结论

### 2.1 当前结果支持的 strongest claim

在四个 teacher-student 形状扫描里，`AOFE_ratio` 与 loss 呈稳定正相关，而原始 `AOFE` 与 loss 呈稳定负相关。这说明：

- 形状改变确实系统性改变了 AGOP 的结构。
- 在这类“固定输出维度、固定 teacher、近似单因子宽度瓶颈”的任务里，`AOFE_ratio` 可以作为形状劣化的有效代理。

对应 Pearson 相关系数如下：

| 任务 | corr(`AOFE_ratio`, loss) | corr(`AOFE`, loss) | 最优形状 |
| --- | ---: | ---: | --- |
| CNN | `+0.961` | `-0.950` | depth=3, channels=128 |
| MLP | `+0.984` | `-0.926` | depth=3, width=672 |
| RNN | `+0.965` | `-0.991` | depth=3, hidden=248 |
| Transformer teacher-student | `+0.870` | `-0.880` | depth=3, d_model=160 |

这组结果非常整齐，足够支持一个局部命题：

> 在 teacher-student 型固定预算 shape sweep 中，更高的 `AOFE_ratio` 往往对应更高的 loss。

### 2.2 当前结果不支持直接写进论文的 claim

`transformer_ntp_shape_sweep.py` 的 WikiText next-token prediction 结果，并不支持“`AOFE_ratio` 越高，loss 越高”这个结论向真实语言建模任务直接迁移。

相反，在 `N >= 1.3M` 的大多数预算上：

- `corr(AOFE_ratio, test_ce) < 0`
- 而 `corr(AOFE, test_ce) > 0`

也就是说，在真实 NTP 任务里，`AOFE_ratio` 的符号和四个 teacher-student 任务相反。

这意味着：

> 目前还不能把“最小化 `AOFE_ratio` 会最小化 loss”写成普适 shape law。

### 2.3 但 NTP 实验确实给出了你最需要的“shape-optimal”证据

尽管 `AOFE_ratio` 迁移失败，NTP 实验本身非常有价值，因为它第一次清楚展示了固定参数预算下的 **内部最优形状**：

- 对 `N >= 1.0M`，最优形状几乎总在 `depth=5~6`
- 对应最优 `alpha = depth / d_model` 集中在 `0.023 ~ 0.047`
- 平均 `alpha* ≈ 0.0331`，中位数 `0.0321`

这说明你的项目已经得到了一条很有论文价值的经验规律：

> 在 TinyGPT + WikiText byte-level NTP + `D≈60N bytes` 的设定下，固定参数预算的最优形状不是最浅最宽，也不是最深最窄，而是一个稳定的中等深宽比带。

## 3. 分实验分析

### 3.1 `data_scaling.py`：原始动机成立，但更支持 AOFE 而不是 AOFE_ratio

从 `results_bottleneck_scaling/summary.npz` 读取到：

- `corr(agop_offdiag_energy_mean, test_loss_mean) = +0.970`
- `corr(agop_offdiag_energy_ratio_mean, test_loss_mean) = +0.221`

因此，这个最早的 data-scaling 结果更强地支持的是：

> loss 与原始副对角能量 `AOFE` 强相关，

而不是：

> loss 与 `AOFE_ratio` 强相关。

这点很关键，因为它和后面的 NTP 结果是相容的。

### 3.2 四个固定预算 shape sweep：结论非常整齐，但本质上还是“单调前沿”

四个任务都得到类似现象：

- 模型越深越窄，loss 越高
- `AOFE_ratio` 越高，loss 越高
- `AOFE` 越低，loss 反而越高

但更值得注意的是：

- 四个任务的最优点全部出现在扫描边界，也就是最浅/最宽的形状
- 没有出现真正的“内部最优 alpha”

因此，这四个任务目前更像是在证明：

> 形状变化会诱导 AGOP 结构变化，并且这种变化与性能退化高度耦合，

而不是在证明：

> 存在一个稳定的 Chinchilla 式 shape-optimal law。

这两者要严格区分。

### 3.3 `transformer_scaling_shape_sweep.py`：多预算 teacher-student 仍然是单调的

这个多预算 teacher-student Transformer 实验对 `AOFE_ratio` 很友好：

| 参数预算 | corr(`AOFE_ratio`, test_mse) | 最优形状 |
| --- | ---: | --- |
| 0.3M | `+0.796` | depth=1, d_model=148 |
| 1.0M | `+0.934` | depth=1, d_model=280 |
| 3.0M | `+0.998` | depth=1, d_model=492 |

但它的问题也很明显：

- 最优点始终落在最浅最宽边界
- 没有出现真实语言任务里那种“太浅也不好”的反转

所以它更适合作为“机制验证”，不适合作为最终 shape law 的主证据。

### 3.4 `transformer_ntp_shape_sweep.py`：最关键，也最复杂

这是整个项目里最重要的一组结果。

#### 3.4.1 是否存在内部最优形状？

存在。除 `N=0.3M` 外，其余预算都出现内部最优点：

| 参数预算 | 最优 shape | `alpha*` | `test_ce` |
| --- | --- | ---: | ---: |
| 0.3M | depth=1, d=128 | `0.0078` | `2.2186` |
| 0.6M | depth=6, d=84 | `0.0714` | `1.7342` |
| 1.0M | depth=5, d=120 | `0.0417` | `1.4459` |
| 1.3M | depth=6, d=128 | `0.0469` | `1.3487` |
| 1.6M | depth=5, d=156 | `0.0321` | `1.2909` |
| 2.0M | depth=5, d=176 | `0.0284` | `1.2340` |
| 2.3M | depth=6, d=172 | `0.0349` | `1.2138` |
| 2.7M | depth=5, d=204 | `0.0245` | `1.1847` |
| 3.0M | depth=5, d=216 | `0.0231` | `1.1598` |

对 `N >= 1.0M`：

- 最优深度只在 `5` 或 `6`
- 最优 `alpha*` 落在 `0.0231 ~ 0.0469`
- 2% 最优损失带的公共交集约为 `alpha ∈ [0.0204, 0.0645]`

这是目前最接近 Chinchilla 风格结论的一条结果。

#### 3.4.2 最优形状带来的收益有多大？

对 `N >= 1.0M`，与极端形状相比：

- 相比最浅的 `depth=1` 宽模型，平均 `test_ce` 改善约 `10.5%`
- 相比最深的 `depth=24` 窄模型，平均 `test_ce` 改善约 `6.6%`

这说明最优形状带来的收益不是噪声级别，而是实质性的。

#### 3.4.3 `AOFE_ratio` 是否仍然是有效代理？

不是。NTP 结果给出了反例。

各预算下 `corr(AOFE_ratio, test_ce)`：

- `0.3M: -0.482`
- `0.6M: +0.513`
- `1.0M: +0.298`
- `1.3M: -0.639`
- `1.6M: -0.593`
- `2.0M: -0.722`
- `2.3M: -0.840`
- `2.7M: -0.762`
- `3.0M: -0.836`

而且在 `N >= 1.3M` 上，这个负相关在统计上已经明显成立。

更直接的反例是：

- 从 `1.3M` 到 `3.0M`，`AOFE_ratio` 最小的模型几乎总是 `depth=1`
- 但最小 loss 的模型稳定出现在 `depth=5~6`

所以在真实 NTP 任务里：

> “最小 `AOFE_ratio`”并不等于“最小 loss”。

#### 3.4.4 NTP 里反而谁更像 loss 代理？

在中高预算上，原始 `AOFE` 的符号更接近你的最初 data-scaling 发现：

- `2.0M: corr(AOFE, test_ce) = +0.805`
- `2.7M: corr(AOFE, test_ce) = +0.708`
- `3.0M: corr(AOFE, test_ce) = +0.820`

这说明对真实语言建模来说，更稳定的可能不是 `AOFE_ratio`，而是：

- 原始 `AOFE`
- 或者 `AOFE` 与对角能量分开建模
- 或者某种比 `AOFE_ratio` 更稳健的归一化

这里我做一个明确的推断：

> `AOFE_ratio` 在 NTP 上失效，可能是因为它把“副对角干扰”和“对角主导强度”的共同变化压缩进了一个分母，导致符号翻转。

这是根据结果做出的推断，不是代码中直接验证过的事实。

## 4. 目前最适合写进论文的结论

### 4.1 可以稳妥主张的部分

1. 形状改变并不会在固定参数量下完全“无代价地互补”；不同深宽比会系统性改变 AGOP 结构与最终 loss。
2. 在多个 teacher-student 任务中，`AOFE_ratio` 与 loss 呈稳定正相关，说明 feature coupling / gradient coupling 与性能退化相关。
3. 在真实语言建模任务中，固定 `N` 存在稳定的中等深宽比最优带，而不是“越宽越好”或“越深越好”。
4. 对 TinyGPT + WikiText + `D≈60N bytes`，经验上推荐的最优带是 `alpha* ≈ 0.02 ~ 0.05`，通常对应 `depth=5~6`。

### 4.2 暂时不要写死的部分

1. 不要写成“`AOFE_ratio` 是普适的 loss 代理”。
2. 不要写成“最小 `AOFE_ratio` 的形状就是最优形状”。
3. 不要写成“scaling law 中 loss 只与参数量有关，与形状无关”。你现在的数据恰恰说明：在固定参数量下，形状仍然显著影响 loss。

### 4.3 更安全、更强的论文表述

我建议把主论断改成：

> 在固定参数量下，模型形状通过改变梯度耦合结构而影响优化与泛化；这种影响在 teacher-student 任务中可由 `AOFE_ratio` 表征，而在真实语言建模中则体现为一个稳定的 shape-optimal 中等深宽比带。

这样既保留了你的核心思想，也和现有结果更一致。

## 5. 对“为什么 scaling law 里 loss 主要看参数量而不显式看形状”的回答

基于你当前项目，更合理的解释不是“形状完全无关”，而是：

1. 参数量决定了可达到性能的大尺度上界。
2. 形状决定了达到这个上界时的表示耦合方式、优化难度与分层特征利用效率。
3. 当 shape 选在一个较优带内时，不同形状的损失差距会被压缩，于是宏观 scaling law 更容易只显式依赖参数量。
4. 一旦离开这个最优带，固定参数量下的 loss 会明显恶化，这说明 shape 只是被平均掉了，不是不存在。

这个版本和你的直觉是一致的，也更贴近当前实证。

## 6. 接下来最值得补的实验

如果你的目标是把这篇工作写成一篇更有说服力的 shape law 论文，我建议优先补下面四件事：

1. 在 NTP 上同时报告 `AOFE`、`diag_energy`、`||AGOP||_F^2`，不要只报告 `AOFE_ratio`。
2. 对 `alpha ∈ [0.02, 0.05]` 做更细密扫描，确认 `alpha*` 是否真的稳定在这一带。
3. 在每个 `N` 上增加多个随机种子，给 `alpha*` 和相关系数加误差条。
4. 拟合一个显式 shape law，例如  
   `L(N, alpha) = L_inf + a N^{-beta} + b (log alpha - log alpha*(N))^2`，
   再看 `alpha*(N)` 是否近似常数或弱幂律。

## 7. 总结

如果只看四个 teacher-student 任务，你的故事会是“`AOFE_ratio` 越高，loss 越高”；  
但把 NTP 也纳入后，更完整的故事变成：

- teacher-student 任务支持 `AOFE_ratio` 作为形状劣化代理；
- 真实语言建模支持“存在稳定最优深宽比带”；
- 但 **NTP 不支持把 `AOFE_ratio` 直接当成普适 loss 代理**。

因此，这个项目现在最强的论文卖点，其实是：

> 你找到了固定参数预算下的 shape-optimal 带，并证明形状会通过梯度耦合结构显著影响 loss；但不同任务下最合适的 coupling 指标并不完全相同。

## 8. 参考文献

- NFA / AGOP 理论来源：[arXiv:2212.13881](https://arxiv.org/abs/2212.13881)
- Chinchilla: [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- 你提到的“损失来自特征干扰”相关工作：[arXiv:2505.10465](https://arxiv.org/abs/2505.10465)
