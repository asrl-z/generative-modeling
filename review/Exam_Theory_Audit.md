# 生成模型基础：理论笔记检查与考前补充

核查日期：2026 年 11 月 19 日。检查对象：`review/Review on Theories.md`。

**结论：已有六个条目基本正确，主要问题是覆盖不足；复杂度与实际运行效率的关系需要加限定。** 如果这份文件仅用于积累论述题答案，它是一个不错的起点；如果将它作为整门课的理论复习提纲，则仍缺多项往年题、教师提示和今年作业中的核心推导。

原笔记未修改。本稿先列检查结果与优先级，再提供可以补入笔记的内容和证明骨架。完整性评价仅针对这份文件，不表示你没有在其他笔记或作业中学过这些内容。

## 1. 依据与使用方法

- 逐条检查原笔记，对照 2024 年卷全部题干、2025 年卷前 4 页题干及其后附答案。
- 浏览七份课件的页级内容结构，核对相关章节，并对含图片公式、算法和疑似错误的关键页做视觉核查。
- 阅读五次作业题干，因为老师的摘要明确要求结合 PPT 与作业复习。
- 教师提示依据《录播考点与出题信息核查》中的 E01–E23。该文件说明引文是自动识别后整理的摘要，并非官方逐字稿；本次没有重新核听原录播。
- 页码均指 **PDF 文件页序，从 1 开始**。来源缩写见文末。

| 优先级 | 含义 | 建议 |
|---|---|---|
| A | 教师明确强调、往年重复出现或今年核心作业题 | 优先做到能独立解释、计算或推导 |
| B | 课件重要补充或作业阅读题，未获单项必考承诺 | A 完成后补；至少能说清机制 |
| C | 明确排除或仅作扩展 | 考前不投入繁长推导时间 |

优先级不是命题概率。**没有证据保证每个 A 项都会出现，也没有证据确认今年一定是五道大题。**

## 2. 原笔记逐条检查

| 原位置 | 判断 | 修改或补充建议 |
|---|---|---|
| VAE 重参数化，第 7–11 行 | 核心准确 | 多维形式写为 $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$，对应对角协方差。你写“不能直接通过普通抽样操作”很准确，不应扩大成所有随机采样都无法求梯度。补 log-variance 版本及 ELBO。 |
| 位置编码，第 17 行 | 方向正确，但不足以回答完整的 2025 年题 | 补无位置编码、无掩码 self-attention 的“置换等变性”，以及一种编码实现；还要解释 diffusion 的 time embedding。 |
| RNN/Transformer 复杂度，第 21–25 行 | 固定其他维度、普通 AR 生成、Transformer 使用 KV cache 时正确 | “因此训练/推理效率更高”过于绝对。区分总运算量与实际耗时；补含 $n,d$ 的复杂度，2025 年明确这样问。 |
| GAN 训练推理，第 31–43 行 | 原始 minimax GAN 的正确答案 | 原目标无需删改。另补 non-saturating loss，说明它是替代的生成器目标；补最优判别器、JS、f-divergence 和 Wasserstein。固定 D 参数时仍须让梯度经过 D 传给 G。 |
| DDPM 训练推理，第 49–81 行 | 公式、采样方向和最后一步不加噪声均正确 | 你的 $\alpha_t$ 是累计乘积，与本课 PPT、作业 4 一致。补单步正向过程、边缘分布证明、time embedding，以及 $L_{\rm simple}$ 与完整负 ELBO 的区别。 |
| 三阶段训练比较，第 87–91 行 | 对本课 AR 语言模型的比较基本完整 | 单问“什么是 Fine-tuning”时，先给总定义。补 REINFORCE、baseline、旧样本复用和 PPO；只有三阶段比较不足以覆盖该模块。 |

### 建议替换的两句话

**训练效率：** 在相同层数、隐藏维度等比较条件下，Transformer 虽然自注意力总运算量关于序列长度为二次，但 teacher forcing 时各位置可并行，通常更能利用 GPU；RNN 存在跨时间步的串行依赖。因此本课讨论的典型设置中 Transformer 训练更高效，实际耗时仍依赖模型规模、序列长度、硬件与实现。

**推理效率：** 普通自回归生成时二者都按 token 依次生成；固定其他维度，RNN 总运算量为 $O(n)$，使用 KV cache 的标准稠密 Transformer 为 $O(n^2)$，所以 RNN 在序列长度增长时的推理运算量更有优势。这不等于任何实际 RNN 都比任何 Transformer 快。

若不使用 KV cache 且每一步完整重算整个前缀，朴素 Transformer 的总运算量可达 $O(n^3)$（固定隐藏维度）。因此应保留你已写出的 cache 前提。

## 3. 缺漏与证据对照

### 3.1 即使只整理论述题，也应补充

| 知识点 | 当前覆盖 | 证据与定位 | 优先级 |
|---|---|---|---|
| VAE 定义、encoder/decoder/prior 分工、训练和生成 | 只有重参数化 | VAE p.89–95、p.102–126；R E03/E19 | A |
| ELBO 两项意义、为何用下界、何时取等 | 未写 | 24/25 Q2(a)、HW1 Q1；R E03/E04 | A |
| Self-attention 的 Q/K/V、softmax 轴与输出 | 未写 | 24 Q1(c)、25 Q1(b)、HW2 Q2；TR p.37–43 | A |
| Transformer 的 attention、FFN、残差、LN、位置编码 | 仅位置编码动机 | TR p.43–93；R E02/E07 | A |
| AR 概率分解、teacher forcing、causal mask | 仅提 teacher forcing | HW2 Q1；AR p.5–15、p.59–79 | A |
| 位置编码与 time embedding 的区别和实现 | 仅位置编码动机 | 25 Q1(a)；TR p.85–93；DF p.111–113 | A |
| RoPE 如何产生相对位置信息 | 未写 | HW2 Q3；TR p.92–93；R E18（“大概”考） | A |
| GAN 训练困难、梯度饱和与 non-saturating loss | 未写 | HW3 Q1(c)；GAN p.88–91 | A |
| Wasserstein / 最优运输的建模含义 | 未写 | **25 Q3(b) 广场舞题**；GAN p.97–110 | A |
| Fine-tuning 总定义、alignment 和奖励来源 | 部分覆盖 | FT p.24–53、p.132–136；R E02/E10 | A |
| REINFORCE → baseline → 旧样本复用 → PPO | 未写 | FT p.68–129；HW5 Q2/Q3；R E10/E14/E15 | A |
| DDPM 正向、逆向过程与可训练参数 | 训练推理已较完整 | 24/25 Q5；DF p.34–61、p.108–117 | A |
| 四类生成模型的训练/生成差异 | 未集中比较 | 各模块总结页 | B |
| Pre-LN/Post-LN、perplexity、KV cache、采样与 greedy | 仅 cache 名称 | TR p.70–79；AR p.91–116、p.138–145 | B |
| WGAN critic、Lipschitz 约束、weight clipping/GP | 未写 | GAN p.111–129 | B，先掌握 OT |
| LDM 两阶段训练与潜空间生成 | 未写 | HW4 Q3；DF p.17 | B，明确作业内容 |
| Gumbel-Softmax、FlashAttention、GigaGAN、KL 正则最优策略 | 未写 | 各次作业阅读题 | B；阅读题不能一概当作不考 |

### 3.2 若作为整门课的理论提纲，还缺这些计算/证明

| 项目 | 应掌握到的程度 | 依据 |
|---|---|---|
| ELBO | Jensen 推导、分解成重构项减 KL、取等条件 | 两年 Q2(a)、HW1 Q1，最高优先级 |
| 高斯 KL | 一般多元公式、对角高斯对标准正态的化简 | HW1 Q2 |
| log-variance 重参数化 | $z=\mu+e^{r/2}\odot\epsilon$，对 $\mu,r$ 求梯度 | HW1 Q3 |
| 注意力复杂度 | $O(nd^2)$ 投影、$O(n^2d)$ 注意力、矩阵存储 $O(n^2)$ | 25 Q1(b)、HW2 Q2 |
| LayerNorm | 每个 token 沿 embedding 维计算，方差分母用 $d$ | 24 Q1(d)、25 Q1(c)；R E17 |
| RoPE | 块旋转矩阵，证明内积的位置依赖只通过位置差进入 | HW2 Q3；R E18 |
| 最优判别器及 JS | 求 $D^*$，代回得 $2\operatorname{JSD}-2\log2$ | HW3 Q1(b)/Q2 |
| 饱和与非饱和损失 | 对 logit 求导，解释链式法则 | HW3 Q1(c) |
| KL 是 f-divergence 特例 | 选 $f(u)=u\log u$，验证凸性并代入 | 24 Q3(b) |
| 正向高斯合成、$q(x_t\mid x_0)$ | 独立高斯线性组合与归纳法 | 两年 Q5(a,b)、HW4 Q1 |
| log-derivative trick、baseline | 从概率求和推期望梯度，证明 baseline 不改期望 | HW5 Q2/Q3；R E14/E15 |
| KL 正则最优策略 | 展开 KL，得到常数减非负项和唯一最优解 | HW5 Q4，短而完整的作业证明 |

## 4. 可直接补入笔记的核心内容

### 4.1 VAE 的完整回答框架

普通 autoencoder 用确定性 encoder 压缩输入，再由 decoder 重构；仅靠重构训练一般没有给出一个方便采样、且与训练潜表示匹配的潜变量先验。VAE 引入概率模型：

$$
p(z),\quad p_\theta(x\mid z),\quad p_\theta(x)=\int p(z)p_\theta(x\mid z)\,dz.
$$

$p(z)$ 通常取 $\mathcal N(0,I)$；decoder 参数化条件分布 $p_\theta(x\mid z)$；encoder 参数化近似后验 $q_\phi(z\mid x)$，近似通常难以精确计算的真后验 $p_\theta(z\mid x)$。

**训练：** 真实样本 $x$ → encoder 输出 $\mu_\phi(x),\sigma_\phi(x)$ → 重参数化采样 $z$ → decoder → 最大化 ELBO，同时更新 encoder 和 decoder。

**生成：** 从先验 $z\sim p(z)$ 采样，再由 $p_\theta(x\mid z)$ 生成 $x$。不需要先给 encoder 一张真实图片。实现中有时直接输出 decoder 给出的条件均值，应与严格的条件分布采样区别开。

### 4.2 ELBO：定义、分解和取等条件

对固定 $x$，假设所需支持集和可积条件成立：

$$
\begin{aligned}
\log p_\theta(x)
&=\log\int q_\phi(z\mid x)\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz\\
&=\log\mathbb E_{q_\phi(z\mid x)}\left[\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]\\
&\geq\mathbb E_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]
\quad\text{（log 凹，Jensen）}\\
&=\mathbb E_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{\rm KL}(q_\phi(z\mid x)\Vert p(z))
\equiv\mathcal L(\theta,\phi;x).
\end{aligned}
$$

更关键的恒等式：

$$
\log p_\theta(x)=\mathcal L(\theta,\phi;x)
+D_{\rm KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x)).
$$

取等当且仅当 $q_\phi(z\mid x)=p_\theta(z\mid x)$ 几乎处处成立。**取等条件不是 $q_\phi(z\mid x)=p(z)$。**

- 重构项鼓励 decoder 在采到的潜变量下给真实样本较高概率。
- KL 正则项约束近似后验靠近先验，让潜空间组织与从先验采样相协调。
- 最大化 ELBO 等价于最小化负重构对数似然加 KL。
- MSE 来自特定 decoder 似然假设，例如固定方差高斯；并非所有 VAE 的重构损失都必然是 MSE。

依据：VAE p.102–126；HW1 Q1；两年 Q2(a)。

### 4.3 高斯 KL 与 log-variance 梯度

若 $q=\mathcal N(\mu_q,\Sigma_q)$、$p=\mathcal N(\mu_p,\Sigma_p)$，协方差正定：

$$
D_{\rm KL}(q\Vert p)=\frac12\left[
\operatorname{tr}(\Sigma_p^{-1}\Sigma_q)
+(\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
-d+\log\frac{\det\Sigma_p}{\det\Sigma_q}\right].
$$

从 $\mathbb E_q[\log q-\log p]$ 展开，利用：

$$
\mathbb E_q[(z-\mu_q)^T\Sigma_q^{-1}(z-\mu_q)]=d,
$$

$$
\mathbb E_q[(z-\mu_p)^T\Sigma_p^{-1}(z-\mu_p)]
=\operatorname{tr}(\Sigma_p^{-1}\Sigma_q)
+(\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p).
$$

对角 $q=\mathcal N(\mu,\operatorname{diag}(\sigma^2))$ 与标准正态 $p$ 的特例：

$$
D_{\rm KL}(q\Vert p)=\frac12\sum_j(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2).
$$

若网络输出 $r_j=\log\sigma_j^2$，**标准差为 $e^{r_j/2}$，不是 $e^{r_j}$**：

$$
z=\mu+e^{r/2}\odot\epsilon,\qquad\epsilon\sim\mathcal N(0,I).
$$

令 $F(\mu,r)=\mathbb E_\epsilon[f(z)]$，求导与期望可交换时：

$$
\nabla_\mu F=\mathbb E_\epsilon[\nabla_z f(z)],\qquad
\nabla_r F=\mathbb E_\epsilon\left[\nabla_z f(z)\odot\frac12 e^{r/2}\odot\epsilon\right].
$$

一般非对角协方差可用 $z=\mu+L\epsilon$，其中 $LL^T=\Sigma$。依据：HW1 Q2/Q3。

### 4.4 自注意力：公式、维度、缩放、复杂度

采用作业的单头同维设定 $X\in\mathbb R^{n\times d}$、$W_Q,W_K,W_V\in\mathbb R^{d\times d}$：

$$
Q=XW_Q,\ K=XW_K,\ V=XW_V,\qquad
S=\frac{QK^T}{\sqrt d},\quad A=\operatorname{softmax}_{\rm row}(S),\quad O=AV.
$$

| 量/步骤 | 维度 | 标准稠密计算量 |
|---|---|---|
| $Q,K,V$ | 各 $n\times d$ | 总阶数 $O(nd^2)$ |
| $QK^T$ | $n\times n$ | $O(n^2d)$ |
| 按行 softmax | $n\times n$ | $O(n^2)$ |
| $AV$ | $n\times d$ | $O(n^2d)$ |

总时间 $O(nd^2+n^2d)$；显式存储注意力矩阵需 $O(n^2)$，加输入输出激活为 $O(n^2+nd)$，这里不计模型参数和其他层。

$A_{ij}$ 是位置 $i$ 读取位置 $j$ 的权重，$o_i=\sum_j A_{ij}v_j$。softmax 沿 key 位置 $j$ 归一化，每行和为 1。

除以 $\sqrt{d_k}$ 的原因：在各坐标独立、零均值、单位方差的简化假设下，$q^Tk$ 方差为 $d_k$；缩放控制分数幅度，缓解 softmax 过度饱和。多头时使用**每个头的 query/key 维度 $d_k$**。

多头使用不同投影，拼接各 head 输出后作输出投影 $W_O$，可学习不同交互模式。依据：TR p.37–44、AR p.120–127、HW2 Q2。

### 4.5 AR、teacher forcing、causal mask 与 KV cache

$$
p_\theta(x_{1:n})=\prod_{t=1}^np_\theta(x_t\mid x_{<t}),\qquad
L_{\rm NLL}=-\sum_{t=1}^n\log p_\theta(x_t\mid x_{<t}).
$$

teacher forcing 使用真实前缀预测下一个 token。Transformer 可一次前向计算所有位置损失，但必须防止未来信息泄漏；RNN 即使使用 teacher forcing，隐藏状态的递归依赖仍存在。

若输入位置 $i$ 的 $x_i$ 用来预测 $x_{i+1}$，允许读取 $j\leq i$：

$$
M_{ij}=\begin{cases}0,&j\leq i,\\-\infty,&j>i,\end{cases}
\qquad A=\operatorname{softmax}_{\rm row}\left(\frac{QK^T}{\sqrt{d_k}}+M\right).
$$

mask 加在 **softmax 之前**，使被屏蔽项权重为零。是否保留对角线取决于输入与标签错位方式，应先交代约定，不能在所有约定下死记“严格下三角”。

无 mask 训练会泄露未来信息，造成虚假的低损失，推理却无法获得这些信息。推理从已有前缀的条件分布逐个采样，再送回模型。

KV cache 保存各层旧位置的 key/value，新一步仅计算新位置的 query/key/value；当前 query 仍需关注已有 keys，因此标准稠密生成总复杂度仍为二次。

依据：AR p.59–81、p.95–118、p.138–145；HW2 Q1。

### 4.6 位置编码与 time embedding

对无位置编码、无因果掩码的标准 self-attention，置换矩阵 $P$ 满足：

$$
\operatorname{Attention}(PX)=P\operatorname{Attention}(X).
$$

这是**置换等变**，即输入换序，输出随之换序；不是“输出完全不变”。逐位置共享参数的 FFN 也不自行提供位置。固定因果掩码引入顺序结构，故上述无掩码证明不能原封不动推广到所有 decoder。

| 编码 | 表示什么 | 目的 | 代表实现 |
|---|---|---|---|
| Positional encoding | token 位置/相对距离 | 区分词序与位置关系 | 正弦/余弦编码加到 token embedding |
| Time embedding | 扩散时间步，固定日程下对应噪声水平 | 让共享去噪网络针对不同噪声强度工作 | 多频率正弦/余弦映射，再经 MLP 注入残差块 |

代表性位置编码：

$$
\operatorname{PE}(i,2j)=\sin(i/10000^{2j/d}),\qquad
\operatorname{PE}(i,2j+1)=\cos(i/10000^{2j/d}).
$$

time embedding 可用

$$
e(t)=[\sin(\omega_1t),\ldots,\sin(\omega_ht),\cos(\omega_1t),\ldots,\cos(\omega_ht)],
$$

再作 MLP 和特征加法或缩放/平移调制。具体频率、坐标排列可依实现变化；DF p.112 采用先 sin 后 cos 的拼接。依据：25 Q1(a)、TR p.83–93、DF p.111–113。

### 4.7 RoPE：本次作业的 half-split 约定

令 $d=2h$，$C_t=\operatorname{diag}(\cos(t\theta))$、$S_t=\operatorname{diag}(\sin(t\theta))$，则

$$
f(x,t)=R_tx,\qquad R_t=\begin{pmatrix}C_t&-S_t\\S_t&C_t\end{pmatrix}.
$$

利用 $R_i^T=R_{-i}$、$R_aR_b=R_{a+b}$：

$$
f(q_i,i)^Tf(k_j,j)=q_i^TR_i^TR_jk_j=q_i^TR_{j-i}k_j.
$$

若要求写为 $q_i^TM(i-j)k_j$，记 $\delta=i-j$：

$$
M(\delta)=R_{-\delta}=
\begin{pmatrix}
\operatorname{diag}(\cos(\delta\theta))&\operatorname{diag}(\sin(\delta\theta))\\
-\operatorname{diag}(\sin(\delta\theta))&\operatorname{diag}(\cos(\delta\theta))
\end{pmatrix}.
$$

位置依赖只通过 $i-j$ 进入。留意位置差的符号，以及“前后半段配对”和“相邻坐标配对”的排列差别。依据：TR p.92–93、HW2 Q3、R E18。

### 4.8 LayerNorm 与 Transformer 组成

对一个 token 的 $x\in\mathbb R^d$：

$$
\mu=\frac1d\sum_jx_j,\quad v=\frac1d\sum_j(x_j-\mu)^2,\qquad
\operatorname{LN}(x)_j=\gamma_j\frac{x_j-\mu}{\sqrt{v+\varepsilon}}+\beta_j.
$$

不是跨整个序列或 batch 归一化，方差分母不是 $d-1$。两年题目取 $\varepsilon=0,\gamma=1,\beta=0$：

| 输入 | 均值 | 方差 | 输出 |
|---|---:|---:|---|
| $(2,1)$ | $3/2$ | $1/4$ | $(1,-1)$ |
| $(0,2)$ | $1$ | $1$ | $(-1,1)$ |
| $(2,0)$ | $1$ | $1$ | $(1,-1)$ |
| $(4,1)$ | $5/2$ | $9/4$ | $(1,-1)$ |

Transformer 组成的答题要点：attention 混合不同位置的信息；FFN 在每个位置独立作非线性变换且共享参数；残差有助信息与梯度传播；LayerNorm 做特征归一化；位置编码提供位置；decoder 还需 causal mask。

记子层为 $F$：Post-LN 为 $\operatorname{LN}(x+F(x))$，Pre-LN 为 $x+F(\operatorname{LN}(x))$。TR p.70–79 讨论优化稳定性差异，可作 B 类简答补充，无需据此把完整梯度定理升级成必考证明。

依据：TR p.48–79、24 Q1(d)、25 Q1(c)、R E17。

### 4.9 GAN：最优判别器、JS 与非饱和损失

你的原始 minimax 目标正确。固定 G，逐点最大化

$$
p_{\rm data}(x)\log D(x)+p_g(x)\log(1-D(x)).
$$

两密度均为正时，对 D 求导并令零：

$$
\frac{p_{\rm data}(x)}{D(x)}-\frac{p_g(x)}{1-D(x)}=0
\quad\Longrightarrow\quad
D^*(x)=\frac{p_{\rm data}(x)}{p_{\rm data}(x)+p_g(x)}.
$$

二阶导非正，确认是最大值。一方密度为零时取相应边界值；两者都为零时该点对目标无影响。

令 $m=(p_{\rm data}+p_g)/2$，代回可得：

$$
\begin{aligned}
C(G)&=\mathbb E_{p_{\rm data}}\log\frac{p_{\rm data}}{p_{\rm data}+p_g}
+\mathbb E_{p_g}\log\frac{p_g}{p_{\rm data}+p_g}\\
&=D_{\rm KL}(p_{\rm data}\Vert m)+D_{\rm KL}(p_g\Vert m)-2\log2\\
&=2\operatorname{JSD}(p_{\rm data}\Vert p_g)-2\log2.
\end{aligned}
$$

这里采用 **HW3 的标准定义**：

$$
\operatorname{JSD}(p\Vert q)=\tfrac12D_{\rm KL}(p\Vert m)+\tfrac12D_{\rm KL}(q\Vert m),\quad m=(p+q)/2.
$$

因为 KL 非负，最小值为 $-2\log2$，当且仅当 $p_g=p_{\rm data}$ 几乎处处时达到，此时在有概率质量的区域 $D^*=1/2$。这是假设判别器可充分优化、生成器分布表达能力不受限制的理论结论，不是保证实际交替训练一定收敛。

**为什么常换生成器损失？** 写 $D=\sigma(a)$，$a$ 为 logit：

$$
\ell_{\rm sat}(a)=\log(1-\sigma(a)),\quad
\frac{d\ell_{\rm sat}}{da}=-\sigma(a)=-D,
$$

$$
\ell_{\rm ns}(a)=-\log\sigma(a),\quad
\frac{d\ell_{\rm ns}}{da}=\sigma(a)-1=D-1.
$$

当假样本 $D\approx0$ 时，原损失的 logit 导数接近零；非饱和损失的 logit 导数接近 $-1$，缓解这一饱和来源。经链式法则仍需乘 $\nabla_{\theta_G}a(G_{\theta_G}(z))$，所以不能宣称它解决所有梯度问题。

**mode collapse**：不同潜变量生成少数相似输出，未覆盖真实分布的多种模式；并非“模型不再输出”。再加上非凸非凹博弈、G/D 能力不匹配、超参数敏感，是 GAN 训练困难的主要回答点。

依据：HW3 Q1/Q2；GAN p.74–91。原论文也说明了非饱和目标的动机，见文末 W2。

### 4.10 KL、f-divergence 与非负性

$$
D_f(p\Vert q)=\int q(x)f\!\left(\frac{p(x)}{q(x)}\right)dx,
$$

其中 $f$ 凸、$f(1)=0$，并满足适当支持条件。取 $f(u)=u\log u$，有 $f(1)=0$、$f''(u)=1/u>0$，代入：

$$
D_f(p\Vert q)=\int q(x)\frac{p(x)}{q(x)}\log\frac{p(x)}{q(x)}dx
=D_{\rm KL}(p\Vert q).
$$

这就是 2024 Q3(b)。再由 Jensen 得

$$
\mathbb E_q[f(p/q)]\geq f(\mathbb E_q[p/q])=f(1)=0.
$$

KL 一般不对称，不满足通常的距离公理；JSD 对称，但“对称”也不等于其本身满足三角不等式。课件若写 f-divergence 的 “linearity”，应理解为它对生成函数 f 的线性关系，不能说 KL 对分布 p、q 是线性的。

依据：GAN p.25–43；24 Q3(b)。

### 4.11 2025 广场舞题：应建模为最优运输/最小代价匹配

假设直线队形给出 $N$ 个初始位置 $a_i\in\mathbb R^2$，圆圈队形给出 $N$ 个目标位置 $b_j\in\mathbb R^2$；每人去一个不同的目标位置，题目要求平均移动距离最短。

最直接模型是排列优化：

$$
\min_{\pi\in S_N}\frac1N\sum_{i=1}^N\|a_i-b_{\pi(i)}\|_2.
$$

或写二元分配变量 $u_{ij}\in\{0,1\}$：

$$
\min_u\frac1N\sum_{i,j}u_{ij}\|a_i-b_j\|_2,
\quad\sum_j u_{ij}=1,\quad\sum_i u_{ij}=1.
$$

对应经验分布 $P=\frac1N\sum_i\delta_{a_i}$、$Q=\frac1N\sum_j\delta_{b_j}$，最优值就是以欧氏距离为运输代价的 $W_1(P,Q)$。一般写法：

$$
W_1(P,Q)=\inf_{\gamma\in\Pi(P,Q)}\mathbb E_{(X,Y)\sim\gamma}\|X-Y\|_2,
$$

其中 $\gamma$ 是边缘分布分别为 P、Q 的联合分布，即运输计划。

给分关键是 **起点、终点、移动代价、一对一约束、最小化目标**。不能让每个人独立去最近点，因为可能多人争同一个目标位置。题目只说平均距离时，不要擅自换成平方距离；若圆心、半径等并未固定，可说明上述以目标队形已给定为假设，未给定的几何参数可以再纳入优化。

依据：25 Q3(b)，GAN p.98–110。卷后附的 KL/f-divergence 答案并不回答这道题。

### 4.12 WGAN 的概念补充

在适当条件（例如有限一阶矩）下，$W_1$ 的对偶为

$$
W_1(P,Q)=\sup_{\|f\|_{\rm Lip}\leq1}\left(\mathbb E_Pf(X)-\mathbb E_Qf(X)\right).
$$

WGAN 用实值 critic 近似这个 f，优化

$$
\min_G\max_{f:\|f\|_{\rm Lip}\leq1}
\mathbb E_{p_{\rm data}}f(x)-\mathbb E_zf(G(z)).
$$

critic 输出评分，不是必须落在 $[0,1]$ 的真假概率。Lipschitz 约束限制评分变化速度，否则可单纯放大 f 让目标无界。weight clipping 限制参数范围；gradient penalty 常对插值点的 $\|\nabla_x f(x)\|_2$ 偏离 1 的程度加惩罚，它们都不等于严格保证整个空间的精确 1-Lipschitz 性。

这部分可作论述补充；优先掌握 2025 题真正用到的运输定义和建模。依据：GAN p.97–129。

### 4.13 DDPM：正向过程与边缘分布证明

**沿用你和本课的记号**：$\alpha_0=1$，$\alpha_t=\prod_{s=1}^t(1-\beta_s)$。原始 DDPM 论文常把这个累计量记作 $\bar\alpha_t$。

单步正向过程为

$$
q(x_t\mid x_{t-1})=\mathcal N\left(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right),
$$

$$
x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\epsilon_t,
\quad\epsilon_t\overset{\rm iid}{\sim}\mathcal N(0,I).
$$

固定 $x_0$，若

$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_0+\sqrt{1-\alpha_{t-1}}\,\eta,
$$

其中 $\eta$ 与本步 $\epsilon_t$ 独立，则

$$
x_t=\sqrt{\alpha_t}x_0
+\sqrt{(1-\beta_t)(1-\alpha_{t-1})}\,\eta
+\sqrt{\beta_t}\epsilon_t.
$$

后两项之和为零均值高斯，协方差是

$$
\left[(1-\beta_t)(1-\alpha_{t-1})+\beta_t\right]I
=(1-\alpha_t)I.
$$

结合 $t=1$ 得到归纳结论：

$$
q(x_t\mid x_0)=\mathcal N\left(\sqrt{\alpha_t}x_0,(1-\alpha_t)I\right).
$$

因此训练时可以直接构造任意一个 $x_t$，无需先执行 t 次加噪。**这个结论是固定 t 的条件边缘分布相同，不代表任意构造出来的所有时间步拥有相同的联合轨迹分布。**

若问“为什么线性组合仍是高斯”，对一维独立标准高斯 $\epsilon_1,\epsilon_2$，特征函数为

$$
\mathbb E e^{iu(a\epsilon_1+b\epsilon_2)}
=e^{-a^2u^2/2}e^{-b^2u^2/2}
=e^{-(a^2+b^2)u^2/2}.
$$

故其分布为 $\mathcal N(0,a^2+b^2)$；只算出均值和方差本身不能证明高斯性。

依据：DF p.34–47、HW4 Q1、两年 Q5(a,b)。这是基础正向推导，**不属于老师排除的繁长后验/ELBO 推导**。

### 4.14 DDPM 原条目还应加的解释

你原有训练、逆向更新公式可保留，再加四点即可形成较完整答案：

1. 正向日程固定，训练的是去噪网络参数。不同 t 通常共享一个网络，通过 time embedding 区分噪声强度。
2. $\alpha_T$ 很小时，$q(x_T\mid x_0)$ 近似标准正态；有限 T 下通常是近似，不是无条件严格相等。
3. $L_{\rm simple}$ 是常用噪声预测 MSE，省略了原变分目标中的时间步权重等，不能写成“与完整负 ELBO 只差一个常数”。
4. 正向训练中知道 $x_0$，所以可构造噪声监督；生成时不知道 $x_0$，用网络预测噪声并计算逆向均值。

反向方差可说明经典固定选择，例如 $\sigma_t^2=\beta_t$ 或

$$
\sigma_t^2=\widetilde\beta_t=\frac{1-\alpha_{t-1}}{1-\alpha_t}\beta_t,
$$

以题目所给设定为准，最后一步按你的写法不再添加随机噪声。不要将所有 diffusion 变体都限定为固定方差；这里只讨论课堂基础 DDPM。

依据：DF p.61、p.102–108、p.115–117；W3。后验分布的繁长配方推导不作为当前优先练习。

### 4.15 Fine-tuning 总定义与三阶段的核心区别

**Fine-tuning 是从预训练模型参数出发，利用目标任务、领域或偏好数据继续训练，使模型适应目标用途。** 可以更新全部参数，也可以只更新部分参数；并非从随机初始化重新训练。

你的三阶段比较可以保留。为防止回答跑题，最重要的对照是：

| 阶段 | 数据与监督 | 目标 | 主要作用 |
|---|---|---|---|
| 本课 AR 语言模型预训练 | 大规模文本；实际下一个 token | 最大化文本似然 | 学习通用语言规律、知识和表征 |
| 指令微调 SFT | 指令和参考回答；参考 token | 条件 next-token 交叉熵 | 学会遵循指令与回答格式 |
| 强化微调 | 模型自己生成回答；回答的奖励 | 最大化期望奖励，可加正则 | 改善任务表现和偏好一致性 |

“SFT 可视作预训练的延伸”从损失形式的角度成立，但不代表两者的数据分布、训练用途和标签组织完全相同。强化微调不要求每个 prompt 配一个唯一参考回答；需要能评价输出的奖励信号。

alignment 的课堂含义：模型输出要符合实际任务/用户期望；较低 next-token loss 不保证整段回答正确、有用或符合偏好。依据：FT p.24–53、HW5 Q1。

### 4.16 REINFORCE / log-derivative trick

固定 prompt，令 $J(\theta)=\sum_s p_\theta(s)R(s)$。假设奖励不显含 $\theta$，概率支持集不随参数变化，求导与求和可交换：

$$
\nabla_\theta\log p_\theta(s)=\frac{\nabla_\theta p_\theta(s)}{p_\theta(s)},
$$

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=\sum_sR(s)\nabla_\theta p_\theta(s)\\
&=\sum_sp_\theta(s)R(s)\nabla_\theta\log p_\theta(s)\\
&=\mathbb E_{s\sim p_\theta}[R(s)\nabla_\theta\log p_\theta(s)].
\end{aligned}
$$

用当前模型采 K 个回答，取样本均值估计梯度，再**梯度上升**更新策略。若优化器做梯度下降，可最小化

$$
L_{\rm policy}=-\frac1K\sum_{k=1}^KR(s_k)\log p_\theta(s_k).
$$

更新时把已采样序列和奖励视为固定监督信号。AR 序列有

$$
\log p_\theta(s\mid q)=\sum_t\log p_\theta(s_t\mid q,s_{<t}),
$$

因此可以用模型可计算的 token log-probability 求梯度，不需要对离散抽样操作求导，也不要求奖励函数对文本可微。

**与 VAE 的区别：** 重参数化用可微样本变换产生路径导数；REINFORCE 改写分布的导数，即使奖励是黑盒也能使用，但估计波动可能很大。若奖励本身显含参数，还会有额外的 $\mathbb E[\nabla_\theta R]$ 项；本作业明确排除了它。

依据：FT p.65–88、HW5 Q2、R E10/E14。

### 4.17 Baseline：不改期望，不保证任意选择都降方差

对固定 prompt，取不依赖当前采样回答 s 的基线 b，则

$$
\mathbb E[b\nabla_\theta\log p_\theta(s)]
=b\sum_s\nabla_\theta p_\theta(s)
=b\nabla_\theta1=0.
$$

因此

$$
\nabla_\theta J=\mathbb E[(R(s)-b)\nabla_\theta\log p_\theta(s)].
$$

可把 $A(s)=R(s)-b$ 视为相对基准的好坏；恰当的基线可降低梯度估计方差。基线可以依赖 prompt/状态；实现策略梯度时，应将其作为固定系数，而不是在该项中额外反传基线的导数。

HW5 Q3 的特例：所有回答奖励都为 1，所以 $J\equiv1$、真实梯度为零。单样本 $\nabla\log p_\theta(s)$ 却可能非零，导致随机更新；取 $b=1$ 后每个样本更新均为零，方差也为零。

不能从这个特例推断任意 b 都降方差。也不能任意选依赖当前回答的 $b(s)$，否则上述零期望证明通常不成立。若将当前同一批样本的平均奖励作为基线，样本间依赖需另作分析，不能直接套“固定 b”的证明。

依据：FT p.89–99、HW5 Q3、R E15。

### 4.18 复用旧样本为什么要概率比

当样本来自旧模型 $p_{\rm old}$，直接计算 $\mathbb E_{p_{\rm old}}[A\nabla\log p_\theta]$ 一般不等于当前模型下的目标梯度。令

$$
\rho_\theta(s)=\frac{p_\theta(s)}{p_{\rm old}(s)},
$$

在旧分布覆盖当前分布所需支持、奖励和基线固定等条件下：

$$
\mathbb E_{p_\theta}[A\nabla\log p_\theta]
=\mathbb E_{p_{\rm old}}[\rho_\theta A\nabla\log p_\theta].
$$

对应可最大化的样本目标为

$$
L(\theta)=\frac1K\sum_k\rho_\theta(s_k)A_k,
$$

因为 $\nabla\rho_\theta=\rho_\theta\nabla\log p_\theta$。分母 old probability、已采样回答、奖励与优势在此轮更新中固定，仅分子随当前参数变化。

这是 FT p.100–120 的“gradient correction”，目的在于复用数据。不要误写成对 $\rho_\theta A\log p_\theta$ 全部求导；那会出现额外项。

### 4.19 PPO：clip 的作用与训练循环

课堂的简化 PPO 最大化：

$$
L_{\rm clip}(\theta)=\frac1K\sum_k
\min\left(\rho_k(\theta)A_k,
\operatorname{clip}(\rho_k(\theta),1-\varepsilon,1+\varepsilon)A_k\right).
$$

- $A_k>0$ 时，不继续奖励概率比超过 $1+\varepsilon$ 的增大。
- $A_k<0$ 时，不继续奖励概率比低于 $1-\varepsilon$ 的减小。
- min 取较保守的 surrogate 值；clip 旨在抑制过大的有利方向更新，**不保证实际概率比永远被硬限制在区间内**。

训练循环：复制当前模型为 old → 采一批回答 → 计算奖励、优势、保存 old log probabilities → 用 clipped objective 做多轮 minibatch 更新 → 重新采样。

课堂演示有整段回答比率；token-level PPO 用给定前缀下的单 token 概率比。**把整段比率 clip 和把每个 token 比率 clip 不是同一个操作。** 完整 PPO 还可包含 learned baseline、value loss 等；本课重点是这条简化推导链，不宜把所有实现细节都升级成必考。

依据：FT p.123–130、p.137；W4。

### 4.20 奖励模型与 diffusion 的强化微调

FT p.132–136 区分 learned reward 和 rule-based reward：奖励模型根据人类偏好学习评分；规则奖励可检验数学答案或运行代码测试。奖励只需提供数值，不要求能对生成内容求导。

偏好对 $(q,s^+,s^-)$ 的代表性奖励模型损失：

$$
L_{\rm RM}=-\mathbb E\log\sigma\left(r_\phi(q,s^+)-r_\phi(q,s^-)\right).
$$

奖励模型训练完后，在一次策略更新阶段固定。不要把“更新奖励模型参数”和“更新生成策略参数”混成同一次监督学习。

对 diffusion，可把一次去噪转移看作动作：状态含 $(x_t,t,q)$，动作是 $x_{t-1}$，用最终图像的质量/文本一致性等给奖励。若

$$
p_\theta(x_{t-1}\mid x_t,q)=\mathcal N(\mu_\theta,\sigma_t^2I),\quad\sigma_t>0,
$$

新旧策略使用相同固定方差，给定同一条旧转移：

$$
\log\rho_t=
\frac{\|x_{t-1}-\mu_{\rm old}\|_2^2-\|x_{t-1}-\mu_\theta\|_2^2}{2\sigma_t^2}.
$$

这是 FT p.138–139 的直接高斯密度比计算。必须有可计算密度的随机转移；$\sigma_t=0$ 的确定性步骤不能直接套此式。这里的 RL 微调内容不因“Diffusion 的 ELBO 不考”而自动被排除，R E21 明确提到文本和 diffusion 相关 RL。

### 4.21 其他短小但值得准备的课件概念

**Perplexity：** 对长度 n 的序列，$\operatorname{PPL}=\exp(-\frac1n\sum_t\log p_\theta(x_t\mid x_{<t}))$。同一数据与分词方式下，较低通常表示模型给测试文本更高平均概率；不同 tokenizer 的 PPL 不宜直接横比。依据：AR p.91–94。

**生成策略：** stochastic sampling 按条件分布随机抽样；greedy 每步取最大概率 token，因此不保证整段序列概率全局最大；top-k 限制到概率最高的 k 个候选再归一化采样。能说明多样性与确定性的区别即可。依据：AR p.101–116。

**VAE/DAE 区别：** VAE 的核心随机性在潜变量分布，并有变分目标；denoising autoencoder 在输入上加噪并学习重构干净输入。不能仅因“都有噪声”就视为同一个模型。依据：VAE p.130–142，B 类补充。

| 模型 | 训练核心 | 生成方式 | 能否直接计算样本密度 |
|---|---|---|---|
| AR | 真实前缀下的 next-token 似然 | 逐 token 采样 | 离散序列概率可按链式法则计算 |
| VAE | 重构项减近似后验到先验的 KL | 先验采 z，再经 decoder | 定义了边缘密度，但一般难以精确积分 |
| GAN | G 与 D 的对抗目标 | 噪声经 G | 通常不直接给出可计算的样本密度 |
| DDPM | 按时间步训练去噪网络 | 从高斯噪声逐步逆向采样 | 一般不直接获得精确边缘似然，可作变分评价 |

该表用于解释基本机制；不能把生成质量、训练稳定性等经验比较说成无条件定理。

## 5. 五次作业的阅读题不能整体漏掉

没有依据将所有“Paper reading”都排除。以下是最低限度的补充；其中 HW5 Q4 本身已给出要证明的式子，适合练习，不需要先通读整篇 DPO 论文。

### 5.1 HW1 Q4：Gumbel-Max 与 Gumbel-Softmax

给定正的类别权重 $\pi_i$，独立采 $U_i\sim\operatorname{Uniform}(0,1)$，令 $g_i=-\log(-\log U_i)$。则

$$
k=\arg\max_i(\log\pi_i+g_i),\qquad
\Pr(k=i)=\frac{\pi_i}{\sum_j\pi_j}.
$$

一个简短证明：$E_i=-\log U_i\sim\operatorname{Exp}(1)$，于是 $T_i=E_i/\pi_i\sim\operatorname{Exp}(\pi_i)$，且上述 argmax 等价于 $\arg\min_iT_i$。独立指数竞争给出

$$
\Pr(T_i=\min_jT_j)=\int_0^\infty\pi_i e^{-\pi_i t}\prod_{j\ne i}e^{-\pi_jt}dt
=\frac{\pi_i}{\sum_j\pi_j}.
$$

Gumbel-Softmax 用连续松弛替代不可微的硬 argmax：

$$
y_i=\frac{\exp((\log\pi_i+g_i)/\tau)}{\sum_j\exp((\log\pi_j+g_j)/\tau)},\quad\tau>0.
$$

$\tau\to0$ 时趋近 one-hot；有限温度的软样本不与原离散分类变量完全同分布。这与高斯 VAE 重参数化的精确分布保持不同。用于离散潜变量的可微近似训练；温度控制接近离散程度与梯度行为。依据：HW1 Q4；W7。

### 5.2 HW2 Q4：FlashAttention

核心是按块将 Q/K/V 从 HBM 搬到 SRAM，在片上进行注意力计算，用在线 softmax 累积精确输出，避免显式写出完整 $n\times n$ 注意力矩阵；反向时结合重计算降低存储需求。

它优化访存与额外显存，保持稠密注意力结果（允许正常浮点误差）。**不是靠删掉注意力项作近似，也不把精确稠密 attention 的 $O(n^2d)$ 算术复杂度改成 $O(nd)$。**

依据：HW2 Q4；W6 第 3 节。这也帮助理解“FLOPs 阶数较低”不等于“实际运行一定较快”。

### 5.3 HW3 Q3：GigaGAN

相对于原始 GAN 的简单无条件生成器/真假判别器框架，GigaGAN 面向大规模文本到图像任务，加入文本条件、style mapping、多尺度图像合成；生成器使用 attention 和自适应卷积核选择，多尺度判别器结合图像和文本评价。系统还可配合学习到的上采样器。

最低限度应能解释“新增的条件信息、生成器结构、判别器如何评价多尺度图像”，不要只答“参数更多、图更清晰”。上述技术点来自作者项目页，不将运行速度数字或实验排行榜当成考试必背内容。依据：HW3 Q3；W8。

### 5.4 HW4 Q3：Latent Diffusion Model

1. 第一阶段训练 autoencoder，学到压缩图像的 encoder E 和恢复图像的 decoder D。
2. 第二阶段固定 autoencoder，在编码出的潜变量 $z_0=E(x)$ 上加噪，训练潜空间去噪网络；本作业的基础答案使用噪声预测 MSE，条件模型还可输入文本条件。
3. 生成时从潜空间高斯噪声 $z_T$ 开始逐步去噪得到 $z_0$，最后调用 decoder 得到图像 $D(z_0)$。

主要收益是将昂贵的扩散计算放到较小的潜空间。注意 **噪声加在潜变量上，decoder 在完成潜空间去噪后使用**；不必每一步都解码成图像。

老师排除 VQ-VAE 扩展，不等于排除作业中 LDM 的这一基本流程。依据：HW4 Q3；W9 第 3 节。

### 5.5 HW5 Q4：KL 正则化最优策略

固定 prompt，有限回答集合上 $p_{\rm ref}(s)>0$、$\beta>0$，令

$$
F(p)=\mathbb E_pR-\beta D_{\rm KL}(p\Vert p_{\rm ref}),\qquad
p^*(s)=\frac{p_{\rm ref}(s)e^{R(s)/\beta}}Z,
\quad Z=\sum_sp_{\rm ref}(s)e^{R(s)/\beta}.
$$

从 $\log p^*=\log p_{\rm ref}+R/\beta-\log Z$ 展开：

$$
\begin{aligned}
D_{\rm KL}(p\Vert p^*)
&=\sum_sp(s)\left[\log\frac{p(s)}{p_{\rm ref}(s)}-\frac{R(s)}\beta+\log Z\right]\\
&=D_{\rm KL}(p\Vert p_{\rm ref})-\frac1\beta\mathbb E_pR+\log Z.
\end{aligned}
$$

所以 $F(p)=\beta\log Z-\beta D_{\rm KL}(p\Vert p^*)$。KL 非负，且仅在两分布相同时为零，故唯一最优分布就是 $p^*$。

直觉：在参考分布基础上，按奖励的指数重新加权；$\beta$ 控制偏离参考分布的惩罚强度。此处 $\beta$ 与 diffusion 的 $\beta_t$ 无关。不要把这道“最优策略证明”误认为必须完整复现 DPO 的训练算法。

依据：HW5 Q4；W5 附录 A.1。

## 6. 明确收缩的考试范围

| 内容 | 摘要的证据 | 处理方式 |
|---|---|---|
| Normalizing Flow | R E21：今年删去 Flow，加入 Fine-tuning 相关 RL | 两年卷的 Flow 大题不继续占据今年复习重点 |
| VQ-VAE 扩展机制 | R E20：明确说这部分不考 | 不优先背 codebook/straight-through 的详细机制；基础 VAE 仍是重点 |
| Diffusion 完整 ELBO 推导 | R E04 明确排除 | 不重建完整多步 ELBO 展开 |
| 给定 $x_t,x_0$ 推 $x_{t-1}$ 后验的繁琐推导 | R E06 | HW4 Q2、24 Q5(c) 不优先重做长推导；认识相关符号即可 |
| Diffusion 的 score matching 联系及更深入 ODE/SDE | R E16 | 不作为本次考前补漏重点 |

仍然要学的是：VAE/ELBO、DDPM 基本训练与推理、正向边缘分布证明、time embedding，以及 FT 课件里的 diffusion 强化微调。**“Diffusion 的 ELBO 不考”不等于“Diffusion 不考”。**

## 7. 材料本身存在的坑

### 7.1 2025 年卷后的答案与题干错位

该文件前 4 页为题干，从 p.5 开始附有答案文字，并出现排版源码残留。没有可靠来源证明这些后附答案的官方身份，不能直接当作标准答案。

| 前面题干 | 后附答案的情况 | 复习处理 |
|---|---|---|
| Q1(c) 是 LayerNorm | p.8 的 Solution 1(c) 却是自回归高斯联合似然，另列了 (d) LayerNorm | 按题干和实际计算核对，不背编号 |
| Q3(b) 是广场舞最短移动建模 | p.11 的 Solution 3(b) 却证明 KL 是 f-divergence 特例 | 用本稿最优运输/分配模型回答 |
| Q5 是标准正向过程、边缘分布和训练推理 | p.14–17 讨论一般 $a_t,b_t$ 递推、线性高斯后验系数等不同问题 | 不能把这段视为本卷 Q5 的逐问参考答案 |

p.5 还把 self-attention 叫作 permutation-invariant，但紧接着描述的是输出随输入同样置换，这实际是 permutation-equivariant。p.7 用“通常 $n\gg d$”删去投影复杂度，也不是任意题目都允许的假设；完整答案保留 $O(nd^2+n^2d)$。

### 7.2 AR 课件 p.126 的复杂度标注

页面把 $AV$ 标为 $O(nd^2)$。但 $A\in\mathbb R^{n\times n}$、$V\in\mathbb R^{n\times d}$，标准矩阵乘法为 **$O(n^2d)$**。这是按实际维度可直接核验的笔误；不能因为 PPT 这样写就照抄。投影 $XW_V$ 才是 $O(nd^2)$。

### 7.3 GAN 课件与作业的 JS 差一个系数

GAN p.43 将 $D_{\rm KL}(P\Vert M)+D_{\rm KL}(Q\Vert M)$ 称为 JS，而 HW3 定义的 JSD 带两个 $1/2$。因此课件该表达式等于标准 JSD 的两倍。回答时先写清定义，再使用一致的 $C(G)$ 公式；采用 HW3 定义时应为 $2\operatorname{JSD}-2\log2$。

### 7.4 GAN 课件 p.125 的数学表述需修正

“连续函数在有界闭区域有有限 Lipschitz 常数”一般不成立。例如 $f(x)=\sqrt x$ 在 $[0,1]$ 连续，但

$$
\frac{|f(h)-f(0)|}{|h-0|}=\frac1{\sqrt h}\to\infty.
$$

可用的充分条件例如：在凸区域上可微且梯度范数一致有界；或有限层网络的权重算子范数有界、激活函数具有有限 Lipschitz 常数。这个纠正用于防止记错数学结论，不表示该反例本身必考。

### 7.5 VAE 的“隐式模型”标签不要混用

GAN p.16–17 将 VAE 与 GAN 一起放入难以直接算样本密度的分类。严格讨论概率建模时，VAE 有明确的联合密度 $p(z)p_\theta(x\mid z)$，困难在边缘化和后验推断；它和通常只给采样器的 GAN 并不完全一样。答题优先写清“定义了什么、能否精确求边缘概率”，避免仅靠显式/隐式二分标签。

## 8. 考前使用顺序与自测

如果时间有限，按以下顺序补漏，而不是从第一页重新抄整套 PPT：

1. **VAE / HW1 Q1–Q3**：ELBO、取等、高斯 KL、log-variance 梯度。原重参数化段保留。
2. **Transformer / AR**：注意力公式与含 n,d 的复杂度、LayerNorm、mask、RoPE、position/time embedding。用两年 Q1 与 HW2 Q1–Q3 检验。
3. **Fine-tuning / HW5**：保留三阶段比较，重点补 log-derivative、baseline、概率比、PPO 和 KL 正则最优策略。
4. **GAN**：保留训练推理，补 $D^*$、JS、non-saturating、f-divergence 和 2025 OT 建模。
5. **Diffusion**：保留训练推理，补单步正向与边缘高斯证明；不再投入被排除的繁长推导。
6. 最后补作业阅读题和 B 类短概念。

这是根据资料的复习排序，不是预测出题顺序。你如果某部分已经能独立完成，可直接跳到尚不会的项。

可用以下问题闭卷自测：

- [ ] 能不用看笔记推 ELBO，并分清“KL 正则里的 prior”和“取等条件里的 true posterior”吗？
- [ ] 能写出多元高斯 KL，且不会写反行列式比和协方差逆吗？
- [ ] encoder 输出 log variance 时，知道哪里出现 $1/2$ 吗？
- [ ] 能写出 Q/K/V、softmax 轴及各矩阵维度，并区分 $nd^2$ 和 $n^2d$ 吗？
- [ ] 能解释 mask 与 teacher forcing，正确说明允许看哪些 token 吗？
- [ ] 能手算四个 LayerNorm 向量并写出 RoPE 的矩阵与位置差符号吗？
- [ ] 能各给一种 positional/time embedding 实现并说明目的差异吗？
- [ ] 能推 $D^*$、GAN 与 JS 的关系，以及两个生成器损失对 logit 的导数吗？
- [ ] 能把广场舞题写成有一对一约束的最小运输代价模型吗？
- [ ] 能证明正向边缘分布，而不是只背 DDPM 的结论吗？
- [ ] 能解释为什么训练直接采 t，不必每次逐步跑完正向链吗？
- [ ] 能从求和推 REINFORCE，证明 baseline 不改期望，并说明奖励无需可微吗？
- [ ] 能说明 old/current 概率比、clip 的目的和更新方向吗？
- [ ] 能完成 HW5 Q4 的“常数减 KL”证明吗？

摘要 E09/E23 还提醒任课教师中文姓名为 **贺笛**。这里仅转述已有课程材料，不据此推断本次具体分值布局。

## 9. 来源索引

以下本地文件均相对于项目根目录 `Generative Modeling`。页码以 PDF 页序计。证据是“往年出现”时，只代表历史出现；证据是教师摘要时，保留其原有的确定性和限定词。

| 缩写 | 文件 | 本稿主要使用位置 |
|---|---|---|
| N | `review/Review on Theories.md` | 全文 91 行 |
| R | `review/录播考点与出题信息核查.pdf` | p.1–7；E01–E23，尤其 E03–E06、E10–E21 |
| 24 | `review/Generative_Model_Midterm_2024.pdf` | p.1–3 全部题干 |
| 25 | `review/Generative_Model_Midterm_2025.pdf` | p.1–4 题干；p.5–17 后附答案用于核对错位问题 |
| IN | `slides/generative-01-intro-2.pdf` | p.28–46 课程结构、考核与范围 |
| VAE | `slides/generative-03-autoencoders-4.pdf` | p.24–32、p.70–95、p.102–129、p.130–143 |
| TR | `slides/generative-04-Transformer-2.pdf` | p.37–93 |
| AR | `slides/generative-05-AR-4.pdf` | p.5–15、p.59–81、p.91–146 |
| GAN | `slides/generative-06-GAN-4.pdf` | p.15–17、p.25–43、p.44–91、p.93–137 |
| FT | `slides/generative-07-rlhf 4.pdf` | p.24–53、p.56–139 |
| DF | `slides/generative-11-diffusion 4.pdf` | p.17、p.34–61、p.102–117；p.118–160 用于辨识扩展边界 |
| HW1 | `assignment/Assignment1.pdf` | Q1–Q4，1 页 |
| HW2 | `assignment/Assignment 2.pdf` | Q1–Q4，1 页 |
| HW3 | `assignment/Assignment3.pdf` | Q1–Q3，1 页 |
| HW4 | `assignment/Assignment4.pdf` | Q1–Q3，1 页 |
| HW5 | `assignment/Assignment5.pdf` | Q1–Q4，1 页 |

原论文仅用于核对技术细节，**不用于扩大老师给出的考试范围**：

- W1：[Attention Is All You Need](https://arxiv.org/html/1706.03762v7)，第 3–4 节，attention、位置编码与复杂度。
- W2：[Generative Adversarial Nets](https://arxiv.org/html/1406.2661v1)，第 3–4 节，非饱和目标与理论最优性。
- W3：[Denoising Diffusion Probabilistic Models](https://arxiv.org/html/2006.11239v2)，第 2–3 节与 Algorithms 1–2，尤其第 3.4 节的简化损失。
- W4：[Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347)，第 3 节公式 (7)，clipped surrogate。
- W5：[Direct Preference Optimization](https://arxiv.org/html/2305.18290v3)，附录 A.1，KL 正则最优策略。
- W6：[FlashAttention](https://arxiv.org/html/2205.14135v2)，第 3 节，分块、重计算、访存与显存。
- W7：[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/html/1611.01144v5)，第 2 节，Gumbel-Max 和连续松弛。
- W8：[GigaGAN 作者项目页](https://mingukkang.github.io/GigaGAN/)，Architecture 部分。
- W9：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/html/2112.10752v2)，第 3 节，感知压缩与潜空间扩散。

本稿中的短证明、矩阵维度检查和反例为独立展开；课堂考试范围的判断均以本地题干、作业和教师摘要为依据。它是复习补充稿，不是官方标准答案。
