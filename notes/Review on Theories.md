# 主要知识点概念辨析

## VAE

### VAE的重参数化

VAE的重参数化指的是在采样均值为 $\mu$、标准差为 $\sigma$ 的潜变量（latent code）$h$ 时，不直接从 $\mathcal N(\mu,\sigma^2)$ 中采样 $h$，而是先采样一个符合标准高斯分布的噪声 $\epsilon$，随后让 $h=\mu+\sigma\epsilon$。

这样采样的 $h$ 同样符合原来的高斯分布，而且对于固定的噪声，可以计算出 $h$ 对于 $\mu$ 和 $\sigma$ 的导数，从而梯度可以穿过采样这一步，从而可以求出损失 $L$ 对 encoder 参数的梯度。

如果直接采样，梯度则不能直接通过普通抽样操作反向传播，因此要用重参数化，通过将随机性放入与参数无关的噪声中，使模型在保持采样分布不变的同时，可以通过反向传播训练 encoder。

## AR

### 为什么Transformer需要Positional Embedding

Token 的顺序和相对位置会影响序列的含义，而标准 self-attention 不编码位置信息，FFN 也在各位置进行相同变换，因此需要 positional embedding 为模型输入位置信息。

### RNN和Transformer训练和推理的时间复杂度对比

假设模型层数、隐藏维度等保持固定，仅考虑序列长度对计算复杂度的影响。

**训练：** 对于长度为 $n$ 的序列，RNN 需要沿序列递归计算隐藏状态，训练计算复杂度为 $O(n)$；Transformer 的自注意力机制训练计算复杂度为 $O(n^2)$。RNN 在训练时，同一序列的隐藏状态存在顺序依赖，难以跨位置并行；Transformer 在 teacher forcing 训练中可以同时计算各位置，并利用高效的 GPU 矩阵运算。因此，Transformer 的训练效率更高。

**推理：** 对于自回归生成长度为 $n$ 的序列，二者都需要逐 token 生成，不能跨生成步骤并行。RNN 每一步只需要更新固定维度的隐藏状态，单步计算复杂度为 $O(1)$，总推理计算复杂度为 $O(n)$。使用 KV cache 的 Transformer 在第 $t$ 步仍需对已有前缀计算注意力，单步计算复杂度为 $O(t)$，总推理计算复杂度为 $O(n^2)$。因此，RNN 的推理效率更高。

## GAN

### GAN的训练和推理

**训练：** 从数据集 $\mathcal D=\{x_i\}_{i=1}^n$ 中采样真实数据，并采样噪声 $\epsilon\sim\mathcal N(0,I)$，通过生成器得到假样本 $f_G(\epsilon)$。判别器 $f_D$ 估计输入为真实样本的概率。训练目标为 $\min_{f_G}\max_{f_D}V(f_D,f_G)$，其中

$$
V(f_D,f_G)=
\frac{1}{n}\sum_{i=1}^{n}\log f_D(x_i)
+
\mathbb E_{\epsilon\sim\mathcal N(0,I)}
\left[\log\left(1-f_D(f_G(\epsilon))\right)\right].
$$

使用小批量真实样本和噪声样本估计梯度，交替进行以下更新：固定生成器参数，通过梯度上升更新判别器参数以最大化 $V$；固定判别器参数，通过梯度下降更新生成器参数以最小化 $V$。重复上述过程直至训练结束。

**推理：** 固定训练好的生成器参数，采样新的噪声 $\epsilon\sim\mathcal N(0,I)$，输出生成样本 $\hat x=f_G(\epsilon)$。此时只需使用生成器。

## Diffusion

### DDPM的训练与推理

设噪声日程 $\beta_1,\ldots,\beta_T$ 固定，记 $\alpha_t=\prod_{s=1}^{t}(1-\beta_s)$。

**训练：** 从数据集中采样真实数据 $x_0$，均匀采样时间步 $t\sim\mathrm{Uniform}(\{1,\ldots,T\})$，并采样噪声 $\epsilon\sim\mathcal N(0,I)$，构造加噪样本

$$
x_t=\sqrt{\alpha_t}\,x_0+\sqrt{1-\alpha_t}\,\epsilon.
$$

噪声预测网络 $\epsilon_\theta(x_t,t)$ 根据加噪样本和时间步预测噪声，训练目标为

$$
\min_\theta L_{\mathrm{simple}}(\theta)
=
\mathbb E_{x_0,t,\epsilon}
\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right].
$$

使用小批量样本估计梯度，通过梯度下降更新参数 $\theta$，重复至训练结束。

**推理：** 固定训练好的网络参数，先采样 $x_T\sim\mathcal N(0,I)$，再按 $t=T,T-1,\ldots,1$ 逐步反向采样：

$$
x_{t-1}
=
\frac{1}{\sqrt{1-\beta_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\alpha_t}}\,
\epsilon_\theta(x_t,t)
\right)
+\sigma_t z_t.
$$

其中，$\sigma_t$ 为预设的反向采样标准差；当 $t>1$ 时，每步独立采样 $z_t\sim\mathcal N(0,I)$，最后一步取 $z_1=0$。最终输出生成样本 $x_0$。

## Fine-tuning

### 预训练、指令微调与强化微调

预训练使用新闻、论文、书籍和代码等大规模文本数据，以预测下一个 token 为训练目标。监督信号来自文本中实际出现的下一个 token，因此属于自监督学习，通常通过最小化交叉熵损失训练。它使模型学习通用的语言规律、知识和表征，为后续微调提供初始化参数。

指令微调使用指令与参考回答组成的数据，例如 QA 交互数据。在预训练参数的基础上，模型通过监督学习预测参考回答中的 token，并最小化交叉熵损失，从而学会遵循指令、适应问答格式，并将已有知识用于具体任务。从继续使用下一个 token 预测损失的角度，也可以将其视为预训练的延伸。

强化微调针对输入提示，让模型自行生成回答，再由奖励函数评价回答。奖励可以来自人类反馈、学习得到的奖励模型或规则。模型以最大化回答的期望奖励为目标调整参数，使生成结果更符合用户偏好和实际任务要求。