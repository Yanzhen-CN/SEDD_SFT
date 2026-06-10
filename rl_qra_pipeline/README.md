# rl_qra_pipeline: verifier-guided local ratio policy optimization

这版 `rl_qra_pipeline` 不再使用 hybrid，也不强依赖大量真实 rollout。核心思想是：SEDD 在每个 noisy / mask state 下会输出完整的 token-level ratio field，因此我们可以直接读取当前模型对所有候选 transition 的概率，然后用 verifier / GT 构造 reward-improved local actions。

## 为什么这样做

严格 on-policy RL 当然可以：SEDD 的反向扩散链可以看成 stochastic policy trajectory。但是对短答案任务，纯 rollout 样本效率很低。例如 GT 是 `(3,4]`，当前状态是 `(3,4[MASK]`，如果模型当前更偏向 `)`，那必须反复采样很多次才可能采到正确的 `]` 并获得正 reward。

所以本 pipeline 不说“假设模型采到了正确 token”，而是明确采用：

> verifier-guided local policy improvement。

也就是直接在关键 mask/noisy state 上 forward 当前模型，得到

```text
pi_theta(y | x_t, t, position)
```

然后：

```text
correct / reward-approved token      -> increase logprob
high-probability wrong token          -> decrease probability
wrong answer type                     -> strong negative
same type but wrong content           -> weak negative
structured component correct/wrong    -> local positive/negative
```

## 与 SEDD ratio 的关系

SEDD 的 DTransformer 输出 `log_score`，代码里转成：

```python
score = exp(log_score)
```

这个 score 对应论文里的 concrete score / ratio field：

```text
s_theta(x_t, t)_{i,y} ≈ p_t(x_t with position i changed to y) / p_t(x_t)
```

采样器再用 ratio、transition matrix 和 step size 构造 reverse transition probability。训练时我们优化的是 token action 的 logprob，因此梯度会回到 DTransformer 参数，间接调高或调低对应 token 替换方向的 ratio。

## Loss

对一个局部状态 `s=(x_t,t,i)`：

```text
y+ = verifier / GT 指定的正确 token
y- = 当前模型高概率但错误的 token
```

使用：

```text
L = - w+ log pi_theta(y+ | s)
    + w- sum_y- penalty(y-) [-log(1 - pi_theta(y- | s))]
```

含义是：

```text
提高正确 token 的 transition probability / ratio；
降低高概率错误 token 的 transition probability / ratio。
```

## Type-aware reward strategy

### single_letter / single_integer

这类答案不需要复杂 chain reward。核心是类型和 exact match：

```text
exact match       -> 强正向
same type wrong   -> 弱负向或接近中性
wrong type        -> 强负向
```

例如 GT=`4`：

```text
Pred token 4  -> 强鼓励
Pred token 5  -> 同为整数，轻微压低
Pred token B  -> 类型错误，强烈压低
```

### signed_decimal / interval

这类答案有结构，因此构造多个局部 mask states。

例如 GT=`(3,4]`：

```text
[MASK]3,4]      -> 训练 left bracket
(3,4[MASK]      -> 训练 right bracket
(3,[MASK]]      -> 训练 right value
```

正确组件得到正向 logprob 更新，错误组件得到负向抑制。这样可以做到：前面 `3`、`4` 已经对了就鼓励，最后括号错了只惩罚括号相关 transition。

## File structure

```text
rl_qra_pipeline/
  answer_specs.py           # answer type parser and potential functions
  reward_type_aware.py      # final answer / reasoning reward utilities
  state_builder.py          # encode sample, build SEDD transition probability
  guided_ratio_update.py    # core guided ratio loss
  train_rl_qra.py           # training entry
  run_rl_qra.py             # thin wrapper
  rl_qra_config.yaml        # config
```

## Run

```bash
python rl_qra_pipeline/run_rl_qra.py \
  --config rl_qra_pipeline/rl_qra_config.yaml \
  --run-name guided_ratio_v1
```

Smoke test：

```yaml
training:
  steps: 1
  batch_size: 1

guided:
  states_per_sample: 1
  topk_negative: 2
```

## How to explain in report

这不是无偏的 on-policy REINFORCE，而是更适合 SEDD 的 RL-style local policy improvement。作业要求是探索 RL 如何整合进离散扩散模型；本方法保留 RL 的核心思想——用 reward/verifier 改进 policy，而不是普通 SFT 的统一模仿 loss。同时，它利用了 SEDD 的关键特点：每一步输出完整 ratio field，可以直接对采样和未采样的 candidate transition 做局部策略更新，从而避免纯 rollout 的低样本效率问题。
