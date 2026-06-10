# rl_qra_pipeline: RRPI ratio-reward policy improvement

这版 `rl_qra_pipeline` 已经从旧的 **GT-guided token boost** 改成 **RRPI（Ratio-Reward Policy Improvement）**。

核心变化：

- 旧版：直接把 GT token 当正样本，full-vocab top-k 当负样本，reward 基本没有真正进入更新目标。
- 新版：直接读取 SEDD 输出的 ratio field，得到局部 transition policy `pi_theta(a | state, t, position)`；然后枚举 answer-type-aware candidate actions，对每个 candidate answer 打 reward，用 reward softmax 构造目标策略 `q(a)`，最后优化 `KL(q || pi_theta)`。

也就是说，现在的梯度链路是清楚的：

```text
SEDD ratio field -> local policy pi_theta
candidate action -> candidate answer -> verifier reward R(a)
reward R(a) -> target policy q(a)=softmax(R(a)/tau)
KL(q || pi_theta) -> update SEDD parameters
```

这不是完整 on-policy REINFORCE，也不需要等待真实 sampler 采到正确 token。它利用 SEDD 每一步输出完整 ratio/probability field 的特点，做低方差的局部 reward-guided policy improvement。

## 训练命令

从 repo 根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_rl_qra.py \
  --config rl_qra_pipeline/rl_qra_config.yaml \
  --run-name rrpi_smoke \
  --start QRA
```

正式后台训练：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u run_rl_qra.py \
  --config rl_qra_pipeline/rl_qra_config.yaml \
  --run-name rrpi_qra_v1 \
  --start QRA \
  > rl_qra_rrpi.log 2>&1 &

tail -f rl_qra_rrpi.log
```

## 输出结构

以 `--start QRA` 为例：

```text
rl_qra_pipeline/modelparameter/rl_QRA/<timestamp>_<run-name>/
  metrics.csv
  rrpi_debug.jsonl
  run_info.json
  best_RL_QRA_rrpi.pth
  last_RL_QRA_rrpi.pth
```

同时同步：

```text
rl_qra_pipeline/modelparameter/rl_QRA/best.pth
rl_qra_pipeline/modelparameter/rl_QRA/last.pth
```

## log 指标解释

```text
target_logp     reference/GT token 当前 log probability，越接近 0 越好
p_target        reference/GT token 当前 probability
modelR          在候选集合中，模型当前最偏好的 candidate 的 reward
bestR           候选集合里最高 reward
gap             bestR - worstR，表示这个位置是否有明显 reward 差异
targets         本 step 实际构造了多少个 candidate-policy update site
```

如果 `modelR` 长期明显低于 `bestR`，说明模型当前 ratio policy 偏向低 reward candidate；RRPI 应该逐步把 `modelR` 拉高。

## debug 可视化

每隔 `rrpi.debug_every` step，日志会展示一个具体 update：

```text
[RRPI DEBUG step=20]
GT: (3,4]
state_answer: (3,4<mask>
position=...
target=']' p_target=...
best_reward=1.000 model_choice_reward=0.780 reward_gap=0.220
candidate_table: token | reward | q_target | pi_model | answer | source
 * ']' | R=+1.000 q=0.90 pi=0.13 | '(3,4]' | gt
   ')' | R=+0.780 q=0.10 pi=0.62 | '(3,4)' | type
```

这个表就是 reward 如何指导梯度更新的直接证据。

## 设计说明

SEDD 的反向过程由 concrete score / ratio field 构造 reverse transition probability。RRPI 把这个 reverse transition probability 当作局部 policy。由于 QRA 的最终答案通常很短，answer action space 可被类型化枚举，例如：

```text
single_letter: A/B/C/D/E
single_integer: digits/sign
signed_decimal: digits/sign/dot
interval: brackets/comma/digits/sign/dot
```

因此不用进行高方差完整 rollout，而是直接枚举候选动作并计算 reward-improved target policy。这比旧版 full-vocab top-k negative 更可解释，也更接近“reward 直接更新 ratio policy”的目的。
