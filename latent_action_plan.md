# RoboTwin Latent Action 研究进展与实施草案

> 状态：讨论中，尚未开始实现。本文保存截至 2026-08-23 的研究结论、当前选择和待决问题。

## 目标

在 RoboTwin 上引入 latent action：

1. 继续使用现有 V-JEPA 2.1 encoder + `JEPAPredictor` video expert；
2. 先让 ActionDiT 学习低维 latent action；
3. 再把 latent action 解码为 RoboTwin 的 normalized `32 × 14D` qpos action；
4. 对外推理接口保持 `[32,14]`，继续使用现有 normalizer 反归一化和 RoboTwin qpos 执行。

## 当前决定

- 第一版 latent 标签来源：**冻结的 DreamDojo 预训练 LAM**。
- 只使用 DreamDojo LAM 提取 pseudo-action，不移植其 Cosmos/WAN world model。
- 使用确定性 `z_mu`，不使用 posterior sample。
- DreamDojo LAM 输出连续 32D latent，和当前 ActionDiT 的连续 flow-matching 比离散 LAPA/UniVLA token 更自然兼容。
- DreamDojo 没有 latent→robot-action decoder；需要在本项目中单独训练 contextual 14D decoder。
- decoder 设计参考 UniVLA/villa-X：输入 latent + 当前 proprio/qpos + V-JEPA visual context，而不是简单线性 `32→14`。
- 保留 dataset 原始 normalized `[32,14]` action contract，不把 latent 表示泄漏到部署层。

## 时间对齐结论

“和 H-FastWAM 动作数量对齐”指最终输出仍为 32 个动作，不要求 latent token 数等于 32。

当前一个样本为：

```text
33 states
32 low-level qpos actions
9 sampled RGB frames（索引 0,4,...,32）
action_video_freq_ratio = 4
```

推荐第一版：

```text
9 个视频帧
→ 8 个相邻帧 DreamDojo latent
→ latent target [8,32]
→ 每个 latent 解码 4 个 14D 动作
→ decoder [8,4,14]
→ reshape [32,14]
```

形式化对齐：

```text
z[k] = LAM.encode(frame[4k], frame[4k+4]).z_mu
z[k] ↔ action[4k : 4k+4]
```

必须先用原始 timestamp 验证 `action[t]` 对应 state `t→t+1`，不能只依赖 tensor shape 假设。

## DreamDojo 调研结论

- LAM 是连续 Gaussian VAE，默认 latent 维度 32。
- encoder 从帧变化抽取 `z_mu`；decoder 重建未来视频帧，不输出机器人动作。
- LAM 规模约 700M、checkpoint 约 8.5GB，不适合在线放在每个训练 batch 内。
- 推荐离线预计算 latent cache。
- 代码许可证 Apache-2.0；预训练权重使用 NVIDIA Open Model License，使用前需要确认项目可接受。

候选比较：

- **DreamDojo**：最适合当前 continuous ActionDiT 的第一版 latent labeler。
- **UniVLA**：公开 contextual latent→robot decoder 的最佳结构参考，但原生是离散 token/VLM 技术栈。
- **villa-X**：概念上最匹配（latent expert + robot expert），但 Actor 代码未公开，只作设计参考。
- **LAPA**：可作离散 latent baseline，不建议作为第一版主线。

## 当前代码事实

### 数据和动作

- RoboTwin action：normalized `[B,N,32,14]`。
- 14D 顺序：左臂 6 joints + 左 gripper + 右臂 6 joints + 右 gripper。
- normalization 是 global z-score，输出 decoder 必须仍处在 normalized 14D 域。

关键文件：

- `configs/data/robotwin_interleaved_webdataset.yaml`
- `src/fastwam/datasets/lerobot/webdataset_robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.py`

### 模型

- Action expert：`ActionDiT`，当前直接对 `[32,14]` 做 flow matching。
- Video expert：仓库自定义 `JEPAPredictor`，预测 V-JEPA future latents，loss 为 L1。
- Action/video expert 参数不共享，但每层在 MoT 中混合 attention。
- 最适合的 latent 插入边界：normalized action 进入 flow scheduler 之前；推理 decode 在 denoising 完成、`infer_action()` 返回之前。

关键文件：

- `src/fastwam/models/wan22/action_dit.py`
- `src/fastwam/models/wan22/jepa_predictor.py`
- `src/fastwam/models/hfastwam/hfastwam.py`
- `src/fastwam/models/hfastwam/hfastwam_idm.py`
- `src/fastwam/models/wan22/mot.py`

### V-JEPA predictor 并非一定仍是随机状态

模型配置单独实例化时 `JEPAPredictor` 随机初始化，但已有 checkpoint 已经通过 video L1 训练过 predictor：

- InternData `step_104000`
- RoboTwin transfer run `robotwin_ft_interndata_step_104000_4x8_b48` 的后续 checkpoint

checkpoint transfer 只重置 20D/14D 不兼容的 action input/output 与 proprio，不删除 predictor。应先评估候选 checkpoint 的：

- one-step V-JEPA latent L1；
- two-step autoregressive rollout L1；
- 对比 random predictor 与 static-copy baseline。

如果候选优于 baseline，Stage 1/2 冻结 predictor；如果不优于，则先单独用 video L1 训练 predictor，不能让 latent loss“顺便”污染 predictor。

## 推荐阶段计划

### Phase 0：冻结基线和 schema

固定：

```yaml
latent_dim: 32
latent_horizon: 8
physical_action_dim: 14
physical_horizon: 32
actions_per_latent: 4
latent_target: z_mu
```

记录当前 direct-action checkpoint 的 video loss、offline action 指标、RoboTwin success rate、推理耗时、normalizer/checkpoint hash。

### Phase 1：DreamDojo latent 小规模探针与缓存

1. 对少量跨任务、跨 episode、含尾部 padding 的样本运行冻结 LAM。
2. 确认三相机 canvas/head camera、resize、RGB 范围、帧序等预处理。
3. 用原始 parquet timestamp 验证每个 latent 与 4 个动作严格对齐。
4. 比较正确帧对和时间打乱帧对的未来帧重建。
5. 离线生成 per-transition cache；dataset 按窗口聚合成 `[8,32]`。
6. manifest 记录 LAM checkpoint/hash、代码 revision、数据 hash、预处理、时间 stride、latent stats、schema version、license 标识。

Stop/Go：

- timestamp/off-by-one 错配必须为 0；
- latent finite、非坍缩、重复提取一致；
- 正确配对重建明显优于打乱配对；
- 任何对齐或预处理不确定时停止，不进入训练。

### Phase 2：选择/训练 V-JEPA predictor 起点

1. 在固定 val subset 上比较现有 predictor checkpoint。
2. checkpoint 必须显式配置，缺失时 fail-fast，禁止静默随机初始化。
3. 若已有效，冻结 V-JEPA encoder/predictor；latent/action loss 对 video K/V stop-gradient。
4. 若无效，先做 predictor-only video L1 训练，达标后再进入 Stage 1。

Stop/Go：candidate predictor 的 one-step 与 rollout L1 均应明显优于 random/static-copy；冻结阶段前后参数 checksum 必须完全相同。

### Stage 1：训练 latent ActionDiT

Dataset 额外提供：

```text
latent_action        [8,32]
latent_action_is_pad [8]
```

ActionDiT 改为：

```text
action representation dim = 32
latent token horizon = 8
objective = existing continuous flow matching
```

保留 physical `[32,14]` action，不覆盖现有 `segments["action"]`。latent mask 由两个视频端点和对应四个 physical actions 的有效性共同决定。

Stop/Go：sampled latent 的 held-out MSE/MAE、cosine、variance 应优于 per-dimension mean baseline；建议至少 10% 相对 MSE 改善。V-JEPA encoder/predictor checksum 不变。

### Stage 2：训练 contextual 14D decoder

建议轻量 small-transformer/cross-attention decoder：

```text
inputs:
  predicted/oracle latent [8,32]
  current normalized qpos/proprio
  current observable V-JEPA visual context
output:
  [8,4,14] → [32,14]
```

先做：

1. **Oracle decoder**：输入缓存的真实 DreamDojo `z_mu`，冻结所有 experts，只训练 decoder。
2. **Generated-latent evaluation**：输入 Stage 1 生成的 latent，测量相对 oracle 的退化。

第一版只使用当前可观测 visual context，不用 teacher-forced future proprio，避免泄漏。

Stop/Go：oracle decoder 必须优于 repeat-current-qpos 和 train-mean baseline（建议 normalized MAE 至少改善 20%）；generated-latent 相对 oracle 的退化建议不超过 25%；分别报告关节和 gripper 指标。

### 可选 Stage 3：联合微调

仅当 Stage 1 和 Stage 2 分别通过后：

```text
L = λ_latent * latent_flow_loss
  + λ_qpos * masked_14D_action_loss
```

首轮仍冻结 V-JEPA encoder/predictor。只有证明 predictor 领域适配有收益时，才允许 video L1 用更小 LR 更新 predictor，同时继续阻断 latent/qpos loss 对 predictor 的梯度。

## Checkpoint 与测试

新 checkpoint schema 至少记录：

```text
checkpoint_schema_version
action_representation=latent
latent_dim=32
latent_horizon=8
physical_action_dim=14
physical_horizon=32
actions_per_latent=4
latent decoder state/config
latent cache signature
DreamDojo checkpoint hash
latent normalization stats
predictor source checkpoint/hash
```

旧 14D checkpoint 迁移到 32D latent ActionDiT 时，显式删除旧：

```text
mixtures.action.action_encoder.*
mixtures.action.head.*
```

保留 ActionDiT backbone、MoT 和 predictor；重新初始化 32D input/output head。

必须覆盖测试：

- `9 frames → 8 latents → 32 actions`；
- physical/latent padding mask；
- IDM mask 使用 8 个 action tokens；
- Stage 1/2 gradient isolation；
- train/infer shape；
- checkpoint 新 schema round-trip 和旧 checkpoint 转换；
- cache 签名/hash 不符时 fail-fast；
- deploy normalization round-trip；
- alignment checker 从“ActionDiT 必须14D”改为“latent expert 32D + decoder 14D”。

## 预计涉及文件

现有文件：

- `configs/model/hfastwam_idm_vjepa21_predictor_full_condition.yaml`
- `configs/data/robotwin_interleaved_webdataset.yaml`
- `src/fastwam/datasets/lerobot/webdataset_robot_video_dataset.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/models/hfastwam/hfastwam.py`
- `src/fastwam/models/hfastwam/hfastwam_idm.py`
- `src/fastwam/models/wan22/action_dit.py`
- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/trainer.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.py`
- `scripts/check_robotwin_eval_alignment.py`
- `scripts/prepare_interndata_checkpoint_for_robotwin.py`
- `tests/test_hfastwam_masks.py`

预计新增：

- `scripts/precompute_robotwin_dreamdojo_latents.py`
- `scripts/check_robotwin_latent_alignment.py`
- `src/fastwam/utils/latent_action_cache.py`
- `src/fastwam/models/hfastwam/latent_action_decoder.py`
- `tests/test_latent_action_pipeline.py`
- 一份独立的 RoboTwin latent-action model config

## 下次讨论待决问题

1. predictor 起点用哪个 checkpoint：`step_014000`，还是先评估后选更新 checkpoint。
2. DreamDojo checkpoint 的精确版本、本地路径，以及是否接受 NVIDIA Open Model License。
3. DreamDojo LAM 输入使用三相机拼接 canvas 还是仅 head camera；先用小规模重建探针决定。
4. 若 `z_mu` 不接近单位高斯，是否采用固定逐维标准化（推荐采用并写入 manifest/checkpoint）。
5. decoder v1 是否确认只用当前可观测 visual context（推荐）。
6. Stage 2 loss 第一版是否用 masked SmoothL1（推荐）。
7. RoboTwin 固定验收任务、episode 数、seed 和成功率门槛。
8. optional joint fine-tune 是否纳入第一轮范围；建议第一轮不纳入，等独立两阶段通过后再决定。
