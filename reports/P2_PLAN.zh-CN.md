# F1609_v7 P2：C0 / S1 匹配语义短探针计划

日期：2026-07-16  
计划状态：`等待用户确认；尚未准备 P2 sandbox、未训练、未读取 locked 视频`

## 1. 实验名称与当前问题

- 共同前缀：`F1609_v7_P2_matched_probe5`
- 控制支：`F1609_v7_P2_C0_control5`
- 实验支：`F1609_v7_P2_S1_semantic5`

P1 已证明受保护 optimizer anchor 没有明显破坏 V0，但 P1 本身并未产生实质改善。P2 只回答一个问题：

> 在完全相同的受保护训练条件下，semantic identity loss 是否比普通 DLC + Teacher protection 多产生真实、可测的语义错侧修复？

## 2. 假设

若 semantic identity loss 有效，则 S1 相对同 epoch 的 C0 应在 target114 上产生至少 0.25 px 的平均额外改善，并减少已知反向语义 inversion；这种改善不能以 fixed81 Native 退化、新增飞点或 clean-margin 反转为代价。

若 C0 与 S1 都没有 material movement，则当前表示/损失仍不足，不能通过增加 epoch 继续试错。

## 3. 冻结基线与输入

### P1 共同 anchor

- checkpoint：`snapshot-001.pt`
- SHA256：`4D26549AE7C6CA48DC7D3DD27341B6D414E0382923C426D5E43A6FA9C452DB7B`
- model state SHA256：`572FE6D3D0490E5FF7CDA3DFD341DD9E0AF184AAF26B1DA52FE8B8968CF05650`
- 必须同时克隆 model state 与 34 项 optimizer state；不得 fresh optimizer。

### V0 与 P1 Gate D 证据

- V0 checkpoint SHA256：`504521C281BF9C1E524B470C590300CFD1FB18D5535C7A119A53DD9DF8A4BC79`
- P1 Gate D summary SHA256：`844552722E839C7ED59C64EA44A4CB7B6A6978F568B4AF66F31FC6A4B9CB3FBF`
- P1 Gate D 状态：`p1_gate_d_pass_anchor_accepted_for_p2_plan`

### 数据

- GT-fix v1：1629 行；固定 split 为 1548 train / 81 internal test。
- GT-fix H5 SHA256：`6EAA40301A67AAFE75D681F35BC77C1202C71200C93AFF461ACE4D2ED6D024E1`。
- corrected22：22 个 train frame；114 个 supervision target point；62 个同帧 context point。
- P2 两支都使用全部 1548 train rows。每个 epoch 每行恰好出现一次。
- fixed81 只评估，不进入 optimizer。

### Locked 数据

以下视频继续保持 0 训练、0 loss 校准、0 阈值选择、0 prediction/render：

- `04_water`
- `05_acetone`
- `09_acetone`
- `test_video.mp4`
- `test_video_bgclean_staticdot.mp4`

## 4. 两支唯一差异

### C0

`DLC native supervised loss + frozen V0 Teacher protection`

### S1

`C0 total loss + 0.007028677557221117 * semantic_identity_loss`

semantic loss 使用同帧人工 GT heatmap 的弯曲、伸缩后真实位置，配对为：

- Head / Tail
- T1 / A7
- T2 / A5
- T3 / A3

它只要求本通道在自身 GT 位置的 heatmap 响应高于反向语义 GT 位置，不建立直线身体轴，不替代 8 个独立 DLC 输出。

权重 `0.007028677557221117` 来自 P0 hard-batch equal-gradient-norm 校准；P2 不搜索多个权重，也不根据 fixed81 或 locked 结果回调权重。

## 5. 共同训练约束

- C0、S1 从同一 P1 checkpoint bit-exact 克隆。
- 共用同一个 DataLoader batch；每个 step 在同一 augmented input 上依次更新 C0 与 S1，避免只靠重设 seed 模拟匹配。
- 共用冻结 V0 Teacher；Teacher output 对同一 augmented input 只计算一次。
- 可训练参数严格为完整 `backbone.model.layer4.*` 与原生 heatmap/locref heads，共 34 个 tensor。
- layer1-3、其余参数与 Teacher 全部冻结。
- batch size 8；seed 42；workers 0；pin memory false；drop_last false。
- 每 epoch `ceil(1548/8)=194` step；5 epochs 每支恰好 970 optimizer steps。
- corrected22 每帧每支恰好 exposure 5 次；fixed81 exposure 0。
- LR 保持 anchor 中的 `1e-6`；scheduler 为 None。
- 每 epoch 原子保存 C0/S1 checkpoint，均含 optimizer state；不覆盖 P1 anchor。

### Teacher 保护的 target mask

Teacher 不是 corrected22 的答案。两支采用完全相同的保护 mask：

- corrected22 的 114 个 supervision target channel：Teacher heatmap/locref protection 权重为 0；仍保留 DLC 人工 GT supervision。
- corrected22 同帧其余 62 个 context channel及所有 unchanged channel：保留完整 Teacher protection。
- S1 的 semantic loss 不使用 Teacher 坐标。

这项 mask 是两支共同条件，不是 S1 特权；它用于消除“Teacher 已知错误峰与人工 GT 直接对抗”的混杂因素。

## 6. 实施顺序

### Gate A：prepare-only

1. 创建两个互不覆盖的 sandbox/output 目录。
2. 验证 P1 checkpoint、model state、optimizer state、GT、split、manifest 与代码 hash。
3. 构建两个 Student clone 与一个冻结 Teacher；验证初始 state、optimizer state 与 parameter IDs 完全匹配。
4. 生成固定 5-epoch batch plan，证明每 epoch 1548 行恰好一次、locked/fixed81 为 0。
5. 验证 114/62 channel mask 精确，不执行 backward/step，不保存候选 checkpoint。

### Gate B：discarded paired smoke

1. 使用同一非 locked batch 对 C0/S1 各执行一次完整 forward/backward/optimizer step。
2. 两支输入 tensor、annotations、Teacher output 和 native/protection loss 必须一致；S1 只多 semantic loss。
3. 验证 loss/gradient 有限，34 个 trainable tensor 更新，冻结参数与 Teacher bit-exact。
4. 验证 target mask 对 114 target channel 生效，context protection 未被误屏蔽。
5. 丢弃两个 smoke state；不写正式 checkpoint，不计入 970 steps。

### Gate C：正式 paired 5 epochs

1. 重新从 P1 anchor 克隆两支，不复用 smoke state。
2. 对同一 batch 依次更新 C0、S1；每支严格 970 steps。
3. 每 epoch 原子保存一次两支 checkpoint；记录 loss、semantic margin、exposure、参数与 optimizer hash。
4. 任一非有限值、样本错序、branch 输入不一致、mask 错位或冻结参数变化立即停止两支。

### Gate D：internal-only 评估

对 epoch1-5 的 C0/S1 使用同一离线路径评估：

- fixed81 Native、Semantic Preservation、>30 px、margin protection；
- corrected22 的 target114 与 context62；
- 不运行 locked 视频。

## 7. P2 评分与通过门槛

### A. 匹配差异：S1 必须真的优于 C0

按同 epoch 比较 `error_C0 - error_S1`：

- target114 mean improvement >= 0.25 px；
- 至少 15/114 个 target point 改善 >= 0.5 px；
- target114 reverse-semantic inversion 数至少比 C0 少 1；
- 只改善 margin、不改善坐标，不计为通过。

### B. corrected/context 保护

- context62 中 S1 相对同 epoch C0 新增 >=5 px 退化：0；
- S1 target114 新增 >30 px 严重退化：0；
- C0/S1 都输出完整 22 frame、176 point，不允许缺点或非有限值。

### C. fixed81 绝对门槛

- mAP >=80.23；mAR >=85.40；RMSE <=7.25；RMSE pcutoff <=6.21。
- 新增 >30 px frame/point/high-confidence point：0/0/0。
- high-error frame/point 不高于 V0。
- endpoint、T-chain、posterior ratio 各 <=1.02。
- all-point ratio <=1.01；最大单 bodypart ratio <=1.02。
- reverse-margin inversion 不高于 V0 的 6。
- 421 个 V0 clean positive-margin 支持点中新增 sign reversal：0。

### D. epoch 选择规则

- 仅在同时通过 A/B/C 的 epoch 中选择 S1。
- 若多个 epoch 通过，选择 target114 mean differential improvement 最大者；差值在 0.01 px 内时选更早 epoch。
- C0 只作为匹配归因控制，不晋升。
- P2 通过也只允许进入 P3 计划，不直接成为最终模型。

## 8. 停止条件

任一情况发生即停止，不增加 epoch、不降低门槛：

- Gate A/B 任一契约失败；
- C0/S1 输入、batch 顺序、初始 model/optimizer state 不匹配；
- semantic weighted gradient 在冻结 hard smoke 上相对 native gradient 超出 0.5x-2.0x；
- Teacher、冻结层或 P1 anchor 被修改；
- 两支均无 material movement；
- S1 学会 target114 但 fixed81 protection 失败；
- 任何 locked 数据被读取用于本阶段。

## 9. 计算、磁盘与人工工作

- 服务器：远端 RTX 4090，遵守 DLC priority/owner。
- 预计总时间：约 1.5-3 小时。
- checkpoint：两支各 5 个，预计约 2.2-2.5 GB；启动前要求 C 盘至少 20 GiB 可用。
- 用户无需新增标注、无需查看 locked 视频。
- 只有 internal 指标出现无法解释的边界样本时，才生成少量高分辨率 GT panel 请求人工复核。

## 10. 回滚与结果记录

- P1 anchor、V0、P0/P0b/P1 所有结果只读，不覆盖。
- C0、S1、smoke、正式 run 和评估各用独立目录。
- 每阶段保存 state、日志、代码/input hash、sample exposure、optimizer/model hash 和同步 manifest。
- 失败保留证据，但不得把失败 checkpoint 改名为候选或继续训练。

## 11. 后续分支

- S1 通过：另写 P3 计划，沿同一 S1 optimizer trajectory 先延长到累计 epoch30。
- C0/S1 都不学习：停止当前 semantic loss，不增加 epoch；重新讨论表示能力。
- S1 学会但保护失败：分解 Teacher mask/梯度冲突，不放宽 fixed81 门槛。
- P3 内部通过后：才单独确认最小 locked short validation。
- 长视频保护通过但 test_video 仍差：另立 `F1609_v7.1_domain_robustness`，只用非 locked 训练图像设计通用增强。

## 12. 当前授权边界

本文件只完成 P2 的冻结实验设计。当前尚未：

- 创建 P2 sandbox；
- 上传或启动 P2 runner；
- 执行 paired smoke 或正式 5-epoch 训练；
- 预测或渲染 locked 视频；
- 晋升任何模型。

用户确认本计划后，建议一次授权整个 Gate A -> Gate D 串行梯级：自动通过才前进，任一 gate 失败立即停止并报告，无需在每个无争议小步骤重复等待人工确认。

