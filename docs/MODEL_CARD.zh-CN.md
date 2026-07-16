# F1609 模型卡

## 冻结版本

最终默认模型是 F1609_v0_raw snapshot-best-120。更新版本不会因为编号更大而自动晋升。

## 输出定义

模型输出 8 个独立 DLC 通道及 likelihood：Head、T1、T2、T3、A3、A5、A7、Tail。骨架顺序用于展示和评估，不取代独立点识别。

## 数据合同

- 人工标注：1629 帧。
- 固定 train：1548 帧。
- 固定 internal test：81 帧。
- locked 视频进入训练：0。
- locked 视频不得用于 loss weight、threshold 或 epoch 选择。

数据和视频不随仓库公开。

## fixed81 指标

- mAP：83.2312
- mAR：88.3951
- RMSE：6.0381 px
- RMSE with pcutoff：5.6477 px
- pcutoff：0.6

这些指标不能充分发现高置信度语义错侧，因此不是唯一验收依据。

## Tail repair v2

Tail repair v2 是可选时间后处理，只修改 Tail。触发包括低 likelihood、Head/Tail 坍缩、Tail 距 A7 过远和相邻帧跳变。修复点来自稳定 Tail 锚点的插值或保持。

它已在 16 个 locked 短验证片段上回归，用户明确确认 Tail 问题在该审核范围内解决。它不是虫体 mask 或解剖轴模型，不能对新域作普遍保证。

## 已知限制

- 困难帧可能发生 Head/Tail 互换。
- T1/T2/T3 和 A3/A5/A7 可能落在反向对应体节。
- 高 likelihood 可能对应语义错误。
- 短 stress 视频可能比同环境长视频更难。
- 当前模型不输出虫体 mask，不能严格证明点在虫体内部。
- 自动结构指标可能漏检，重要部署应保留人工审核。

## 禁止的宣传结论

- 不声称所有姿态无错误。
- 不声称 Tail repair 具有普遍解剖学正确性。
- 不把 rejected P2 checkpoint 称为改进模型。
