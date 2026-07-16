# F1609_v7 P2 最终实验结论

## 状态

Gate C 和 Gate D 均完整运行。结论为 rejected，不是运行中断。

## 设计

- C0：DLC native loss 加 V0 Teacher protection。
- S1：与 C0 相同，仅增加固定 semantic identity loss。
- 两支从同一 model 和独立克隆的 optimizer anchor 出发。
- 每个 batch 共享输入、增强和 Teacher output。
- 5 epochs，每支 970 steps。
- fixed81 只评估，locked 读取和预测为 0。

## Gate C

全部运行完整性门槛通过：optimizer storage 独立；epoch step 精确为 385、579、773、967、1161；5 对 checkpoint 原子写入；Teacher、冻结层和 anchor 不变。

## Gate D

五个 S1 epoch 都保护了 fixed81，但没有 material semantic movement。

target114 mean improvement，单位 px：

- epoch 1：-0.00000968
- epoch 2：+0.00010713
- epoch 3：+0.00021162
- epoch 4：-0.00066165
- epoch 5：-0.00087602

全部 epoch 的改善至少 0.5 px 点数为 0/114；C0 和 S1 的 target reverse inversion 都是 28；reduction 为 0。

门槛是 mean improvement 至少 0.25 px、至少 15/114 点改善 0.5 px、inversion 至少减少 1。没有 epoch 接近门槛。

## 决策

- 不运行 P3。
- 不降低门槛。
- 不晋升 P2 checkpoint。
- 最终冻结 F1609_v0_raw snapshot-best-120。
- P2 实现只作为研究证据保留。
