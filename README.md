# F1609 insect 8-point pose tracking

F1609 是一个基于 DeepLabCut 3 PyTorch engine 的单虫体 8 点姿态追踪项目。

关键点顺序：

Head, T1, T2, T3, A3, A5, A7, Tail

发布状态：2026-07-17 冻结。

## 最终交付

- 默认模型：F1609_v0_raw snapshot-best-120。
- 可选后处理：Tail repair v2，仅修改 Tail。
- checkpoint 内容哈希与自动验证命令。
- 脱敏的 DLC project 和 model config 模板。
- 项目准备、视频推理、Tail repair 和发布检查入口。
- Tail-only 修改合同的自动测试。
- 模型卡、复现说明和失败实验的完整结论。

训练图像、标签、源视频、locked 视频和 checkpoint 本体不进入 Git。checkpoint 单独作为 Release artifact 分发。

## checkpoint

文件名：snapshot-best-120.pt

SHA256：504521C281BF9C1E524B470C590300CFD1FB18D5535C7A119A53DD9DF8A4BC79

文件大小：95,869,810 bytes

公开下载：

https://github.com/LYM0905/F1609-pose-tracking/releases/download/v1.0.0/snapshot-best-120.pt

也可以使用 GitHub CLI：

    gh release download v1.0.0 --repo LYM0905/F1609-pose-tracking --pattern snapshot-best-120.pt

校验：

    python scripts/verify_checkpoint.py path/to/snapshot-best-120.pt

## 快速使用

安装：

    python -m pip install -e .[dlc]

创建便携 DLC 项目：

    python scripts/prepare_dlc_project.py       --checkpoint path/to/snapshot-best-120.pt       --destination work/f1609-dlc-project

推理：

    python scripts/run_inference.py       --config work/f1609-dlc-project/config.yaml       --video path/to/video.mp4       --output outputs/predictions       --filter

可选 Tail repair：

    python scripts/apply_tail_repair.py       --input outputs/predictions/video_filtered.h5       --output outputs/predictions/video_tail_repair_v2.h5       --log outputs/predictions/video_tail_repair_v2_log.csv

原始 DLC 预测不会被覆盖。

## 为什么最终仍是 best120

best120 是至今人工 locked 视频审核中整体最强且最稳定的模型。后续 resume、局部再训练和 semantic loss 实验没有在保护旧能力的同时产生可测的语义改善。

最新 P2 实验完整运行了 5 个 matched C0/S1 epoch。S1 相对 C0 的 target114 mean improvement 仅在约 -0.00088 到 +0.00021 px 之间；改善至少 0.5 px 的点为 0/114；反向语义 inversion 没有减少。因此 P3 没有启动，P2 checkpoint 不属于发布模型。

## 验证定位

- 04_water：整体改善明显，接近可接受。
- 09_acetone：后段链条更合理。
- 05_acetone：总体正向改善，仍有局部体节语义错误。
- test_video_bgclean_staticdot：best120 优于同训练后期 checkpoint，但困难帧仍可能出现 Head/Tail 和 T/A 链条错误。
- Tail repair v2：完成 16 个 locked 短片回归，用户明确确认 Tail 问题在该审核范围内解决。

## 发布流程 smoke test

发布包已在服务器 DeepLabCut 3 环境完成端到端 smoke test：

- 输入：24 张非 locked 训练图组成的临时视频；
- checkpoint 校验：通过；
- portable DLC project：成功创建；
- analyze_videos：24/24 帧完成；
- median filtering：完成；
- Tail repair CLI：完成；
- 输出：24/24 行、列结构一致、全部数值 finite。
## 重要限制

- Head/Tail 互换仍可能发生。
- T1/T2/T3 与 A3/A5/A7 仍可能落到反向对应体节。
- likelihood 高不代表语义一定正确。
- Tail repair v2 是时间插值，不是图像解剖模型。
- 自动指标用于筛查，视觉关键应用仍需人工抽查。

详见 docs/MODEL_CARD.zh-CN.md 和 reports/P2_RESULT.zh-CN.md。

## 仓库检查

    python -m unittest discover -s tests -v
    python scripts/verify_release.py

本仓库当前按 public 发布，但尚未附加许可证，因此默认保留全部权利。第三方可以查看和 clone；如需复用、修改或再分发，应先获得项目所有者许可或等待后续许可证。

checkpoint 不进入 Git 历史，已作为 `v1.0.0` GitHub Release asset 公开下载。模型权重尚未附加独立许可证；进一步复用、修改或再分发仍需遵守项目所有者许可。
