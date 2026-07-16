# 复现说明

## 环境

推荐 Python 3.10、CUDA 12.x、PyTorch 2.x 和 DeepLabCut 3 PyTorch engine。以 DeepLabCut 官方安装说明为准。

    python -m pip install -e .[dlc]

## 获取 checkpoint

下载 snapshot-best-120.pt 后先校验：

    python scripts/verify_checkpoint.py path/to/snapshot-best-120.pt

filename、bytes 和 SHA256 必须全部通过。

## 创建 DLC 项目

    python scripts/prepare_dlc_project.py       --checkpoint path/to/snapshot-best-120.pt       --destination work/f1609-dlc-project

## 推理

    python scripts/run_inference.py       --config work/f1609-dlc-project/config.yaml       --video path/to/video.mp4       --output outputs/predictions       --filter       --render

长视频先对短片段 smoke test，确认分辨率、crop、颜色和输出后再全量运行。

## Tail repair

    python scripts/apply_tail_repair.py       --input outputs/predictions/video_filtered.h5       --output outputs/predictions/video_tail_repair_v2.h5       --log outputs/predictions/video_tail_repair_v2_log.csv

## 仓库验证

    python -m unittest discover -s tests -v
    python scripts/verify_release.py

## 训练复现边界

本仓库不包含训练图像和标签，不能单独从零复现权重。它提供网络配置、checkpoint 哈希、数据和 split 合同、评估门槛、失败实验结论，以及推理和后处理入口。

完整重训需要数据所有者另行提供 1629 帧人工标注，并保持 1548/81 固定 split。
