from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    import deeplabcut

    args.output.mkdir(parents=True, exist_ok=True)
    videos = [str(args.video.resolve())]
    common = {
        "shuffle": args.shuffle,
        "videotype": args.video.suffix,
        "destfolder": str(args.output.resolve()),
    }
    deeplabcut.analyze_videos(str(args.config.resolve()), videos, save_as_csv=True, **common)
    if args.filter:
        deeplabcut.filterpredictions(str(args.config.resolve()), videos, save_as_csv=True, **common)
    if args.render:
        deeplabcut.create_labeled_video(str(args.config.resolve()), videos, filtered=args.filter, **common)


if __name__ == "__main__":
    main()
