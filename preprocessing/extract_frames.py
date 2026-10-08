#!/usr/bin/env python
"""视频抽帧工具。

抽帧策略:在视频中段均匀取 k 帧,两端各留 margin 比例避开头尾的黑场/转场。
关键约束:同一视频与其 mask 视频必须用**同一组帧下标**,否则标签会错位。
因此 build_dataset.py 统一调用本模块的 pick_indices() 生成下标,再分别读取。
"""
import argparse
import os
import sys

import cv2

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def pick_indices(n_frames, k, margin=0.10):
    """返回 k 个从 n_frames 中均匀选取的帧下标,两端各跳过 margin 比例。"""
    if n_frames <= 0 or k <= 0:
        return []
    lo = int(n_frames * margin)
    hi = max(lo + 1, int(n_frames * (1 - margin)))
    if k == 1:
        return [(lo + hi) // 2]
    span = hi - lo
    if span < k:                      # 视频太短,退化为逐帧
        return list(range(lo, min(hi + 1, n_frames)))
    return sorted({lo + round(i * span / (k - 1)) for i in range(k)})


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'无法打开视频: {path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def read_frames(path, indices):
    """按给定下标读取帧,返回 {idx: frame(BGR)}。缺失的帧不会出现在结果中。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f'无法打开视频: {path}')
    out = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out[i] = fr
    cap.release()
    return out


def main():
    ap = argparse.ArgumentParser(description='视频抽帧工具(可单独运行用于检查)')
    ap.add_argument('video')
    ap.add_argument('--k', type=int, default=15, help='抽取帧数')
    ap.add_argument('--margin', type=float, default=0.10)
    ap.add_argument('--save', default=None, help='保存目录(可选)')
    args = ap.parse_args()

    n, w, h = video_meta(args.video)
    idx = pick_indices(n, args.k, args.margin)
    print(f'{os.path.basename(args.video)}: {w}x{h}, 共 {n} 帧')
    print(f'选取 {len(idx)} 帧: {idx}')

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        frames = read_frames(args.video, idx)
        for i, fr in frames.items():
            cv2.imwrite(os.path.join(args.save, f'{i:05d}.jpg'), fr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'已保存 {len(frames)} 帧到 {args.save}')


if __name__ == '__main__':
    main()
