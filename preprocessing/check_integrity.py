#!/usr/bin/env python
"""扫描 data/raw 下所有 mp4 的完整性。

背景:早期版本的下载器在连接中断时会把截断的 mp4 当作完整文件保存
(cv2 打开报 "moov atom not found")。此脚本用于找出这类损坏文件。

判定标准:cv2 能打开、首帧能解码、帧数 > 0。三项任一失败即视为损坏。

输出损坏文件清单到 _debug/bad_files.txt,可直接交给 download.py 重下。
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def check(path):
    """返回 (path, ok, reason)。"""
    try:
        if os.path.getsize(path) == 0:
            return path, False, 'zero_bytes'
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return path, False, 'open_failed'
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, fr = cap.read()
        cap.release()
        if not ok or fr is None:
            return path, False, 'decode_failed'
        if n <= 0:
            return path, False, 'zero_frames'
        return path, True, ''
    except Exception as e:
        return path, False, f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description='扫描视频文件完整性')
    ap.add_argument('--root', default='data/raw')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='_debug/bad_files.txt')
    args = ap.parse_args()

    files = []
    for dp, _, fns in os.walk(args.root):
        for fn in fns:
            if fn.lower().endswith('.mp4'):
                files.append(os.path.join(dp, fn))
    print(f'扫描 {len(files)} 个 mp4 ...')

    bad = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for path, ok, reason in ex.map(check, files, chunksize=16):
            done += 1
            if not ok:
                bad.append((path, reason))
            if done % 1000 == 0 or done == len(files):
                print(f'  {done}/{len(files)}  损坏 {len(bad)}')

    print(f'\n损坏文件: {len(bad)} 个')
    if bad:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            for p, r in bad:
                f.write(p + '\n')
        # 按目录归类,便于判断影响范围
        from collections import Counter
        by_dir = Counter(os.path.dirname(p).replace(os.sep, '/') for p, _ in bad)
        print('\n按目录分布:')
        for d, c in by_dir.most_common():
            print(f'  {c:5d}  {d}')
        print(f'\n清单已写入 {args.out}')
    else:
        print('全部文件完整。')


if __name__ == '__main__':
    main()
