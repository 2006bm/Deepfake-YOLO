# -*- coding: utf-8 -*-
"""c40 压缩鲁棒性实验 —— 只读预检(不抽帧、不改任何文件)。

四项检查:
  1. 文件映射: test 的每个 c23 视频都有对应 c40 视频(文件名一致)
  2. 视频完整性: 抽样用 cv2 读帧数/分辨率/帧率, 对比 c23 vs c40
  3. 帧对应: 相同帧索引解码 c23/c40, 检查尺寸一致
  4. mask 对齐前提: 确认 c40 与 c23 分辨率一致(分辨率变了 mask 坐标才失效)

用法: python -I evaluate/c40_precheck.py
"""
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RAW = 'data/raw'
ORIG_C23 = f'{RAW}/original_sequences/youtube/c23/videos'
ORIG_C40 = f'{RAW}/original_sequences/youtube/c40/videos'
MANIP_C23 = f'{RAW}/manipulated_sequences/{{m}}/c23/videos'
MANIP_C40 = f'{RAW}/manipulated_sequences/{{m}}/c40/videos'
METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def video_meta(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return dict(n=n, w=w, h=h, fps=round(fps, 2))


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def main():
    # ---- 提取 test 清单 ----
    rows = list(csv.DictReader(open('data/dataset_v1/metadata.csv', encoding='utf-8')))
    test = [r for r in rows if r['split'] == 'test']
    orig_ids = sorted({r['video_id'] for r in test if r['is_fake'] == '0'})
    manip = defaultdict(set)          # method -> set of src_tgt
    frame_of = {}                     # (method_or_original, name) -> 抽样帧索引
    for r in test:
        fr = int(r['frame'])
        if r['is_fake'] == '0':
            frame_of[('original', r['video_id'])] = fr
        else:
            manip[r['method']].add(r['video_id'])
            frame_of[(r['method'], r['video_id'])] = fr

    print(f'test: original {len(orig_ids)}, manipulated 各方法 {[len(manip[m]) for m in METHODS]}')

    # ============ 检查 1: 文件映射 ============
    print()
    print('=' * 84)
    print('检查 1: 文件映射 (c23 存在 + c40 存在)')
    print('=' * 84)
    miss = []
    def chk(base_c23, base_c40, name):
        c23 = os.path.join(base_c23, name + '.mp4')
        c40 = os.path.join(base_c40, name + '.mp4')
        s23, s40 = os.path.exists(c23), os.path.exists(c40)
        if not (s23 and s40):
            miss.append(('c23' if not s23 else 'c40', name))
        return s23, s40
    for vid in orig_ids:
        chk(ORIG_C23, ORIG_C40, vid)
    for m in METHODS:
        for v in manip[m]:
            chk(MANIP_C23.format(m=m), MANIP_C40.format(m=m), v)
    print(f'  映射缺失: {len(miss)} / {len(orig_ids) + sum(len(manip[m]) for m in METHODS)}')
    if miss:
        for side, name in miss[:10]:
            print(f'    [{side}] {name}')

    # ============ 检查 2+3: 视频完整性 + 帧对应(抽样) ============
    print()
    print('=' * 84)
    print('检查 2+3: 抽样视频完整性 + 帧对应 (每种方法 5 个 + original 5 个)')
    print('=' * 84)
    print(f'{"类型":<10}{"视频":<12}{"c23帧数":>8}{"c40帧数":>8}{"帧数一致":>9}'
          f'{"c23尺寸":>10}{"c40尺寸":>10}{"尺寸一致":>9}{"解码帧":>7}')
    rng = np.random.default_rng(0)
    samples = [('original', v) for v in rng.choice(orig_ids, 5, replace=False)]
    for m in METHODS:
        for v in rng.choice(sorted(manip[m]), 5, replace=False):
            samples.append((m, v))
    n_frame_mismatch = n_size_mismatch = n_decode_fail = 0
    for typ, name in samples:
        if typ == 'original':
            p23, p40 = os.path.join(ORIG_C23, name + '.mp4'), os.path.join(ORIG_C40, name + '.mp4')
        else:
            p23 = os.path.join(MANIP_C23.format(m=typ), name + '.mp4')
            p40 = os.path.join(MANIP_C40.format(m=typ), name + '.mp4')
        m23, m40 = video_meta(p23), video_meta(p40)
        if m23 is None or m40 is None:
            n_decode_fail += 1
            print(f'{typ:<10}{name:<12}  [解码失败 c23={m23 is not None} c40={m40 is not None}]')
            continue
        fm = 'OK' if m23['n'] == m40['n'] else '不一致'
        sz = 'OK' if (m23['w'], m23['h']) == (m40['w'], m40['h']) else '不一致'
        if fm != 'OK':
            n_frame_mismatch += 1
        if sz != 'OK':
            n_size_mismatch += 1
        # 帧对应: 用该样本在 metadata 里的 frame 索引解码
        fr = frame_of.get((typ, name))
        ok_decode = 'OK'
        if fr is not None:
            f23 = read_frame(p23, fr)
            f40 = read_frame(p40, fr)
            if f23 is None or f40 is None:
                ok_decode = '失败'
                n_decode_fail += 1
            elif f23.shape != f40.shape:
                ok_decode = '尺寸不同'
        else:
            ok_decode = '无frame'
        print(f'{typ:<10}{name:<12}{m23["n"]:>8}{m40["n"]:>8}{fm:>9}'
              f'{(m23["w"],m23["h"])!s:>10}{(m40["w"],m40["h"])!s:>10}{sz:>9}{ok_decode:>7}')
    print(f'\n  帧数不一致: {n_frame_mismatch}   尺寸不一致: {n_size_mismatch}   解码失败: {n_decode_fail}')

    # ============ 检查 4: mask 对齐前提 ============
    print()
    print('=' * 84)
    print('检查 4: mask 对齐前提 (c40 分辨率 == c23 分辨率?)')
    print('=' * 84)
    print('  若分辨率一致, GT 掩码坐标可直接复用(掩码是归一化坐标, 与压缩级别无关)')
    print('  结论: ' + ('分辨率一致, 掩码可复用' if n_size_mismatch == 0
                       else f'有 {n_size_mismatch} 个抽样分辨率不一致, 需进一步排查'))


if __name__ == '__main__':
    main()
