#!/usr/bin/env python
"""配对频谱分析 —— 增强之前, NeuralTextures 的高频证据到底存不存在?

动机
----
FFT λ=1.0 的假设是"NT 的伪影藏在高频里, 放大高频能帮模型看到它"。
直接训练验证的代价是 2.6 小时 GPU, 而且一旦失败分不清是"高频里没东西"
还是"增强方式不对"。这里先用零成本的方式回答前置问题:

    NT 人脸区域的高频含量, 和**同一张脸的未伪造版本**比, 有区别吗?

配对设计(这是本脚本的关键)
--------------------------
FF++ 的 real 图与 fake 图来自同一批源视频, 且 Deepfakes / FaceSwap /
NeuralTextures 的抽帧位置与 real **完全对齐**:

    Real_001_f00046.jpg  <->  NeuralTextures_001_xxx_f00046.jpg

同一个人、同一姿态、同一光照、同一背景。所以可以在**同一帧上**取
GT 人脸区域, 分别算 real 版和 fake 版的高频能量, 得到配对差

    Δ = E_h(fake 区域) − E_h(real 同区域)

身份/姿态/光照/背景/压缩世代这些混淆项被配对直接消掉, 剩下的差异
只可能来自伪造本身。这比"同图非人脸区做对照"强得多 —— 后者会被
背景纹理(头发/树/墙)污染, E_h 的绝对值根本不可比。

判读
----
  若 Δ 的排序 FS > DF > NT 且 NT 接近 0
      -> 高频证据确实随伪造方法变弱, 与 baseline 漏检率排序吻合, FFT 方向有依据
  若各方法 Δ 都接近 0
      -> 增强没有东西可放大, FFT 方向到此为止

同时给出 real–real 配对作为噪声地板(同一视频的两个不同 real 帧),
用来判断 Δ 的量级是否超过自然帧间波动。

用法
----
  python -I evaluate/spectral_paired_analysis.py --data data/dataset_v1 --split val
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def high_freq_fft(bgr, cutoff, hp_cache):
    """与 preprocessing/fft_enhance.py 的 high_freq_fft 同一带通。

    用 cv2.dft 实现以提速(等价于 numpy 的 fft2 + fftshift + 高通掩码 + ifft2)。
    """
    h, w = bgr.shape[:2]
    key = (h, w, cutoff)
    hp = hp_cache.get(key)
    if hp is None:
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        hp = (dist > cutoff).astype(np.float32)
        hp_cache[key] = hp
    out = np.empty(bgr.shape, np.float32)
    for c in range(3):
        f = cv2.dft(bgr[:, :, c].astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f = np.fft.fftshift(f, axes=(0, 1))
        f *= hp[:, :, None]
        f = np.fft.ifftshift(f, axes=(0, 1))
        out[:, :, c] = cv2.idft(f, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return out


def rasterize_label(path, w, h):
    c = np.zeros((h, w), np.uint8)
    if not os.path.exists(path):
        return c
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            xy = np.array(p[1:], dtype=np.float64).reshape(-1, 2)
            cv2.fillPoly(c, [np.round(np.stack([xy[:, 0] * w,
                                                xy[:, 1] * h], 1)).astype(np.int32)], 1)
    return c


def hf_stats(bgr, mask, cutoff, hp_cache):
    """返回 (区域内高频能量均值, 全图高频能量均值)。"""
    hf = high_freq_fft(bgr, cutoff, hp_cache)
    mag = np.sqrt((hf ** 2).sum(axis=2))          # 逐像素跨通道 L2
    sel = mask.astype(bool)
    in_region = float(mag[sel].mean()) if sel.any() else 0.0
    whole = float(mag.mean())
    return in_region, whole


def summarize(name, deltas):
    d = np.asarray(deltas, float)
    if d.size == 0:
        print(f'  {name:<22} n=0')
        return None
    mean, med = float(d.mean()), float(np.median(d))
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    se = sd / np.sqrt(d.size) if d.size > 1 else 0.0
    pos = float((d > 0).mean()) * 100
    cohen = mean / sd if sd > 0 else 0.0
    print(f'  {name:<22} n={d.size:>5}  均值Δ={mean:>8.3f}  中位Δ={med:>8.3f}  '
          f'SD={sd:>7.3f}  95%CI=[{mean-1.96*se:>7.3f},{mean+1.96*se:>7.3f}]  '
          f'Δ>0占比={pos:>5.1f}%  d={cohen:>6.2f}')
    return dict(n=int(d.size), mean=mean, median=med, sd=sd,
                ci_lo=mean - 1.96 * se, ci_hi=mean + 1.96 * se,
                frac_pos=pos, cohen_d=cohen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/dataset_v1',
                    help='必须用**原图**数据集(dataset_v1) —— 测的是增强之前')
    ap.add_argument('--meta', default=None, help='metadata.csv, 默认 <data>/metadata.csv')
    ap.add_argument('--split', default='val',
                    help="划分, 逗号分隔可多选, 'all' 表示全部")
    ap.add_argument('--frame-tol', type=int, default=0,
                    help='帧号容差。0=要求帧号完全一致; >0 时退化为"同源视频内最近帧"配对。'
                         'Face2Face 的抽帧位置与 real 完全错开, 必须靠它才能配上')
    ap.add_argument('--cutoff', type=float, default=0.15,
                    help='FFT 高通截止(与增强实验保持一致)')
    ap.add_argument('--max-per-method', type=int, default=0, help='0=不限制')
    ap.add_argument('--null-max', type=int, default=700,
                    help='real–real 噪声地板的配对上限(0=不限制)')
    ap.add_argument('--out', default='evaluate/spectral_paired_val.csv')
    args = ap.parse_args()

    meta = args.meta or os.path.join(args.data, 'metadata.csv')

    def img_path(stem):
        return os.path.join(args.data, 'images', split_of[stem], stem + '.jpg')

    def lab_path(stem):
        return os.path.join(args.data, 'labels', split_of[stem], stem + '.txt')

    # ---- 建立配对索引 ----
    want = None if 'all' in args.split.split(',') else set(args.split.split(','))
    split_of = {}                                     # stem -> split
    real_at = defaultdict(dict)                       # src -> {frame: stem}
    fake_at = {}                                      # (method, src, frame) -> stem
    with open(meta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if want is not None and r['split'] not in want:
                continue
            stem = os.path.basename(r['image'])[:-4]
            split_of[stem] = r['split']
            fr = int(r['frame'])
            if r['is_fake'] == '0':
                real_at[r['video_id']][fr] = stem
            else:
                src = r['video_id'].split('_')[0]
                fake_at[(r['method'], src, fr)] = stem

    print(f'数据集 {args.data}  划分 {args.split}  cutoff={args.cutoff}  '
          f'帧容差 {args.frame_tol}')
    print(f'源视频 {len(real_at)} 个, 伪造样本 {len(fake_at)} 张\n')

    hp_cache = {}
    rows = []
    skipped_shape = 0
    per_method = defaultdict(list)

    for m in METHODS:
        pairs = []
        for (mm, src, fr) in fake_at:
            if mm != m:
                continue
            rf = real_at.get(src)
            if not rf:
                continue
            if fr in rf:
                pairs.append((src, fr, fr, 0))
            elif args.frame_tol:
                near = min(rf, key=lambda x: abs(x - fr))
                if abs(near - fr) <= args.frame_tol:
                    pairs.append((src, fr, near, abs(near - fr)))
        pairs.sort()
        if args.max_per_method:
            pairs = pairs[:args.max_per_method]
        for i, (src, fr, rfr, off) in enumerate(pairs):
            fake_stem = fake_at[(m, src, fr)]
            real_stem = real_at[src][rfr]
            fi = cv2.imread(img_path(fake_stem))
            ri = cv2.imread(img_path(real_stem))
            if fi is None or ri is None:
                continue
            if fi.shape != ri.shape:
                skipped_shape += 1
                continue
            h, w = fi.shape[:2]
            mask = rasterize_label(lab_path(fake_stem), w, h)
            if mask.sum() == 0:
                continue
            eh_f, whole_f = hf_stats(fi, mask, args.cutoff, hp_cache)
            eh_r, whole_r = hf_stats(ri, mask, args.cutoff, hp_cache)
            rows.append(dict(
                real_stem=real_stem, fake_stem=fake_stem, method=m,
                src=src, frame=fr, frame_real=rfr, frame_off=off,
                gt_px=int(mask.sum()),
                eh_fake=eh_f, eh_real=eh_r, delta=eh_f - eh_r,
                delta_norm=(eh_f - eh_r) / whole_r if whole_r else 0.0,
                whole_real=whole_r, whole_fake=whole_f))
            per_method[m].append(eh_f - eh_r)
            if (i + 1) % 50 == 0:
                print(f'\r  {m} 配对 {i+1}/{len(pairs)}', end='', flush=True)
        print(f'\r  {m} 配对完成 {len(pairs)} 对            ')

    # ---- real–real 噪声地板 ----
    null_rows = []
    for src, frames in real_at.items():
        if args.null_max and len(null_rows) >= args.null_max:
            break
        fs = sorted(frames)
        if len(fs) < 2:
            continue
        # 用该视频任一样本的 GT 掩码(人脸位置在同一源视频内一致)
        cand = [k for k in fake_at if k[1] == src]
        if not cand:
            continue
        m, _, fr0 = cand[0]
        stem0 = fake_at[(m, src, fr0)]
        ri = cv2.imread(img_path(real_at[src][fr0]))
        if ri is None:
            continue
        h, w = ri.shape[:2]
        mask = rasterize_label(lab_path(stem0), w, h)
        if mask.sum() == 0:
            continue
        for a, b in zip(fs[:-1], fs[1:]):
            ia = cv2.imread(img_path(real_at[src][a]))
            ib = cv2.imread(img_path(real_at[src][b]))
            if ia is None or ib is None or ia.shape != ib.shape:
                continue
            ea, _ = hf_stats(ia, mask, args.cutoff, hp_cache)
            eb, _ = hf_stats(ib, mask, args.cutoff, hp_cache)
            null_rows.append(eb - ea)

    # ---- 报告 ----
    print()
    print('=' * 108)
    print(f'配对高频能量差 Δ = E_h(fake 的 GT 人脸区) − E_h(real 同区域)   [cutoff={args.cutoff}]')
    print('=' * 108)
    summary = {}
    for m in METHODS:
        s = summarize(m, per_method.get(m, []))
        if s:
            summary[m] = s
    s_null = summarize('real–real 噪声地板', null_rows)

    print()
    print('=' * 108)
    print('按 baseline 漏检率排序对照(漏检越高的方法, Δ 是否越小?)')
    print('=' * 108)
    fail = {'FaceSwap': 1.7, 'Face2Face': 4.3, 'Deepfakes': 4.5, 'NeuralTextures': 27.1}
    print(f'{"method":<18}{"baseline漏检率":>16}{"n":>7}{"中位Δ":>12}{"Δ>0占比":>12}{"Cohen d":>11}')
    for m in sorted(METHODS, key=lambda x: fail[x]):
        s = summary.get(m)
        if not s:
            continue
        print(f'{m:<18}{fail[m]:>15.1f}%{s["n"]:>7}{s["median"]:>12.3f}'
              f'{s["frac_pos"]:>11.1f}%{s["cohen_d"]:>11.2f}')

    if skipped_shape:
        print(f'\n注: 因 real/fake 图像尺寸不一致跳过 {skipped_shape} 对')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    if not rows:
        print('\n没有配到任何样本 —— 检查 --split / --frame-tol / 路径。未写入文件。')
        return
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f'\n逐对结果已写入 {args.out}  ({len(rows)} 对)')


if __name__ == '__main__':
    main()
