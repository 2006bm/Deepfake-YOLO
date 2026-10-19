#!/usr/bin/env python
"""Step 3.2: NT 硬子群的定量归因。

对每一对 (Real_xxx_fNNNNN, NeuralTextures_xxx_yyy_fNNNNN) —— 同源视频、同帧号 —— 计算
一组可观测变量, 然后比较"低分 NT"与"高分 NT"在这组变量上的差异, 找出硬子群的关联因素。

变量(按合作者指定):
  GT 面积占比        人脸尺寸是否相关
  面区高频能量       复用 Step 1 的频域口径(fft_enhance.high_freq_fft, cutoff 0.15)
  Real−Fake 差异强度  伪造对该区域的实际改动有多大(注意: 仅用于解释机制, **不可用作部署特征**,
                     因为部署时拿不到同一帧的真实版本)
  面区边界梯度差      融合边界是否更不明显
  图像模糊度         排除低质量输入
  源视频 ID          是否集中于少数身份/视频

纪律
----
  1. 全部分组只用**模型分数**, 不引入任何标签信息; 阈值 0.25 只用于分组定义, 不做优化。
  2. 源视频集中度按"每视频低分个数 vs 二项零假设"检验, 而不是只看 top-k 占比。
  3. 每个变量的组间比较给出效应量(Cliff's δ), 不只给 p 值 —— n 小时 p 值不说明问题大小。

用法
----
  python -I evaluate/nt_subgroup_attribution.py --data data/dataset_v1 --split val
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'preprocessing'))
from fft_enhance import high_freq_fft                        # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def rasterize(path, w, h):
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


def hf_mean(bgr, mask, cutoff=0.15):
    hf = high_freq_fft(bgr, cutoff)
    mag = np.sqrt((hf ** 2).sum(axis=2))
    sel = mask.astype(bool)
    return float(mag[sel].mean()) if sel.any() else 0.0


def blur_var(bgr, mask):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    sel = mask.astype(bool)
    return float(lap[sel].var()) if sel.any() else 0.0


def boundary_grad(bgr, mask):
    """掩码边界环上的平均梯度幅值 —— 融合边界不明显的伪造, 这里会更低。"""
    k = np.ones((5, 5), np.uint8)
    ring = (cv2.dilate(mask, k, iterations=2) > 0) & (cv2.erode(mask, k, iterations=2) == 0)
    if not ring.any():
        return 0.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2)[ring].mean())


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (a.size * b.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--conf-csv', default='evaluate/conf_analysis_val.csv')
    ap.add_argument('--out', default='evaluate/nt_subgroup_attribution_val.csv')
    args = ap.parse_args()

    want = set(args.split.split(','))
    split_of, real_at, fake_at = {}, defaultdict(dict), {}
    with open(os.path.join(args.data, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] not in want:
                continue
            stem = os.path.basename(r['image'])[:-4]
            split_of[stem] = r['split']
            fr = int(r['frame'])
            if r['is_fake'] == '0':
                real_at[r['video_id']][fr] = stem
            elif r['method'] == 'NeuralTextures':
                fake_at[(r['video_id'].split('_')[0], fr)] = stem

    conf = {r['stem']: float(r['max_conf'])
            for r in csv.DictReader(open(args.conf_csv, encoding='utf-8'))}

    def ip(stem):
        return os.path.join(args.data, 'images', split_of[stem], stem + '.jpg')

    def lp(stem):
        return os.path.join(args.data, 'labels', split_of[stem], stem + '.txt')

    pairs = sorted((src, fr, real_at[src][fr], fstem)
                   for (src, fr), fstem in fake_at.items() if fr in real_at.get(src, {}))
    print(f'配对 {len(pairs)} 对 (NT, {args.split})')

    rows = []
    for i, (src, fr, rstem, fstem) in enumerate(pairs):
        ri, fi = cv2.imread(ip(rstem)), cv2.imread(ip(fstem))
        if ri is None or fi is None or ri.shape != fi.shape:
            continue
        oh, ow = fi.shape[:2]
        mask = rasterize(lp(fstem), ow, oh)
        if mask.sum() == 0:
            continue
        rows.append(dict(
            src=src, frame=fr, real_stem=rstem, fake_stem=fstem,
            conf=conf.get(fstem, np.nan),
            gt_frac=float(mask.mean()),
            hf_fake=hf_mean(fi, mask), hf_real=hf_mean(ri, mask),
            blur_fake=blur_var(fi, mask), blur_real=blur_var(ri, mask),
            edge_fake=boundary_grad(fi, mask), edge_real=boundary_grad(ri, mask)))
        if (i + 1) % 40 == 0:
            print(f'\r  {i+1}/{len(pairs)}', end='', flush=True)
    print('\r  完成            ')

    for r in rows:
        r['delta_hf'] = r['hf_fake'] - r['hf_real']
        r['delta_edge'] = r['edge_fake'] - r['edge_real']
        r['delta_blur'] = r['blur_fake'] - r['blur_real']
        r['group'] = 'low' if r['conf'] < args.conf_thr else 'high'

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    low = [r for r in rows if r['group'] == 'low']
    high = [r for r in rows if r['group'] == 'high']
    print(f'\n低分子群 n={len(low)}   高分组 n={len(high)}   (阈值 {args.conf_thr})\n')

    print('=' * 104)
    print('变量归因(低分 vs 高分)。Cliff\'s δ: |δ|>0.33 中等, >0.47 大')
    print('=' * 104)
    print(f'{"变量":<22}{"低分中位":>12}{"高分中位":>12}{"Mann-Whitney p":>16}'
          f'{"Cliff δ":>10}{"解读":>26}')
    VAR = [('gt_frac', 'GT 面积占比'), ('hf_fake', '面区高频能量(伪造)'),
           ('delta_hf', 'Real−Fake 高频差'), ('blur_fake', '面区模糊度(伪造)'),
           ('edge_fake', '面区边界梯度(伪造)'), ('delta_edge', 'Real−Fake 边界梯度差'),
           ('conf', '实例分数(用于分组)')]
    for key, name in VAR:
        a = [r[key] for r in low if r[key] == r[key]]
        b = [r[key] for r in high if r[key] == r[key]]
        if len(a) < 2 or len(b) < 2:
            continue
        p = st.mannwhitneyu(a, b, alternative='two-sided').pvalue
        d = cliffs_delta(a, b)
        verdict = ('差异大' if abs(d) > 0.47 else '中等差异' if abs(d) > 0.33
                   else '弱差异' if abs(d) > 0.15 else '几乎无差异')
        print(f'{name:<22}{np.median(a):>12.4f}{np.median(b):>12.4f}{p:>16.4f}'
              f'{d:>+10.3f}{verdict:>26}')

    print()
    print('=' * 104)
    print('源视频集中度: 每视频低分个数 vs 二项零假设(按该视频的配对数)')
    print('=' * 104)
    tot = Counter(r['src'] for r in rows)
    lsc = Counter(r['src'] for r in low)
    pbits = len(low) / len(rows)
    print(f'  配对来自 {len(tot)} 个源视频; 每视频配对数 中位 {int(np.median(list(tot.values())))} '
          f'范围 {min(tot.values())}-{max(tot.values())}')
    print(f'  低分子群覆盖 {len(lsc)} 个视频 ({len(lsc)/len(tot)*100:.0f}%)')
    print(f'  {"低分个数":<10}{"实测视频":>10}{"二项期望":>12}')
    mx = max(tot.values())
    for k in range(mx + 1):
        o = sum(1 for v in tot if lsc.get(v, 0) == k)
        e = sum(st.binom.pmf(k, tot[v], pbits) for v in tot)
        if o or e > 1:
            print(f'  {k:<10}{o:>10}{e:>12.1f}')
    hard = [v for v in tot if lsc.get(v, 0) == tot[v]]
    print(f'  全部配对数都低的视频: {len(hard)} 个 ({len(hard)/len(tot)*100:.0f}%), '
          f'覆盖低分样本 {sum(lsc.get(v,0) for v in hard)} 张 '
          f'({sum(lsc.get(v,0) for v in hard)/max(len(low),1)*100:.0f}%)')
    print(f'\n逐对结果 {args.out}')


if __name__ == '__main__':
    main()
