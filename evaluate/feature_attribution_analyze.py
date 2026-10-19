# -*- coding: utf-8 -*-
"""Step 3.3 分析: 读取特征, 回答"低分 NT 特征接近 real 还是接近高分 NT"。

三个指标:
  a) 线性探针 AUROC  : real vs fake, 按源视频分组 K 折交叉验证(避免同视频相邻帧泄漏)
  b) 配对特征距离    : 同源同帧 real 与 fake 的欧氏距离, 低分 vs 高分
  c) 类间距离        : 低分fake / 高分fake / real 三者质心的成对距离(各层级)

用法: python -I evaluate/feature_attribution_analyze.py
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NPZ = 'evaluate/features_nt_val.npz'


def lr_fit(X, y, lam=1.0, max_iter=200):
    d = X.shape[1]
    def loss(wb):
        w, b = wb[:-1], wb[-1]
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return -(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean() \
               + lam * (w ** 2).sum() / X.shape[0]
    def grad(wb):
        w, b = wb[:-1], wb[-1]
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        dw = (X.T @ (p - y)) / X.shape[0] + 2 * lam * w / X.shape[0]
        db = (p - y).mean()
        return np.concatenate([dw, [db]])
    res = minimize(loss, np.zeros(d + 1), jac=grad, method='L-BFGS-B',
                   options={'maxiter': max_iter})
    return res.x


def auroc(y, s):
    y = y.astype(bool)
    npos = int(y.sum()); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s, kind='stable')
    rk = np.empty(s.size); rk[o] = np.arange(1, s.size + 1)
    return (rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def zscore(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    return (X - mu) / (sd + 1e-8), mu, sd


def main():
    d = np.load(NPZ)
    l1, l2, l3 = d['l1'], d['l2'], d['l3']
    real_idx = d['real_idx'].astype(int)
    fake_idx = d['fake_idx'].astype(int)
    src = np.array(d['src'])
    group = np.array(d['group'])
    conf = np.array(d['conf'], dtype=float)
    n = len(real_idx)
    print(f'配对 {n} 对, 低分 {int((group=="low").sum())} / 高分 {int((group=="high").sum())}')

    uniq_videos = np.unique(src)
    print(f'源视频 {len(uniq_videos)} 个\n')

    print('=' * 92)
    print('a) 线性探针 AUROC (real vs fake, 按源视频 5 折 CV, 逻辑回归 L2)')
    print('=' * 92)
    print(f'{"层级":<6}{"AUROC均值":>10}{"±SD":>8}{"min":>8}{"max":>8}   (特征维数)')
    for name, F in (('L1', l1), ('L2', l2), ('L3', l3)):
        Xr, Xf = F[real_idx], F[fake_idx]
        X = np.vstack([Xr, Xf]); y = np.concatenate([np.zeros(n), np.ones(n)])
        vid = np.concatenate([src, src])
        # 按视频打乱做 5 折(每个视频整体进 train 或 val)
        rng = np.random.default_rng(0)
        order = rng.permutation(len(uniq_videos))
        folds = np.array_split(order, 5)
        aucs = []
        for fold in folds:
            val_vid = set(uniq_videos[fold])
            tr = ~np.isin(vid, list(val_vid))
            va = ~tr
            Xtr, mu, sd = zscore(X[tr]); Xva, _, _ = zscore(X[va], mu, sd)
            wb = lr_fit(Xtr, y[tr])
            s = Xva @ wb[:-1] + wb[-1]
            aucs.append(auroc(y[va], s))
        aucs = np.array(aucs)
        print(f'{name:<6}{aucs.mean():>10.4f}{aucs.std():>8.4f}{aucs.min():>8.4f}'
              f'{aucs.max():>8.4f}   ({F.shape[1]})')

    print()
    print('=' * 92)
    print('b) 配对特征距离 real−fake (欧氏, 标准化后), 低分 vs 高分')
    print('=' * 92)
    print(f'{"层级":<6}{"全体中位":>10}{"低分中位":>10}{"高分中位":>10}{"低/高比":>10}'
          f'{"Mann-Whitney p":>16}')
    from scipy.stats import mannwhitneyu
    for name, F in (('L1', l1), ('L2', l2), ('L3', l3)):
        X = np.vstack([F[real_idx], F[fake_idx]])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        dist = np.linalg.norm(X[real_idx] - X[fake_idx], axis=1)
        lo = dist[group == 'low']; hi = dist[group == 'high']
        p = mannwhitneyu(lo, hi, alternative='two-sided').pvalue
        print(f'{name:<6}{np.median(dist):>10.4f}{np.median(lo):>10.4f}'
              f'{np.median(hi):>10.4f}{np.median(lo)/np.median(hi):>10.3f}{p:>16.4f}')

    print()
    print('=' * 92)
    print('c) 类间距离矩阵 (各群体质心间的欧氏距离, 标准化后)')
    print('   核心: dist(低分fake, real) vs dist(低分fake, 高分fake)')
    print('=' * 92)
    for name, F in (('L1', l1), ('L2', l2), ('L3', l3)):
        X = np.vstack([F[real_idx], F[fake_idx]])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        real = X[:n]; fake = X[n:]
        lo = fake[group == 'low']; hi = fake[group == 'high']
        c_real, c_lo, c_hi = real.mean(0), lo.mean(0), hi.mean(0)
        d_rl = np.linalg.norm(c_real - c_lo)
        d_rh = np.linalg.norm(c_real - c_hi)
        d_lh = np.linalg.norm(c_lo - c_hi)
        print(f'  [{name}]  d(real,低分fake)={d_rl:.3f}  d(real,高分fake)={d_rh:.3f}  '
              f'd(低分,高分)={d_lh:.3f}')
        # 平均逐样本距离(比质心距离更稳健)
        def avgd(a, b):
            return np.linalg.norm(a[:, None] - b[None], axis=2).mean()
        print(f'          逐样本均值: real−低分={avgd(real, lo):.3f}  '
              f'real−高分={avgd(real, hi):.3f}  低分−高分={avgd(lo, hi):.3f}')
    print()

    # 结论导向的判读
    print('=' * 92)
    print('判读(基于 L3, 即分类前特征):')
    print('  若 dist(低分fake, real) < dist(低分fake, 高分fake) → 低分NT特征接近real')
    print('    → 中间表征缺判别信息 → 对比学习 / 难例表征增强')
    print('  若 dist(低分fake, real) > dist(低分fake, 高分fake) → 低分NT特征仍接近高分NT')
    print('    → 表征可分但分类头未利用 → 分类头优化 / 特征融合')
    print('=' * 92)


if __name__ == '__main__':
    main()
