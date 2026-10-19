# -*- coding: utf-8 -*-
"""Step 3.4a: 判别信息跨层保留 + 连续置信度分析。

两个核心问题:
  Q1 跨层信息是否损失: AUROC_L1 − AUROC_L3 = 0.0455 是稳定差异还是噪声?
     -> 用**配对**统计检验(同一视频划分下各层 AUROC 之差), 而非两个独立均值之差。
  Q2 选择效应: 低分/高分的 L3 特征差异, 有多少是"按 conf 分组"造成的选择效应?
     -> 连续分析: 用全部 170 对, 看配对特征距离与 conf 的 Spearman 相关。
        关键: 若 L1 距离与 conf 也显著相关, 说明差异在 backbone 就存在, 非选择效应;
             若只有 L3 相关, 则更像选择效应。

用法: python -I evaluate/feature_attribution_crosslayer.py
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, mannwhitneyu

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
from feature_attribution_analyze import auroc, lr_fit, zscore   # noqa: E402


def main():
    d = np.load('evaluate/features_nt_val.npz')
    l1, l2, l3 = d['l1'], d['l2'], d['l3']
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group']); conf = np.array(d['conf'], float)
    n = len(ri)
    Feats = {'L1': l1, 'L2': l2, 'L3': l3}
    uniq = np.unique(src)

    # ============ Q1: 跨层 AUROC 配对检验 ============
    print('=' * 84)
    print('Q1 跨层判别信息: 配对 AUROC 之差(重复按视频 70/30 划分, 200 次)')
    print('=' * 84)
    rng = np.random.default_rng(42)
    n_rep = 200
    aucs = {k: [] for k in Feats}
    for rep in range(n_rep):
        order = rng.permutation(len(uniq))
        val_vid = set(uniq[order[: int(len(uniq) * 0.3)]])
        tr = ~np.isin(np.tile(src, 2), list(val_vid))
        va = ~tr
        y = np.concatenate([np.zeros(n), np.ones(n)])
        for k, F in Feats.items():
            X = np.vstack([F[ri], F[fi]])
            Xtr, mu, sd = zscore(X[tr]); Xva, _, _ = zscore(X[va], mu, sd)
            wb = lr_fit(Xtr, y[tr])
            aucs[k].append(auroc(y[va], Xva @ wb[:-1] + wb[-1]))
    for k in Feats:
        a = np.array(aucs[k])
        print(f'  {k}: 均值 {a.mean():.4f}  95%CI [{np.percentile(a,2.5):.4f}, {np.percentile(a,97.5):.4f}]')
    # 配对差值
    d13 = np.array(aucs['L1']) - np.array(aucs['L3'])
    d12 = np.array(aucs['L1']) - np.array(aucs['L2'])
    d23 = np.array(aucs['L2']) - np.array(aucs['L3'])
    print(f'\n  配对差 L1−L3: 均值 {d13.mean():+.4f}  95%CI [{np.percentile(d13,2.5):+.4f}, '
          f'{np.percentile(d13,97.5):+.4f}]  差>0 占比 {(d13>0).mean()*100:.1f}%')
    print(f'  配对差 L1−L2: 均值 {d12.mean():+.4f}  差>0 占比 {(d12>0).mean()*100:.1f}%')
    print(f'  配对差 L2−L3: 均值 {d23.mean():+.4f}  差>0 占比 {(d23>0).mean()*100:.1f}%')

    # ============ Q2: 连续置信度分析 ============
    print()
    print('=' * 84)
    print('Q2 连续分析: 配对特征距离 vs 实例置信度(Spearman, 全部 170 对)')
    print('=' * 84)
    print(f'{"层级":<6}{"Spearman ρ":>12}{"p":>10}   判读')
    rhos = {}
    for k, F in Feats.items():
        X = np.vstack([F[ri], F[fi]])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        dist = np.linalg.norm(X[ri] - X[fi], axis=1)
        rho, p = spearmanr(dist, conf)
        rhos[k] = rho
        verdict = ('距离与conf显著正相关→差异在特征层,非选择效应'
                   if p < 0.05 else '不显著')
        print(f'{k:<6}{rho:>12.4f}{p:>10.4f}   {verdict}')

    print()
    print('  关键对照: 若 L1 的 ρ 就显著, 说明"低分样本特征更接近real"在 backbone 已存在,')
    print('  不是由 L3 附近的 conf 分组造成的选择效应。')

    # ============ 补充: 连续 vs 二分的一致性 ============
    print()
    print('=' * 84)
    print('补充: 低分/高分的距离差异(二分) 与 连续 ρ 是否一致')
    print('=' * 84)
    for k, F in Feats.items():
        X = np.vstack([F[ri], F[fi]])
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        dist = np.linalg.norm(X[ri] - X[fi], axis=1)
        lo, hi = dist[group == 'low'], dist[group == 'high']
        dlt = (np.median(lo) - np.median(hi)) / np.median(hi)
        p = mannwhitneyu(lo, hi).pvalue
        print(f'  {k}: 二分(低分/高分中位比 {np.median(lo)/np.median(hi):.3f}, p={p:.4f})'
              f'   连续 ρ={rhos[k]:+.3f}')

    # ============ 视频级 ICC 提示(聚集程度量化) ============
    print()
    print('=' * 84)
    print('附: 低分失败的视频级聚集程度(组内相关)')
    print('=' * 84)
    # 每个视频的"低分比例"方差分解: 组间方差 vs 组内方差
    vid_of = defaultdict(list)
    for i in range(n):
        vid_of[src[i]].append(1.0 if group[i] == 'low' else 0.0)
    gmean = np.mean([np.mean(v) for v in vid_of.values()])
    between = np.var([np.mean(v) for v in vid_of.values()], ddof=1)
    within = np.mean([np.var(v, ddof=1) for v in vid_of.values() if len(v) > 1])
    icc = between / (between + within) if (between + within) > 0 else np.nan
    print(f'  组间方差 {between:.4f}  组内方差 {within:.4f}  ICC(1) ≈ {icc:.3f}')
    print(f'  (ICC 越接近 1 表示低分失败越聚集在同一批视频; 接近 0 表示样本级随机)')


if __name__ == '__main__':
    main()
