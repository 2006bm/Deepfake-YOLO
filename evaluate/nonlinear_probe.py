# -*- coding: utf-8 -*-
"""Step 3.6 第一阶段: 复核测量一致性 + 非线性探针对照。

核心: L1→L2 的 AUROC 下降, 是"信息真的丢失", 还是"信息仍在但变得线性不可分"?

方法: 同一 GroupKFold 协议下, 对比
  线性探针   = 逻辑回归(已做)
  非线性探针 = RBF 核岭回归(γ 在训练折内选)
若 L2 的非线性 AUROC 显著高于其线性 AUROC 且接近 L1, 说明信息在 neck 后被"非线性化",
不是丢失; 若 L2 非线性也低, 才是真丢失。

纪律:
  1. L1/L2/L3 用一致的 GT 区域映射与池化(均为 P5 尺度, 同一 letterbox_mask)。
  2. 标准化/正则化/γ 选择全部在训练折内。
  3. 低分定义固定 conf<0.25, 不重划。
  4. 三层级用相同 GroupKFold 划分。

用法: python -I evaluate/nonlinear_probe.py
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
from feature_attribution_analyze import auroc, lr_fit, zscore   # noqa: E402


def group_kfold(videos, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    return [set(videos[o]) for o in np.array_split(order, k)]


def rbf_kernel(X, Y, gamma):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)


def kernel_ridge_auroc(Xtr, ytr, Xva, yva):
    """RBF 核岭回归, γ 用训练折的中位距离倒数(1/median_pairwise_dist²)。"""
    # γ 在训练折内选: 1/median pairwise squared distance
    sq = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    med = np.median(sq[np.triu_indices(len(Xtr), 1)]) if len(Xtr) > 1 else 1.0
    gamma = 1.0 / (med + 1e-8)
    K = rbf_kernel(Xtr, Xtr, gamma)
    yc = ytr * 2 - 1                              # -> {-1, +1}
    lam = 1e-2 * len(Xtr)
    alpha = np.linalg.solve(K + lam * np.eye(len(Xtr)), yc)
    Kva = rbf_kernel(Xva, Xtr, gamma)
    return auroc(yva, Kva @ alpha)


def linear_probe(Xtr, ytr, Xva, yva):
    Xtr, mu, sd = zscore(Xtr); Xva, _, _ = zscore(Xva, mu, sd)
    wb = lr_fit(Xtr, ytr)
    return auroc(yva, Xva @ wb[:-1] + wb[-1])


def main():
    d = np.load('evaluate/features_nt_val.npz')
    L1, L2, L3 = d['l1'], d['l2'], d['l3']
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group'])
    n = len(ri)
    Xr = {'L1': L1[ri], 'L2': L2[ri], 'L3': L3[ri]}
    Xf = {'L1': L1[fi], 'L2': L2[fi], 'L3': L3[fi]}
    lo = group == 'low'; hi = group == 'high'
    videos = np.unique(src)
    folds = group_kfold(videos, 5, seed=0)

    tasks = {'完整': np.ones(n, bool), '低分': lo, '高分': hi}

    print('=' * 96)
    print('线性 vs 非线性探针 (同一 GroupKFold, 三层级)')
    print('=' * 96)
    print(f'{"任务":<8}{"层级":<6}{"线性AUROC":>12}{"非线性AUROC":>14}{"Δ(非线性−线性)":>16}')

    store = {}
    for tname, m in tasks.items():
        for k in ('L1', 'L2', 'L3'):
            A = Xf[k][m]; B = Xr[k]
            X = np.vstack([A, B]); y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
            vid = np.concatenate([src[m], src])
            lin, nlin = [], []
            for fold in folds:
                tr = ~np.isin(vid, list(fold)); va = ~tr
                lin.append(linear_probe(X[tr], y[tr], X[va], y[va]))
                nlin.append(kernel_ridge_auroc(X[tr], y[tr], X[va], y[va]))
            lin = np.array(lin); nlin = np.array(nlin)
            store[(tname, k)] = (lin, nlin)
            print(f'{tname:<8}{k:<6}{lin.mean():>12.4f}{nlin.mean():>14.4f}'
                  f'{(nlin.mean()-lin.mean()):>+16.4f}')

    # 关键对照: L2 是否"非线性可分但线性不可分"
    print()
    print('=' * 96)
    print('关键: L2 的信息是"非线性化"还是"丢失"?')
    print('=' * 96)
    for tname in tasks:
        l1_l, l1_n = store[(tname, 'L1')]
        l2_l, l2_n = store[(tname, 'L2')]
        print(f'  {tname}: L1线性={l1_l.mean():.3f} L1非线性={l1_n.mean():.3f} | '
              f'L2线性={l2_l.mean():.3f} L2非线性={l2_n.mean():.3f}')
        # L2 非线性能否追平 L1 线性?
        gap = l1_l.mean() - l2_n.mean()
        verdict = ('L2非线性≈L1线性 -> 信息在neck后非线性化, 非丢失'
                   if gap < 0.03 else
                   'L2非线性仍低于L1线性 -> 信息在neck中确实丢失(至少部分)')
        print(f'      L1线性 − L2非线性 = {gap:+.3f}  -> {verdict}')


if __name__ == '__main__':
    main()
