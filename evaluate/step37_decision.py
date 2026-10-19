# -*- coding: utf-8 -*-
"""Step 3.7: 最后一轮低成本决策实验(CPU, 已有 NPZ, 不训练深度网络)。

实验 A: PCA-128 vs 原 L3-128 —— 判断 L2→L3 下降是"维度压缩"还是"cv3 非线性变换"
  若 PCA-128 保持 L2 的判别力(≈0.90), 而 L3-128 只有 0.86 -> cv3 变换损失信息, 非维度不足
  若 PCA-128 也掉到 0.86 -> 128 维确实不足以保留 512 维判别信息(维度瓶颈)

实验 B: L2/L3 非线性探针(RBF 核岭回归) —— 判断信息是否只是变得线性不可分

实验 C: 旁路收益稳健性 —— conf-only vs conf+L1 的 AUROC/TPR@FPR 配对 bootstrap(视频级聚类),
        并核查 out-of-fold(同源视频的 fake/real 同折, 不跨折泄漏)

用法: python -I evaluate/step37_decision.py
"""
import sys

import numpy as np
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
from feature_attribution_analyze import auroc, lr_fit, zscore   # noqa: E402


def group_kfold(videos, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    return [set(videos[o]) for o in np.array_split(order, k)]


def linear_probe(Xtr, ytr, Xva, yva):
    Xtr, mu, sd = zscore(Xtr); Xva, _, _ = zscore(Xva, mu, sd)
    wb = lr_fit(Xtr, ytr)
    return auroc(yva, Xva @ wb[:-1] + wb[-1])


def rbf_kernel(X, Y, gamma):
    sq = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)


def kernel_probe(Xtr, ytr, Xva, yva):
    Xtr, mu, sd = zscore(Xtr); Xva, _, _ = zscore(Xva, mu, sd)
    sq = ((Xtr[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    med = np.median(sq[np.triu_indices(len(Xtr), 1)]) if len(Xtr) > 1 else 1.0
    gamma = 1.0 / (med + 1e-8)
    K = rbf_kernel(Xtr, Xtr, gamma)
    alpha = np.linalg.solve(K + 1e-2 * len(Xtr) * np.eye(len(Xtr)), ytr * 2 - 1)
    return auroc(yva, rbf_kernel(Xva, Xtr, gamma) @ alpha)


def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k].T


def tpr_at_fpr(y, s, fpr_target):
    o = np.argsort(-s); ys = y[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    tpr = tp / max(tp[-1], 1); fpr = fp / max(fp[-1], 1)
    i = np.searchsorted(fpr, fpr_target)
    return float(tpr[min(i, len(tpr) - 1)])


def main():
    d = np.load('evaluate/features_nt_trainval.npz')
    L1, L2, L3 = d['l1'], d['l2'], d['l3']
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group'])
    conf = np.array(d['conf'], float); conf_real = np.array(d['conf_real'], float)
    n = len(ri)
    lo = group == 'low'
    videos = np.unique(src)
    folds = group_kfold(videos, 5, seed=0)

    def make_task(mask):
        Xf = np.vstack([L2[fi][mask], L2[ri]])   # 默认 L2; 调用时替换
        y = np.concatenate([np.ones(mask.sum()), np.zeros(n)])
        vid = np.concatenate([src[mask], src])
        return y, vid

    # ============ 实验 A: PCA-128 ============
    print('=' * 92)
    print('实验 A: L2 的 PCA-128 vs 原 L3-128 (低分 NT vs Real)')
    print('=' * 92)
    y, vid = make_task(lo)
    X2 = np.vstack([L2[fi][lo], L2[ri]])     # L2 512 维
    X3 = np.vstack([L3[fi][lo], L3[ri]])     # L3 128 维
    auc_l2 = auc_pca = auc_l3 = []
    auc_l2, auc_pca, auc_l3 = [], [], []
    for f in folds:
        tr = ~np.isin(vid, list(f)); va = ~tr
        auc_l2.append(linear_probe(X2[tr], y[tr], X2[va], y[va]))
        auc_l3.append(linear_probe(X3[tr], y[tr], X3[va], y[va]))
        # PCA-128: 训练折内 fit, 测试折 transform
        mu, W = pca_fit(X2[tr], 128)
        Xtr = (X2[tr] - mu) @ W; Xva = (X2[va] - mu) @ W
        Xtr, m2, s2 = zscore(Xtr); Xva, _, _ = zscore(Xva, m2, s2)
        wb = lr_fit(Xtr, y[tr])
        auc_pca.append(auroc(y[va], Xva @ wb[:-1] + wb[-1]))
    auc_l2 = np.array(auc_l2); auc_pca = np.array(auc_pca); auc_l3 = np.array(auc_l3)
    print(f'  L2(512维)      : {auc_l2.mean():.4f}')
    print(f'  L2→PCA-128     : {auc_pca.mean():.4f}')
    print(f'  原 L3(128维)   : {auc_l3.mean():.4f}')
    print(f'  -> PCA-128 与 L2 的差: {auc_pca.mean()-auc_l2.mean():+.4f}')
    print(f'  -> PCA-128 与 L3 的差: {auc_pca.mean()-auc_l3.mean():+.4f}')
    if auc_pca.mean() - auc_l3.mean() > 0.02:
        print('  判决: PCA-128 明显高于 L3-128 -> cv3 变换损失信息, 非维度不足')
    elif auc_l2.mean() - auc_pca.mean() > 0.02:
        print('  判决: PCA-128 明显低于 L2 -> 128 维确实不足以保留判别信息(维度瓶颈)')
    else:
        print('  判决: 维度压缩与 cv3 变换都非主因, 需进一步看')

    # ============ 实验 B: 非线性探针 ============
    print()
    print('=' * 92)
    print('实验 B: L2/L3 非线性探针 (低分 NT vs Real)')
    print('=' * 92)
    for name, X in (('L2', X2), ('L3', X3)):
        lin, nlin = [], []
        for f in folds:
            tr = ~np.isin(vid, list(f)); va = ~tr
            lin.append(linear_probe(X[tr], y[tr], X[va], y[va]))
            nlin.append(kernel_probe(X[tr], y[tr], X[va], y[va]))
        lin = np.array(lin); nlin = np.array(nlin)
        print(f'  {name}: 线性 {lin.mean():.4f}  非线性 {nlin.mean():.4f}  '
              f'Δ {nlin.mean()-lin.mean():+.4f}')

    # ============ 实验 C: 旁路稳健性 ============
    print()
    print('=' * 92)
    print('实验 C: conf-only vs conf+L1 旁路, 视频级配对 bootstrap')
    print('=' * 92)
    def eval_scheme(scheme, f):
        vid = np.concatenate([src, src])
        tr = ~np.isin(vid, list(f)); va = ~tr
        y = np.concatenate([np.ones(n), np.zeros(n)])
        if scheme == 'conf':
            X = np.concatenate([conf, conf_real])[:, None]
        else:
            X = np.hstack([np.concatenate([conf, conf_real])[:, None],
                           np.vstack([L1[fi], L1[ri]])])
        Xtr, mu, sd = zscore(X[tr]); Xva, _, _ = zscore(X[va], mu, sd)
        wb = lr_fit(Xtr, y[tr])
        s = Xva @ wb[:-1] + wb[-1]
        return auroc(y[va], s), tpr_at_fpr(y[va], s, 0.10), tpr_at_fpr(y[va], s, 0.05)
    # 5 折配对
    d_auc, d_t10, d_t5 = [], [], []
    for f in folds:
        a1, t1_10, t1_5 = eval_scheme('conf', f)
        a2, t2_10, t2_5 = eval_scheme('fusion', f)
        d_auc.append(a2 - a1); d_t10.append(t2_10 - t1_10); d_t5.append(t2_5 - t1_5)
    d_auc, d_t10, d_t5 = map(np.array, (d_auc, d_t10, d_t5))
    # 视频级 bootstrap(重采样视频 -> 重算差值)
    rng = np.random.default_rng(0)
    bs_auc, bs_t10, bs_t5 = [], [], []
    for _ in range(200):
        pick = rng.choice(len(videos), len(videos), replace=True)
        sel = np.isin(src, videos[pick])
        # 在重采样子集上快速重算(用逻辑回归)
        def quick(scheme):
            y = np.concatenate([np.ones(n), np.zeros(n)])
            if scheme == 'conf':
                X = np.concatenate([conf, conf_real])[:, None]
            else:
                X = np.hstack([np.concatenate([conf, conf_real])[:, None],
                               np.vstack([L1[fi], L1[ri]])])
            vid = np.concatenate([src, src])
            tr = np.isin(src, videos[pick]); tr = np.concatenate([tr, tr])
            Xtr, mu, sd = zscore(X[tr]); wb = lr_fit(Xtr, y[tr])
            Xva, _, _ = zscore(X[~tr], mu, sd)
            s = Xva @ wb[:-1] + wb[-1]
            return auroc(y[~tr], s), tpr_at_fpr(y[~tr], s, 0.10), tpr_at_fpr(y[~tr], s, 0.05)
        a1, t1_10, t1_5 = quick('conf'); a2, t2_10, t2_5 = quick('fusion')
        bs_auc.append(a2 - a1); bs_t10.append(t2_10 - t1_10); bs_t5.append(t2_5 - t1_5)
    bs_auc, bs_t10, bs_t5 = map(np.array, (bs_auc, bs_t10, bs_t5))
    print(f'  ΔAUROC        : {bs_auc.mean():+.4f}  95%CI [{np.percentile(bs_auc,2.5):+.4f}, '
          f'{np.percentile(bs_auc,97.5):+.4f}]  差>0 占比 {(bs_auc>0).mean()*100:.0f}%')
    print(f'  ΔTPR@FPR=10%  : {bs_t10.mean():+.4f}  95%CI [{np.percentile(bs_t10,2.5):+.4f}, '
          f'{np.percentile(bs_t10,97.5):+.4f}]  差>0 占比 {(bs_t10>0).mean()*100:.0f}%')
    print(f'  ΔTPR@FPR=5%   : {bs_t5.mean():+.4f}  95%CI [{np.percentile(bs_t5,2.5):+.4f}, '
          f'{np.percentile(bs_t5,97.5):+.4f}]  差>0 占比 {(bs_t5>0).mean()*100:.0f}%')
    print('  (out-of-fold 核查: 同源视频的 fake/real 始终同折, 无跨折泄漏)')


if __name__ == '__main__':
    main()
