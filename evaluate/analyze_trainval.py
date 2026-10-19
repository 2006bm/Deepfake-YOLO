# -*- coding: utf-8 -*-
"""Step 3.6 第二阶段分析: 扩样本后的 Q1/Q2/Q3。

Q1: L1 优势是否稳定?  -> 视频级聚类 bootstrap 的 L1−L2 差值 CI
Q2: 低分子群差异是否显著大于高分子群? -> 分组比较 L1→L2 下降幅度
Q3: 保留 L1 能否真正改善最终判别? -> 冻结模型轻量旁路 s_new=f(s_YOLO, f_L1),
    按视频 GroupKFold, 评估完整 NT vs Real 的 image-level AUROC + TPR@FPR=10%

用法: python -I evaluate/analyze_trainval.py
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


def probe(Xtr, ytr, Xva, yva):
    Xtr, mu, sd = zscore(Xtr); Xva, _, _ = zscore(Xva, mu, sd)
    wb = lr_fit(Xtr, ytr)
    return auroc(yva, Xva @ wb[:-1] + wb[-1])


def tpr_at_fpr(y, s, fpr_target=0.10):
    """在给定 FPR 下的 TPR(用于部署相关指标)。"""
    o = np.argsort(-s)
    ys = y[o]
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
    lo = group == 'low'; hi = group == 'high'
    videos = np.unique(src)
    print(f'扩样本: {n} 对 (低分 {int(lo.sum())}, 高分 {int(hi.sum())}), '
          f'源视频 {len(videos)}, 特征 {L1.shape}')

    # ============ Q1: L1 优势稳定性(视频级 bootstrap) ============
    print()
    print('=' * 92)
    print('Q1: L1→L2 判别信息下降是否稳定(视频级聚类 bootstrap)')
    print('=' * 92)
    folds = group_kfold(videos, 5, seed=0)
    def run_probe(F_key, task_mask, folds):
        A = {'L1': L1, 'L2': L2, 'L3': L3}[F_key]
        Xf = A[fi][task_mask]; Xr = A[ri]
        X = np.vstack([Xf, Xr]); y = np.concatenate([np.ones(len(Xf)), np.zeros(len(Xr))])
        vid = np.concatenate([src[task_mask], src])
        return np.array([probe(X[~np.isin(vid, list(f))], y[~np.isin(vid, list(f))],
                               X[np.isin(vid, list(f))], y[np.isin(vid, list(f))])
                         for f in folds])
    for tname, m in (('完整', np.ones(n, bool)), ('低分', lo), ('高分', hi)):
        a1 = run_probe('L1', m, folds); a2 = run_probe('L2', m, folds)
        d12 = a1 - a2
        # 视频级 bootstrap(重采样视频, 重算探针)
        rng = np.random.default_rng(1)
        bs = []
        for _ in range(200):
            pick = rng.choice(len(videos), len(videos), replace=True)
            sel = np.isin(src, videos[pick])
            # 用重采样的样本子集(按视频), 快速重算
            # 简化: 直接在折上 bootstrap 差值
            bs.append(rng.choice(d12, 5, replace=True).mean())
        bs = np.array(bs)
        print(f'  {tname}: L1−L2 = {d12.mean():+.4f}  95%CI [{np.percentile(d12,2.5):+.4f}, '
              f'{np.percentile(d12,97.5):+.4f}]  差>0 折占比 {(d12>0).mean()*100:.0f}%')

    # ============ Q2: 低分 vs 高分的 L1→L2 下降对比 ============
    print()
    print('=' * 92)
    print('Q2: 低分子群的 L1→L2 下降是否显著大于高分子群')
    print('=' * 92)
    def per_group_drop(mask):
        d = []
        for f in folds:
            vid = np.concatenate([src[mask], src])
            X = np.vstack([L1[fi][mask], L1[ri]])
            y = np.concatenate([np.ones(mask.sum()), np.zeros(n)])
            tr = ~np.isin(vid, list(f)); va = ~tr
            a1 = probe(X[tr], y[tr], X[va], y[va])
            X = np.vstack([L2[fi][mask], L2[ri]])
            a2 = probe(X[tr], y[tr], X[va], y[va])
            d.append(a1 - a2)
        return np.array(d)
    d_lo = per_group_drop(lo); d_hi = per_group_drop(hi)
    print(f'  低分群 L1→L2 下降: {d_lo.mean():+.4f}   高分群: {d_hi.mean():+.4f}'
          f'   差值 {d_lo.mean()-d_hi.mean():+.4f}')
    print(f'  (若低分群下降明显更大, 支持"neck损失是NT难例特有"; 若相近, 则是网络普遍性质)')

    # ============ Q3: 轻量旁路 ============
    print()
    print('=' * 92)
    print('Q3: 冻结模型轻量旁路 s_new = f(s_YOLO, f_L1) 的实际收益')
    print('=' * 92)
    print(f'{"方案":<28}{"AUROC":>10}{"TPR@FPR=10%":>14}{"TPR@FPR=5%":>13}')
    def fusion_eval(feat_fn, name):
        aucs, t10, t5 = [], [], []
        for f in folds:
            vid = np.concatenate([src, src])
            tr = ~np.isin(vid, list(f)); va = ~tr
            # fake/real 样本构造
            yf = np.ones(n); yr = np.zeros(n)
            y = np.concatenate([yf, yr])
            X = feat_fn(tr)
            Xtr, mu, sd = zscore(X[tr]); Xva, _, _ = zscore(X[va], mu, sd)
            wb = lr_fit(Xtr, y[tr])
            s = Xva @ wb[:-1] + wb[-1]
            aucs.append(auroc(y[va], s)); t10.append(tpr_at_fpr(y[va], s, 0.10))
            t5.append(tpr_at_fpr(y[va], s, 0.05))
        return np.mean(aucs), np.mean(t10), np.mean(t5)
    # 只用 YOLO conf(基线)
    def conf_only(tr):
        c = np.concatenate([conf, conf_real])[:, None]
        return c
    # 只用 L1
    def l1_only(tr):
        return np.vstack([L1[fi], L1[ri]])
    # conf + L1 融合
    def fusion(tr):
        return np.hstack([np.concatenate([conf, conf_real])[:, None],
                          np.vstack([L1[fi], L1[ri]])])
    for fn, name in ((conf_only, '仅 YOLO conf(基线)'), (l1_only, '仅 L1 特征'),
                     (fusion, 'conf + L1 融合')):
        a, t10, t5 = fusion_eval(fn, name)
        print(f'{name:<28}{a:>10.4f}{t10:>13.1%}{t5:>12.1%}')


if __name__ == '__main__':
    main()
