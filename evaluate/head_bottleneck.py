# -*- coding: utf-8 -*-
"""Step 3.5: 分类头信息瓶颈定位 —— 统一协议下的跨层线性探针对照。

目的: 判断"backbone 已有判别信息"究竟在哪个环节没有被充分利用。
  L1 backbone SPPF 区域池化 (已验证的判别基线)
  L2 neck P5 区域池化      (信息是否在特征融合中减弱)
  L3 cv3[2][1] 分类前      (分类分支内部是否保留信息)

纪律:
  1. 三个任务(完整NT/低分NT/高分NT vs Real) × 三个层级, 用**相同**的 GroupKFold 划分。
  2. 标准化、正则化、逻辑回归全部在训练折内; 测试折只验证。
  3. 低分样本定义统一为 conf<0.25。

判决:
  L1高 L2明显下降      -> 信息在 neck 融合中减弱
  L2高 L3明显下降      -> 信息在 cv3 分类分支内部丢失
  L1/L2/L3 都高        -> 信息仍在, 需看 L4(最终分数) 是否丢失 -> 分类映射/训练目标

用法: python -I evaluate/head_bottleneck.py
"""
import sys
from collections import defaultdict

import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
from feature_attribution_analyze import auroc, lr_fit, zscore   # noqa: E402


def group_kfold(videos, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    return [set(videos[o]) for o in np.array_split(order, k)]


def probe(Feat_list, y, vid, folds):
    """给定特征矩阵和标签, 按 GroupKFold 算线性探针 AUROC(每折), 返回数组。"""
    aucs = []
    for fold in folds:
        tr = ~np.isin(vid, list(fold)); va = ~tr
        Xtr, mu, sd = zscore(Feat_list[tr]); Xva, _, _ = zscore(Feat_list[va], mu, sd)
        wb = lr_fit(Xtr, y[tr])
        aucs.append(auroc(y[va], Xva @ wb[:-1] + wb[-1]))
    return np.array(aucs)


def main():
    d = np.load('evaluate/features_nt_val.npz')
    L1, L2, L3 = d['l1'], d['l2'], d['l3']
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group'])
    n = len(ri)
    Xr = {k: F[ri] for k, F in (('L1', L1), ('L2', L2), ('L3', L3))}
    Xf = {k: F[fi] for k, F in (('L1', L1), ('L2', L2), ('L3', L3))}
    lo = group == 'low'; hi = group == 'high'
    videos = np.unique(src)
    folds = group_kfold(videos, 5, seed=0)

    print(f'配对 {n} 对 (低分 {int(lo.sum())}, 高分 {int(hi.sum())}), 视频 {len(videos)}')
    print('GroupKFold K=5, 三层级统一划分/标准化/正则化\n')

    tasks = {
        '完整 NT vs Real': (Xf, Xr, np.ones(n, bool), np.ones(n, bool), src, src),
        '低分 NT vs Real': (Xf, Xr, lo, np.ones(n, bool), src, src),
        '高分 NT vs Real': (Xf, Xr, hi, np.ones(n, bool), src, src),
    }

    # 收集结果, 打印 3(任务) x 3(层级) 表
    res = {}
    print('=' * 88)
    print('统一协议下的线性探针 AUROC (均值 ± 标准差)')
    print('=' * 88)
    print(f'{"任务":<18}' + ''.join(f'{k:>20}' for k in ('L1 backbone', 'L2 neck', 'L3 分类前')))
    for tname, (Xf_t, Xr_t, mf, mr, sf, sr) in tasks.items():
        row = []
        for k in ('L1', 'L2', 'L3'):
            A = Xf_t[k][mf]; B = Xr_t[k][mr]
            X = np.vstack([A, B]); y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
            vid = np.concatenate([sf[mf], sr[mr]])
            aucs = probe(X, y, vid, folds)
            res[(tname, k)] = aucs
            row.append(aucs.mean())
        print(f'{tname:<18}' + ''.join(f'{v:>14.4f}' for v in row))
        print(f'{"":<18}' + ''.join(f'{"±"+format(res[(tname,k)].std(),".4f"):>20}' for k in ('L1','L2','L3')))

    # 跨层差异(配对, 同一折)
    print()
    print('=' * 88)
    print('跨层差异(配对, 同折): L1−L2 与 L2−L3 的 AUROC 之差')
    print('=' * 88)
    print(f'{"任务":<18}{"L1−L2":>12}{"差>0占比":>12}{"L2−L3":>12}{"差>0占比":>12}')
    for tname in tasks:
        d12 = res[(tname, 'L1')] - res[(tname, 'L2')]
        d23 = res[(tname, 'L2')] - res[(tname, 'L3')]
        print(f'{tname:<18}{d12.mean():>+12.4f}{(d12>0).mean()*100:>11.0f}%'
              f'{d23.mean():>+12.4f}{(d23>0).mean()*100:>11.0f}%')

    # 判决
    print()
    print('=' * 88)
    print('判决(基于低分 NT vs Real, 因为那是问题所在)')
    print('=' * 88)
    a1 = res[('低分 NT vs Real', 'L1')]; a2 = res[('低分 NT vs Real', 'L2')]
    a3 = res[('低分 NT vs Real', 'L3')]
    print(f'  低分NT vs Real: L1={a1.mean():.3f}  L2={a2.mean():.3f}  L3={a3.mean():.3f}')
    d12 = a1 - a2; d23 = a2 - a3
    if d12.mean() > 0.05 and (d12 > 0).mean() > 0.8:
        verdict = '信息在 neck 融合中减弱 (L1→L2 明显下降)'
    elif d23.mean() > 0.05 and (d23 > 0).mean() > 0.8:
        verdict = '信息在 cv3 分类分支内部丢失 (L2→L3 明显下降)'
    else:
        verdict = 'L1/L2/L3 判别性相近 -> 信息仍在, 瓶颈在最终分类映射/训练目标(需看 L4)'
    print(f'  -> {verdict}')


if __name__ == '__main__':
    main()
