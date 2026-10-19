# -*- coding: utf-8 -*-
"""Step 3.4b (v2): 通道归因 —— 区分 H1/H2/H3。

核心问题(用户): 原有 backbone 的 512 维特征, 是否已存在能识别**低置信 NT** 的稳定判别通道?
  H1: 低分 NT 在 backbone 中确实缺少可分真假特征   -> 难例表征增强
  H2: backbone 已有判别信息, 但原模型没有效利用     -> 特征选择/判别监督/分类头
  H3: 混合

纪律:
  1. 用**单图特征**(不是配对差)训练线性探针 —— 部署时没有配对真实帧, 配对差只能做机制分析。
  2. GroupKFold(按视频分组), 不做一次随机划分。
  3. 三种特征: 全 512 / 训练折筛选 Top-K / 随机 K(排除维度减少的假象)。
  4. 通道筛选、标准化、分类器训练全部在训练折内完成; 测试折只验证。
  5. 报告各折选中通道的重合率 —— 若不同折筛出完全不同的通道, 不能声称稳定判别通道。
  6. 核心指标是"低分 NT vs 真实"的独立检出率(总体 AUROC 高 ≠ 能识别低分 NT)。

用法: python -I evaluate/channel_attribution.py
"""
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
from feature_attribution_analyze import auroc, lr_fit, zscore   # noqa: E402


def group_kfold(videos, k=5, seed=0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(videos))
    return [set(videos[o]) for o in np.array_split(order, k)]


def probe_auroc(X_tr, y_tr, X_va, y_va, sel=None):
    """在训练折上标准化(+可选通道选择)+逻辑回归, 测试折上算 AUROC。"""
    if sel is not None:
        X_tr, X_va = X_tr[:, sel], X_va[:, sel]
    Xtr, mu, sd = zscore(X_tr); Xva, _, _ = zscore(X_va, mu, sd)
    wb = lr_fit(Xtr, y_tr)
    return auroc(y_va, Xva @ wb[:-1] + wb[-1])


def main():
    d = np.load('evaluate/features_nt_val.npz')
    L1 = d['l1']                                   # (340, 512) 单图特征
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group']); conf = np.array(d['conf'], float)
    n = len(ri)
    Xr, Xf = L1[ri], L1[fi]                        # 各 (170, 512)
    lo = group == 'low'; hi = group == 'high'
    videos = np.unique(src)
    print(f'配对 {n} 对 (低分 {int(lo.sum())}, 高分 {int(hi.sum())}), 视频 {len(videos)}, L1 {L1.shape[1]} 维')

    # ============ 任务定义 ============
    # 任务A: 低分NT vs 真实   任务B: 高分NT vs 真实   任务C: 低分NT vs 高分NT
    # 每个任务的样本 = 若干 fake + 若干 real/其他fake, 用单图特征
    tasks = {
        'A 低分NT vs 真实': (Xf[lo], Xr, '低分', '真实'),
        'B 高分NT vs 真实': (Xf[hi], Xr, '高分', '真实'),
        'C 低分NT vs 高分NT': (Xf[lo], Xf[hi], '低分', '高分'),
    }
    # 每个任务里, 样本对应的源视频(用于 GroupKFold)
    task_src = {
        'A 低分NT vs 真实': (src[lo], src),
        'B 高分NT vs 真实': (src[hi], src),
        'C 低分NT vs 高分NT': (src[lo], src[hi]),
    }

    # ============ GroupKFold 线性探针 ============
    K = 5
    print()
    print('=' * 96)
    print(f'单图特征线性探针 (GroupKFold K={K}, 按源视频分组, 5 折)')
    print('=' * 96)
    print(f'{"任务":<20}{"全512 AUROC":>14}{"Top-32 AUROC":>14}{"Top-64 AUROC":>14}'
          f'{"随机32 AUROC":>14}')

    results = {}
    for tname, (A, B, lab_a, lab_b) in tasks.items():
        sA, sB = task_src[tname]
        # 合并特征和标签: A 是正类(fake/低分), B 是负类(real/高分)
        X = np.vstack([A, B]); y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
        vid = np.concatenate([sA, sB])
        folds = group_kfold(videos, K, seed=0)
        auc_full, auc_top32, auc_top64, auc_rand32 = [], [], [], []
        sel_sets = []
        for fold in folds:
            tr = ~np.isin(vid, list(fold)); va = ~tr
            # 全 512
            auc_full.append(probe_auroc(X[tr], y[tr], X[va], y[va]))
            # 训练折内筛选 Top-K 通道: 用训练折内 A 类(fake)与 B 类(real/其他)的特征差做效应量
            XtrA, XtrB = X[tr][y[tr] == 1], X[tr][y[tr] == 0]
            if len(XtrA) > 0 and len(XtrB) > 0:
                eff = np.abs(XtrA.mean(0) - XtrB.mean(0)) / (np.sqrt(XtrA.var(0) + XtrB.var(0)) + 1e-8)
                for k, lst in ((32, auc_top32), (64, auc_top64)):
                    sel = np.argsort(eff)[::-1][:k]
                    lst.append(probe_auroc(X[tr], y[tr], X[va], y[va], sel))
                    if k == 32:
                        sel_sets.append(set(sel.tolist()))
            # 随机 32 通道
            rng = np.random.default_rng(len(auc_rand32))
            sel = rng.permutation(X.shape[1])[:32]
            auc_rand32.append(probe_auroc(X[tr], y[tr], X[va], y[va], sel))
        results[tname] = dict(full=np.array(auc_full), top32=np.array(auc_top32),
                              top64=np.array(auc_top64), rand32=np.array(auc_rand32),
                              sel_sets=sel_sets)
        print(f'{tname:<20}{auc_full[0] if False else np.mean(auc_full):>14.4f}'
              f'{np.mean(auc_top32):>14.4f}{np.mean(auc_top64):>14.4f}'
              f'{np.mean(auc_rand32):>14.4f}')
        print(f'{"":<20}{"±"+format(np.std(auc_full),".4f"):>14}'
              f'{"±"+format(np.std(auc_top32),".4f"):>14}'
              f'{"±"+format(np.std(auc_top64),".4f"):>14}'
              f'{"±"+format(np.std(auc_rand32),".4f"):>14}')

    # ============ 通道重合率 ============
    print()
    print('=' * 96)
    print('通道筛选稳定性: 各折选出的 Top-32 通道的重合率')
    print('=' * 96)
    for tname in results:
        ss = results[tname]['sel_sets']
        if len(ss) < 2:
            continue
        from itertools import combinations
        pairs = [len(a & b) / 32 for a, b in combinations(ss, 2)]
        print(f'  {tname:<20} 两两重合率 中位 {np.median(pairs)*100:.0f}%  '
              f'范围 {min(pairs)*100:.0f}-{max(pairs)*100:.0f}%  '
              f'(随机期望 {32/512*100:.1f}%)')

    # ============ 关键判决: 低分 NT 能否被单图特征识别 ============
    print()
    print('=' * 96)
    print('判决: 低分 NT vs 真实 的单图特征线性可分性')
    print('=' * 96)
    A = results['A 低分NT vs 真实']['full']
    lo_, hi_ = np.percentile(A, 2.5), np.percentile(A, 97.5)
    print(f'  全 512 通道: AUROC 均值 {A.mean():.4f}  95%CI [{lo_:.4f}, {hi_:.4f}]')
    r32 = results['A 低分NT vs 真实']['rand32']
    print(f'  随机 32 通道: AUROC 均值 {r32.mean():.4f}')
    if A.mean() > 0.75:
        verdict = 'H2 倾向: backbone 已含可区分低分 NT 的信息(单图特征)'
    elif A.mean() < 0.6:
        verdict = 'H1 倾向: backbone 单图特征难以区分低分 NT 与真实'
    else:
        verdict = 'H3 倾向: 混合(部分可分)'
    print(f'  -> {verdict}')

    # ============ ICC + 视频级 bootstrap CI ============
    print()
    print('=' * 96)
    print('附: ICC(1) 及视频级 bootstrap 置信区间(二值低分事件)')
    print('=' * 96)
    vid_of = defaultdict(list)
    for i in range(n):
        vid_of[src[i]].append(1.0 if lo[i] else 0.0)
    def calc_icc(vid_sub):
        vs = [v for v in vid_sub if len(vid_sub[v]) > 1]
        if len(vs) < 2:
            return np.nan
        between = np.var([np.mean(vid_sub[v]) for v in vs], ddof=1)
        within = np.mean([np.var(vid_sub[v], ddof=1) for v in vs])
        return between / (between + within) if (between + within) > 0 else np.nan
    icc_obs = calc_icc(vid_of)
    keys = list(vid_of.keys())
    rng = np.random.default_rng(0)
    icc_bs = []
    for _ in range(1000):
        pick = rng.choice(keys, len(keys), replace=True)
        icc_bs.append(calc_icc({k: vid_of[k] for k in pick}))
    icc_bs = np.array([x for x in icc_bs if x == x])
    print(f'  ICC(1) = {icc_obs:.3f}  95%CI [{np.percentile(icc_bs,2.5):.3f}, '
          f'{np.percentile(icc_bs,97.5):.3f}]  (bootstrap, 按视频重采样)')
    print('  说明: 该 ICC 是对"该配对是否低分"的二值事件计算的, 度量视频内低分事件的相关性,')
    print('        不能直接解释为"失败原因的视频占比"。')


if __name__ == '__main__':
    main()
