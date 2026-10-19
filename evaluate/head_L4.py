# -*- coding: utf-8 -*-
"""Step 3.5 L4: 最终分类输出层的信息保留。

L4 两个指标(对 fake 和 real 都提取):
  L4a 分类响应图: head.cv3[2][2] 输出的 nc=1 分类 logits, 用 GT 掩码区域平均(单值)。
      这是"分类映射"的结果, 但保留了空间区域信息。
  L4b 实例分数:   NMS 前 top-1 anchor 的 sigmoid 置信度(即部署输出的来源)。

目的: 补全 L1(0.868) → L2(0.763) → L3(0.760) 之后的信息流动, 看最终分数还保留多少。
  L3 ≈ L4a      -> cv3 的 128→1 线性映射几乎无损
  L4a > L4b     -> 信息损失在 top-1 anchor 选择 / 实例分数聚合
  L4b ≈ 0.5     -> 最终输出无法区分低分 NT 与真实(这正是原模型的失败)

用法: python -I evaluate/head_L4.py
"""
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'evaluate')
import pixel_auroc as P                                        # noqa: E402


def auroc(y, s):
    y = np.asarray(y, bool)
    npos = int(y.sum()); nneg = y.size - npos
    if npos == 0 or nneg == 0:
        return np.nan
    o = np.argsort(s, kind='stable')
    rk = np.empty(s.size); rk[o] = np.arange(1, s.size + 1)
    return (rk[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def letterbox_mask(mask, feat_hw, new_shape=(640, 640), stride=32):
    oh, ow = mask.shape[:2]
    r = min(new_shape[0] / oh, new_shape[1] / ow); r = min(r, 1.0)
    uw, uh = round(ow * r), round(oh * r)
    dw, dh = new_shape[1] - uw, new_shape[0] - uh
    dw, dh = dw % stride, dh % stride
    top, left = round(dh / 2 - 0.1), round(dw / 2 - 0.1)
    m = cv2.resize(mask, (uw, uh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    lh, lw = uh + dh, uw + dw
    padded = np.zeros((lh, lw), np.float32)
    padded[top:top + uh, left:left + uw] = m
    fh, fw = feat_hw
    return cv2.resize(padded, (fw, fh), interpolation=cv2.INTER_AREA)


def main():
    d = np.load('evaluate/features_nt_val.npz')
    ri = d['real_idx'].astype(int); fi = d['fake_idx'].astype(int)
    src = np.array(d['src']); group = np.array(d['group'])
    real_stems = list(d['real_stem']); fake_stems = list(d['fake_stem'])
    n = len(ri)
    lo = group == 'low'; hi = group == 'high'

    # 建 stem -> 配对索引
    pairs = [(real_stems[i], fake_stems[i]) for i in range(n)]
    fake_of_real = {r: f for r, f in pairs}

    from ultralytics import YOLO
    model = YOLO('runs/segment/cloud_baseline_v1/weights/best.pt')
    dm = model.model.model
    cap = {}
    # L4a: cv3[2][2] 分类 logits (nc=1, P5 尺度)
    hs = [dm[22].cv3[2][2].register_forward_hook(lambda m, i, o: cap.update(l4=o))]
    P._install_head_hook(model)   # 拿 head 输出, 用于 top-1 conf

    def ip(s):
        sp = 'val'
        return f'data/dataset_v1/images/{sp}/{s}.jpg'

    def lp(s):
        return f'data/dataset_v1/labels/val/{s}.txt'

    need = list(dict.fromkeys([s for r, f in pairs for s in (r, f)]))
    l4a = {}; l4b = {}
    for i in range(0, len(need), 8):
        chunk = need[i:i + 8]
        paths = [ip(s) for s in chunk]
        cap.clear(); P._CAP.clear(); P._CAP['by_path'] = {}
        model.predict(paths, imgsz=640, conf=0.001, device='0', verbose=False, max_det=100)
        l4 = cap['l4']                                    # (B, 1, fh, fw)
        ho = P._CAP.get('head')
        for j, stem in enumerate(chunk):
            img = cv2.imread(ip(stem)); oh, ow = img.shape[:2]
            if stem.startswith('Real_'):
                mask = P.rasterize_label(lp(fake_of_real[stem]), ow, oh)
            else:
                mask = P.rasterize_label(lp(stem), ow, oh)
            fh, fw = l4.shape[-2:]
            mf = letterbox_mask(mask, (fh, fw))
            l4_np = l4[j, 0].cpu().numpy()                    # (fh, fw) P5 分类 logit
            sel = mf > 0.5
            l4a[stem] = float(l4_np[sel].max()) if sel.any() else 0.0
            # L4b: NMS 前 top-1 conf
            det = ho[0][0][j]
            l4b[stem] = float(det[4].max().cpu())
        print(f'\r  {min(i+8, len(need))}/{len(need)}', end='', flush=True)
    print()
    for h in hs:
        h.remove()

    # 组装成配对数组
    La, Lb = [], []
    for r, f in pairs:
        La.append((l4a[f], l4a[r])); Lb.append((l4b[f], l4b[r]))
    La = np.array(La); Lb = np.array(Lb)                  # (n, 2) = (fake, real)

    # 三个任务的 AUROC(标量直接排序, 无训练)
    def task_auc(mask):
        f = np.concatenate([La[mask, 0]]); r = np.concatenate([La[np.ones(n, bool), 1]])
        y = np.concatenate([np.ones(mask.sum()), np.zeros(n)])
        return auroc(y, np.concatenate([La[mask, 0], La[:, 1]])), \
               auroc(np.concatenate([np.ones(mask.sum()), np.zeros(n)]),
                     np.concatenate([Lb[mask, 0], Lb[:, 1]]))

    print('=' * 84)
    print('L4 最终输出的判别能力 (AUROC, 标量直接排序; 括号内为 L4a分类logit / L4b实例分数)')
    print('=' * 84)
    for tname, m in (('完整 NT vs Real', np.ones(n, bool)),
                     ('低分 NT vs Real', lo),
                     ('高分 NT vs Real', hi)):
        fa = auroc(np.concatenate([np.ones(m.sum()), np.zeros(n)]),
                   np.concatenate([La[m, 0], La[:, 1]]))
        fb = auroc(np.concatenate([np.ones(m.sum()), np.zeros(n)]),
                   np.concatenate([Lb[m, 0], Lb[:, 1]]))
        print(f'  {tname:<18} L4a={fa:.4f}   L4b={fb:.4f}')

    # 信息流动总览
    print()
    print('=' * 84)
    print('完整信息流动 (低分 NT vs Real 的判别能力)')
    print('=' * 84)
    L1 = 0.8682; L2 = 0.7629; L3 = 0.7603
    fa = auroc(np.concatenate([np.ones(lo.sum()), np.zeros(n)]),
               np.concatenate([La[lo, 0], La[:, 1]]))
    fb = auroc(np.concatenate([np.ones(lo.sum()), np.zeros(n)]),
               np.concatenate([Lb[lo, 0], Lb[:, 1]]))
    print(f'  L1 backbone {L1:.3f} → L2 neck {L2:.3f} → L3 分类前 {L3:.3f} '
          f'→ L4a 分类logit {fa:.3f} → L4b 实例分数 {fb:.3f}')


if __name__ == '__main__':
    main()
