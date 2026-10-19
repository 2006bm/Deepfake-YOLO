# -*- coding: utf-8 -*-
"""Step 3.3: NT 低置信样本的特征空间归因。

核心问题: 低分 NT 的中间特征, 是更接近真实人脸, 还是更接近高分伪造样本?

三层级 hook(均在 P5 尺度, stride 32, 区域池化后 512 维, 可直接比较):
  L1 backbone 最深  : layer 9  SPPF    (基础视觉表征)
  L2 neck 融合输出  : layer 21 C2f     (多尺度融合后)
  L3 head 分类前    : layer 22 内部 feats[2]  (cv2/cv3 之后、分类 logits 之前)

三个指标:
  a) 线性探针 AUROC  : 冻结特征 + 逻辑回归, 按源视频分组划分训练/验证(避免同视频相邻帧泄漏)
  b) 配对特征距离    : 同一源视频同帧的 real vs fake 特征欧氏距离
  c) 类间距离        : 低分fake / 高分fake / real 三者的成对距离矩阵

用法:
  python -I evaluate/feature_attribution.py --mode smoke   # 每方法 2 对, 验证 hook/池化/维度
  python -I evaluate/feature_attribution.py --mode full    # 全部 val 精确配对 + 线性探针
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel_auroc as P                                        # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def letterbox_mask(mask, feat_hw, new_shape=(640, 640), stride=32):
    """把原图 mask 映射到特征图空间, 返回软覆盖比例 (0~1) 的 (fh, fw)。

    与 ultralytics LetterBox(auto=True, center=True, stride=32) 逐位一致:
      r = min(new_shape/old); new_unpad = round(old*r); pad 取 stride 余数后居中。
    """
    oh, ow = mask.shape[:2]
    r = min(new_shape[0] / oh, new_shape[1] / ow)
    r = min(r, 1.0)                          # scaleup=False
    uw, uh = round(ow * r), round(oh * r)
    dw, dh = new_shape[1] - uw, new_shape[0] - uh
    dw, dh = dw % stride, dh % stride        # auto=True
    top, left = round(dh / 2 - 0.1), round(dw / 2 - 0.1)
    m = cv2.resize(mask, (uw, uh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    lh, lw = uh + dh, uw + dw                # 实际 letterbox 尺寸
    padded = np.zeros((lh, lw), np.float32)
    padded[top:top + uh, left:left + uw] = m
    fh, fw = feat_hw
    out = cv2.resize(padded, (fw, fh), interpolation=cv2.INTER_AREA)  # 软覆盖比例
    return out


def pooled_feat(feat, mask_feat):
    """用软掩码做区域加权平均池化, 得到 (C,) 向量。feat 已是 (C, H, W)。"""
    w = torch.from_numpy(mask_feat).to(feat.device, feat.dtype)
    num = (feat * w).sum(dim=(1, 2))
    den = w.sum() + 1e-6
    return num / den


def make_hooks(dm):
    cap = {}

    def h(name):
        return lambda mod, inn, out: cap.update({name: out})

    hs = [dm[9].register_forward_hook(h('l1_backbone')),
          dm[21].register_forward_hook(h('l2_neck'))]
    # L3 = cls_head(cv3) 分类映射前: cv3[2][1] 是 P5 尺度、128 通道、分类 logits 之前的特征。
    # (注意: head.forward_head 返回的 dict['feats'] 只是 neck 输入的引用, 不是分类前特征)
    hs.append(dm[22].cv3[2][1].register_forward_hook(h('l3_head')))
    # 额外 hook head 输出, 拿 NMS 前 det[4](已 sigmoid 的分类置信度)做 top-1 conf
    def head_hook(mod, inn, out):
        cap['det'] = out[0][0] if isinstance(out, tuple) else out
    hs.append(dm[22].register_forward_hook(head_hook))
    return cap, hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='smoke', choices=['smoke', 'full'])
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--conf-csv', default='evaluate/conf_analysis_val.csv')
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--per-method', type=int, default=0)
    ap.add_argument('--out', default='evaluate/features_nt_val.npz')
    args = ap.parse_args()

    if 'cloud_baseline_v1' in args.weights and 'dataset_v2' in args.data:
        sys.exit('权重/数据分布错配, 拒绝运行')

    # ---- 建配对 + 分组 ----
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

    def ip(s):
        return os.path.join(args.data, 'images', split_of[s], s + '.jpg')

    def lp(s):
        return os.path.join(args.data, 'labels', split_of[s], s + '.txt')

    pairs = sorted((src, fr, real_at[src][fr], fstem)
                   for (src, fr), fstem in fake_at.items() if fr in real_at.get(src, {}))
    # 尺寸检查: real 与 fake 尺寸不一致的配对跳过(与 real_fake_control 口径一致)
    ok_pairs = []
    for (src, fr, rstem, fstem) in pairs:
        ri = cv2.imread(ip(rstem)); fi = cv2.imread(ip(fstem))
        if ri is not None and fi is not None and ri.shape == fi.shape:
            ok_pairs.append((src, fr, rstem, fstem))
    pairs = ok_pairs
    real_mask_of = {rstem: fstem for (_, _, rstem, fstem) in pairs}
    if args.per_method:
        pairs = pairs[:args.per_method]
    print(f'配对 {len(pairs)} 对 ({args.split})  mode={args.mode}')

    # ---- 模型 + hook ----
    from ultralytics import YOLO
    model = YOLO(args.weights)
    dm = model.model.model
    cap, hooks = make_hooks(dm)

    need = sorted({p[2] for p in pairs} | {p[3] for p in pairs})
    F = {'l1': [], 'l2': [], 'l3': []}
    meta = []
    # 断点续跑: 检查 checkpoint
    import pickle
    ckpt_path = args.out + '.ckpt.pkl'
    done_stems = set()
    if os.path.exists(ckpt_path):
        with open(ckpt_path, 'rb') as f:
            ck = pickle.load(f)
        F = ck['F']; meta = ck['meta']; done_stems = ck['done_stems']
        print(f'  断点续跑: 已恢复 {len(meta)} 张图')

    for i in range(0, len(need), 8):
        chunk = [s for s in need[i:i + 8] if s not in done_stems]
        if not chunk:
            continue
        paths = [ip(s) for s in chunk]
        cap.clear()
        model.predict(paths, imgsz=640, conf=0.001, device='0', verbose=False, max_det=100)
        det = cap.get('det')
        for j, stem in enumerate(chunk):
            img = cv2.imread(ip(stem))
            oh, ow = img.shape[:2]
            if stem.startswith('Real_'):
                # real 帧用配对 fake 的 GT 掩码(归一化坐标, 套到同位置)
                mask = P.rasterize_label(lp(real_mask_of[stem]), ow, oh)
            else:
                mask = P.rasterize_label(lp(stem), ow, oh)
            l1 = cap['l1_backbone'][j]          # (C, fh, fw)
            l2 = cap['l2_neck'][j]
            l3 = cap['l3_head'][j]              # cls_head cv3[2][1], 128 通道
            for key, feat in (('l1', l1), ('l2', l2), ('l3', l3)):
                fh, fw = feat.shape[-2:]
                mf = letterbox_mask(mask, (fh, fw))
                F[key].append(pooled_feat(feat, mf).detach().cpu().numpy())
            top1 = float(det[j, 4].max().cpu()) if det is not None else np.nan
            meta.append(dict(stem=stem, is_real=stem.startswith('Real_'),
                             src=stem.split('_')[1],
                             frame=int(stem.split('_f')[-1]), top1=top1))
            done_stems.add(stem)
        if (i // 8 + 1) % 50 == 0:
            with open(ckpt_path, 'wb') as f:
                pickle.dump(dict(F=F, meta=meta, done_stems=done_stems), f)
        print(f'\r  提取 {len(meta)}/{len(need)}', end='', flush=True)
    print('\n')
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    for h in hooks:
        h.remove()

    for k in F:
        F[k] = np.stack(F[k])
        print(f'  {k} 特征矩阵 {F[k].shape}')

    if args.mode == 'smoke':
        # sanity: 三层级特征的数值范围和两两相关性
        for k in F:
            print(f'  {k}: L2范数 均值 {np.linalg.norm(F[k], axis=1).mean():.2f}')
        return

    # ---- 配对重组 + 指标 ----
    real_idx = {m['stem']: i for i, m in enumerate(meta) if m['is_real']}
    fake_idx = {m['stem']: i for i, m in enumerate(meta) if not m['is_real']}
    rows = []
    for (src, fr, rstem, fstem) in pairs:
        if rstem not in real_idx or fstem not in fake_idx:
            continue
        c = meta[fake_idx[fstem]]['top1']           # fake 侧的 NMS 前 top-1 conf
        rows.append(dict(src=src, real_stem=rstem, fake_stem=fstem, conf=c,
                         conf_real=meta[real_idx[rstem]]['top1'],
                         group='low' if c < args.conf_thr else 'high'))
    print(f'有效配对 {len(rows)} 对 (低分 {sum(r["group"]=="low" for r in rows)})')

    np.savez(args.out,
             l1=F['l1'], l2=F['l2'], l3=F['l3'],
             real_stem=[r['real_stem'] for r in rows],
             fake_stem=[r['fake_stem'] for r in rows],
             src=[r['src'] for r in rows],
             conf=[r['conf'] for r in rows],
             conf_real=[r['conf_real'] for r in rows],
             group=[r['group'] for r in rows],
             real_idx=[real_idx[r['real_stem']] for r in rows],
             fake_idx=[fake_idx[r['fake_stem']] for r in rows])
    print(f'特征已保存 {args.out}')


if __name__ == '__main__':
    main()
