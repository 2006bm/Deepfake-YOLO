#!/usr/bin/env python
"""从 best.pt 计算 Dice / Mask IoU —— YOLO 官方指标里没有这两项。

为什么必须自己算:论文要区分两件不同的事 ——
  "发现伪造区域"      → mAP50 / mAP50-95(YOLO 自带)
  "伪造区域定位得准"  → Dice / Mask IoU(本脚本)

这直接回应答辩/审稿最常见的质疑:"你在做检测还是定位?"

实现要点:
  1. 单类别定位任务,把一张图内所有预测实例合并为一个二值掩码,
     与同样合并的 GT 掩码比较 —— 回答的是"模型认为哪里被伪造"。
  2. 预测掩码取 results[0].masks.xy(原始图像像素坐标的多边形),
     而不是 masks.data(模型输入分辨率的张量),避免 resize 引入误差。
  3. 逐图记录 method 与 mask_area_ratio,便于按伪造方法和区域大小分组分析 ——
     这是回答"哪类样本拖了性能"的关键,且无需重新训练。

用法:
    python evaluate/metrics.py --weights runs/segment/baseline_v1/weights/best.pt
    python evaluate/metrics.py --split val --limit 200      # 快速抽查
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

BINS = [(0.0, 0.05, '<5%'), (0.05, 0.20, '5-20%'), (0.20, 2.0, '≥20%')]


def rasterize_label(label_path, w, h):
    """YOLO-Seg 标签 → 二值掩码(单类别,合并所有实例)。"""
    canvas = np.zeros((h, w), np.uint8)
    if not os.path.exists(label_path):
        return canvas
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:
                continue
            xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
            pts = np.stack([xy[:, 0] * w, xy[:, 1] * h], axis=1)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas


def rasterize_pred(result, w, h):
    """模型预测 → 二值掩码(合并所有实例)。"""
    canvas = np.zeros((h, w), np.uint8)
    if result.masks is None or result.masks.xy is None:
        return canvas
    for poly in result.masks.xy:
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(canvas, [np.round(poly).astype(np.int32)], 1)
    return canvas


def iou_dice(a, b):
    inter = int((a & b).sum())
    union = int((a | b).sum())
    sa, sb = int(a.sum()), int(b.sum())
    iou = inter / union if union else float('nan')
    dice = 2 * inter / (sa + sb) if (sa + sb) else float('nan')
    return iou, dice


def stats(vals):
    v = np.array([x for x in vals if not np.isnan(x)])
    if v.size == 0:
        return None
    return dict(n=int(v.size), mean=float(v.mean()),
                median=float(np.median(v)), std=float(v.std()))


def main():
    ap = argparse.ArgumentParser(description='Dice / Mask IoU 评测')
    ap.add_argument('--weights',
                    default='runs/segment/baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--metadata', default='data/dataset_v1/metadata.csv')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--device', default='0')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='evaluate/seg_metrics.csv')
    args = ap.parse_args()

    from ultralytics import YOLO

    # metadata 提供 method 与 mask_area_ratio,用于分组分析
    meta = {}
    if os.path.exists(args.metadata):
        with open(args.metadata, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['split'] == args.split:
                    meta[os.path.basename(r['image'])[:-4]] = r

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)
    stems = sorted(f[:-4] for f in os.listdir(lab_dir) if f.endswith('.txt'))
    fake = [s for s in stems if not s.startswith('Real_')]
    if args.limit:
        fake = fake[:args.limit]

    print(f'权重   {args.weights}')
    print(f'划分   {args.split}   伪造样本 {len(fake)} 张')
    print(f'阈值   conf={args.conf}   imgsz={args.imgsz}\n')

    model = YOLO(args.weights)
    rows = []
    for i in range(0, len(fake), args.batch):
        chunk = fake[i:i + args.batch]
        paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
        results = model.predict(paths, imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)
        for stem, res in zip(chunk, results):
            h, w = res.orig_shape
            gt = rasterize_label(os.path.join(lab_dir, stem + '.txt'), w, h)
            pr = rasterize_pred(res, w, h)
            iou, dice = iou_dice(gt, pr)
            m = meta.get(stem, {})
            rows.append(dict(
                stem=stem,
                method=m.get('method', stem.split('_')[0]),
                area=float(m.get('mask_area_ratio', 0) or 0),
                iou=iou, dice=dice,
                gt_px=int(gt.sum()), pred_px=int(pr.sum()),
            ))
        done = min(i + args.batch, len(fake))
        print(f'\r  推理 {done}/{len(fake)}', end='', flush=True)
    print('\n')

    ious = [r['iou'] for r in rows]
    dices = [r['dice'] for r in rows]
    S = stats(ious)
    D = stats(dices)

    print('=' * 62)
    print('总体(Pred vs GT 掩码,单类别合并)')
    print('=' * 62)
    print(f'  Mask IoU  均值 {S["mean"]:.4f}  中位数 {S["median"]:.4f}')
    print(f'  Dice      均值 {D["mean"]:.4f}  中位数 {D["median"]:.4f}')
    blank = sum(1 for r in rows if r['pred_px'] == 0)
    print(f'  模型未预测出任何区域的样本: {blank} / {len(rows)} '
          f'({blank/len(rows)*100:.1f}%)')

    print('\n按伪造方法:')
    print(f'  {"method":<16}{"n":>6}{"IoU":>10}{"Dice":>10}')
    by_m = defaultdict(list)
    for r in rows:
        by_m[r['method']].append(r)
    for m in sorted(by_m):
        s, d = stats([x['iou'] for x in by_m[m]]), stats(
            [x['dice'] for x in by_m[m]])
        print(f'  {m:<16}{s["n"]:>6}{s["mean"]:>10.4f}{d["mean"]:>10.4f}')

    print('\n按伪造区域面积占比:')
    print(f'  {"bin":<10}{"n":>6}{"IoU":>10}{"Dice":>10}')
    for lo, hi, name in BINS:
        sub = [r for r in rows if lo <= r['area'] < hi]
        if not sub:
            continue
        s, d = stats([x['iou'] for x in sub]), stats([x['dice'] for x in sub])
        print(f'  {name:<10}{s["n"]:>6}{s["mean"]:>10.4f}{d["mean"]:>10.4f}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f'\n逐样本结果已写入 {args.out}')


if __name__ == '__main__':
    main()
