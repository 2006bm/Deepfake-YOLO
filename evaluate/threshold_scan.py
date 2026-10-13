#!/usr/bin/env python
"""阶段 3:验证集阈值扫描 + 置信度校准分析。

目标:确定阈值对性能的影响,为 FFT 是否值得做提供依据。

关键设计:
  1. **一次低阈值推理,离线扫描全部阈值** —— 避免重复推理 9 次。
  2. **纳入真实图像** —— 降低阈值会引入误检,只看伪造样本的命中率会得出
     片面结论("降阈值就能提升"),必须同时看真实图像的误检率。
  3. 图像级检测指标(TP/FP/FN)与像素级定位指标(IoU/Dice)分开报告。

指标定义:
  图像级: 命中 = 该伪造图至少有 1 个 conf>=t 的预测
          FP   = 真实图有 conf>=t 的预测
          Prec/Rec/F1 基于图像级 TP/FP/FN
  像素级: 合并该图所有 conf>=t 的预测掩码,与 GT 多边形比较(伪造图)
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

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.40, 0.50]


def rasterize_label(path, w, h):
    c = np.zeros((h, w), np.uint8)
    if not os.path.exists(path):
        return c
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            xy = np.array(p[1:], dtype=np.float64).reshape(-1, 2)
            pts = np.stack([xy[:, 0] * w, xy[:, 1] * h], axis=1)
            cv2.fillPoly(c, [np.round(pts).astype(np.int32)], 1)
    return c


def poly_ras(poly, w, h):
    c = np.zeros((h, w), np.uint8)
    if poly is not None and len(poly) >= 3:
        cv2.fillPoly(c, [np.round(poly).astype(np.int32)], 1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--probe-conf', type=float, default=0.01)
    ap.add_argument('--out-prefix', default='evaluate/threshold_scan')
    args = ap.parse_args()

    from ultralytics.utils import SETTINGS
    SETTINGS['sync'] = False
    from ultralytics import YOLO

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)
    all_stems = sorted(f[:-4] for f in os.listdir(lab_dir) if f.endswith('.txt'))
    fake = [s for s in all_stems if not s.startswith('Real_')]
    real = [s for s in all_stems if s.startswith('Real_')]
    print(f'划分 {args.split}: 伪造 {len(fake)}  真实 {len(real)}')
    print(f'探测阈值 {args.probe_conf},离线扫描 {THRESHOLDS}\n')

    model = YOLO(args.weights)
    # 逐样本记录: 方法/是否真实/每条预测的 conf + 其掩码(以 conf 降序保存)
    samples = {}
    for tag, stems in (('fake', fake), ('real', real)):
        for i in range(0, len(stems), args.batch):
            chunk = stems[i:i + args.batch]
            paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
            results = model.predict(paths, imgsz=args.imgsz, conf=args.probe_conf,
                                    device='0', verbose=False, max_det=100)
            for stem, r in zip(chunk, results):
                h, w = r.orig_shape
                confs, masks = [], []
                if r.masks is not None and r.masks.xy is not None:
                    bconf = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                    for k, poly in enumerate(r.masks.xy):
                        c = float(bconf[k]) if k < len(bconf) else 0.0
                        confs.append(c)
                        masks.append(poly_ras(poly, w, h))
                gt = (rasterize_label(os.path.join(lab_dir, stem + '.txt'), w, h)
                      if tag == 'fake' else None)
                samples[stem] = dict(
                    method=stem.split('_')[0], is_real=(tag == 'real'),
                    confs=confs, masks=masks, gt=gt, w=w, h=h)
            done = min(i + args.batch, len(stems))
            print(f'\r  {tag} 推理 {done}/{len(stems)}', end='', flush=True)
    print('\n')

    # ---- 离线扫描阈值 ----
    rows = []
    for t in THRESHOLDS:
        tp = fn = fp = tn = 0
        ious, dices = defaultdict(list), defaultdict(list)
        hit = defaultdict(lambda: [0, 0])
        fp_per_real = []
        for stem, s in samples.items():
            confs = np.array(s['confs']) if s['confs'] else np.array([])
            keep = np.where(confs >= t)[0] if confs.size else np.array([], int)
            if s['is_real']:
                n = len(keep)
                fp_per_real.append(n)
                if n > 0:
                    fp += 1
                else:
                    tn += 1
                continue
            hit[s['method']][1] += 1
            if len(keep) == 0:
                fn += 1
                continue
            tp += 1
            hit[s['method']][0] += 1
            pred = np.zeros((s['h'], s['w']), np.uint8)
            for k in keep:
                pred |= s['masks'][k]
            inter = int((s['gt'] & pred).sum())
            union = int((s['gt'] | pred).sum())
            sa, sb = int(s['gt'].sum()), int(pred.sum())
            i = inter / union if union else 0.0
            d = 2 * inter / (sa + sb) if (sa + sb) else 0.0
            ious[s['method']].append(i)
            dices[s['method']].append(d)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        alliou = [x for m in METHODS for x in ious[m]]
        alldice = [x for m in METHODS for x in dices[m]]
        rows.append(dict(
            thr=t, tp=tp, fp=fp, fn=fn, tn=tn,
            prec=prec, rec=rec, f1=f1,
            iou_mean=float(np.mean(alliou)) if alliou else 0.0,
            dice_mean=float(np.mean(alldice)) if alldice else 0.0,
            fp_rate=fp / (fp + tn) if (fp + tn) else 0.0,
            fp_per_img=float(np.mean(fp_per_real)) if fp_per_real else 0.0,
            **{f'hit_{m}': (hit[m][0] / hit[m][1] if hit[m][1] else 0.0) for m in METHODS},
            **{f'iou_{m}': (float(np.mean(ious[m])) if ious[m] else 0.0) for m in METHODS},
        ))

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    with open(f'{args.out_prefix}.csv', 'w', newline='', encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)

    # ---- 打印 ----
    print('=' * 100)
    print('图像级检测指标(TP=伪造图命中, FP=真实图误检)')
    print('=' * 100)
    print(f'{"thr":>5}{"TP":>6}{"FN":>6}{"FP":>6}{"Prec":>8}{"Rec":>8}{"F1":>8}'
          f'{"真实误检率":>11}{"每图误检":>10}')
    for r in rows:
        print(f'{r["thr"]:>5.2f}{r["tp"]:>6}{r["fn"]:>6}{r["fp"]:>6}'
              f'{r["prec"]:>8.4f}{r["rec"]:>8.4f}{r["f1"]:>8.4f}'
              f'{r["fp_rate"]*100:>10.2f}%{r["fp_per_img"]:>10.2f}')

    print()
    print('=' * 100)
    print('各方法命中率')
    print('=' * 100)
    print(f'{"thr":>5}' + ''.join(f'{m:>17}' for m in METHODS))
    for r in rows:
        print(f'{r["thr"]:>5.2f}' + ''.join(f'{r[f"hit_{m}"]*100:>16.1f}%' for m in METHODS))

    print()
    print('=' * 100)
    print('各方法 Mask IoU(合并该图所有 conf>=t 的预测)')
    print('=' * 100)
    print(f'{"thr":>5}{"总体":>10}' + ''.join(f'{m:>17}' for m in METHODS))
    for r in rows:
        print(f'{r["thr"]:>5.2f}{r["iou_mean"]:>10.4f}'
              + ''.join(f'{r[f"iou_{m}"]:>17.4f}' for m in METHODS))

    print(f'\n结果已写入 {args.out_prefix}.csv')


if __name__ == '__main__':
    main()
