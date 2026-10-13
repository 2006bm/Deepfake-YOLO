#!/usr/bin/env python
"""阶段 2b:定量区分 NeuralTextures 漏检的三种失败机制。

问题:NT 漏检 22.5%。可视化提示两类失败可能机制不同,需要定量确认:
  A. 完全无预测 (6.5%)  -> 模型对该区域零响应
  B. 低置信     (16.1%) -> 有预测但 conf < 0.25

对 B 类进一步拆分:
  B1. 定位问题 —— 即使忽略阈值,预测区域与 GT 的 IoU 也很低(预测偏了)
  B2. 置信度问题 —— IoU 其实不低,只是置信度没到阈值(可以用阈值/校准解决)

同时对 A 类检查:它们的 GT 面积是否显著小于正常样本?
若是,则"看不到"可能与目标过小有关,而非伪造类型本身。

用法: python evaluate/failure_mode_analysis.py --split val
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
            cv2.fillPoly(c, [np.round(np.stack([xy[:, 0] * w, xy[:, 1] * h], 1)).astype(np.int32)], 1)
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--conf-csv', default='evaluate/conf_analysis_val.csv')
    ap.add_argument('--conf', type=float, default=0.25, help='评测阈值')
    ap.add_argument('--out', default='evaluate/failure_modes_val.csv')
    args = ap.parse_args()

    from ultralytics.utils import SETTINGS
    SETTINGS['sync'] = False
    from ultralytics import YOLO

    recs = [r for r in csv.DictReader(open(args.conf_csv))]
    for r in recs:
        r['max_conf'] = float(r['max_conf'])
    nt = [r for r in recs if r['method'] == 'NeuralTextures']
    others = [r for r in recs if r['method'] != 'NeuralTextures']

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)

    # 只对 NT 样本做无阈值推理;其他方法只需要 GT 面积做对照
    model = YOLO(args.weights)
    rows = []
    stems = [r['stem'] for r in nt]
    for i in range(0, len(stems), 16):
        chunk = stems[i:i + 16]
        paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
        res = model.predict(paths, imgsz=640, conf=0.01, device='0',
                            verbose=False, max_det=100)
        for stem, r in zip(chunk, res):
            img = cv2.imread(os.path.join(img_dir, stem + '.jpg'))
            h, w = img.shape[:2]
            gt = rasterize_label(os.path.join(lab_dir, stem + '.txt'), w, h)
            pred = np.zeros((h, w), np.uint8)
            if r.masks is not None and r.masks.xy is not None:
                for poly in r.masks.xy:
                    if poly is not None and len(poly) >= 3:
                        cv2.fillPoly(pred, [np.round(poly).astype(np.int32)], 1)
            u = int((gt | pred).sum())
            iou = float((gt & pred).sum() / u) if u else 0.0
            rows.append(dict(stem=stem, gt_px=int(gt.sum()),
                             pred_px=int(pred.sum()), iou_notresh=iou))
        print(f'\r  推理 {min(i+16, len(stems))}/{len(stems)}', end='', flush=True)
    print('\n')

    # 按 conf_analysis 的分类打标签
    mp = {r['stem']: r['max_conf'] for r in nt}
    for r in rows:
        mc = mp[r['stem']]
        r['max_conf'] = mc
        if mc == 0:
            r['group'] = 'A_no_pred'
        elif mc < args.conf:
            r['group'] = 'B_low_conf'
        else:
            r['group'] = 'C_ok'

    g = defaultdict(list)
    for r in rows:
        g[r['group']].append(r)

    print('=' * 72)
    print('NeuralTextures 失败机制拆分 (验证集, 阈值 %.2f)' % args.conf)
    print('=' * 72)
    print(f'{"组":<14}{"n":>6}{"GT面积中位":>12}{"无阈值IoU中位":>15}{"IoU>0.5占比":>12}')
    for k in ['A_no_pred', 'B_low_conf', 'C_ok']:
        v = g.get(k, [])
        if not v:
            continue
        gt = np.median([x['gt_px'] for x in v])
        iou = np.median([x['iou_notresh'] for x in v])
        hi = np.mean([x['iou_notresh'] > 0.5 for x in v]) * 100
        print(f'{k:<14}{len(v):>6}{gt:>12.0f}{iou:>15.4f}{hi:>11.1f}%')

    print()
    print('=' * 72)
    print('B 类(低置信)内部再拆分 —— 到底是定位问题还是置信度问题?')
    print('=' * 72)
    b = g.get('B_low_conf', [])
    if b:
        good = [x for x in b if x['iou_notresh'] > 0.5]
        bad = [x for x in b if x['iou_notresh'] <= 0.5]
        print(f'  B1 定位问题 (无阈值 IoU<=0.5): {len(bad):>3} 张 '
              f'({len(bad)/len(b)*100:.1f}%)  中位 IoU {np.median([x["iou_notresh"] for x in bad]) if bad else 0:.4f}')
        print(f'  B2 置信度问题 (无阈值 IoU>0.5): {len(good):>3} 张 '
              f'({len(good)/len(b)*100:.1f}%)  中位 IoU {np.median([x["iou_notresh"] for x in good]) if good else 0:.4f}')
        print()
        print('  解读: B2 是"其实找对了、只是置信度低" -> 可通过阈值/校准解决')
        print('        B1 是"确实没找对"             -> 属于定位能力问题')

    print()
    print('=' * 72)
    print('A 类(完全无预测)的 GT 面积 vs 全体 NT')
    print('=' * 72)
    a = g.get('A_no_pred', [])
    allnt = rows
    if a:
        print(f'  A 类 GT 面积中位数   : {np.median([x["gt_px"] for x in a]):>8.0f} px')
        print(f'  全体 NT GT 面积中位数: {np.median([x["gt_px"] for x in allnt]):>8.0f} px')
        ratio = np.median([x['gt_px'] for x in a]) / max(np.median([x['gt_px'] for x in allnt]), 1)
        print(f'  比值                 : {ratio:.3f}  '
              f'-> {"面积偏小,与目标尺度有关" if ratio < 0.7 else "面积与整体相当,不是尺度问题"}')

    # 对照组:其他方法的 GT 面积
    print()
    print('  对照(其他方法的 GT 面积中位数):')
    for m in ['Deepfakes', 'Face2Face', 'FaceSwap']:
        sub = [r for r in others if r['method'] == m][:80]
        if sub:
            print(f'    {m:<16} 需重算,跳过(本表仅含 NT)')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=['stem', 'group', 'max_conf',
                                            'gt_px', 'pred_px', 'iou_notresh'])
        wtr.writeheader()
        wtr.writerows(rows)
    print(f'\n逐样本结果已写入 {args.out}')


if __name__ == '__main__':
    main()
