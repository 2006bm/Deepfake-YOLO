#!/usr/bin/env python
"""NeuralTextures 漏检率是否受置信度阈值影响 —— 逐样本置信度分析。

背景:测试集评测用 conf=0.25,得到 NeuralTextures 漏检率 27.1%(其他方法 1.7~4.5%)。
但"没预测出区域"未必等于"没学到特征" —— 可能是模型给出了预测,只是置信度低于阈值。

方法:用极低阈值(0.01)跑一次推理,记录每个样本的**最高预测置信度**,
然后离线统计不同阈值下的命中率,从而把"阈值权衡"与"特征缺失"区分开。

**按科研规范在验证集上做** —— 测试集只用于最终确认评估,不用来调参。
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights',
                    default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--probe-conf', type=float, default=0.01,
                    help='探测阈值,设很低以捕获所有弱预测')
    ap.add_argument('--out', default='evaluate/conf_analysis_val.csv')
    args = ap.parse_args()

    from ultralytics import YOLO

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)
    stems = sorted(f[:-4] for f in os.listdir(lab_dir)
                   if f.endswith('.txt') and not f.startswith('Real_'))
    print(f'权重   {args.weights}')
    print(f'划分   {args.split}  伪造样本 {len(stems)} 张')
    print(f'探测阈值 conf={args.probe_conf}\n')

    model = YOLO(args.weights)
    rows = []
    for i in range(0, len(stems), args.batch):
        chunk = stems[i:i + args.batch]
        paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
        results = model.predict(paths, imgsz=args.imgsz, conf=args.probe_conf,
                                device=args.device, verbose=False, max_det=100)
        for stem, res in zip(chunk, results):
            method = stem.split('_')[0]
            conf = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.array([])
            rows.append(dict(stem=stem, method=method,
                             n_pred=int(len(conf)),
                             max_conf=float(conf.max()) if len(conf) else 0.0))
        print(f'\r  推理 {min(i + args.batch, len(stems))}/{len(stems)}',
              end='', flush=True)
    print('\n')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['stem', 'method', 'n_pred', 'max_conf'])
        w.writeheader()
        w.writerows(rows)
    print(f'逐样本置信度已写入 {args.out}\n')

    # ---- 核心分析:不同阈值下的命中率 ----
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    by = defaultdict(list)
    for r in rows:
        by[r['method']].append(r['max_conf'])

    print('=' * 74)
    print('不同置信度阈值下的"命中率"(该类别中至少产生一个预测的样本比例)')
    print('=' * 74)
    hdr = f'{"method":<16}' + ''.join(f'{t:>8.2f}' for t in thresholds)
    print(hdr)
    print('-' * len(hdr))
    for m in METHODS:
        v = np.array(by.get(m, []))
        if v.size == 0:
            continue
        cells = ''.join(f'{(v >= t).mean() * 100:>7.1f}%' for t in thresholds)
        print(f'{m:<16}{cells}')
    print()
    print('(漏检率 = 100% - 命中率)')
    print()
    print('=' * 74)
    print('置信度分布: 完全无预测 / 低置信 / 高置信')
    print('=' * 74)
    print(f'{"method":<16}{"n":>6}{"无预测":>10}{"<0.25":>10}{"≥0.25":>10}')
    for m in METHODS:
        v = np.array(by.get(m, []))
        if v.size == 0:
            continue
        zero = int((v == 0).sum())
        low = int(((v > 0) & (v < 0.25)).sum())
        high = int((v >= 0.25).sum())
        print(f'{m:<16}{v.size:>6}{zero:>10}{low:>10}{high:>10}')
    print()
    print('  无预测 = 模型完全没输出任何框(阈值降到 0.01 也没有)')
    print('  <0.25  = 有预测但置信度低于评测阈值 —— 这部分是"阈值代价",不是"特征缺失"')


if __name__ == '__main__':
    main()
