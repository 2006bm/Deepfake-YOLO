#!/usr/bin/env python
"""阶段 2:NeuralTextures 漏检样本可视化。

背景:NT 在验证集上漏检 22.5%,其中 16.1% 是低置信(阈值代价)、6.5% 完全无预测。
本脚本把这两类样本各抽若干张画出来,用于人工判断:
  - GT 的伪造区域到底长什么样?
  - 模型给出的低置信预测落在哪里?偏了还是没覆盖?
  - 完全无预测的那些,肉眼能否看出伪影?

输出三栏拼图:原图+GT轮廓 / 原图+模型预测 / 原图+GT与预测叠加
"""
import argparse
import os
import sys

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def rasterize_label(label_path, w, h):
    canvas = np.zeros((h, w), np.uint8)
    if not os.path.exists(label_path):
        return canvas
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            xy = np.array(p[1:], dtype=np.float64).reshape(-1, 2)
            pts = np.stack([xy[:, 0] * w, xy[:, 1] * h], axis=1)
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights',
                    default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--csv', default='evaluate/conf_analysis_val.csv',
                    help='由 conf_analysis.py 生成的逐样本置信度')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--n-weak', type=int, default=4, help='低置信样本数')
    ap.add_argument('--n-none', type=int, default=4, help='无预测样本数')
    ap.add_argument('--out', default='_debug/nt_miss_analysis.jpg')
    args = ap.parse_args()

    import csv as _csv
    from ultralytics.utils import SETTINGS
    SETTINGS['sync'] = False
    from ultralytics import YOLO

    # 读逐样本置信度,挑出 NT 的两类问题样本
    recs = [r for r in _csv.DictReader(open(args.csv))
            if r['method'] == 'NeuralTextures']
    for r in recs:
        r['max_conf'] = float(r['max_conf'])
    none_pred = sorted([r for r in recs if r['max_conf'] == 0],
                       key=lambda r: r['stem'])
    weak = sorted([r for r in recs if 0 < r['max_conf'] < args.conf],
                  key=lambda r: r['max_conf'])
    print(f'NT 样本 {len(recs)}  无预测 {len(none_pred)}  低置信 {len(weak)}')

    picks = ([(r['stem'], 'NO-PRED', r['max_conf'])
              for r in none_pred[:args.n_none]] +
             [(r['stem'], 'LOW-CONF', r['max_conf'])
              for r in weak[:args.n_weak]])

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)
    model = YOLO(args.weights)

    rows = []
    for stem, tag, mc in picks:
        ip = os.path.join(img_dir, stem + '.jpg')
        img = cv2.imread(ip)
        h, w = img.shape[:2]
        gt = rasterize_label(os.path.join(lab_dir, stem + '.txt'), w, h)
        res = model.predict(ip, imgsz=640, conf=0.01, device='0',
                            verbose=False, max_det=100)[0]

        pred = np.zeros((h, w), np.uint8)
        if res.masks is not None and res.masks.xy is not None:
            for poly in res.masks.xy:
                if poly is not None and len(poly) >= 3:
                    cv2.fillPoly(pred, [np.round(poly).astype(np.int32)], 1)

        # 三栏:GT / 预测 / 叠加
        c1 = img.copy()
        cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c1, cs, -1, (0, 255, 0), 2)
        c2 = img.copy()
        cs2, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(c2, cs2, -1, (0, 0, 255), 2)
        c3 = img.copy()
        c3[gt > 0] = (0.55 * c3[gt > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        c3[pred > 0] = (0.55 * c3[pred > 0] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        overlap = (gt > 0) & (pred > 0)
        c3[overlap] = (0.4 * c3[overlap] + 0.6 * np.array([0, 255, 255])).astype(np.uint8)

        for c, txt in ((c1, f'{tag}  GT (green)'),
                       (c2, f'{tag}  PRED (red)'),
                       (c3, f'overlap (yellow)  maxconf={mc:.3f}')):
            cv2.putText(c, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
        rows.append([c1, c2, c3])

    th = 260
    tiles = []
    for r in rows:
        rs = [cv2.resize(x, (int(x.shape[1] * th / x.shape[0]), th)) for x in r]
        tiles.append(np.hstack(rs))
    wmax = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0))) for t in tiles]
    grid = np.vstack(tiles)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'拼图已保存 {args.out}   形状 {grid.shape}')
    print('  绿=GT  红=模型预测  黄=重合')


if __name__ == '__main__':
    main()
