#!/usr/bin/env python
"""Step 3.1: NT 硬样本可视化 —— 低分 / 高分 各取代表性样本, 三列对照。

列: 配对真实帧 | NT 伪造帧 + GT 轮廓 | NT 伪造帧 + 模型 soft mask + 实例分数

选样纪律: 低分组按实例分数升序取, 但**强制来自不同源视频**, 避免连续抽取同一视频的相邻帧
(相邻帧高度相关, 视觉上会让人误以为发现了共性)。

用法
----
  python -I evaluate/visualize_nt_subgroups.py --data data/dataset_v1 --split val --n 12
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

TH = 240          # 每格高度
LABEL_H = 26


def pick(rows, n, key, reverse):
    """按 key 排序取样, 但每个源视频最多取 1 张, 保证覆盖不同视频。"""
    out, seen = [], set()
    for r in sorted(rows, key=key, reverse=reverse):
        if r['src'] in seen:
            continue
        seen.add(r['src'])
        out.append(r)
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--conf-csv', default='evaluate/conf_analysis_val.csv')
    ap.add_argument('--conf-thr', type=float, default=0.25)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--out', default='_debug/nt_subgroups.jpg')
    args = ap.parse_args()

    if 'cloud_baseline_v1' in args.weights and 'dataset_v2' in args.data:
        sys.exit('权重/数据分布错配, 拒绝运行')

    ops = P._install_patch()

    split_of, real_at, fake_at = {}, defaultdict(dict), {}
    with open(os.path.join(args.data, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] != args.split:
                continue
            stem = os.path.basename(r['image'])[:-4]
            split_of[stem] = r['split']
            fr = int(r['frame'])
            if r['is_fake'] == '0':
                real_at[r['video_id']][fr] = stem
            elif r['method'] == 'NeuralTextures':
                fake_at[(r['video_id'].split('_')[0], fr)] = stem

    conf = {r['stem']: float(r['max_conf'])
            for r in csv.DictReader(open(args.conf_csv, encoding='utf-8'))}

    def ip(s):
        return os.path.join(args.data, 'images', split_of[s], s + '.jpg')

    def lp(s):
        return os.path.join(args.data, 'labels', split_of[s], s + '.txt')

    pairs = [dict(src=src, fr=fr, rstem=real_at[src][fr], fstem=fstem,
                  conf=conf.get(fstem, np.nan))
             for (src, fr), fstem in fake_at.items() if fr in real_at.get(src, {})]
    pairs = [p for p in pairs if p['conf'] == p['conf']]
    low = pick([p for p in pairs if p['conf'] < args.conf_thr], args.n, lambda r: r['conf'], False)
    high = pick([p for p in pairs if p['conf'] >= args.conf_thr], args.n, lambda r: r['conf'], True)
    sel = low + high
    print(f'低分 {len(low)} 张 / 高分 {len(high)} 张, 共 {len({s["src"] for s in sel})} 个不同源视频')

    from ultralytics import YOLO
    model = YOLO(args.weights)
    P._install_head_hook(model)

    tiles = []
    for i in range(0, len(sel), 8):
        chunk = sel[i:i + 8]
        paths = [ip(c['fstem']) for c in chunk]
        P._CAP.clear()
        P._CAP['by_path'] = {}
        model.predict(paths, imgsz=640, conf=0.001, device='0', verbose=False, max_det=100)
        ho = P._CAP.get('head')
        lh, lw = P._CAP.get('shape', (640, 640))
        for j, c in enumerate(chunk):
            fi = cv2.imread(ip(c['fstem']))
            ri = cv2.imread(ip(c['rstem']))
            oh, ow = fi.shape[:2]
            mask = P.rasterize_label(lp(c['fstem']), ow, oh)
            det, protos = ho[0][0][j], ho[0][1][j]
            k = int(det[4].argmax())
            coeffs = det[5:, k].unsqueeze(0)
            x, y, w, h = det[:4, k].tolist()
            box = torch.tensor([[x - w / 2, y - h / 2, x + w / 2, y + h / 2]],
                               device=det.device, dtype=det.dtype)
            soft = P.to_orig(P._soft_from_coeffs(coeffs, protos, box, lh, lw, False),
                             lh, lw, oh, ow)[0].cpu().numpy()
            s = soft - soft.min()
            s = s / max(s.max(), 1e-9)
            heat = cv2.applyColorMap((s * 255).astype(np.uint8), cv2.COLORMAP_JET)
            resp = cv2.addWeighted(fi, 0.55, heat, 0.45, 0)
            gt = fi.copy()
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt, cnts, -1, (0, 255, 0), 2)
            tiles.append([ri, gt, resp, c])

    def fit(im, h):
        return cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))

    def tile(img, txt, w):
        if img.shape[1] != w:
            img = np.pad(img, ((0, 0), (0, w - img.shape[1]), (0, 0)))
        bar = np.full((LABEL_H, w, 3), 20, np.uint8)
        cv2.putText(bar, txt, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (230, 230, 230), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    out, by_grp = [], {'low': [], 'high': []}
    for i, (ri, gt, resp, c) in enumerate(tiles):
        g = 'low' if c['conf'] < args.conf_thr else 'high'
        by_grp[g].append([
            fit(ri, TH), fit(gt, TH), fit(resp, TH),
            f'配对真实帧  {c["rstem"][:26]}', 'NT 伪造帧 + GT',
            f'soft mask   conf={c["conf"]:.3f}'])

    # 两组必须统一宽度才能纵向拼接
    W = max(t.shape[1] for g in by_grp.values() for r in g for t in r[:3])
    for g in by_grp.values():
        for r in g:
            for k in range(3):
                if r[k].shape[1] < W:
                    r[k] = np.pad(r[k], ((0, 0), (0, W - r[k].shape[1]), (0, 0)))

    for tag, g, col in (('低分 (conf<0.25) —— 这些就是漏检来源', 'low', (60, 60, 255)),
                        ('高分 (conf>=0.25)', 'high', (60, 220, 60))):
        want = W * 3 + 16
        band = np.full((36, want, 3), (40, 40, 40), np.uint8)
        cv2.putText(band, tag, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
        out.append(band)
        for r in by_grp[g]:
            t = [tile(r[k], r[3 + k], W) for k in range(3)]
            gap = np.full((t[0].shape[0], 8, 3), 255, np.uint8)
            out.append(np.hstack([t[0], gap, t[1], gap, t[2]]))
        out.append(np.full((12, want, 3), 255, np.uint8))

    grid = np.vstack(out)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'拼图已保存 {args.out}  尺寸 {grid.shape[1]}x{grid.shape[0]}')


if __name__ == '__main__':
    main()
