#!/usr/bin/env python
"""阶段 3b:置信度分布与校准分析。

严格区分两个概念:
  置信度分布偏移 —— 不同方法的置信度分布位置/形状不同(比较均值即可)
  置信度校准偏差 —— 预测置信度**系统性地低于实际命中率**(需要可靠性曲线/ECE)
只有后者成立时,才能声称"模型对 NT 过度保守"。

输出:
  1. 各方法置信度分布统计(均值/中位数/四分位/低置信段占比)
  2. 可靠性曲线 + ECE + Brier score(逐预测)
  3. 真实图像误检分析(每图最高置信度、高置信误检数、误检面积)
  4. PR 曲线(图像级 + 分方法)
  5. 汇总 CSV
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
FINE_THR = [round(x, 3) for x in np.arange(0.01, 0.96, 0.02)]
BINS = [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]


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
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--probe-conf', type=float, default=0.005)
    ap.add_argument('--iou-hit', type=float, default=0.5)
    ap.add_argument('--out-prefix', default='evaluate/calibration')
    args = ap.parse_args()

    from ultralytics.utils import SETTINGS
    SETTINGS['sync'] = False
    from ultralytics import YOLO

    lab_dir = os.path.join(args.data, 'labels', args.split)
    img_dir = os.path.join(args.data, 'images', args.split)
    all_stems = sorted(f[:-4] for f in os.listdir(lab_dir) if f.endswith('.txt'))
    fake = [s for s in all_stems if not s.startswith('Real_')]
    real = [s for s in all_stems if s.startswith('Real_')]
    print(f'伪造 {len(fake)}  真实 {len(real)}\n')

    model = YOLO(args.weights)
    # 逐图记录;逐预测记录 (conf, iou_vs_gt)
    per_image, per_pred = [], []
    for tag, stems in (('fake', fake), ('real', real)):
        for i in range(0, len(stems), args.batch):
            chunk = stems[i:i + args.batch]
            paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
            res = model.predict(paths, imgsz=640, conf=args.probe_conf,
                                device='0', verbose=False, max_det=100)
            for stem, r in zip(chunk, res):
                h, w = r.orig_shape
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
                ious = []
                gt = None
                if tag == 'fake':
                    gt = rasterize_label(os.path.join(lab_dir, stem + '.txt'), w, h)
                if r.masks is not None and r.masks.xy is not None:
                    for k, poly in enumerate(r.masks.xy):
                        c = float(confs[k]) if k < len(confs) else 0.0
                        if tag == 'fake' and poly is not None and len(poly) >= 3:
                            pm = np.zeros((h, w), np.uint8)
                            cv2.fillPoly(pm, [np.round(poly).astype(np.int32)], 1)
                            u = int((gt | pm).sum())
                            iou = float((gt & pm).sum() / u) if u else 0.0
                        else:
                            iou = float('nan')
                        ious.append((c, iou))
                per_image.append(dict(
                    stem=stem, method=stem.split('_')[0], is_real=(tag == 'real'),
                    n_pred=len(ious),
                    max_conf=max([c for c, _ in ious], default=0.0),
                    best_iou=max([x for _, x in ious if not np.isnan(x)], default=0.0),
                    gt_px=int(gt.sum()) if gt is not None else None))
                for c, iou in ious:
                    per_pred.append(dict(stem=stem, method=stem.split('_')[0],
                                         is_real=(tag == 'real'), conf=c, iou=iou))
            print(f'\r  {tag} {min(i+args.batch, len(stems))}/{len(stems)}',
                  end='', flush=True)
    print('\n')

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    fr = [x for x in per_image if not x['is_real']]
    rr = [x for x in per_image if x['is_real']]

    # ---- 1. 各方法置信度分布 ----
    print('=' * 92)
    print('1. 各方法「每图最高预测置信度」分布')
    print('=' * 92)
    print(f'{"method":<16}{"n":>6}{"均值":>9}{"中位":>9}{"Q1":>9}{"Q3":>9}'
          f'{"低置信段":>10}{"零预测":>9}')
    dist_rows = []
    for m in METHODS:
        v = np.array([x['max_conf'] for x in fr if x['method'] == m])
        pos = v[v > 0]
        low = ((v > 0) & (v < 0.25)).sum()
        d = dict(method=m, n=len(v), mean=v.mean(), median=np.median(v),
                 q1=np.percentile(v, 25), q3=np.percentile(v, 75),
                 low_frac=low / len(v) if len(v) else 0,
                 zero_frac=(v == 0).sum() / len(v) if len(v) else 0)
        dist_rows.append(d)
        print(f'{m:<16}{d["n"]:>6}{d["mean"]:>9.4f}{d["median"]:>9.4f}'
              f'{d["q1"]:>9.4f}{d["q3"]:>9.4f}{d["low_frac"]*100:>9.1f}%'
              f'{d["zero_frac"]*100:>8.1f}%')

    print()
    print('=' * 92)
    print('1b. 命中样本 vs 未命中样本 的置信度对比(阈值 0.25,命中=Iou>=0.5)')
    print('=' * 92)
    print(f'{"method":<16}{"命中n":>8}{"命中中位conf":>14}{"未命中n":>10}{"未命中中位conf":>16}')
    for m in METHODS:
        sub = [x for x in fr if x['method'] == m]
        h = [x for x in sub if x['best_iou'] >= args.iou_hit and x['max_conf'] >= 0.25]
        mi = [x for x in sub if not (x['best_iou'] >= args.iou_hit and x['max_conf'] >= 0.25)]
        hm = np.median([x['max_conf'] for x in h]) if h else 0
        mm = np.median([x['max_conf'] for x in mi]) if mi else 0
        print(f'{m:<16}{len(h):>8}{hm:>14.4f}{len(mi):>10}{mm:>16.4f}')

    # ---- 2. 可靠性 / ECE / Brier (逐预测,伪造图) ----
    print()
    print('=' * 92)
    print('2. 可靠性分析(逐预测,仅伪造图;正确 = 该预测与 GT 的 IoU >= %.1f)' % args.iou_hit)
    print('=' * 92)
    print(f'{"method":<16}{"n_pred":>8}{"均值conf":>10}{"实际精度":>10}'
          f'{"ECE":>9}{"Brier":>9}   判定')
    rel_rows = []
    for m in METHODS:
        ps = [p for p in per_pred if p['method'] == m and not p['is_real']]
        if not ps:
            continue
        conf = np.array([p['conf'] for p in ps])
        y = np.array([1.0 if (not np.isnan(p['iou']) and p['iou'] >= args.iou_hit) else 0.0
                      for p in ps])
        ece = 0.0
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
            if sel.sum() == 0:
                continue
            ece += sel.sum() / len(conf) * abs(conf[sel].mean() - y[sel].mean())
        brier = float(np.mean((conf - y) ** 2))
        gap = conf.mean() - y.mean()
        verdict = ('置信度偏低(校准偏差)' if gap < -0.05
                   else '置信度偏高' if gap > 0.05 else '基本校准')
        rel_rows.append(dict(method=m, n=len(ps), mean_conf=conf.mean(),
                             acc=y.mean(), ece=ece, brier=brier, gap=gap,
                             verdict=verdict))
        print(f'{m:<16}{len(ps):>8}{conf.mean():>10.4f}{y.mean():>10.4f}'
              f'{ece:>9.4f}{brier:>9.4f}   {verdict}')

    # ---- 3. 真实图像误检 ----
    print()
    print('=' * 92)
    print('3. 真实图像误检分析(n=%d)' % len(rr))
    print('=' * 92)
    rc = np.array([x['max_conf'] for x in rr])
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.40):
        sel = rc >= t
        print(f'  conf>={t:.2f}: 误检 {sel.sum():>5} 张 ({sel.mean()*100:>6.2f}%)  '
              f'每图误检数 {np.mean([x["n_pred"] for x in rr if x["max_conf"] >= t]) if sel.sum() else 0:.2f}')
    print()
    hi = rc[rc >= 0.5]
    print(f'  真实图中最高预测置信度: 中位 {np.median(rc):.4f}  '
          f'P90 {np.percentile(rc,90):.4f}  P99 {np.percentile(rc,99):.4f}  最大 {rc.max():.4f}')
    print(f'  高置信误检(conf>=0.5): {len(hi)} 张 ({len(hi)/len(rr)*100:.2f}%)'
          f'  -> 降阈值的代价主要是{"少量高置信误检" if len(hi)>0 else "低分噪声"}')

    # ---- 4. PR 曲线数据 ----
    pr_rows = []
    for t in FINE_THR:
        tp = sum(1 for x in fr if x['max_conf'] >= t)
        fn = len(fr) - tp
        fp = sum(1 for x in rr if x['max_conf'] >= t)
        tn = len(rr) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        row = dict(threshold=round(t, 3), precision=prec, recall=rec, f1=f1,
                   false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
                   avg_fp_per_real_image=float(np.mean(
                       [x['n_pred'] for x in rr if x['max_conf'] >= t])) if fp else 0.0)
        for m in METHODS:
            sub = [x for x in fr if x['method'] == m]
            row[f'recall_{m}'] = sum(1 for x in sub if x['max_conf'] >= t) / len(sub) if sub else 0.0
        pr_rows.append(row)

    best_f1 = max(pr_rows, key=lambda r: r['f1'])
    ok20 = [r for r in pr_rows if r['threshold'] <= 0.25 and r['false_positive_rate'] < 0.10]
    print()
    print('=' * 92)
    print('4. 工作阈值选择')
    print('=' * 92)
    print(f'  F1 最优: thr={best_f1["threshold"]:.2f}  F1={best_f1["f1"]:.4f}  '
          f'P={best_f1["precision"]:.4f} R={best_f1["recall"]:.4f}  '
          f'真实误检率={best_f1["false_positive_rate"]*100:.2f}%')
    if ok20:
        b20 = max(ok20, key=lambda r: r['recall_NeuralTextures'])
        print(f'  误检<10% 约束下 NT Recall 最高: thr={b20["threshold"]:.2f}  '
              f'NT召回={b20["recall_NeuralTextures"]*100:.1f}%  '
              f'F1={b20["f1"]:.4f}  真实误检率={b20["false_positive_rate"]*100:.2f}%')

    # ---- 保存 ----
    with open(f'{args.out_prefix}_stats.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pr_rows[0].keys())); w.writeheader(); w.writerows(pr_rows)
    with open(f'{args.out_prefix}_dist.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dist_rows[0].keys())); w.writeheader(); w.writerows(dist_rows)
    with open(f'{args.out_prefix}_reliability.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rel_rows[0].keys())); w.writeheader(); w.writerows(rel_rows)
    with open(f'{args.out_prefix}_per_image.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys())); w.writeheader(); w.writerows(per_image)
    print(f'\n已写入: {args.out_prefix}_{{stats,dist,reliability,per_image}}.csv')

    # ---- 绘图 ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        ax = axes[0][0]
        for m in METHODS:
            v = np.array([x['max_conf'] for x in fr if x['method'] == m])
            ax.hist(v, bins=np.arange(0, 1.02, 0.05), alpha=0.5, label=m, density=True)
        ax.set_title('Per-image max confidence distribution (fake)')
        ax.set_xlabel('max confidence'); ax.set_ylabel('density'); ax.legend(fontsize=8)

        ax = axes[0][1]
        for r in rel_rows:
            ax.bar(r['method'], [r['mean_conf']], alpha=0.6, label='mean conf')
        ax.set_title('Mean confidence vs actual precision (per-prediction)')
        x = np.arange(len(rel_rows))
        ax.bar(x + 0.35, [r['acc'] for r in rel_rows], width=0.35, alpha=0.6,
               color='orange', label='actual precision')
        ax.set_xticks(x + 0.175); ax.set_xticklabels([r['method'][:9] for r in rel_rows],
                                                     rotation=20, fontsize=8)
        ax.legend(fontsize=8)

        ax = axes[1][0]
        ax.plot([r['recall'] for r in pr_rows], [r['precision'] for r in pr_rows], 'k-', lw=2)
        ax.set_title('Image-level PR curve'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.grid(alpha=0.3)

        ax = axes[1][1]
        for m in METHODS:
            ax.plot([r['recall_'+m] for r in pr_rows],
                    [r['precision'] for r in pr_rows], label=m, lw=1.5)
        ax.set_title('PR curve (overall precision, per-method recall)')
        ax.set_xlabel('method recall'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{args.out_prefix}_plots.jpg', dpi=95,
                    pil_kwargs={'quality': 88})
        print(f'图已保存 {args.out_prefix}_plots.jpg')
    except Exception as e:
        print(f'绘图失败(不影响数值结果): {e}')


if __name__ == '__main__':
    main()
