#!/usr/bin/env python
"""真实帧对照实验 —— 模型是"知道人脸被伪造"，还是"只知道哪里有人脸"?

背景
----
Step 2 的判决是: NT 的分割表征不是瓶颈(失败样本的像素 AUROC 也有 0.98), 瓶颈在实例
置信度层级。但像素 AUROC 高本身有歧义 —— 本任务的 GT 是整张人脸区域, 所以它相当程度上
衡量的是"能不能找到脸", 这个能力四种方法都有。

本实验用**配对真实帧**把这两件事分开:
  对每一对 (Real_xxx_fNNNNN, NeuralTextures_xxx_yyy_fNNNNN) —— 同一源视频、同一帧号、
  同一张脸的未伪造版本 —— 把伪造样本的 GT 人脸掩码**原样**套到真实帧上, 比较同一个模型
  对"真实人脸区域"和"伪造人脸区域"的:
    A. 空间响应   : 像素 AUROC / AUPRC / 面区平均软响应 / top-1 框与参考区 IoU
    B. 实例置信度 : NMS 前 top-1 候选分数, 以及配对差 Δs = s_伪 − s_真
                  外加基于实例分数的 image-level AUROC(直接把 Real 当负类、伪造当正类)

判读
----
  空间响应一致 + Δs 明显为正   -> 有真假判别信号, 只是工作点不佳 -> 做分数校准/决策
  空间响应一致 + Δs 高度重叠   -> 模型主要在用人脸先验        -> 做真假判别表征
  NT 空间响应强于 Real 但 Δs 弱 -> 分割响应没转化成实例分数   -> 做分支间信息交互
  NT 空间响应本身不稳           -> 空间表征有缺陷            -> 做局部表征增强

三条纪律(按合作者要求)
--------------------
  1. 全部指标**无阈值**, 真实帧不参与任何阈值选择。
  2. 配对坐标先**实测**再使用 —— 用面区之外的背景做相位相关, 逐对报告位移, 位移非零的
     样本单列, 不与正常样本混算。
  3. 实例分数一律取 **NMS 前 top-1**(`det[4].max()`, det[4] 已是 sigmoid 后的置信度),
     不用 conf=0.01 截断, 否则会丢掉 A_no_pred 那批最有价值的失败样本。
     raw head 分数与 NMS 后实例置信度**分列报告**, 不混为一谈。

用法
----
  python -I evaluate/real_fake_control.py --data data/dataset_v1 --split val --frame-tol 0
"""
import argparse
import csv
import os
import sys
import zlib
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pixel_auroc as P                                       # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['NeuralTextures', 'Face2Face', 'Deepfakes', 'FaceSwap']


def alignment(real, fake, mask):
    """用**面区之外**的背景做相位相关, 返回 (dx, dy, 峰值响应, 背景相关系数)。

    人脸重演只改面区, 背景应当逐像素一致; 若真实帧与伪造帧存在几何变换, 这里会露出来。
    """
    g1 = cv2.cvtColor(real, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g2 = cv2.cvtColor(fake, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = (mask == 0).astype(np.float32)
    a, b = g1 * bg, g2 * bg
    try:
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
    except cv2.error:
        return np.nan, np.nan, np.nan, np.nan
    sel = bg > 0
    if sel.sum() < 100:
        return dx, dy, resp, np.nan
    av, bv = g1[sel], g2[sel]
    corr = float(np.corrcoef(av, bv)[0, 1]) if av.std() > 0 and bv.std() > 0 else np.nan
    return float(dx), float(dy), float(resp), corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--frame-tol', type=int, default=0)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-per-method', type=int, default=0)
    ap.add_argument('--max-neg', type=int, default=100000)
    ap.add_argument('--out', default='evaluate/real_fake_control_val.csv')
    args = ap.parse_args()

    if 'fft_v1' in args.weights and 'dataset_v2' not in args.data:
        sys.exit('权重是 FFT 模型但数据不是 dataset_v2: 分布错配, 拒绝运行')
    if 'cloud_baseline_v1' in args.weights and 'dataset_v2' in args.data:
        sys.exit('权重是 baseline 但数据是 dataset_v2: 分布错配, 拒绝运行')

    ops = P._install_patch()

    want = None if 'all' in args.split.split(',') else set(args.split.split(','))
    split_of = {}

    def ipath(stem):
        return os.path.join(args.data, 'images', split_of[stem], stem + '.jpg')

    def lpath(stem):
        return os.path.join(args.data, 'labels', split_of[stem], stem + '.txt')

    real_at, fake_at = defaultdict(dict), {}
    with open(os.path.join(args.data, 'metadata.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if want is not None and r['split'] not in want:
                continue
            stem = os.path.basename(r['image'])[:-4]
            split_of[stem] = r['split']
            fr = int(r['frame'])
            if r['is_fake'] == '0':
                real_at[r['video_id']][fr] = stem
            else:
                fake_at[(r['method'], r['video_id'].split('_')[0], fr)] = stem

    # ---- 建配对(逐个方法, 真实帧可能被多个方法共用, 去重) ----
    pairs = []
    for m in METHODS:
        got = []
        for (mm, src, fr), fstem in fake_at.items():
            if mm != m:
                continue
            rf = real_at.get(src)
            if not rf:
                continue
            if fr in rf:
                got.append((src, fr, fr, 0, fstem, rf[fr]))
            elif args.frame_tol:
                near = min(rf, key=lambda x: abs(x - fr))
                if abs(near - fr) <= args.frame_tol:
                    got.append((src, fr, near, abs(near - fr), fstem, rf[near]))
        got.sort()
        if args.max_per_method:
            got = got[:args.max_per_method]
        pairs += [(m,) + g for g in got]

    need = sorted({p[6] for p in pairs} | {p[5] for p in pairs})   # (…, fstem, rstem)
    print(f'权重 {args.weights}')
    print(f'数据 {args.data}  划分 {args.split}  帧容差 {args.frame_tol}')
    print(f'配对 {len(pairs)} 对, 需要推理 {len(need)} 张图\n')

    from ultralytics import YOLO
    model = YOLO(args.weights)
    P._install_head_hook(model)

    # ---- 一次批量推理, 取每张图的: 未截断的 top-1 实例分数 + 软掩码 ----
    info = {}
    for i in range(0, len(need), args.batch):
        chunk = need[i:i + args.batch]
        paths = [ipath(s) for s in chunk]
        P._CAP.clear()
        P._CAP['by_path'] = {}
        model.predict(paths, imgsz=args.imgsz, conf=0.001, device='0',
                      verbose=False, max_det=100)
        ho = P._CAP.get('head')
        lh, lw = P._CAP.get('shape', (args.imgsz, args.imgsz))
        by_stem = {os.path.basename(str(k))[:-4]: v
                   for k, v in P._CAP['by_path'].items() if k is not None}
        for j, stem in enumerate(chunk):
            img = cv2.imread(ipath(stem))
            oh, ow = img.shape[:2]
            det, protos = ho[0][0][j], ho[0][1][j]
            k = int(det[4].argmax())
            conf = float(det[4, k])                       # det[4] 已是 sigmoid 后的值
            coeffs = det[5:, k].unsqueeze(0)
            x, y, w, h = det[:4, k].tolist()
            box = torch.tensor([[x - w / 2, y - h / 2, x + w / 2, y + h / 2]],
                               device=det.device, dtype=det.dtype)
            soft = P.to_orig(P._soft_from_coeffs(coeffs, protos, box, lh, lw, False),
                             lh, lw, oh, ow)[0].cpu().numpy().astype(np.float64)
            nms = by_stem.get(stem)
            info[stem] = dict(soft=soft, conf=conf,
                              box_xyxy_lb=box[0].tolist(), lh=lh, lw=lw,
                              n_det=0 if not nms else int(nms['crop'].shape[0]))
        print(f'\r  推理 {min(i+args.batch, len(need))}/{len(need)}', end='', flush=True)
    print('\n')

    # ---- 配对比较 ----
    rows = []
    for (m, src, fr, rfr, off, fstem, rstem) in pairs:
        ri = cv2.imread(ipath(rstem))
        fi = cv2.imread(ipath(fstem))
        if ri is None or fi is None or ri.shape != fi.shape:
            continue
        oh, ow = fi.shape[:2]
        mask = P.rasterize_label(lpath(fstem), ow, oh)
        if mask.sum() == 0:
            continue
        y = mask.reshape(-1).astype(bool)
        sel = mask.astype(bool)
        dx, dy, resp, corr = alignment(ri, fi, mask)

        # 参考区域 = 伪造样本的 GT 人脸掩码, 原样套到配对真实帧上
        sr, sf = info[rstem]['soft'], info[fstem]['soft']
        yr, yr2, _ = P.subsample(y, sr.reshape(-1), args.max_neg,
                                 zlib.crc32((rstem + 'r').encode()))
        yf, yf2, _ = P.subsample(y, sf.reshape(-1), args.max_neg,
                                 zlib.crc32((fstem + 'f').encode()))
        ar, pr = P.metrics(yr, yr2)
        af, pf = P.metrics(yf, yf2)

        # top-1 检测框 与 参考区域 的 IoU(框与 GT 掩码外接矩形比, 衡量"框到没到那儿")
        def box_iou(stem):
            bx = info[stem]['box_xyxy_lb']
            lh_, lw_ = info[stem]['lh'], info[stem]['lw']
            gx, gy = ow / lw_, oh / lh_                   # letterbox -> 原图
            x1, y1, x2, y2 = bx[0] * gx, bx[1] * gy, bx[2] * gx, bx[3] * gy
            ys, xs = np.where(sel)
            if xs.size == 0:
                return np.nan
            g = [xs.min(), ys.min(), xs.max(), ys.max()]
            ix1, iy1 = max(x1, g[0]), max(y1, g[1])
            ix2, iy2 = min(x2, g[2]), min(y2, g[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = ((x2 - x1) * (y2 - y1) + (g[2] - g[0]) * (g[3] - g[1]) - inter)
            return inter / ua if ua > 0 else np.nan

        rows.append(dict(
            method=m, src=src, frame=fr, frame_off=off, real_stem=rstem, fake_stem=fstem,
            gt_px=int(mask.sum()), gt_frac=float(mask.mean()),
            auroc_real=ar, auprc_real=pr, auroc_fake=af, auprc_fake=pf,
            resp_real=float(sr[sel].mean()), resp_fake=float(sf[sel].mean()),
            conf_real=info[rstem]['conf'], conf_fake=info[fstem]['conf'],
            dconf=info[fstem]['conf'] - info[rstem]['conf'],
            iou_box_real=box_iou(rstem), iou_box_fake=box_iou(fstem),
            n_det_real=info[rstem]['n_det'], n_det_fake=info[fstem]['n_det'],
            shift_dx=dx, shift_dy=dy, shift_resp=resp, bg_corr=corr))

    if not rows:
        sys.exit('没有配对成功')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- 报告
    def st(v):
        v = np.asarray([x for x in v if x == x], float)
        return (np.nan, np.nan, np.nan, 0) if v.size == 0 else \
            (float(v.mean()), float(np.median(v)), float(v.std(ddof=1)), v.size)

    print('=' * 116)
    print('0. 几何对齐实测(面区之外背景的相位相关) —— 必须近似 (0,0) 才能把 GT 掩码直接套用')
    print('=' * 116)
    dx, dy = np.abs([r['shift_dx'] for r in rows]), np.abs([r['shift_dy'] for r in rows])
    cc = st([r['bg_corr'] for r in rows])
    print(f'  |dx| 中位 {np.median(dx):.2f}px  p95 {np.percentile(dx,95):.2f}px  最大 {dx.max():.2f}px')
    print(f'  |dy| 中位 {np.median(dy):.2f}px  p95 {np.percentile(dy,95):.2f}px  最大 {dy.max():.2f}px')
    print(f'  背景相关系数 均值 {cc[0]:.4f} 中位 {cc[1]:.4f}')
    bad = [r for r in rows if max(abs(r['shift_dx']), abs(r['shift_dy'])) > 1.0]
    print(f'  位移 >1px 的配对: {len(bad)} / {len(rows)}')

    print()
    print('=' * 116)
    print('A. 空间响应对照(参考区域 = 该配对伪造样本的 GT 人脸掩码, 原样套到真实帧上)')
    print('=' * 116)
    print(f'{"method":<16}{"n":>5}{"AUROC_real":>12}{"AUROC_fake":>12}{"ΔAUROC":>9}'
          f'{"AUPRC_real":>12}{"AUPRC_fake":>12}{"ΔAUPRC":>9}')
    for m in METHODS:
        rs = [r for r in rows if r['method'] == m]
        if not rs:
            continue
        a = st([r['auroc_real'] for r in rs]); b = st([r['auroc_fake'] for r in rs])
        pa = st([r['auprc_real'] for r in rs]); pb = st([r['auprc_fake'] for r in rs])
        print(f'{m:<16}{len(rs):>5}{a[0]:>12.4f}{b[0]:>12.4f}{b[0]-a[0]:>+9.4f}'
              f'{pa[0]:>12.4f}{pb[0]:>12.4f}{pb[0]-pa[0]:>+9.4f}')

    print()
    print(f'{"method":<16}{"n":>5}{"面区软响应_real":>16}{"面区软响应_fake":>16}{"比值":>9}'
          f'{"框IoU_real":>12}{"框IoU_fake":>12}')
    for m in METHODS:
        rs = [r for r in rows if r['method'] == m]
        if not rs:
            continue
        rr = st([r['resp_real'] for r in rs]); rf = st([r['resp_fake'] for r in rs])
        br = st([r['iou_box_real'] for r in rs]); bf = st([r['iou_box_fake'] for r in rs])
        print(f'{m:<16}{len(rs):>5}{rr[0]:>16.3f}{rf[0]:>16.3f}'
              f'{(rf[0]/rr[0] if rr[0] else float("nan")):>9.3f}{br[1]:>12.4f}{bf[1]:>12.4f}')

    print()
    print('=' * 116)
    print('B. 实例置信度对照(NMS 前 top-1, 未截断; 以及基于实例分数的 image-level AUROC)')
    print('=' * 116)
    print(f'{"method":<16}{"n":>5}{"conf_real中位":>14}{"conf_fake中位":>14}{"Δs中位":>10}'
          f'{"Δs>0占比":>10}{"image-level AUROC":>18}')
    for m in METHODS:
        rs = [r for r in rows if r['method'] == m]
        if not rs:
            continue
        cr = st([r['conf_real'] for r in rs]); cf = st([r['conf_fake'] for r in rs])
        d = np.array([r['dconf'] for r in rs], float)
        s = np.concatenate([[r['conf_real'] for r in rs], [r['conf_fake'] for r in rs]])
        lab = np.concatenate([np.zeros(len(rs)), np.ones(len(rs))]).astype(bool)
        il = P.metrics(lab, s)[0]
        print(f'{m:<16}{len(rs):>5}{cr[1]:>14.4f}{cf[1]:>14.4f}{np.median(d):>+10.4f}'
              f'{(d>0).mean()*100:>9.1f}%{il:>18.4f}')

    print(f'\n逐对结果 {args.out}  ({len(rows)} 对)')


if __name__ == '__main__':
    main()
