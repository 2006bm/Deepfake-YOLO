#!/usr/bin/env python
"""Step 2: Pixel-level AUROC / AUPRC 诊断 —— 无阈值的像素排序能力。

回答的判决性问题
----------------
NT 的失败是"置信度/决策阈值"问题, 还是"分割表征本身可分性不足"?
前三项诊断(阈值扫描 / 校准 / A-B-C 分组)都在**某个工作点下**测量, 只有像素级
AUROC/AUPRC 是**无阈值**的排序度量, 能直接回答这个问题。

为什么不能直接用 ultralytics 的输出
-----------------------------------
8.4.160 的流程(源码核对, 非假设):
  ops.py:530   masks = masks_in @ protos        <- **没有 sigmoid**
  ops.py:534   interpolate 到 letterbox 尺寸
  ops.py:541   masks.gt_(0.0).byte()            <- 二值化阈值是 **0**, 且在裁剪之前
  segment/predict.py:105  传入的 pred 已经是 **NMS 之后**
  实测: Results.masks.data 存的是 letterbox 分辨率(384x640), 不是原图

所以 `sigmoid(coeffs @ proto)` 与 `mask_threshold=0` 这类写法都是错的。
正确做法是 monkeypatch `ops.process_mask`, 在它二值化**之前**截下浮点响应,
这样得到的软掩码与线上二值掩码严格同源、同坐标系。

三种对照(区分三种失败机制)
--------------------------
  raw     : 头内部 top-1 anchor 的浮点响应, **不裁剪** -> 分割头的空间排序能力
  cropped : 同一 anchor, 裁剪到其解码框         -> 加检测框约束后的排序能力
  nms     : NMS 后实际进入 process_mask 的实例    -> 线上真实候选选择

用法
----
  # 冒烟验证(每方法 10 张, 检查尺寸/数值域/坐标还原/与线上二值掩码的一致性)
  python -I evaluate/pixel_auroc.py --mode smoke --per-method 10

  # 全量
  python -I evaluate/pixel_auroc.py --mode full --out-prefix evaluate/pixel_auroc_val
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
import torch.nn.functional as F

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
_CAP = {}
_HOOKED = False


# ---------------------------------------------------------------- mask 重建
def _install_patch():
    """monkeypatch 两处, 在二值化之前截下浮点响应。

    逐图调用 process_mask 的 Python 开销约 3.8 s/图(实测), 批量推理是唯一可行的
    加速方式。批量下 process_mask 仍被逐图调用, 所以用 construct_result 补丁记录
    "当前是哪张图", 把结果按路径归位 —— 哪张图没有检测到就自然缺席, 不会错位。
    """
    from ultralytics.utils import ops
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    if getattr(ops.process_mask, '_patched', False):
        return ops
    orig = ops.process_mask
    orig_cr = SegmentationPredictor.construct_result

    def patched(protos, masks_in, bboxes, shape, upsample=False):
        c, mh, mw = protos.shape
        _CAP['shape'] = tuple(shape)
        cur = _CAP.get('cur')
        if masks_in.shape[0] == 0:
            _CAP.setdefault('by_path', {})[cur] = None
            return orig(protos, masks_in, bboxes, shape, upsample)
        raw = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)   # 无 sigmoid
        if upsample:
            raw = F.interpolate(raw[None], shape, mode='bilinear')[0]
        else:
            wr, hr = mw / shape[1], mh / shape[0]
            bboxes = bboxes * torch.tensor([[wr, hr, wr, hr]], device=bboxes.device)
        _CAP.setdefault('by_path', {})[cur] = dict(
            up=raw.detach(), crop=ops.crop_mask(raw.clone(), bboxes).detach())
        return orig(protos, masks_in, bboxes, shape, upsample)

    def patched_cr(self, pred, img, orig_img, img_path, proto):
        _CAP['cur'] = img_path
        return orig_cr(self, pred, img, orig_img, img_path, proto)

    patched._patched = True
    ops.process_mask = patched
    SegmentationPredictor.construct_result = patched_cr
    return ops


def _install_head_hook(model):
    """挂住 Segment 头, 取 NMS 之前的原始预测张量与原型。"""
    global _HOOKED
    if _HOOKED:
        return
    model.model.model[-1].register_forward_hook(
        lambda mod, inp, out: _CAP.update(head=out))
    _HOOKED = True


def _top1_pre_nms(head_out, b):
    """从头部输出取 NMS 前 top-1 anchor 的 (coeffs, box_xyxy)。

    注意 det[4] 已经是 **sigmoid 之后的类别置信度**(实测与 sigmoid(dict['scores'])
    逐位相同), 不要再套一层 sigmoid —— 虽然单调变换不改变 argmax, 但会让报告出来的
    置信度值失真。
    """
    det, protos = head_out[0][0][b], head_out[0][1][b]      # (37,5040), (32,96,160)
    k = int(det[4].argmax())
    conf = float(det[4, k])
    coeffs = det[5:, k].unsqueeze(0)                         # (1,32)
    x, y, w, h = det[:4, k].tolist()
    box = torch.tensor([[x - w / 2, y - h / 2, x + w / 2, y + h / 2]],
                       device=det.device, dtype=det.dtype)
    return coeffs, box, conf, protos


def _soft_from_coeffs(coeffs, protos, box, lh, lw, crop):
    c, mh, mw = protos.shape
    raw = (coeffs @ protos.float().view(c, -1)).view(1, mh, mw)
    raw = F.interpolate(raw[None], (lh, lw), mode='bilinear')[0]   # (1,lh,lw)
    if crop:
        from ultralytics.utils import ops
        raw = ops.crop_mask(raw.clone(), box)
    return raw


def to_orig(m, lh, lw, oh, ow):
    """letterbox 分辨率 -> 原图分辨率(与 scale_masks 同一套 ratio/pad 公式)。"""
    from ultralytics.utils import ops
    if m is None:
        return None
    return ops.scale_masks(m[None].float(), (oh, ow), ratio_pad=None,
                           padding=True)[0]


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
            cv2.fillPoly(c, [np.round(np.stack([xy[:, 0] * w,
                                                xy[:, 1] * h], 1)).astype(np.int32)], 1)
    return c


# ---------------------------------------------------------------- 指标
def metrics(y, s):
    """一次排序同时算出 AUROC 与 AUPRC。

    AUROC 用并列感知的平均秩(Mann-Whitney U); AUPRC 用"每个正类的精度取平均"
    (即阶梯式 PR 曲线下面积)。两者对单调变换不变, 所以用原始 logits 值域即可,
    不需要 sigmoid。
    """
    npos = int(y.sum())
    if npos == 0:
        return np.nan, np.nan
    n = s.size
    order = np.argsort(s, kind='stable')            # 升序, 只排一次
    ss = s[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1))
    ends = np.concatenate((starts[1:], [n]))
    ranks = np.empty(n, np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    nneg = n - npos
    auroc = ((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
             if nneg else np.nan)
    ys = y[order][::-1].astype(np.float64)          # 降序
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    auprc = float((tp / (tp + fp))[ys > 0].sum() / npos)
    return auroc, auprc


def subsample(y, s, max_neg, seed):
    """为了速度, 对**背景像素**做无偏随机抽样(AUROC/AUPRC 是排序度量, 抽样
    只影响精度不影响期望; 正类全部保留)。固定种子保证可复现。"""
    npos = int(y.sum())
    nneg = y.size - npos
    if max_neg <= 0 or nneg <= max_neg:
        return y, s, 1.0
    rng = np.random.default_rng(seed)
    neg = np.flatnonzero(~y)
    # 用伯努利掩码而不是 rng.choice(replace=False) —— 后者对 90 万元素要几秒,
    # 这里是 O(n) 向量化且不产生重复样本
    sel = neg[rng.random(neg.size) < (max_neg / neg.size)]
    keep = np.concatenate((np.flatnonzero(y), sel))
    keep.sort()
    return y[keep], s[keep], sel.size / neg.size


# micro 口径: 把所有图的像素合并计算。逐图保存百万级分数会吃掉十几 GB 内存,
# 所以用分数直方图(8192 桶, 区间 ±128, 观测值域约 [-20, +30])累积 —— 桶宽 0.031,
# 量化误差远小于图间差异。桶内按并列处理。
MICRO_BINS = 8192
MICRO_RANGE = 128.0


def micro_accumulate(pos_hist, neg_hist, y, s):
    idx = np.clip(((s + MICRO_RANGE) * (MICRO_BINS / (2 * MICRO_RANGE))).astype(np.int32),
                  0, MICRO_BINS - 1)
    pos_hist += np.bincount(idx[y], minlength=MICRO_BINS)
    neg_hist += np.bincount(idx[~y], minlength=MICRO_BINS)


def micro_metrics(pos_hist, neg_hist):
    npos, nneg = int(pos_hist.sum()), int(neg_hist.sum())
    if npos == 0 or nneg == 0:
        return np.nan, np.nan
    cnt = pos_hist + neg_hist
    cum_before = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    sum_ranks = float((pos_hist * (cum_before + (cnt + 1) / 2.0)).sum())
    auroc = (sum_ranks - npos * (npos + 1) / 2.0) / (npos * nneg)
    p, n = pos_hist[::-1], neg_hist[::-1]           # 降序扫桶
    tp, fp = np.cumsum(p), np.cumsum(n)
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    return auroc, float((prec * p).sum() / npos)


def cluster_ci(vals, clusters, n_boot=1000, seed=0):
    """按源视频聚类的 bootstrap 置信区间(帧之间不独立)。"""
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    v, cl = v[ok], np.asarray(clusters)[ok]
    if v.size == 0:
        return np.nan, np.nan
    uniq = np.unique(cl)
    idx = {u: np.flatnonzero(cl == u) for u in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        sel = np.concatenate([idx[u] for u in pick])
        out.append(v[sel].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='smoke', choices=['smoke', 'full'])
    ap.add_argument('--weights', default='runs/segment/cloud_baseline_v1/weights/best.pt')
    ap.add_argument('--data', default='data/dataset_v1')
    ap.add_argument('--split', default='val')
    ap.add_argument('--imgsz', type=int, default=640)
    # 必须批量推理: 逐图调用 model.predict 的 Python 开销达 3.8 s/图(实测),
    # 批量 16 可降到约 0.15 s/图。捕获用 construct_result 补丁按路径归位,
    # 不会因为某些图没有检测而错位。
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--probe-conf', type=float, default=0.01)
    ap.add_argument('--per-method', type=int, default=0, help='每种方法最多取多少张(0=全部)')
    ap.add_argument('--out-prefix', default='evaluate/pixel_auroc_val')
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--max-neg', type=int, default=100000,
                    help='macro 指标里每图保留的背景像素上限(0=全部, 全量约 0.92M/图, 很慢)')
    args = ap.parse_args()

    # 训练分布与评估分布必须配对 —— 这是本项目踩过的坑, 这里显式断言
    if 'fft_v1' in args.weights and 'dataset_v2' not in args.data:
        sys.exit('权重是 FFT 模型但数据不是 dataset_v2: 训练/评估分布错配, 拒绝运行')
    if 'cloud_baseline_v1' in args.weights and 'dataset_v2' in args.data:
        sys.exit('权重是 baseline 但数据是 dataset_v2: 训练/评估分布错配, 拒绝运行')

    ops = _install_patch()

    img_dir = os.path.join(args.data, 'images', args.split)
    lab_dir = os.path.join(args.data, 'labels', args.split)
    meta = os.path.join(args.data, 'metadata.csv')
    src_of = {}
    with open(meta, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['split'] == args.split and r['is_fake'] == '1':
                src_of[os.path.basename(r['image'])[:-4]] = r['video_id'].split('_')[0]

    stems = sorted(s[:-4] for s in os.listdir(lab_dir)
                   if s.endswith('.txt') and not s.startswith('Real_'))
    if args.per_method:
        by = defaultdict(list)
        for s in stems:
            by[s.split('_')[0]].append(s)
        stems = [s for m in METHODS for s in by[m][:args.per_method]]

    print(f'权重 {args.weights}')
    print(f'数据 {args.data}   划分 {args.split}   样本 {len(stems)}')
    print(f'模式 {args.mode}   探测阈值 conf={args.probe_conf}\n')

    from ultralytics import YOLO
    model = YOLO(args.weights)
    _install_head_hook(model)

    per_image = []
    smoke_log = []
    micro = {m: {t: [np.zeros(MICRO_BINS, np.int64), np.zeros(MICRO_BINS, np.int64)]
                 for t in ('raw', 'crop', 'nms')} for m in METHODS}

    # 增量落盘: 中途被打断也保住已经算完的样本
    FIELDS = ['stem', 'method', 'src', 'gt_px', 'n_px', 'gt_frac', 'top1_conf', 'n_det',
              'auroc_raw', 'auprc_raw', 'auroc_crop', 'auprc_crop', 'auroc_nms', 'auprc_nms']
    per_img_fh = per_img_writer = None
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)
    if args.mode == 'full':
        per_img_fh = open(f'{args.out_prefix}_per_image.csv', 'w', newline='', encoding='utf-8')
        per_img_writer = csv.DictWriter(per_img_fh, fieldnames=FIELDS)
        per_img_writer.writeheader()
    for i in range(0, len(stems), args.batch):
        chunk = stems[i:i + args.batch]
        paths = [os.path.join(img_dir, s + '.jpg') for s in chunk]
        _CAP.clear()
        _CAP['by_path'] = {}
        res = model.predict(paths, imgsz=args.imgsz, conf=args.probe_conf,
                            device='0', verbose=False, max_det=100)
        head_out = _CAP.get('head')
        lh, lw = _CAP.get('shape', (args.imgsz, args.imgsz))
        by_stem = {os.path.basename(str(k))[:-4]: v
                   for k, v in _CAP['by_path'].items() if k is not None}
        for j, (stem, r) in enumerate(zip(chunk, res)):
            oh, ow = r.orig_shape
            gt = rasterize_label(os.path.join(lab_dir, stem + '.txt'), ow, oh)
            y = gt.reshape(-1).astype(bool)
            cap = by_stem.get(stem)

            # --- V3: 线上路径。NMS 输出按置信度降序, 取 [:1] 即"NMS 后最高分实例" ---
            v_nms = None
            if cap and cap['crop'].shape[0]:
                v_nms = to_orig(cap['crop'][:1], lh, lw, oh, ow)

            # --- V1/V2: NMS 前 top-1 anchor ---
            v_raw = v_crop = None
            top1_conf = np.nan
            if head_out is not None and j < head_out[0][0].shape[0]:
                coeffs, box, top1_conf, protos = _top1_pre_nms(head_out, j)
                v_raw = to_orig(_soft_from_coeffs(coeffs, protos, box, lh, lw, False),
                                lh, lw, oh, ow)
                v_crop = to_orig(_soft_from_coeffs(coeffs, protos, box, lh, lw, True),
                                 lh, lw, oh, ow)

            if args.mode == 'smoke':
                b = (r.masks.data[0].cpu().numpy().astype(bool)
                     if r.masks is not None and r.masks.data.shape[0] else None)
                re_bin = (cap['crop'][0].cpu().numpy() > 0
                          if cap and cap['crop'].shape[0] else None)
                mism = float((b != re_bin).mean()) if (b is not None and re_bin is not None) else -1.0
                # 还原到原图后, 与 r.masks.xy 多边形光栅化对比
                iou_ok = -1.0
                if v_nms is not None and r.masks is not None and r.masks.xy is not None:
                    ref = np.zeros((oh, ow), np.uint8)
                    for poly in r.masks.xy:
                        if poly is not None and len(poly) >= 3:
                            cv2.fillPoly(ref, [np.round(poly).astype(np.int32)], 1)
                    mine = (v_nms[0].cpu().numpy() > 0).astype(np.uint8)
                    u = int((ref | mine).sum())
                    iou_ok = float((ref & mine).sum() / u) if u else 0.0
                smoke_log.append((stem, lh, lw, oh, ow,
                                  None if v_nms is None else float(v_nms.min()),
                                  None if v_nms is None else float(v_nms.max()),
                                  float(top1_conf), mism, iou_ok))

            row = dict(stem=stem, method=stem.split('_')[0], src=src_of.get(stem, '?'),
                       gt_px=int(gt.sum()), n_px=int(gt.size), gt_frac=float(gt.mean()),
                       top1_conf=float(top1_conf),
                       n_det=int(r.masks.data.shape[0]) if r.masks is not None else 0)
            for tag, v in (('raw', v_raw), ('crop', v_crop), ('nms', v_nms)):
                if v is None:
                    row[f'auroc_{tag}'] = row[f'auprc_{tag}'] = np.nan
                else:
                    s = v[0].cpu().numpy().astype(np.float64).reshape(-1)
                    # micro 用全部像素(直方图累积很便宜), macro 用抽样后的(排序很贵)
                    micro_accumulate(micro[row['method']][tag][0],
                                     micro[row['method']][tag][1], y, s)
                    yy, ss_, _ = subsample(y, s, args.max_neg, zlib.crc32(stem.encode()))
                    row[f'auroc_{tag}'], row[f'auprc_{tag}'] = metrics(yy, ss_)
            if per_img_writer is not None:
                per_img_writer.writerow(row)
            per_image.append(row)
        print(f'\r  推理 {min(i+args.batch, len(stems))}/{len(stems)}', end='', flush=True)
    print('\n')

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    if args.mode == 'smoke':
        print('=' * 118)
        print('冒烟验证: 尺寸 / 数值域 / 坐标还原 / 与线上二值掩码的一致性')
        print('=' * 118)
        print(f'{"stem":<38}{"letterbox":>12}{"原图":>12}{"V3值域":>18}{"top1conf":>10}'
              f'{"二值≠线上":>11}{"还原IoU":>9}')
        for (stem, lh, lw, oh, ow, lo, hi, tc, mism, acc) in smoke_log:
            rng = f'[{lo:.2f},{hi:.2f}]' if lo is not None else 'None'
            print(f'{stem[:37]:<38}{f"{lh}x{lw}":>12}{f"{oh}x{ow}":>12}{rng:>18}'
                  f'{tc:>10.3f}{mism*100:>11.2f}%{acc*100:>9.1f}%')
        bad = [s for s in smoke_log if 0 <= s[8] and s[8] > 1e-6]
        print(f'\n  V3 二值化后与线上 masks.data 不一致的样本: {len(bad)} / {len(smoke_log)}')
        print('  (应为 0 —— 非 0 说明浮点重建路径与线上不等价)')
        print('\n  逐图指标:')
        print(f'  {"stem":<38}{"AUROC_raw":>11}{"AUROC_crop":>12}{"AUROC_nms":>11}'
              f'{"AUPRC_raw":>11}{"AUPRC_nms":>11}')
        for r in per_image[:24]:
            print(f'  {r["stem"][:37]:<38}{r["auroc_raw"]:>11.4f}{r["auroc_crop"]:>12.4f}'
                  f'{r["auroc_nms"]:>11.4f}{r["auprc_raw"]:>11.4f}{r["auprc_nms"]:>11.4f}')
        return

    # ---------------------------- 全量汇总 ----------------------------
    if per_img_fh is not None:
        per_img_fh.close()

    print('=' * 122)
    print('分方法 Pixel AUROC / AUPRC')
    print('  macro = 逐图算再平均(每图等权, 95%CI 按源视频聚类 bootstrap)')
    print('  micro = 合并全部像素(像素等权, 直方图累积)')
    print('=' * 122)
    print(f'{"method":<16}{"n":>5}{"GT占比":>8}{"变体":>7}{"AUROC_macro":>13}{"95%CI":>17}'
          f'{"AUROC_micro":>13}{"AUPRC_macro":>13}{"95%CI":>17}{"AUPRC_micro":>12}{"AUPRC基线":>11}')
    summary = []
    for m in METHODS:
        rows = [r for r in per_image if r['method'] == m]
        if not rows:
            continue
        gf = float(np.mean([r['gt_frac'] for r in rows]))
        gf_micro = sum(r['gt_px'] for r in rows) / max(sum(r['n_px'] for r in rows), 1)
        for tag in ('raw', 'crop', 'nms'):
            a = np.array([r[f'auroc_{tag}'] for r in rows], float)
            p = np.array([r[f'auprc_{tag}'] for r in rows], float)
            cl = [r['src'] for r in rows]
            alo, ahi = cluster_ci(a, cl, args.n_boot)
            plo, phi = cluster_ci(p, cl, args.n_boot)
            am, pm = micro_metrics(*micro[m][tag])
            print(f'{m:<16}{len(rows):>5}{gf:>8.3f}{tag:>7}{np.nanmean(a):>13.4f}'
                  f'{f"[{alo:.3f},{ahi:.3f}]":>17}{am:>13.4f}{np.nanmean(p):>13.4f}'
                  f'{f"[{plo:.3f},{phi:.3f}]":>17}{pm:>12.4f}{gf_micro:>11.3f}')
            summary.append(dict(method=m, variant=tag, n=len(rows),
                                gt_frac_macro=gf, gt_frac_micro=gf_micro,
                                auroc_macro=float(np.nanmean(a)), auroc_lo=alo, auroc_hi=ahi,
                                auroc_micro=am,
                                auprc_macro=float(np.nanmean(p)), auprc_lo=plo, auprc_hi=phi,
                                auprc_micro=pm, auprc_random_macro=gf,
                                auprc_random_micro=gf_micro,
                                n_nan_auroc=int(np.isnan(a).sum())))
        print()

    with open(f'{args.out_prefix}_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f'逐图结果 {args.out_prefix}_per_image.csv')
    print(f'汇总结果 {args.out_prefix}_summary.csv')


if __name__ == '__main__':
    main()
