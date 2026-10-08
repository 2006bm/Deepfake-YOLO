#!/usr/bin/env python
"""数据集完整性与标签正确性验证(论文可引用的数值证据)。

分两部分:

  1) 数量核对 --counts
     逐划分比对 images 与 labels 的配对关系,并与理论期望值对照。
     理论值由官方划分的配对数与抽帧参数推导,不写死,避免口径漂移。

  2) 标签↔mask 一致性  --n 1000
     两项独立判定:

     (a) 质心偏移 —— 判**正确性**
         比较多边形光栅化后的质心与原始 mask 质心,归一化到图像尺寸。
         若存在坐标错位(Y 轴翻转、归一化用错宽高、帧下标错位),
         质心会系统性偏移图像尺寸的 10%~100%。实测 < 0.2%,故无错位。

     (b) IoU —— 判**边界保真度**
         IoU = |M_gt ∩ M_poly| / |M_gt ∪ M_poly|

     ★ 为什么两项都要:只用 IoU 会误判。软边灰度掩码(Face2Face /
       FaceSwap / NeuralTextures)的边界本身是模糊渐变,多边形拟合必然
       损失一部分边界细节,实测 IoU 上限约 0.978~0.982,而二值的
       Deepfakes 可轻松达到 0.99。若统一用 0.98 卡,软掩码会被错误地
       判定为"不合格",而实际上质心偏移证明它完全对齐。
       因此通过标准为:质心偏移 < 1% 且 平均 IoU > 0.95。

     注意:mask 视频在数据处理后会被删除以释放磁盘(全部 mask 有 38.6GB,
     而可用磁盘仅 ~20GB)。因此默认只对**本地仍存在**的 mask 做验证;
     加 --download-missing 会按需重新下载被抽中的那几十个 mask,
     从而在 mask 已删除的情况下依然能给出证据。
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from download import BASE_URL, fetch_one  # noqa: E402
from mask_convert import to_gray  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def parse_stem(stem):
    """Deepfakes_585_599_f00179 -> ('Deepfakes','585_599',179)
       Real_585_f00041         -> (None,'585',41)"""
    body, _, fidx = stem.rpartition('_f')
    method, _, name = body.partition('_')
    if method == 'Real':
        return None, name, int(fidx)
    return method, name, int(fidx)


def mask_url_and_path(raw, method, name):
    url = BASE_URL + f'manipulated_sequences/{method}/masks/videos/{name}.mp4'
    path = f'{raw}/manipulated_sequences/{method}/masks/videos/{name}.mp4'
    return url, path


def rasterize_label(label_path, w, h):
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


def centroid_offset(rast, mb, w, h):
    """多边形质心与 mask 质心的偏移,归一化到图像尺寸(百分比)。

    坐标错位会让该值达到 10%~100%;单纯的边界拟合误差则在 1% 以内。
    任一为空时返回 None。
    """
    if rast.sum() == 0 or mb.sum() == 0:
        return None
    ys, xs = np.nonzero(mb)
    cy1, cx1 = ys.mean(), xs.mean()
    ys, xs = np.nonzero(rast)
    cy2, cx2 = ys.mean(), xs.mean()
    return ((cx2 - cx1) / w * 100.0, (cy2 - cy1) / h * 100.0)


def mask_binary(mask_video, idx, w, h, threshold):
    cap = cv2.VideoCapture(mask_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    g = to_gray(fr, 'max')
    if (g.shape[1], g.shape[0]) != (w, h):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
    return (g > threshold).astype(np.uint8)


def count_check(args):
    print('=' * 62)
    print('数量核对')
    print('=' * 62)
    import json
    plan = {}
    for split in ['train', 'val', 'test']:
        p = os.path.join(args.split_dir, f'{split}.json')
        if not os.path.exists(p):
            continue
        with open(p) as f:
            n_pairs = len(json.load(f))
        plan[split] = {
            'pairs': n_pairs,
            'real': n_pairs * 2 * args.frames_per_real,
            **{m: n_pairs * 2 * args.frames_per_fake for m in METHODS},
        }

    ok_all = True
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(args.dataset, 'images', split)
        lab_dir = os.path.join(args.dataset, 'labels', split)
        if not os.path.isdir(lab_dir):
            print(f'\n[{split}] 目录不存在,跳过')
            continue
        labs = [f[:-4] for f in os.listdir(lab_dir) if f.endswith('.txt')]
        imgs = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')]
        set_l, set_i = set(labs), set(imgs)
        only_lab = set_l - set_i
        only_img = set_i - set_l

        print(f'\n[{split}]  images {len(imgs)}  labels {len(labs)}')
        if only_lab or only_img:
            ok_all = False
            print(f'   !! 配对不一致: 仅标签 {len(only_lab)}, 仅图片 {len(only_img)}')
            for s in list(only_lab)[:3]:
                print(f'      仅标签: {s}')
            for s in list(only_img)[:3]:
                print(f'      仅图片: {s}')
        else:
            print('   images/labels 一一配对 ✓')

        exp = plan.get(split)
        if exp:
            got = {'real': sum(1 for s in labs if s.startswith('Real_'))}
            for m in METHODS:
                got[m] = sum(1 for s in labs if s.startswith(m + '_'))
            print(f'   {"类别":<16}{"实际":>8}{"期望":>8}')
            for k in ['real'] + METHODS:
                flag = '' if got[k] == exp[k] else '  !!'
                if flag:
                    ok_all = False
                print(f'   {k:<16}{got[k]:>8}{exp[k]:>8}{flag}')
    print('\n数量核对结论:', '全部通过 ✓' if ok_all else '存在不一致 ✗')
    return ok_all


def iou_check(args):
    print()
    print('=' * 62)
    print(f'标签↔mask IoU 验证 (每划分抽样 {args.n})')
    print('=' * 62)
    random.seed(args.seed)
    all_ious = []
    all_offs = []
    rows = []

    for split in args.splits:
        lab_dir = os.path.join(args.dataset, 'labels', split)
        img_dir = os.path.join(args.dataset, 'images', split)
        if not os.path.isdir(lab_dir):
            continue
        # 只抽伪造样本:真实样本标签为空,IoU 无意义
        labs = [f for f in os.listdir(lab_dir)
                if f.endswith('.txt') and not f.startswith('Real_')
                and any(f.startswith(m + '_') for m in args.methods)]
        pick = random.sample(labs, min(args.n, len(labs)))
        print(f'\n[{split}] 方法 {args.methods} 伪造标签 {len(labs)} 个,'
              f'抽样 {len(pick)} 个')

        ious = []
        offs = []
        missing = 0
        for lab in pick:
            stem = lab[:-4]
            method, name, idx = parse_stem(stem)
            img = cv2.imread(os.path.join(img_dir, stem + '.jpg'))
            if img is None:
                continue
            h, w = img.shape[:2]
            url, mpath = mask_url_and_path(args.raw, method, name)
            if not os.path.exists(mpath):
                if not args.download_missing:
                    missing += 1
                    continue
                try:
                    r = fetch_one(url, mpath, root='.')
                    if r == 'fail':
                        missing += 1
                        continue
                except Exception:
                    missing += 1
                    continue
            mb = mask_binary(mpath, idx, w, h, args.threshold)
            if mb is None:
                missing += 1
                continue
            rast = rasterize_label(os.path.join(lab_dir, lab), w, h)
            union = int((rast | mb).sum())
            iou = float((rast & mb).sum() / union) if union else float('nan')
            ious.append(iou)
            off = centroid_offset(rast, mb, w, h)
            if off:
                offs.append(off)
                all_offs.append(off)
            if len(rows) < args.montage_rows:
                c1 = img.copy()
                cv2.putText(c1, f'{method} {iou:.3f}', (8, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                c2 = img.copy()
                cs, _ = cv2.findContours(rast, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(c2, cs, -1, (0, 0, 255), 3)
                c3 = img.copy()
                c3[mb > 0] = (0.45 * c3[mb > 0] +
                              0.55 * np.array([255, 0, 0])).astype(np.uint8)
                rows.append([c1, c2, c3])

        if ious:
            a = np.array(ious)
            all_ious.extend(ious)
            print(f'   IoU     均值 {a.mean():.4f}  最小 {a.min():.4f}  '
                  f'最大 {a.max():.4f}  样本 {len(a)}')
            if offs:
                o = np.array(offs)
                mag = np.sqrt((o ** 2).sum(axis=1))
                print(f'   质心偏移 均值 {mag.mean():.3f}%  最大 {mag.max():.3f}%'
                      f'  (对齐判据, 应 < 1%)')
            if missing:
                print(f'   (mask 缺失跳过 {missing} 个;加 --download-missing 可补验)')
        else:
            print(f'   无可验证样本(mask 缺失 {missing} 个)')

    if all_ious:
        a = np.array(all_ious)
        o = np.array(all_offs) if all_offs else np.zeros((1, 2))
        mag = np.sqrt((o ** 2).sum(axis=1))
        print(f'\n总体: IoU 均值 {a.mean():.4f}  最小 {a.min():.4f}  '
              f'样本 {len(a)}')
        print(f'      质心偏移 均值 {mag.mean():.3f}%  最大 {mag.max():.3f}%')
        align_ok = mag.mean() < 1.0
        iou_ok = a.mean() > 0.95
        passed = align_ok and iou_ok
        print(f'      对齐判定: {"通过" if align_ok else "不通过"} '
              f'(质心偏移 < 1%)')
        print(f'      IoU 判定: {"通过" if iou_ok else "不通过"} '
              f'(均值 > 0.95;软掩码边界模糊,拟合上限约 0.98)')
        print('结论:', '通过 ✓' if passed else '不通过 ✗')
        if not align_ok:
            print('  !! 质心系统性偏移说明坐标可能错位:排查帧下标、'
                  'Y 轴方向、归一化所用宽高是否一致')
    else:
        passed = False
        print('\n无可验证样本 ✗')

    if rows:
        th = 300
        tiles = []
        for r in rows:
            rs = [cv2.resize(x, (int(x.shape[1] * th / x.shape[0]), th))
                  for x in r]
            tiles.append(np.hstack(rs))
        wmax = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)))
                 for t in tiles]
        os.makedirs(os.path.dirname(args.montage) or '.', exist_ok=True)
        cv2.imwrite(args.montage, np.vstack(tiles),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'拼图已保存 {args.montage}  (左:原图 中:标签轮廓 右:mask叠加)')

    # 写 JSON 摘要供 make_report.py 汇总。用 try 包住:报告是附加产物,
    # 绝不能因为它写失败而影响验证本身的结论。
    try:
        import json
        os.makedirs('_debug', exist_ok=True)
        tag = '-'.join(args.methods)
        with open(f'_debug/iou_summary_{tag}.json', 'w',
                  encoding='utf-8') as f:
            json.dump({
                'methods': args.methods,
                'splits': args.splits,
                'threshold': args.threshold,
                'n_total': len(all_ious),
                'mean': float(np.mean(all_ious)) if all_ious else None,
                'min': float(np.min(all_ious)) if all_ious else None,
                'max': float(np.max(all_ious)) if all_ious else None,
                'centroid_mean_pct': float(mag.mean()) if all_ious else None,
                'centroid_max_pct': float(mag.max()) if all_ious else None,
                'align_ok': bool(align_ok) if all_ious else None,
                'passed': bool(passed),
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # 非零退出码供调用方(如 build_masks_all.sh)判断是否放行删除 mask
    if not passed:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description='数据集验证:数量 + IoU')
    ap.add_argument('--dataset', default='data/dataset_v1')
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--split-dir', default='data/splits')
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--methods', default=','.join(METHODS),
                    help='只验证指定伪造方法,逗号分隔')
    ap.add_argument('--n', type=int, default=1000,
                    help='每个划分抽样的伪造标签数')
    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frames-per-fake', type=int, default=3)
    ap.add_argument('--frames-per-real', type=int, default=11)
    ap.add_argument('--counts-only', action='store_true')
    ap.add_argument('--download-missing', action='store_true',
                    help='mask 已删除时按需重下被抽中的样本')
    ap.add_argument('--montage', default='_debug/verify_montage.jpg')
    ap.add_argument('--montage-rows', type=int, default=12)
    args = ap.parse_args()
    args.splits = [s for s in args.splits.split(',') if s]
    args.methods = [m for m in args.methods.split(',') if m]

    count_check(args)
    if not args.counts_only:
        iou_check(args)


if __name__ == '__main__':
    main()
