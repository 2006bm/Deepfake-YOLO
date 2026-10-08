#!/usr/bin/env python
"""生成 data/dataset_v1/metadata.csv。

为什么不只靠文件名解析:分方法统计 mAP/Dice、按划分抽样、查某视频的帧分布、
按伪造面积分档分析,这些都要反复扫描 23000 个文件名并做字符串解析,既慢又易错。
一次生成清单表,后续所有分析直接读表。

字段:
  image            图片相对路径
  label            标签相对路径
  split            train / val / test
  method           伪造方法;真实样本为 Real
  video_id         伪造样本为 "源_目标"(如 585_599);真实样本为原始视频号
  frame            帧下标
  width, height    图片分辨率(FF++ 不同视频分辨率不同:854x480 ~ 1280x720)
  is_fake          1=伪造(标签非空) 0=真实(标签为空)
  n_instances      标签中的多边形个数
  n_points         标签中的总点数
  mask_area_ratio  伪造区域占整图面积比

mask_area_ratio 的算法:标签坐标已归一化到 [0,1],因此在归一化空间用鞋带公式
算出的多边形面积**本身就等于面积占比**,无需读取原图计算像素面积。
这是后续"小面积伪造是否更难检测"分档分析的输入。
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

SPLITS = ['train', 'val', 'test']


def parse_stem(stem):
    body, _, fidx = stem.rpartition('_f')
    method, _, name = body.partition('_')
    return method, name, int(fidx)


def poly_area_norm(pts):
    """归一化坐标多边形的面积(鞋带公式),数值即面积占比。"""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def read_label_stats(path):
    """返回 (多边形个数, 总点数, 面积占比)。"""
    n_inst = n_pts = 0
    area = 0.0
    try:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7:
                    continue
                xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
                if len(xy) < 3:
                    continue
                n_inst += 1
                n_pts += len(xy)
                area += poly_area_norm(xy)
    except OSError:
        pass
    return n_inst, n_pts, min(area, 1.0)


def image_size(path):
    """只读文件头取尺寸,不解码整图(PIL 懒加载),23000 张也很快。"""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def build_row(job):
    """单个样本的元数据。作为多进程工作单元(每行互不依赖)。"""
    split, stem, dataset = job
    method, name, frame = parse_stem(stem)
    img_rel = os.path.join('images', split, stem + '.jpg')
    lab_rel = os.path.join('labels', split, stem + '.txt')
    n_inst, n_pts, area = read_label_stats(os.path.join(dataset, lab_rel))
    w, h = image_size(os.path.join(dataset, img_rel))
    return {
        'image': img_rel.replace(os.sep, '/'),
        'label': lab_rel.replace(os.sep, '/'),
        'split': split,
        'method': method,
        'video_id': name,
        'frame': frame,
        'width': w,
        'height': h,
        'is_fake': 1 if method != 'Real' else 0,
        'n_instances': n_inst,
        'n_points': n_pts,
        'mask_area_ratio': round(area, 6),
    }


def main():
    ap = argparse.ArgumentParser(description='生成数据集 metadata.csv')
    ap.add_argument('--dataset', default='data/dataset_v1')
    ap.add_argument('--out', default=None,
                    help='默认写到 {dataset}/metadata.csv')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    out = args.out or os.path.join(args.dataset, 'metadata.csv')

    jobs = []
    for split in SPLITS:
        lab_dir = os.path.join(args.dataset, 'labels', split)
        if not os.path.isdir(lab_dir):
            continue
        for fn in sorted(os.listdir(lab_dir)):
            if fn.endswith('.txt'):
                jobs.append((split, fn[:-4], args.dataset))

    print(f'扫描 {len(jobs)} 个样本 (workers={args.workers}) ...')
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            rows = list(ex.map(build_row, jobs, chunksize=64))
    else:
        rows = [build_row(j) for j in jobs]
    rows = [r for r in rows if not (r['width'] == 0 and r['height'] == 0)]

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'metadata.csv 已写入 {out}  共 {len(rows)} 行')

    # ---- 摘要,可直接贴进 dataset_report.md ----
    methods = ['Real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    by = defaultdict(Counter)
    for r in rows:
        by[r['split']][r['method']] += 1
    print('\n按划分与方法统计:')
    print(f'  {"split":<8}' + ''.join(f'{m:>16}' for m in methods) + f'{"合计":>10}')
    for split in SPLITS:
        if split not in by:
            continue
        print(f'  {split:<8}' +
              ''.join(f'{by[split][m]:>16}' for m in methods) +
              f'{sum(by[split].values()):>10}')
    print(f'  {"合计":<8}' + ''.join(
        f'{sum(by[s][m] for s in by):>16}' for m in methods) + f'{len(rows):>10}')

    print('\n分辨率分布:')
    for (w_, h_), c in Counter((r['width'], r['height']) for r in rows
                               ).most_common():
        print(f'  {w_}x{h_}: {c}')

    fake = [r for r in rows if r['is_fake']]
    real = [r for r in rows if not r['is_fake']]
    print(f'\n伪造 {len(fake)} / 真实 {len(real)}')
    if fake:
        print(f'平均每图 {sum(r["n_instances"] for r in fake)/len(fake):.2f} '
              f'个多边形, {sum(r["n_points"] for r in fake)/len(fake):.1f} 个点')
        ar = np.array([r['mask_area_ratio'] for r in fake])
        print(f'伪造面积占比: 均值 {ar.mean()*100:.2f}%  '
              f'中位数 {np.median(ar)*100:.2f}%  '
              f'最小 {ar.min()*100:.3f}%  最大 {ar.max()*100:.2f}%')
        bins = [(0, 0.05), (0.05, 0.20), (0.20, 1.01)]
        print('面积分档:')
        for lo, hi in bins:
            c = int(((ar >= lo) & (ar < hi)).sum())
            print(f'  [{lo*100:.0f}%, {hi*100:.0f}%): {c} 张 '
                  f'({c/len(ar)*100:.1f}%)')
    empty = [r for r in fake if r['n_instances'] == 0]
    if empty:
        print(f'!! 警告: {len(empty)} 个伪造样本标签为空(mask 可能过小被面积过滤)')


if __name__ == '__main__':
    main()
