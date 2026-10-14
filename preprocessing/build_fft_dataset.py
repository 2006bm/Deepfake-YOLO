#!/usr/bin/env python
"""从 dataset_v1 生成频域增强数据集 dataset_v2。

严格保持:目录结构、文件命名、train/val/test 划分、标签文件**原样复制**。
只有 images/ 下的图片经过增强 —— 这样训练配置只需换 data 路径,
两组实验的唯一变量就是输入图。

独立输出到 dataset_v2,dataset_v1 保持冻结不被触碰。

多进程:单张 FFT 增强约 0.35~0.5 秒(每通道一次 FFT),单进程跑 2.3 万张要 2 小时以上。
"""
import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fft_enhance import enhance  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

SPLITS = ['train', 'val', 'test']


def _work(job):
    src, dst, lam, method, sigma, cutoff = job
    bgr = cv2.imread(src)
    if bgr is None:
        return 'read_fail', src
    out = enhance(bgr, lam, method, sigma, cutoff)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return 'ok', src


def main():
    ap = argparse.ArgumentParser(description='生成频域增强数据集')
    ap.add_argument('--src', default='data/dataset_v1')
    ap.add_argument('--dst', default='data/dataset_v2')
    ap.add_argument('--lam', type=float, default=1.0)
    ap.add_argument('--method', default='fft', choices=['fft', 'residual'])
    ap.add_argument('--sigma', type=float, default=1.0, help='residual 模式的高斯半径')
    ap.add_argument('--cutoff', type=float, default=0.15, help='fft 模式的高通截止(归一化)')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0, help='每个划分只处理前 N 张(冒烟用)')
    args = ap.parse_args()

    print(f'方法={args.method}  λ={args.lam}  '
          f'{"cutoff="+str(args.cutoff) if args.method == "fft" else "sigma="+str(args.sigma)}')
    print(f'{args.src}  ->  {args.dst}   workers={args.workers}')

    jobs = []
    counts = {}
    for sp in SPLITS:
        idir = os.path.join(args.src, 'images', sp)
        ldir = os.path.join(args.src, 'labels', sp)
        if not os.path.isdir(idir):
            continue
        imgs = sorted(f for f in os.listdir(idir) if f.endswith('.jpg'))
        if args.limit:
            imgs = imgs[:args.limit]
        counts[sp] = len(imgs)
        # 标签原样复制(与图片同名同结构)
        odir = os.path.join(args.dst, 'labels', sp)
        os.makedirs(odir, exist_ok=True)
        for f in os.listdir(ldir):
            shutil.copy2(os.path.join(ldir, f), os.path.join(odir, f))
        for f in imgs:
            jobs.append((os.path.join(idir, f),
                         os.path.join(args.dst, 'images', sp, f),
                         args.lam, args.method, args.sigma, args.cutoff))

    # dataset.yaml 与 metadata.csv 一并带过去,并把 path 指向 dataset_v2。
    # 注意:yaml 里写的是**正斜杠**路径,而 os.path.abspath 在 Windows 返回反斜杠,
    # 直接替换会静默失败(实测踩过)。统一按正斜杠比较替换。
    srcyaml = os.path.join(args.src, 'dataset.yaml')
    if os.path.exists(srcyaml):
        src_fwd = os.path.abspath(args.src).replace('\\', '/')
        dst_fwd = os.path.abspath(args.dst).replace('\\', '/')
        txt = open(srcyaml, encoding='utf-8').read()
        if src_fwd not in txt:
            print(f'!! 警告: dataset.yaml 中未找到 "{src_fwd}",path 可能仍指向旧数据集,'
                  f'请手工确认')
        txt = txt.replace(src_fwd, dst_fwd)
        with open(os.path.join(args.dst, 'dataset.yaml'), 'w', encoding='utf-8') as f:
            f.write(txt)
    for extra in ('metadata.csv',):
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.dst, extra))

    print(f'待处理 {len(jobs)} 张  (每划分: {counts})')
    print('完成前不中断;可用 Ctrl-C,重跑会自动跳过已存在的输出')

    # 跳过已生成(可续跑)
    todo = [j for j in jobs if not os.path.exists(j[1])]
    print(f'其中 {len(jobs) - len(todo)} 张已存在,实际处理 {len(todo)} 张')
    if not todo:
        print('全部已生成')
        return

    t0 = time.time()
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (st, _) in enumerate(ex.map(_work, todo, chunksize=8), 1):
            if st == 'ok':
                ok += 1
            else:
                fail += 1
            if i % 200 == 0 or i == len(todo):
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate / 60 if rate else 0
                print(f'\r  {i}/{len(todo)}  成功{ok} 失败{fail}  '
                      f'{rate:.0f} 张/秒  剩余 {eta:.1f} 分钟', end='', flush=True)
    print()
    print(f'完成: {ok} 成功 / {fail} 失败,耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
