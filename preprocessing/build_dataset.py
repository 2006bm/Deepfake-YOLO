#!/usr/bin/env python
"""构建 YOLO-Seg 数据集:FF++ 视频 + mask → 图片 + polygon 标签。

关键正确性约束(已实测验证):
  1. mask 必须与**输出的伪造视频**配对读取,不能与原始源视频配对。
     Face2Face/NeuralTextures 的输出视频跟随*目标*视频的长度与分辨率
     (如 585_599 是 398 帧,而源视频 585 只有 359 帧),Deepfakes 则跟随*源*视频,
     但 mask 恒等于输出视频。因此代码里对二者做逐项元数据断言。
  2. 视频与 mask 使用**同一组帧下标**,由 pick_indices 统一生成。
  3. 真实样本(original)写出空标签文件,YOLO 会当作背景图处理。

输出为扁平结构,方法名编码进文件名,便于按伪造方法分组汇报指标:
  images/{split}/{method}_{name}_f{idx:05d}.jpg
  labels/{split}/{method}_{name}_f{idx:05d}.txt
  images/{split}/Real_{id}_f{idx:05d}.jpg

以"单个视频"为最小并行单位(每个视频写互不相同的文件,无共享状态),
可用 --workers 多进程加速;结果与单进程完全一致。
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# 必须在导入同目录模块之前插入路径:Windows 多进程会重新导入本模块,
# 子进程的 sys.path[0] 与主进程不同,不显式插入会 ImportError。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2  # noqa: E402

from extract_frames import pick_indices, read_frames, video_meta  # noqa: E402
from mask_convert import mask_to_polygons, polygons_to_label  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def load_split(split_dir, split):
    with open(os.path.join(split_dir, f'{split}.json')) as f:
        return json.load(f)


def save_sample(out, split, stem, frame, label):
    ip = os.path.join(out, 'images', split, stem + '.jpg')
    lp = os.path.join(out, 'labels', split, stem + '.txt')
    if os.path.exists(ip) and os.path.exists(lp):
        return False
    os.makedirs(os.path.dirname(ip), exist_ok=True)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    cv2.imwrite(ip, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(lp, 'w', encoding='utf-8') as f:
        f.write(label)
    return True


def _do_fake(job):
    (raw, out, split, method, name, k, threshold, compression) = job
    vp = f'{raw}/manipulated_sequences/{method}/{compression}/videos/{name}.mp4'
    mp = f'{raw}/manipulated_sequences/{method}/masks/videos/{name}.mp4'
    if not (os.path.exists(vp) and os.path.exists(mp)):
        return 0, 0, f'{method}_{name}: missing'
    vmeta, mmeta = video_meta(vp), video_meta(mp)
    if vmeta != mmeta:
        return 0, 0, f'{method}_{name}: meta_mismatch {vmeta} vs {mmeta}'
    idx = pick_indices(vmeta[0], k)
    vf, mf = read_frames(vp, idx), read_frames(mp, idx)
    n_new = n_tot = 0
    skipped = []
    for i in idx:
        if i not in vf or i not in mf:
            continue
        label = polygons_to_label(mask_to_polygons(mf[i], threshold))
        if not label.strip():
            # 伪造样本但该帧 mask 为空 —— 必须跳过而不是保存。
            # 实测 FF++ 的 FaceSwap 部分视频存在"隔帧为空"的 mask
            # (如 601_653 的奇数帧有内容、偶数帧全空),若抽到空帧仍保存,
            # 就等于给模型一个"此处无伪造"的错误监督信号。
            skipped.append(i)
            continue
        stem = f'{method}_{name}_f{i:05d}'
        n_tot += 1
        if save_sample(out, split, stem, vf[i], label):
            n_new += 1
    # 记录被跳过的具体帧,便于他人复现时理解为什么样本数少于理论值
    warn = None
    if skipped:
        warn = (f'skip empty mask: method={method} video={name} '
                f'frames=[{",".join(str(x) for x in skipped)}] '
                f'reason=empty_polygon')
    return n_new, n_tot, warn


def _do_real(job):
    (raw, out, split, vid, k) = job
    vp = f'{raw}/original_sequences/youtube/c23/videos/{vid}.mp4'
    if not os.path.exists(vp):
        return 0, 0, f'Real_{vid}: missing'
    n, _, _ = video_meta(vp)
    idx = pick_indices(n, k)
    vf = read_frames(vp, idx)
    n_new = n_tot = 0
    for i in idx:
        if i not in vf:
            continue
        stem = f'Real_{vid}_f{i:05d}'
        n_tot += 1
        if save_sample(out, split, stem, vf[i], ''):
            n_new += 1
    return n_new, n_tot, None


def main():
    ap = argparse.ArgumentParser(description='构建 YOLO-Seg 数据集')
    ap.add_argument('--split', default='train',
                    help='划分名,对应 {split-dir}/{split}.json')
    ap.add_argument('--methods', default=','.join(METHODS))
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--split-dir', default='data/splits')
    ap.add_argument('--out', default='data/dataset_v1')
    ap.add_argument('--compression', default='c23', choices=['c23', 'c40'])
    ap.add_argument('--frames-per-fake', type=int, default=3)
    ap.add_argument('--frames-per-real', type=int, default=11)
    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个配对')
    ap.add_argument('--no-real', action='store_true', help='不生成真实样本')
    ap.add_argument('--workers', type=int, default=6,
                    help='并行进程数;1 表示单进程')
    args = ap.parse_args()

    methods = [m for m in args.methods.split(',') if m]
    pairs = load_split(args.split_dir, args.split)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f'划分 {args.split}: {len(pairs)} 个配对, 方法 {methods}')
    print(f'伪造每视频 {args.frames_per_fake} 帧, 真实每视频 '
          f'{args.frames_per_real} 帧, threshold={args.threshold}, '
          f'workers={args.workers}')

    jobs = []
    for pair in pairs:
        if not args.no_real:
            for vid in pair:
                jobs.append(('real', (args.raw, args.out, args.split, vid,
                                      args.frames_per_real)))
        for m in methods:
            for src, tgt in (pair, pair[::-1]):
                jobs.append(('fake', (args.raw, args.out, args.split, m,
                                      f'{src}_{tgt}', args.frames_per_fake,
                                      args.threshold, args.compression)))

    fn = {'real': _do_real, 'fake': _do_fake}
    tot_new = tot_all = 0
    warns = []
    done = 0
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for kind, res in zip((j[0] for j in jobs),
                                 ex.map(_dispatch, jobs, chunksize=8)):
                n_new, n_tot, err = res
                tot_new += n_new
                tot_all += n_tot
                if err:
                    warns.append(err)
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f'  视频 {done}/{len(jobs)}  新增 {tot_new} / '
                          f'处理 {tot_all}')
    else:
        for kind, payload in jobs:
            n_new, n_tot, err = fn[kind](payload)
            tot_new += n_new
            tot_all += n_tot
            if err:
                warns.append(err)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f'  视频 {done}/{len(jobs)}  新增 {tot_new} / '
                      f'处理 {tot_all}')

    print(f'\n完成: 新增 {tot_new} 张, 合计处理 {tot_all} 张')
    if warns:
        print(f'\n告警 {len(warns)} 条(前 10 条):')
        for w in warns[:10]:
            print('  ' + w)


def _dispatch(job):
    """单个视频的处理入口。**吞掉异常**并转成告警字符串。

    不能让一个损坏视频炸掉整批:ProcessPoolExecutor.map 一旦抛异常就会
    中止整个 map,前面已处理的成果虽在磁盘上但剩余全部不执行。
    """
    kind, payload = job
    label = payload[3] if kind == 'real' else f'{payload[3]}_{payload[4]}'
    try:
        return (_do_real if kind == 'real' else _do_fake)(payload)
    except Exception as e:
        return 0, 0, f'{label}: {type(e).__name__}: {e}'


if __name__ == '__main__':
    main()
