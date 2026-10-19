# -*- coding: utf-8 -*-
"""c40 抽帧 —— 按 metadata 的精确帧索引从 c40 视频抽帧, 复用 dataset_v1 标签。

原则(冻结协议):
  1. 帧索引直接取自 metadata.csv 的 frame 列, **不重新 pick_indices**。
  2. 只处理 test 划分(鲁棒性实验 = c23 训练 + c40 test)。
  3. 输出到新目录 data/dataset_c40, **不覆盖** dataset_v1。
  4. label 直接复制(掩码是归一化坐标, 与压缩级别无关, 预检已确认分辨率一致)。
  5. 断点续跑: 已存在的输出图片跳过。

用法: python -I preprocessing/extract_c40.py
"""
import csv
import os
import shutil
import sys
import time

import cv2

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

RAW = 'data/raw'
ORIG_C40 = f'{RAW}/original_sequences/youtube/c40/videos'
MANIP_C40 = f'{RAW}/manipulated_sequences/{{m}}/c40/videos'
SRC_IMAGES = 'data/dataset_v1/images'
SRC_LABELS = 'data/dataset_v1/labels'
DST_IMAGES = 'data/dataset_c40/images/test'
DST_LABELS = 'data/dataset_c40/labels/test'
JPEG_Q = 95


def c40_video_path(method, video_id):
    if method == 'Real':
        return os.path.join(ORIG_C40, video_id + '.mp4')
    return os.path.join(MANIP_C40.format(m=method), video_id + '.mp4')


def main():
    rows = list(csv.DictReader(open('data/dataset_v1/metadata.csv', encoding='utf-8')))
    test = [r for r in rows if r['split'] == 'test']
    print(f'test 样本 {len(test)} 个')

    os.makedirs(DST_IMAGES, exist_ok=True)
    os.makedirs(DST_LABELS, exist_ok=True)

    # 按 (method, video_id) 分组, 每组处理完立即释放句柄, 避免累积 700 个句柄耗尽内存
    from collections import defaultdict
    groups = defaultdict(list)
    for r in test:
        groups[(r['method'], r['video_id'])].append(r)
    print(f'{len(groups)} 个视频')

    ok = skip = fail = 0
    t0 = time.time()
    done_count = 0
    for gi, ((method, video_id), grp) in enumerate(groups.items()):
        # 跳过本组已全部完成的
        pending = []
        for r in grp:
            stem = os.path.basename(r['image'])[:-4]
            dst_img = os.path.join(DST_IMAGES, stem + '.jpg')
            if os.path.exists(dst_img):
                skip += 1
                done_count += 1
            else:
                pending.append((r, stem))
        if not pending:
            continue
        p = c40_video_path(method, video_id)
        cap = cv2.VideoCapture(p) if os.path.exists(p) else None
        if cap is None or not cap.isOpened():
            fail += len(pending)
            for r, stem in pending:
                print(f'  [失败] 无视频: {method} {video_id}')
            continue
        for r, stem in pending:
            fr = int(r['frame'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            got, img = cap.read()
            if not got:
                fail += 1
                print(f'  [失败] 解码帧 {fr}: {method} {video_id}')
                continue
            cv2.imwrite(os.path.join(DST_IMAGES, stem + '.jpg'), img,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
            shutil.copy(os.path.join(SRC_LABELS, r['split'], stem + '.txt'),
                        os.path.join(DST_LABELS, stem + '.txt'))
            ok += 1
            done_count += 1
        cap.release()
        if (gi + 1) % 50 == 0:
            print(f'\r  视频 {gi+1}/{len(groups)}  完成 {ok} 跳过 {skip} 失败 {fail}  '
                  f'{time.time()-t0:.0f}s', end='', flush=True)
    print(f'\r  完成 {ok} 跳过 {skip} 失败 {fail}  共 {len(test)} 张, '
          f'耗时 {time.time()-t0:.0f}s')

    # 生成 dataset.yaml(仅 test 划分, 用于 c40 评测)
    yaml = f"""# c40 压缩鲁棒性评测数据集(仅 test 划分)
# 图像从 c40 视频按 dataset_v1 的精确帧索引抽取; 标签直接复用 dataset_v1。
path: D:/科研论文课程/Deepfake-YOLO/data/dataset_c40
test: images/test
names:
  0: fake_region
"""
    with open('data/dataset_c40/dataset.yaml', 'w', encoding='utf-8') as f:
        f.write(yaml)
    print(f'完成, 图片 {DST_IMAGES}, 标签 {DST_LABELS}, yaml data/dataset_c40/dataset.yaml')


if __name__ == '__main__':
    main()
