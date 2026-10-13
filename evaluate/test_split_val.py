#!/usr/bin/env python
"""在测试集上跑官方 YOLO 指标 —— 与 Dice/IoU 统一到同一划分。

背景:训练时 results.csv 里的 mAP 是**验证集**上的,而 Dice/IoU 是**测试集**上的。
两组数字不能并列成"同一模型的性能"。本脚本在测试集上补一份官方 mAP,
使 mAP 与 Dice/IoU 同划分、可直接对照。
"""
import os
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

# 必须关掉 sync:ultralytics 默认会把结果上传到它的服务器,
# 网络不通时会静默挂起(实测卡住近 1 小时,CPU 仅用 28 秒)。
from ultralytics.utils import SETTINGS
SETTINGS['sync'] = False

from ultralytics import YOLO  # noqa: E402


def main():
    w = 'runs/segment/cloud_baseline_v1/weights/best.pt'
    print(f'权重: {w}', flush=True)
    print('在测试集上评估(约 3200 张) ...', flush=True)

    m = YOLO(w)
    r = m.val(data='data/dataset_v1/dataset.yaml', split='test',
              imgsz=640, batch=16, device='0', plots=False, verbose=False,
              project='runs/segment', name='cloud_baseline_v1_testval')

    b, s = r.box, r.seg
    print()
    print('=' * 62)
    print('测试集(test)官方指标 —— best.pt (ep68)')
    print('=' * 62)
    print(f'  Box  : P {b.mp:.4f}  R {b.mr:.4f}  '
          f'mAP50 {b.map50:.5f}  mAP50-95 {b.map:.5f}')
    print(f'  Mask : P {s.mp:.4f}  R {s.mr:.4f}  '
          f'mAP50 {s.map50:.5f}  mAP50-95 {s.map:.5f}')
    print()
    print(f'  fitness(box95+mask95) = {b.map + s.map:.5f}')
    print()
    print('对照(同一测试集,由 evaluate/metrics.py 计算):')
    print('  Dice 0.8798 / Mask IoU 0.8564')
    print()
    print('对照(验证集,训练时 results.csv 记录的 best.pt 值):')
    print('  Mask mAP50 0.9744 / mAP50-95 0.8743 @ep68')


if __name__ == '__main__':
    main()
