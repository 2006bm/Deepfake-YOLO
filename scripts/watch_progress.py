#!/usr/bin/env python
"""实时查看 baseline 训练进度。

原始训练日志用 \r 反复刷新同一行(进度条),直接 tail -f 会刷屏且看不清。
本脚本改为每 30 秒清屏重绘一张干净的进度面板。

用法:
    python scripts/watch_progress.py
    python scripts/watch_progress.py --interval 10
    python scripts/watch_progress.py --run runs/segment/baseline_v1
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

COLS = [
    ('epoch', 'ep', 5),
    ('time', '耗时(s)', 8),
    ('train/box_loss', 'box', 7),
    ('train/seg_loss', 'seg', 7),
    ('train/cls_loss', 'cls', 7),
    ('metrics/precision(M)', 'P(M)', 7),
    ('metrics/recall(M)', 'R(M)', 7),
    ('metrics/mAP50(M)', 'mAP50', 8),
    ('metrics/mAP50-95(M)', 'mAP50-95', 9),
]


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def gpu_line():
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout.strip()
        if out:
            t, p, u, m = [x.strip() for x in out.split(',')]
            return f'GPU {t}°C(墙87)  {p}W(限125)  利用率 {u}%  显存 {m}MiB'
    except Exception:
        pass
    return 'GPU 状态不可用'


def ram_line():
    """系统内存。dataloader worker 数过多会耗尽内存导致 cv2 OOM 崩溃
    (实测 8 worker 在全量数据下崩溃),因此这里给出预警。"""
    try:
        import psutil
        vm = psutil.virtual_memory()
        used = vm.used / 1024 ** 3
        total = vm.total / 1024 ** 3
        avail = vm.available / 1024 ** 3
        warn = '  !! 内存紧张' if avail < 1.5 else ''
        return f'内存 {used:.1f}/{total:.1f} GB 已用,可用 {avail:.1f} GB{warn}'
    except Exception:
        return '内存状态不可用'


def tail_log(path, n=2):
    import re
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8000))
            chunk = f.read().decode('utf-8', errors='replace')
        # 去掉 ANSI 转义序列(进度条用的 [K / [34m 等),否则会污染显示
        chunk = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', chunk)
        parts = [p.strip() for p in chunk.replace('\r', '\n').split('\n') if p.strip()]
        return parts[-n:]
    except Exception:
        return []


def epoch_seconds(rows):
    """每个 epoch 的实际耗时。results.csv 的 time 列是累计值,需做差。"""
    ts = []
    prev = 0.0
    for r in rows:
        try:
            cur = float(r['time'])
        except Exception:
            continue
        ts.append(cur - prev)
        prev = cur
    return ts


def main():
    ap = argparse.ArgumentParser(description='训练进度实时面板')
    ap.add_argument('--run', default='runs/segment/baseline_v1')
    ap.add_argument('--log', default='logs_baseline3.txt')
    ap.add_argument('--interval', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    res = os.path.join(args.run, 'results.csv')
    while True:
        clear()
        print('=' * 78)
        print(f'  Baseline 训练进度    {datetime.now():%Y-%m-%d %H:%M:%S}')
        print('=' * 78)

        rows = []
        if os.path.exists(res):
            try:
                with open(res, encoding='utf-8') as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                pass

        if rows:
            done = len(rows)
            per = epoch_seconds(rows)
            # 用最近若干个 epoch 的中位数估计,避免被首个 epoch 的缓存开销带偏
            recent = sorted(per[-5:])
            med = recent[len(recent) // 2] if recent else 0
            eta = f'  剩余约 {(args.epochs - done) * med / 3600:.1f} 小时' if med else ''
            print(f'  已完成 {done} / {args.epochs} epoch'
                  f'   ({med/60:.1f} 分钟/epoch{eta})')
            print()
            hdr = '  ' + ' '.join(f'{alias:>{w}}' for _, alias, w in COLS)
            print(hdr)
            print('  ' + '-' * (len(hdr) - 2))
            for r in rows[-8:]:
                cells = []
                for key, _, w in COLS:
                    v = r.get(key, '')
                    try:
                        v = f'{float(v):.{w-3}f}'
                    except Exception:
                        pass
                    cells.append(f'{v:>{w}}')
                print('  ' + ' '.join(cells))
        else:
            print('  尚无完成的 epoch(第一个 epoch 含数据缓存与预热,约需 12 分钟)')

        print()
        print(f'  {gpu_line()}')
        print(f'  {ram_line()}')
        print()
        for l in tail_log(args.log):
            print(f'  {l[:100]}')
        print()
        print(f'  (每 {args.interval} 秒刷新,Ctrl-C 退出;训练在后台独立运行,退出本面板不影响训练)')
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n已退出面板(训练仍在后台运行)')
