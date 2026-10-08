#!/usr/bin/env python
"""最终一致性校验:文件系统 / metadata.csv / verification_report.json 三方对齐。

为什么要单独做这一步:本项目的数据是分批、可中断、多进程构建的
(下载中断、进程被杀、逐方法处理都发生过)。单个环节看起来都对,
不代表三方统计口径一致。冻结数据集前做一次交叉校验,把"数字对不上"
这类问题挡在训练之前 —— 这类错误一旦进入论文,复盘成本极高。

校验内容:
  1. 每个划分中 images/*.jpg 与 labels/*.txt 的 stem 必须一一对应
  2. metadata.csv 行数 == 标签总数,且其 image/label 路径全部真实存在
  3. metadata.csv 的分划分、分方法计数与文件系统实际计数一致
  4. verification_report.json 的 totals 与前述一致(若存在)
  5. 伪造样本标签非空、真实样本标签为空

退出码 0 表示全部通过,1 表示存在不一致。
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

SPLITS = ['train', 'val', 'test']
METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def stems(d, ext):
    if not os.path.isdir(d):
        return set()
    return {f[:-len(ext)] for f in os.listdir(d) if f.endswith(ext)}


def main():
    ap = argparse.ArgumentParser(description='数据集三方一致性校验')
    ap.add_argument('--dataset', default='data/dataset_v1')
    ap.add_argument('--report', default='verification_report.json')
    args = ap.parse_args()

    problems = []

    def bad(msg):
        problems.append(msg)
        print(f'  !! {msg}')

    # ---- 1. images / labels 配对 ----
    print('[1] images / labels 一一配对')
    fs_counts = {}
    for s in SPLITS:
        i = stems(os.path.join(args.dataset, 'images', s), '.jpg')
        l = stems(os.path.join(args.dataset, 'labels', s), '.txt')
        fs_counts[s] = len(l)
        only_i, only_l = i - l, l - i
        if only_i or only_l:
            bad(f'{s}: 仅图片 {len(only_i)} 个, 仅标签 {len(only_l)} 个')
            for x in list(only_i)[:2]:
                print(f'       仅图片: {x}')
            for x in list(only_l)[:2]:
                print(f'       仅标签: {x}')
        else:
            print(f'  {s}: {len(i)} 对,配对正确')

    # ---- 2 & 3. metadata.csv ----
    print('\n[2] metadata.csv 与文件系统对照')
    mpath = os.path.join(args.dataset, 'metadata.csv')
    if not os.path.exists(mpath):
        bad(f'metadata.csv 不存在: {mpath}')
        rows = []
    else:
        with open(mpath, encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        print(f'  metadata 行数 {len(rows)};文件系统标签数 '
              f'{sum(fs_counts.values())}')
        if len(rows) != sum(fs_counts.values()):
            bad(f'metadata 行数({len(rows)}) != 文件系统标签数'
                f'({sum(fs_counts.values())})')

        missing_img = missing_lab = 0
        for r in rows:
            if not os.path.exists(os.path.join(args.dataset, r['image'])):
                missing_img += 1
            if not os.path.exists(os.path.join(args.dataset, r['label'])):
                missing_lab += 1
        if missing_img:
            bad(f'metadata 引用了 {missing_img} 个不存在的图片')
        if missing_lab:
            bad(f'metadata 引用了 {missing_lab} 个不存在的标签')

        mc = Counter(r['split'] for r in rows)
        for s in SPLITS:
            if mc.get(s, 0) != fs_counts.get(s, 0):
                bad(f'{s}: metadata {mc.get(s, 0)} != 文件系统 '
                    f'{fs_counts.get(s, 0)}')
        print(f'  分划分计数一致: {dict(mc)}')

        # 每方法计数
        per = Counter((r['split'], r['method']) for r in rows)
        print('  分方法计数:')
        for s in SPLITS:
            line = '    ' + f'{s:<7}' + ''.join(
                f'{m[:4]}:{per.get((s, m), 0):<6}' for m in METHODS)
            print(line + f'Real:{per.get((s, "Real"), 0)}')

    # ---- 4. verification_report.json ----
    print('\n[3] verification_report.json 对照')
    if not os.path.exists(args.report):
        print(f'  ({args.report} 不存在,跳过 —— 冻结前应先生成)')
    else:
        with open(args.report, encoding='utf-8') as f:
            vr = json.load(f)
        t = vr.get('totals', {})
        n_lab = sum(fs_counts.values())
        if t.get('images') != n_lab:
            bad(f'json totals.images({t.get("images")}) != 标签总数({n_lab})')
        else:
            print(f'  totals.images {t.get("images")} 与文件系统一致')
        for s in SPLITS:
            jc = vr.get('counts', {}).get(s)
            if not jc:
                continue
            js = sum(jc.values())
            if js != fs_counts.get(s, 0):
                bad(f'json {s} 合计({js}) != 文件系统 {fs_counts.get(s, 0)}')
        print(f'  json 与文件系统分划分计数一致')

    # ---- 5. 空标签语义 ----
    print('\n[4] 空标签语义')
    if rows:
        real_empty = sum(1 for r in rows
                         if r['method'] == 'Real' and r['n_instances'] == '0')
        fake_empty = sum(1 for r in rows
                         if r['method'] != 'Real' and r['n_instances'] == '0')
        n_real = sum(1 for r in rows if r['method'] == 'Real')
        n_fake = len(rows) - n_real
        print(f'  真实 {n_real} 中空标签 {real_empty} '
              f'({real_empty/max(n_real,1)*100:.2f}%),应为 100%')
        print(f'  伪造 {n_fake} 中空标签 {fake_empty} '
              f'({fake_empty/max(n_fake,1)*100:.2f}%),应为 0%')
        if n_real and real_empty != n_real:
            bad(f'{n_real - real_empty} 个真实样本标签非空')
        if fake_empty:
            bad(f'{fake_empty} 个伪造样本标签为空')

    print('\n' + '=' * 58)
    if problems:
        print(f'结论: 发现 {len(problems)} 处不一致 ✗  冻结前必须修复')
        sys.exit(1)
    print('结论: 三方一致,数据集可以冻结 ✓')


if __name__ == '__main__':
    main()
