#!/usr/bin/env python
"""实验1:Baseline —— YOLOv8s-seg 训练。

两档:
  --stage smoke   5 epoch + 10% 数据。目的**不是看指标**,而是确认
                  管线能跑通:loss 下降、mask 正常、显存够、速度可接受。
  --stage full    100 epoch 全量数据,正式基线。

Ultralytics 默认会保存 runs/segment/<name>/ 下的
  results.csv      逐 epoch 指标(务必保留,论文图表的数据源)
  weights/best.pt  最优权重
  weights/last.pt  最后一轮权重
  args.yaml        本次运行的全部参数(复现依据)
不要只留 best.pt —— 没有 args.yaml 就无法复现。
"""
import argparse
import os
import sys

from ultralytics import YOLO

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

PRESETS = {
    'smoke': dict(epochs=5, fraction=0.10, name='smoke_v1'),
    'full': dict(epochs=100, fraction=1.00, name='baseline_v1'),
}


def main():
    ap = argparse.ArgumentParser(description='YOLOv8s-seg Baseline 训练')
    ap.add_argument('--stage', default='smoke', choices=list(PRESETS))
    ap.add_argument('--data', default='data/dataset_v1/dataset.yaml')
    ap.add_argument('--model', default='yolov8s-seg.pt')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16,
                    help='显存不足时降到 8 或 4')
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4,
                    help='dataloader 进程数。每个 worker 会独立加载一份 '
                         'Python/numpy/cv2(约 0.5GB 常驻),本机 15GB 内存下 '
                         '设 8 会在全量数据 + mosaic 增强时耗尽内存并崩溃'
                         '(实测 cv2 OutOfMemoryError),4 是安全值')
    ap.add_argument('--project', default='runs/segment')
    ap.add_argument('--name', default=None, help='默认随 stage 自动命名')
    ap.add_argument('--resume', action='store_true',
                    help='从上次中断处的 weights/last.pt 继续训练')
    ap.add_argument('--save-period', type=int, default=0,
                    help='每 N 轮额外保存 weights/epoch{N}.pt 做冗余。0=关闭。'
                         '训练被强杀时正在写的 last.pt 会损坏(实测 zip 无中央目录),'
                         '有编号的副本可用于回退')
    args = ap.parse_args()

    p = PRESETS[args.stage]
    name = args.name or p['name']

    # 必须用绝对路径:ultralytics 会把相对 project 与其 runs_dir 设置拼接,
    # 传 'runs/segment' 会得到 runs/segment/runs/segment/<name> 的重复嵌套。
    project = os.path.abspath(args.project)

    print(f'=== Baseline 训练 [{args.stage}] ===')
    print(f'  data    {args.data}')
    print(f'  model   {args.model}')
    print(f'  imgsz   {args.imgsz}   batch {args.batch}   device {args.device}')
    print(f'  workers {args.workers}')
    print(f'  epochs  {p["epochs"]}   fraction {p["fraction"]}')
    print(f'  输出     {os.path.join(project, name)}')
    print()

    if args.resume:
        # 从上次中断处继续。ultralytics 用 last.pt 里记录的 epoch 与
        # optimizer 状态续跑,已完成轮次不会重来。
        last = os.path.join(project, name, 'weights', 'last.pt')
        if not os.path.exists(last):
            print(f'错误: 找不到 {last},无从续训')
            sys.exit(1)
        print(f'从 {last} 续训')
        model = YOLO(last)
        model.train(resume=True)
        out = os.path.join(project, name)
        print(f'\n=== 续训结束 ===\n  结果目录 {out}')
        for f in ['results.csv', 'args.yaml',
                  'weights/best.pt', 'weights/last.pt']:
            fp = os.path.join(out, f)
            print(f'  {"OK " if os.path.exists(fp) else "缺失"} {f}')
        return

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=p['epochs'],
        fraction=p['fraction'],
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project,
        name=name,
        # 与 configs/baseline.yaml 保持一致,保证 smoke 与 full 同口径
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3.0,
        patience=20,
        amp=True,
        seed=0,
        deterministic=True,
        # 不做旋转:旋转会引入新的边缘,污染高频特征
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        plots=True,
        save_period=args.save_period if args.save_period > 0 else -1,
    )

    out = os.path.join(project, name)
    print(f'\n=== 训练结束 ===')
    print(f'  结果目录 {out}')
    for f in ['results.csv', 'args.yaml',
              'weights/best.pt', 'weights/last.pt']:
        fp = os.path.join(out, f)
        print(f'  {"OK " if os.path.exists(fp) else "缺失"} {f}')


if __name__ == '__main__':
    main()
