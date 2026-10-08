#!/usr/bin/env bash
# 阶段2a: 下载全部视频文件(不含 mask)。
#
# 为什么先下视频不下 mask:
#   mask 总计 38.6GB,是体积大头,必须"边下边转边删"才能装进 39.4GB 的 D: 盘。
#   视频只有 ~10GB,可以一次性下完,且抽帧后还能删。
#
# 预计耗时: 当前实测 1.6~2.3 MB/s 下约 1.5 小时;拥塞时可能到 8 小时。
# 可随时中断,重跑本脚本会跳过已完成文件。
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/边妙/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONIOENCODING=utf-8

# c23: 训练 + 内部测试;c40: 压缩鲁棒性测试
for c in c23 c40; do
  echo "########## original ($c) ##########"
  "$PY" preprocessing/download.py --type videos --dataset original --compression "$c" --workers 8
  for ds in Deepfakes Face2Face FaceSwap NeuralTextures; do
    echo "########## $ds ($c) ##########"
    "$PY" preprocessing/download.py --type videos --dataset "$ds" --compression "$c" --workers 8
  done
done

echo "########## 视频下载全部完成 ##########"
du -sh data/raw
