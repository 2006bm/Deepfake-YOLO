#!/usr/bin/env bash
# 实时查看 baseline 训练进度。等价于:
#   python scripts/watch_progress.py --interval 30
# 放在这里是为了让命令短一些(否则每次要敲完整的 python 解释器路径)。
cd "$(dirname "$0")/.."
PY="C:/Users/边妙/AppData/Local/Programs/Python/Python313/python.exe"
exec "$PY" -u scripts/watch_progress.py --interval "${1:-30}"
