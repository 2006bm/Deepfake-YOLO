#!/usr/bin/env bash
# GPU 温度/功耗监控。每 30 秒采样一次,输出 CSV。
#
# 为什么要监控:笔记本散热能力有限,4060 Laptop 的温度墙是 87°C,
# 越过后会降频,训练时间会明显拉长。有这份数据才能判断:
#   - 散热是否足够(是否频繁触碰 87°C)
#   - 是否需要散热垫或降低 batch
#   - "31 小时"的估算是否成立
#
# 用法: bash scripts/monitor_gpu.sh [输出文件] [采样间隔秒]
# 停止: Ctrl-C 或 kill
set -u
OUT="${1:-logs_gpu.csv}"
INTERVAL="${2:-30}"

echo "timestamp,temp_c,power_w,util_pct,mem_used_mib,clock_mhz" > "$OUT"
echo "监控中 -> $OUT  (每 ${INTERVAL}s 一次, Ctrl-C 停止)"

while true; do
  line=$(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,clocks.sm 2>/dev/null \
         --format=csv,noheader,nounits | tr -d ' ')
  if [ -n "$line" ]; then
    printf '%s,%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$line" >> "$OUT"
  fi
  sleep "$INTERVAL"
done
