#!/usr/bin/env python
"""在云主机上启动训练并校验它确实跑起来了。

用 nohup + & 启动,这样 SSH/网络断开不影响训练 —— 这正是本地踩过的坑
(会话结束、窗口关闭、电池切换都会杀掉训练进程),云上必须避开。

密码从环境变量 CLOUD_PASS 读取。
用法:  CLOUD_PASS=xxx python scripts/start_cloud_training.py
"""
import os
import sys
import time

import paramiko

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

HOST = 'connect.cqa1.seetacloud.com'
PORT = 49447
USER = 'root'
WORK = '/root/autodl-tmp/Deepfake-YOLO'
PY = '/root/miniconda3/bin/python'


def run(c, cmd, timeout=120):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace').strip()
    err = e.read().decode('utf-8', 'replace').strip()
    return out or err


def main():
    pw = os.environ.get('CLOUD_PASS')
    if not pw:
        print('错误: 未设置 CLOUD_PASS')
        return 1

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=60)

    print('已有的训练进程:', run(c, 'pgrep -af "train_baseline" || echo 无') or '无')

    # nohup + & : 训练脱离 SSH 会话,断网/关终端都不影响
    # workers=8 (云上 15 vCPU / 80GB,本地 4 是为了迁就 15GB 内存)
    cmd = (f'cd {WORK} && nohup {PY} -u train/train_baseline.py '
           f'--stage full --batch 16 --workers 8 --save-period 10 '
           f'> logs_train.txt 2>&1 & echo "已启动 pid=$!"')
    print(run(c, cmd))
    print('等待 120 秒确认训练进入正常迭代 ...')
    time.sleep(120)

    print('\n--- 进程 ---')
    print(run(c, 'pgrep -af train_baseline || echo "无进程(异常!)"'))
    print('\n--- GPU ---')
    print(run(c, 'nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu '
                 '--format=csv,noheader'))
    print('\n--- 日志尾部 ---')
    print(run(c, "tail -c 500 logs_train.txt | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -4"))

    c.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
