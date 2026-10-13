#!/usr/bin/env python
"""在 AutoDL 云主机上部署训练环境(解压数据、修正路径、装依赖)。

密码从环境变量 CLOUD_PASS 读取 —— 不写入本文件。
用法:  CLOUD_PASS=xxx python scripts/deploy_to_cloud.py
"""
import os
import sys

import paramiko

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

HOST = 'connect.cqa1.seetacloud.com'
PORT = 49447
USER = 'root'
WORK = '/root/autodl-tmp/Deepfake-YOLO'
PY = '/root/miniconda3/bin/python'
PIP = '/root/miniconda3/bin/pip'
TAR = '/root/autodl-tmp/dataset_v1.tar'


def run(c, cmd, timeout=300, label=None):
    if label:
        print(f'--- {label} ---')
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace').strip()
    err = e.read().decode('utf-8', 'replace').strip()
    code = o.channel.recv_exit_status()
    if out:
        print(out)
    if err:
        print('STDERR:', err[:600])
    print(f'  (exit {code})')
    return code


def main():
    pw = os.environ.get('CLOUD_PASS')
    if not pw:
        print('错误: 未设置 CLOUD_PASS')
        return 1

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=60)
    sftp = c.open_sftp()

    run(c, f'mkdir -p {WORK}/train {WORK}/data', label='创建工作目录')

    sftp.put('train/train_baseline.py', f'{WORK}/train/train_baseline.py')
    print(f'已上传 train_baseline.py -> {WORK}/train/')

    # tar 顶层是 dataset_v1/,所以解压到 data/ 下 -> data/dataset_v1/
    if run(c, f'mkdir -p {WORK}/data && tar -xf {TAR} -C {WORK}/data',
           timeout=900, label='解压数据 (3GB, 约 1 分钟)') != 0:
        print('解压失败'); return 1

    run(c, f'du -sh {WORK}/data/dataset_v1; '
           f'ls {WORK}/data/dataset_v1/; '
           f'ls {WORK}/data/dataset_v1/images/ ; '
           f'ls {WORK}/data/dataset_v1/labels/',
        label='校验数据')

    # dataset.yaml 里 path 是本机绝对路径,必须改成云上路径
    run(c, f'sed -i "s|^path:.*|path: {WORK}/data/dataset_v1|" '
           f'{WORK}/data/dataset_v1/dataset.yaml && '
           f'grep "^path:" {WORK}/data/dataset_v1/dataset.yaml',
        label='修正 dataset.yaml 路径')

    run(c, f'{PIP} install -q ultralytics 2>&1 | tail -3; '
           f'{PY} -c "import ultralytics;print(\'ultralytics\', ultralytics.__version__)"',
        timeout=1200, label='安装 ultralytics')

    sftp.close()
    c.close()
    print('\n部署完成。下一步: 运行 scripts/start_cloud_training.py 启动训练')
    return 0


if __name__ == '__main__':
    sys.exit(main())
