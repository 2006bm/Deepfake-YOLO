#!/usr/bin/env python
"""上传 dataset_v1.tar 到 AutoDL 云主机。

密码从环境变量 CLOUD_PASS 读取 —— 不写入本文件,避免凭据落盘。
用法:  CLOUD_PASS=xxx python scripts/upload_to_cloud.py
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
LOCAL = 'dataset_v1.tar'
REMOTE = '/root/autodl-tmp/dataset_v1.tar'


def main():
    pw = os.environ.get('CLOUD_PASS')
    if not pw:
        print('错误: 未设置 CLOUD_PASS 环境变量')
        return 1

    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=pw, timeout=60)

    # 先清掉可能存在的半截文件,避免续传歧义
    _, o, _ = c.exec_command('mkdir -p /root/autodl-tmp && rm -f ' + REMOTE)
    o.channel.recv_exit_status()

    sftp = c.open_sftp()
    size = os.path.getsize(LOCAL)
    print(f'本地 {LOCAL}  {size / 1024 ** 3:.2f} GB')
    print(f'上传 -> {REMOTE}')

    t0 = time.time()

    def cb(done, total):
        el = max(time.time() - t0, 0.001)
        speed = done / el / 1024 / 1024
        pct = done / total * 100
        eta = (total - done) / (done / el) if done else 0
        sys.stdout.write(
            f'\r  {pct:5.1f}%  {done / 1024 ** 3:.2f}/{total / 1024 ** 3:.2f} GB  '
            f'{speed:6.1f} MB/s  剩余 {eta / 60:5.1f} 分钟')
        sys.stdout.flush()

    sftp.put(LOCAL, REMOTE, callback=cb)
    print()

    st = sftp.stat(REMOTE)
    ok = st.st_size == size
    print(f'远端 {st.st_size} 字节 / 本地 {size} 字节  ->  '
          f'{"一致 ✓" if ok else "不一致 ✗"}')
    print(f'耗时 {(time.time() - t0) / 60:.1f} 分钟')
    sftp.close()
    c.close()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
