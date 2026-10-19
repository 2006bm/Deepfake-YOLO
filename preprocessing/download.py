#!/usr/bin/env python
"""FaceForensics++ 多线程断点续传下载器。

替代官方 faceforensics_download_v4.py。官方脚本是单线程 urllib 且无断点续传,
中断后单个文件重下,不适合 48GB / 43 小时级别的下载。

本脚本特性:
  - 多线程并发(默认 8,实测 kaldir 服务器 ~300-350 KB/s 已饱和)
  - HTTP Range 断点续传(服务器已确认支持 206 Partial Content)
  - 失败指数退避重试
  - 磁盘空间守卫,低于阈值暂停
  - 已下载文件自动跳过,可随时 Ctrl-C 后重跑

用法示例:
    python download.py --type videos --dataset original  --compression c23
    python download.py --type videos --dataset all       --compression c23
    python download.py --type masks  --dataset Deepfakes
    python download.py --list-only --type masks --dataset all
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# Windows 控制台默认 GBK,中文输出会乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')

BASE_URL = 'https://kaldir.vc.in.tum.de/faceforensics/v3/'
# 只有 kaldir(EU2) 实测可达;canis(EU) 与 falas(CA) 均连接超时
FILELIST_URL = BASE_URL + 'misc/filelist.json'

MANIPULATIONS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
COMPRESSIONS = ['c23', 'c40']

CHUNK = 256 * 1024
_lock = threading.Lock()
_state = {'ok': 0, 'skip': 0, 'fail': 0, 'bytes': 0}


def log(msg):
    with _lock:
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()


def free_gb(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def load_pairs(cache='filelist.json'):
    """返回 [[id, id], ...],共 500 对 / 1000 个原始视频。"""
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    with urllib.request.urlopen(FILELIST_URL, timeout=60) as r:
        pairs = json.loads(r.read().decode('utf-8'))
    with open(cache, 'w') as f:
        json.dump(pairs, f)
    return pairs


def build_tasks(datasets, ftype, compression, pairs, outdir, video_list=None):
    """生成 (url, 本地目标路径) 列表。

    目录布局与官方脚本输出保持一致,便于后续脚本对接:
      original     : data/raw/original_sequences/youtube/{c}/videos/{id}.mp4
      manipulated  : data/raw/manipulated_sequences/{M}/{c}/videos/{src}_{tgt}.mp4
      masks        : data/raw/manipulated_sequences/{M}/masks/videos/{src}_{tgt}.mp4

    video_list: 可选 set, 精确筛选下载目标(格式 'method:identifier')。
      original   : 'original:{vid}'(纯数字 ID)
      manipulated: '{method}:{src}_{tgt}'(双 ID 文件名, 不能只匹配单个 ID, 且区分方法)
    """
    if video_list is not None:
        video_list = set(video_list)
    tasks = []
    for ds in datasets:
        if ds == 'original':
            if ftype == 'masks':
                continue  # FF++ 不为原始视频提供 mask
            for pair in pairs:
                for vid in pair:
                    if video_list is not None and f'original:{vid}' not in video_list:
                        continue
                    url = (BASE_URL + 'original_sequences/youtube/'
                           f'{compression}/videos/{vid}.mp4')
                    dest = os.path.join(outdir, 'original_sequences', 'youtube',
                                        compression, 'videos', f'{vid}.mp4')
                    tasks.append((url, dest))
        else:
            for pair in pairs:
                for src, tgt in (pair, pair[::-1]):
                    stem = f'{src}_{tgt}'
                    if video_list is not None and f'{ds}:{stem}' not in video_list:
                        continue
                    name = f'{stem}.mp4'
                    if ftype == 'videos':
                        url = (BASE_URL + f'manipulated_sequences/{ds}/'
                               f'{compression}/videos/{name}')
                        dest = os.path.join(outdir, 'manipulated_sequences', ds,
                                            compression, 'videos', name)
                    else:
                        url = (BASE_URL + f'manipulated_sequences/{ds}/'
                               f'masks/videos/{name}')
                        dest = os.path.join(outdir, 'manipulated_sequences', ds,
                                            'masks', 'videos', name)
                    tasks.append((url, dest))
    return tasks


def fetch_one(url, dest, retries=6, min_free_gb=3.0, root='.'):
    part = dest + '.part'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        with _lock:
            _state['skip'] += 1
        return 'skip'

    for attempt in range(retries):
        if free_gb(root) < min_free_gb:
            log(f'[磁盘守卫] 剩余空间 < {min_free_gb}GB,暂停下载等待人工处理')
            return 'fail'
        try:
            done = os.path.getsize(part) if os.path.exists(part) else 0
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'ffpp-downloader/1.0')
            if done:
                req.add_header('Range', f'bytes={done}-')
            with urllib.request.urlopen(req, timeout=120) as r:
                if done and r.status == 206:
                    mode = 'ab'
                    # 续传时总长度只在 Content-Range 里
                    expect = None
                    cr = r.headers.get('Content-Range')
                    if cr and '/' in cr:
                        try:
                            expect = int(cr.rsplit('/', 1)[1])
                        except ValueError:
                            expect = None
                else:
                    mode = 'wb'
                    done = 0
                    cl = r.headers.get('Content-Length')
                    expect = int(cl) if cl and cl.isdigit() else None
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        with _lock:
                            _state['bytes'] += len(chunk)
            # 关键校验:连接中断时 read() 会静默结束,若不比对字节数,
            # 截断的 mp4 会被当成完整文件保存(cv2 打开报 moov atom not found)。
            actual = os.path.getsize(part)
            if expect is not None and actual != expect:
                raise IOError(f'下载不完整: {actual}/{expect} 字节')
            os.replace(part, dest)
            with _lock:
                _state['ok'] += 1
            return 'ok'
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt == retries - 1:
                with _lock:
                    _state['fail'] += 1
                log(f'[失败] {url} :: {type(e).__name__}: {e}')
                return 'fail'
            time.sleep(wait)
    return 'fail'


def main():
    ap = argparse.ArgumentParser(
        description='FaceForensics++ 多线程断点续传下载器')
    ap.add_argument('--dataset', default='original',
                    help='original | ' + ' | '.join(MANIPULATIONS) + ' | all')
    ap.add_argument('--type', default='videos', choices=['videos', 'masks'])
    ap.add_argument('--compression', default='c23', choices=COMPRESSIONS)
    ap.add_argument('--workers', type=int, default=8,
                    help='并发线程数(实测 >8 无收益)')
    ap.add_argument('--limit', type=int, default=0,
                    help='只下载前 N 个文件,用于阶段1小批量验证')
    ap.add_argument('--outdir', default='data/raw')
    ap.add_argument('--min-free-gb', type=float, default=3.0)
    ap.add_argument('--video-list', default=None,
                    help='精确筛选文件: 每行一个视频标识(original 用纯数字 ID, '
                         'manipulated 用 src_tgt), 只下载匹配的')
    ap.add_argument('--list-format', default='text', choices=['text', 'csv', 'json'],
                    help='--list-only 的输出格式')
    ap.add_argument('--list-only', action='store_true')
    args = ap.parse_args()

    datasets = MANIPULATIONS + ['original'] if args.dataset == 'all' \
        else [args.dataset]

    video_list = None
    if args.video_list:
        with open(args.video_list) as f:
            video_list = {ln.strip() for ln in f if ln.strip()}
        print(f'video-list 加载 {len(video_list)} 个标识')

    pairs = load_pairs()
    tasks = build_tasks(datasets, args.type, args.compression, pairs,
                        args.outdir, video_list)
    if args.limit:
        tasks = tasks[:args.limit]

    print(f'任务数 : {len(tasks)}')
    print(f'目标目录: {os.path.abspath(args.outdir)}')
    print(f'并发数 : {args.workers}   磁盘剩余: {free_gb("."):.1f} GB')
    if args.list_only:
        if args.list_format == 'text':
            for url, dest in tasks[:20]:
                print(f'  {url}\n    -> {dest}')
            if len(tasks) > 20:
                print(f'  ... 其余 {len(tasks) - 20} 个')
            return
        # 结构化清单: 方法 / 文件名 / 压缩 / 目标路径 / 下载地址
        rows = []
        for url, dest in tasks:
            parts = dest.replace('\\', '/').split('/')
            # data/raw/[original_sequences|manipulated_sequences]/...
            if 'original_sequences' in parts:
                method = 'original'
            else:
                method = parts[parts.index('manipulated_sequences') + 1]
            filename = os.path.basename(dest)
            rows.append(dict(method=method, filename=filename,
                             compression=args.compression, dest=dest, url=url))
        if args.list_format == 'csv':
            import csv as _csv
            w = _csv.DictWriter(sys.stdout, fieldnames=['method', 'filename',
                                                        'compression', 'dest', 'url'])
            w.writeheader(); w.writerows(rows)
        else:
            import json as _json
            print(_json.dumps(rows, ensure_ascii=False, indent=1))
        return

    if free_gb('.') < args.min_free_gb:
        print(f'错误: 磁盘剩余空间不足 {args.min_free_gb} GB,已中止')
        return

    t0 = time.time()
    last = [t0, 0]
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fetch_one, u, d, 6, args.min_free_gb, '.')
                    for u, d in tasks]
            for i, _ in enumerate(as_completed(futs), 1):
                now = time.time()
                if now - last[0] >= 10 or i == len(tasks):
                    with _lock:
                        s = dict(_state)
                    el = now - t0
                    rate = s['bytes'] / el / 1024 if el else 0
                    eta = ''
                    print(f'\r进度 {i}/{len(tasks)}  完成{s["ok"]} '
                          f'跳过{s["skip"]} 失败{s["fail"]}  '
                          f'{s["bytes"]/1024**3:.2f}GB  {rate:.0f}KB/s  '
                          f'剩余{free_gb("."):.1f}GB', end='', flush=True)
                    last[0] = now
    except KeyboardInterrupt:
        print('\n已中断。重新运行本命令即可从断点继续。')
        return

    el = time.time() - t0
    with _lock:
        s = dict(_state)
    print(f'\n完成: {s["ok"]} 下载 / {s["skip"]} 已存在 / {s["fail"]} 失败')
    print(f'耗时 {el/60:.1f} 分钟,共 {s["bytes"]/1024**3:.2f} GB,'
          f'均速 {s["bytes"]/el/1024:.0f} KB/s')
    if s['fail']:
        print('存在失败项,直接重跑本命令即可重试(已完成文件会跳过)')


if __name__ == '__main__':
    main()
