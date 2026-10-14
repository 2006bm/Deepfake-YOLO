#!/usr/bin/env python
"""频域(高频)增强 —— 生成本项目的改进实验输入。

设计原则(与 configs/fft.yaml 一致):
  **不改网络结构**。YOLOv8 默认 3 通道输入,改 4 通道要动第一层,
  风险和调试成本都不适合本项目。改为在**输入端**把高频信息以增强图的
  形式注入,保持 3 通道:

      I_h = I - lowpass(I)                 # 高频残差
      I'  = clip(I + λ·I_h, 0, 255)        # 增强后的 RGB,仍是三通道

  依据:Deepfake 的融合边界会破坏局部高频纹理。该依据目前是**假设**,
  必须由实验验证;若无效则按计划放弃,题目为"定位与鲁棒性研究",不受影响。

两种高频提取:
  --method residual   空域近似: I - GaussianBlur(I)   (快,等价一阶高通)
  --method fft        频域:     FFT → 高通掩码 → IFFT (更接近"频域增强"的表述)

参数:
  --lam    增强强度 λ。0 等于原图(可用于自检)
  --sigma  空域模式的高斯半径 / 频域模式的截止频率(相对半径)
"""
import argparse
import os
import sys

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def high_freq_residual(bgr, sigma):
    """I_h = I - GaussianBlur(I)。空域一阶高通,快且与频域结果高度一致。"""
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return bgr.astype(np.float32) - blur.astype(np.float32)


def high_freq_fft(bgr, cutoff):
    """FFT 高通:保留高于 cutoff 的频率分量(按到频谱中心的归一化距离)。"""
    h, w = bgr.shape[:2]
    out = np.empty_like(bgr, dtype=np.float32)
    # 到频谱中心的归一化距离场,只需算一次
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    hp = (dist > cutoff).astype(np.float32)          # 高通掩码
    for c in range(3):
        f = np.fft.fftshift(np.fft.fft2(bgr[:, :, c].astype(np.float32)))
        out[:, :, c] = np.real(np.fft.ifft2(np.fft.ifftshift(f * hp)))
    return out


def enhance(bgr, lam, method='residual', sigma=1.0, cutoff=0.15):
    """返回增强后的三通道图。lam=0 时输出与输入一致(自检用)。"""
    hf = (high_freq_residual(bgr, sigma) if method == 'residual'
          else high_freq_fft(bgr, cutoff))
    out = bgr.astype(np.float32) + lam * hf
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description='FFT / 高频增强')
    ap.add_argument('--src', required=True, help='输入图片或目录')
    ap.add_argument('--out', default=None, help='输出目录(默认 _debug/fft_preview)')
    ap.add_argument('--lam', type=float, default=1.0, help='增强强度 λ')
    ap.add_argument('--method', default='residual', choices=['residual', 'fft'])
    ap.add_argument('--sigma', type=float, default=1.0)
    ap.add_argument('--cutoff', type=float, default=0.15)
    ap.add_argument('--preview', action='store_true', help='生成对比拼图')
    ap.add_argument('--n-preview', type=int, default=4)
    args = ap.parse_args()

    if os.path.isdir(args.src):
        files = sorted(f for f in os.listdir(args.src)
                       if f.lower().endswith(('.jpg', '.png', '.jpeg')))
        files = [os.path.join(args.src, f) for f in files]
    else:
        files = [args.src]

    out_dir = args.out or '_debug/fft_preview'
    os.makedirs(out_dir, exist_ok=True)
    print(f'方法={args.method}  λ={args.lam}  '
          f'{"sigma="+str(args.sigma) if args.method=="residual" else "cutoff="+str(args.cutoff)}')
    print(f'输入 {len(files)} 张 -> {out_dir}')

    rows = []
    for i, fp in enumerate(files):
        bgr = cv2.imread(fp)
        if bgr is None:
            continue
        enh = enhance(bgr, args.lam, args.method, args.sigma, args.cutoff)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(fp)), enh,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        if args.preview and len(rows) < args.n_preview:
            hf = (high_freq_residual(bgr, args.sigma) if args.method == 'residual'
                  else high_freq_fft(bgr, args.cutoff))
            hfv = np.clip(np.abs(hf).sum(axis=2), 0, 255).astype(np.uint8)
            hfv = cv2.applyColorMap(hfv, cv2.COLORMAP_JET)
            rows.append([bgr, hfv, enh])
        if (i + 1) % 100 == 0:
            print(f'\r  {i+1}/{len(files)}', end='', flush=True)
    print(f'\r  完成 {len(files)} 张            ')

    if args.preview and rows:
        th = 260
        tiles = []
        for r in rows:
            rs = [cv2.resize(x, (int(x.shape[1]*th/x.shape[0]), th)) for x in r]
            tiles.append(np.hstack(rs))
        wmax = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0,0),(0,wmax-t.shape[1]),(0,0))) for t in tiles]
        grid = np.vstack(tiles)
        for k, lab in enumerate(['ORIGINAL', 'HIGH-FREQ |I_h|', "ENHANCED I'"]):
            cv2.putText(grid, lab, (10 + k*(grid.shape[1]//3), 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        p = os.path.join(out_dir, '_preview.jpg')
        cv2.imwrite(p, grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'对比拼图 {p}  (左:原图 中:高频残差 右:增强图)')


if __name__ == '__main__':
    main()
