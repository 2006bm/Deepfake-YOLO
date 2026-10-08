#!/usr/bin/env python
"""FF++ mask → YOLO-Seg polygon 标签转换。

FF++ 四类 mask 的实际格式差异很大(实测):
  Deepfakes       : 严格二值(0/255),非零占比 ~3.3%
  Face2Face       : 软边灰度,228 级,范围 0~254,非零 ~4.0%
  NeuralTextures  : 与 Face2Face **逐像素完全相同**(同一套人脸区域掩码)
  FaceSwap        : 软边灰度,三通道不一致(存在色度编码痕迹),非零 ~4.3%

因此二值化阈值是一个真实的建模选择,不是无所谓的细节:
  --threshold 0   保留软边全部范围(默认,标签更完整,但含较淡的边缘光晕)
  --threshold 127 只取核心区域(F2F/NT/FS 会明显收缩,Deepfakes 不受影响)
两套阈值都会生成,便于后续做敏感性分析。
"""
import argparse
import os
import sys

import cv2
import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        _s.reconfigure(encoding='utf-8', errors='replace')


def to_gray(mask_bgr, mode='max'):
    """把 mask 帧转成单通道。mode: gray | max | any

    默认 max:取三通道最大值。FaceSwap 的 mask 三通道不一致,用 max 可以
    完整捕获任一通道上的标记,避免 gray 平均后边缘被削弱。
    """
    if mask_bgr.ndim == 2:
        return mask_bgr
    if mode == 'gray':
        return cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    return mask_bgr.max(axis=2)


def mask_to_polygons(mask_bgr, threshold=0, min_area_px=100, approx_eps=0.002,
                     morph=True, channel_mode='max'):
    """把 mask 帧转成归一化多边形列表 [[(x,y), ...], ...]。

    返回的坐标已除以图像宽高,落在 [0,1],可直接写入 YOLO 标签文件。
    """
    g = to_gray(mask_bgr, channel_mode)
    h, w = g.shape[:2]
    binary = (g > threshold).astype(np.uint8)

    if morph:
        # 先开后闭:去掉椒盐噪点,再填掉区域内部的针孔
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = approx_eps * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:           # YOLO 多边形至少 3 个点
            continue
        norm = approx.astype(np.float64) / np.array([w, h], dtype=np.float64)
        np.clip(norm, 0.0, 1.0, out=norm)
        polys.append(norm)
    return polys


def polygons_to_label(polys, class_id=0):
    """拼成 YOLO-Seg 标签文本,每行一个多边形。"""
    lines = []
    for p in polys:
        coords = ' '.join(f'{v:.6f}' for v in p.reshape(-1))
        lines.append(f'{class_id} {coords}')
    return '\n'.join(lines) + ('\n' if lines else '')


def draw_polygons(image, polys, color=(0, 0, 255), thickness=2):
    """把多边形画回图上,用于人工核对标签是否与 mask 对齐。"""
    vis = image.copy()
    h, w = vis.shape[:2]
    for p in polys:
        pts = (p * np.array([w, h])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, color, thickness)
    return vis


def main():
    ap = argparse.ArgumentParser(description='FF++ mask → YOLO polygon 转换')
    ap.add_argument('mask_video')
    ap.add_argument('--threshold', type=int, default=0)
    ap.add_argument('--min-area', type=int, default=100, help='最小轮廓面积(像素)')
    ap.add_argument('--approx-eps', type=float, default=0.002)
    ap.add_argument('--channel-mode', default='max',
                    choices=['gray', 'max'])
    ap.add_argument('--frames', type=int, default=5, help='检查多少帧')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.mask_video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{os.path.basename(args.mask_video)}: {n} 帧')
    step = max(1, n // args.frames)
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            break
        polys = mask_to_polygons(fr, args.threshold, args.min_area,
                                 args.approx_eps, channel_mode=args.channel_mode)
        npts = sum(len(p) for p in polys)
        print(f'  帧{i:4d}: {len(polys)} 个多边形, 共 {npts} 个点')
    cap.release()


if __name__ == '__main__':
    main()
