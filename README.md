# Deepfake-YOLO

**基于 YOLOv8-Seg 的 Deepfake 人脸伪造区域定位与鲁棒性研究**

Deepfake Facial Manipulation Localization and Robustness Analysis Based on
YOLOv8 Segmentation

> 题目刻意不写"频域增强":FFT 是否有效尚未验证。若实验证明有效,升级为
> "基于 YOLOv8-Seg 与频域增强的...";若无效,本题仍然成立,不影响交付。

## 快速开始

```bash
git clone https://github.com/2006bm/Deepfake-YOLO.git
cd Deepfake-YOLO
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**然后必须先自行申请数据集(见下节)** —— 本仓库不含任何数据。

## 数据申请(必读)

本仓库**不包含任何 FF++ 数据**,也不会包含。原因有两个:数据集体积达
~13GB(视频 9.8GB + 图片数据集 3.2GB),远超 GitHub 限制;更重要的是
**FF++ 的服务条款禁止再分发** —— 它要求填写申请表、审批通过后才提供下载脚本,
这一设计的目的就是限制传播。把数据传到公开仓库既违反条款,也会让你的账号
和后续引用出问题。

| 数据集 | 申请入口 | 用途 |
|---|---|---|
| FaceForensics++ | https://github.com/ondyari/FaceForensics | 主数据集(训练 + 定位 + 鲁棒性) |
| Celeb-DF v2 | https://github.com/yuezunli/celeb-deepfakeforensics | 可选:跨数据集泛化 |

申请通过后,官方会给你下载脚本。**本项目不使用官方脚本** ——
它是单线程且无断点续传,中断后单个文件重下,不适合 48GB / 数十小时级别的下载。
改用本仓库的 `preprocessing/download.py`(多线程 + 断点续传 + 字节数校验)。

## 项目目标

不是"输入图片→判断真假",而是进一步**定位被篡改的人脸区域**:

```
输入人脸图像 → 判断是否 Deepfake → 若是,输出伪造区域的 Mask
```

为此采用 YOLO **Segmentation**(而非 Detection/Classification),单类别
`0 = fake_region`。

## 技术路线

```
FaceForensics++
   ├── c23 视频 (训练 + 内部测试)
   ├── c40 视频 (压缩鲁棒性测试)
   └── mask 视频 (定位监督信号)
        ↓  抽帧(视频与 mask 用同一组帧下标)
        ↓  mask → YOLO-Seg polygon 标签
        ↓  官方视频级划分 720/140/140
   YOLOv8n-seg  小规模验证(仅工程验证,不进论文)
        ↓
   YOLOv8s-seg  Baseline  (640×640, batch 8, AMP)
        ↓
   FFT 高频增强预处理 → YOLOv8s-seg(可选创新,实验2验证)
        ↓
   ┌────────────┼────────────┐
   定位实验    压缩鲁棒性    跨方法/跨数据集泛化
   mAP/IoU/Dice  c23→c40     FF++→Celeb-DF v2
```

## 目录结构

```
Deepfake-YOLO/
├── preprocessing/          数据管线
│   ├── download.py         多线程断点续传下载器(替代官方单线程脚本)
│   ├── extract_frames.py   抽帧(视频与 mask 共用帧下标)
│   ├── mask_convert.py     mask → YOLO polygon
│   ├── build_dataset.py    端到端构建数据集
│   └── verify_labels.py    标签↔mask IoU 验证 + 可视化抽查
├── scripts/
│   ├── download_videos_all.sh   阶段2a:全量视频
│   └── build_masks_all.sh       阶段2b:逐方法 mask 边下边转边删
├── configs/
│   ├── baseline.yaml       实验1 基线
│   └── fft.yaml            实验2 频域增强
├── data/
│   ├── splits/             官方 720/140/140 划分(train/val/test.json)
│   ├── raw/                下载的原始视频与 mask(可删)
│   └── dataset_v1/       YOLO 训练数据 + dataset.yaml
└── train/ evaluate/ models/    (待填)
```

## 数据管线用法

```bash
PY="C:/Users/边妙/AppData/Local/Programs/Python/Python313/python.exe"

# 阶段2a:下载全部视频(~10GB)
bash scripts/download_videos_all.sh

# 阶段2b:逐方法下载 mask → 转标签 → 删除 mask(~38.6GB,峰值受控)
bash scripts/build_masks_all.sh
```

两个脚本都可随时 Ctrl-C,重跑自动跳过已完成文件。

## 交付流程:每次拿到数据必须先做的三步

下载"完成"不等于文件可用。本项目曾出现 5 个 mp4 尾部被截断
(丢 moov 索引)却仍被当作成功的情况,直到训练阶段才暴露。
因此下载后、处理前必须执行完整性扫描:

```bash
# 1) 完整性扫描 —— 找出截断/损坏的 mp4 并重下
$PY preprocessing/check_integrity.py
# 损坏文件清单写入 _debug/bad_files.txt,删除后重跑 download.py 即可补齐

# 2) 数量核对 —— images/labels 配对 + 与理论期望值对照
$PY preprocessing/verify_labels.py --counts-only

# 3) 标签正确性 —— 抽样算 polygon↔mask 的 IoU + 质心偏移
$PY preprocessing/verify_labels.py --n 250 --download-missing

# 4) 三方一致性 —— 文件系统 / metadata.csv / verification_report.json
$PY preprocessing/check_consistency.py
```

第 3 步的 `--download-missing` 是必要的:mask 在处理后会被删除以释放磁盘
(全部 mask 38.6GB > 可用磁盘),所以要验证 IoU 只能按需重下被抽中的那几十个
mask(约 40 个/方法 ≈ 1.6GB)。

### 标签验证的判据:为什么不能只看 IoU

验证用**两项独立判定**,缺一不可:

| 判据 | 阈值 | 作用 |
|---|---|---|
| 质心偏移 | < 1% | 判**正确性**。坐标错位(Y 轴翻转、归一化用错宽高、帧下标错位)会让质心偏移 10%~100% |
| IoU 均值 | > 0.95 | 判**边界保真度**。多边形对 mask 轮廓的拟合程度 |

只用 IoU 会误判:软边灰度掩码(Face2Face / FaceSwap / NeuralTextures)的边界
本身是模糊渐变,多边形拟合必然损失边界细节,实测上限约 0.978~0.982;
而二值的 Deepfakes 可轻松达到 0.99。若统一用 0.98 卡,软掩码会被错误地判定
为"不合格"——而质心偏移(实测 0.176%)恰恰证明它完全对齐。

期望结果:

| 划分 | images | labels | real | 每类伪造 |
|---|---|---|---|---|
| train | 16560 | 16560 | 7920 | 2160 |
| val | 3220 | 3220 | 1540 | 420 |
| test | 3220 | 3220 | 1540 | 420 |
| **合计** | **23000** | **23000** | 11000 | 12000 |

## 三个必须知道的坑(都已在代码中处理)

### 1. mask 必须与**输出的伪造视频**配对,不能与原始源视频配对

Face2Face / NeuralTextures 的输出视频跟随**目标**视频的长度与分辨率:
`585_599.mp4` 是 398 帧,而源视频 `585.mp4` 只有 359 帧。Deepfakes 则跟随源视频。
但 **mask 恒等于输出视频**。按源视频读 mask 会导致整批标签错位。
`build_dataset.py` 已对二者做元数据断言。

### 2. 四类 mask 的格式与语义都不一致

| 方法 | 灰度级 | 非零占比 | 性质 |
|---|---|---|---|
| Deepfakes | 2 | 3.3% | **真二值**篡改区域 |
| Face2Face | 228 | 4.0% | 软边灰度 |
| NeuralTextures | 228 | 4.0% | **与 Face2Face 逐像素完全相同** |
| FaceSwap | 157 | 4.3% | 软边灰度,三通道不一致 |

Face2Face 与 NeuralTextures 共用同一套人脸区域掩码。因此**跨方法直接比较
IoU/mAP 不完全公平**,论文需设一节"GT 语义差异"讨论,并做阈值(0 vs 127)
敏感性消融。定位任务仍是四类混训单类别。

### 3. 下载速度波动可达 10 倍以上

实测同一批文件在 300 KB/s ~ 4.6 MB/s 之间波动,官方三台服务器只有
`kaldir.vc.in.tum.de` 可达。估时不要把单次测速当成稳定值。

## 复现要求

记录并固定:随机种子、Python / PyTorch / CUDA / Ultralytics 版本、训练参数、
官方 split、抽帧规则、Git commit。换机器后应能仅靠
`pip install -r requirements.txt` + 两个下载脚本重建全流程。

## 标签验证

每次重建数据集后必须跑一次,确认无错位:

```bash
$PY preprocessing/verify_labels.py --split val --n 20
```

健康指标:**polygon 与 mask 的 IoU 应 > 0.98**(实测均值 0.9923)。
若明显低于此值,说明抽帧下标错位或坐标归一化有问题,不要继续训练。

### smoke 测试集(不要删)

`data/splits/smoke.json` 与 `data/dataset_v1/{images,labels}/smoke/`
是一套 52 张的**管线单元测试集**,不是垃圾数据。任何改动(FFT 增强、数据增强、
标签转换、换 YOLO 版本)之后,先跑它,10 秒内就能发现标签错位或读取失败。

```bash
$PY preprocessing/verify_labels.py --split smoke --n 52
```

重新生成:

```bash
$PY preprocessing/build_dataset.py --split smoke \
    --methods Deepfakes,Face2Face,NeuralTextures \
    --frames-per-fake 3 --frames-per-real 4
```


## 训练

```bash
# 冒烟:确认流程能跑通(约 1000 张)
$PY -c "
from ultralytics import YOLO
YOLO('yolov8s-seg.pt').train(data='data/dataset_v1/dataset.yaml',
    epochs=10, imgsz=640, batch=8, name='smoke')"

# 正式基线
yolo segment train cfg=configs/baseline.yaml
```

配置见 `configs/baseline.yaml`(Baseline)与 `configs/fft.yaml`(频域增强)。
两份配置**除输入图像外全部参数相同**,以保证对比实验的唯一变量是输入。

### 评价指标:必须同时报告两类

YOLO-Seg 输出的是 polygon。只报 mAP 会被问"你在做检测还是定位",
因此论文需同时给出:

| 类别 | 指标 | 回答的问题 |
|---|---|---|
| 检测 | mAP50、mAP50-95 | 能不能找到伪造区域 |
| 定位 | Mask IoU、Dice | 找得准不准(边界贴不贴合) |
| 轻量性 | Params、FLOPs、FPS | 模型能不能实际部署 |

另可基于 `metadata.csv` 的 `mask_area_ratio` 做**分档误差分析**
(<5% / 5-20% / ≥20% 各档的 Dice/IoU),用于回答"小面积伪造是否更难定位"。

## 当前状态

- [x] 项目结构、下载器、数据管线、标签验证、一致性校验
- [x] 全部视频下载(10,000 个 / 9.8 GB,完整性扫描通过)
- [x] GPU 环境(torch 2.6.0+cu124 / CUDA 可用 / RTX 4060 8GB)
- [x] Deepfakes / Face2Face / FaceSwap 标签与验证
- [ ] NeuralTextures 标签(进行中)
- [ ] dataset_v1 冻结
- [ ] YOLOv8s-seg baseline 训练
- [ ] 频域增强对比实验

## 许可

代码 MIT(见 `LICENSE`)。**数据集不在许可范围内,本仓库不分发任何 FF++ 数据。**
