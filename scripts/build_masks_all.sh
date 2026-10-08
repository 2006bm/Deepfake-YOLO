#!/usr/bin/env bash
# 阶段2b: 逐方法处理 mask —— 下载 → 转 YOLO 标签 → 抽样验证 IoU → 删除 mask。
#
# 为什么"边下边转边删":
#   全部 mask 共 38.6GB(FaceSwap 一类就 15.9GB),而 D: 盘只有 ~30GB 可用。
#   逐个方法处理可把峰值压到"最大单方法 15.9GB",转完即删。
#
# 为什么验证必须在删除之前:
#   mask 一旦删除就无法事后算 IoU。早期版本先删后想验证,导致只能重新下载
#   采样 mask 补救。现在把 IoU 抽样验证插在删除之前,验证不过就保留 mask。
#
# 幂等性:
#   已处理完的方法(标签数达标)会整体跳过,不会重复下载已删除的 mask。
#
# 可随时中断,重跑会从断点继续。
set -u
cd "$(dirname "$0")/.."
PY="C:/Users/边妙/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONIOENCODING=utf-8

EXPECTED=3000      # 每方法标签数: train 2160 + val 420 + test 420
TOLERANCE=5        # 容许少量缺席。已知实例:NeuralTextures_567_606 的视频 337 帧、
                   # mask 340 帧(build_dataset 断言拒绝处理),少 3 张。
                   # 占比 0.013%,对训练无影响;为它放宽核心断言反而风险更大。
IOU_SAMPLE=250     # 每方法 IoU 抽样数
FIRST=0            # 真实样本是否已生成(Deepfakes/Face2Face 已完成 -> 已生成)

count_labels() {
  local m=$1 n=0
  for s in train val test; do
    local c
    c=$(ls "data/dataset_v1/labels/$s" 2>/dev/null | grep -c "^${m}_" || true)
    n=$((n + c))
  done
  echo "$n"
}

for m in Deepfakes Face2Face FaceSwap NeuralTextures; do
  echo "############################################################"
  echo "########## $m ##########"
  echo "############################################################"

  have=$(count_labels "$m")
  if [ "$have" -ge $((EXPECTED - TOLERANCE)) ]; then
    echo "--- 已完成($have/$EXPECTED 标签),整体跳过 ---"
    [ "$have" -gt 0 ] && FIRST=1
    echo
    continue
  fi

  echo "--- 1) 下载 $m 的 mask (已有 $have/$EXPECTED 标签) ---"
  if ! "$PY" preprocessing/download.py --type masks --dataset "$m" --workers 8; then
    echo "!!! $m mask 下载失败,中止"; exit 1
  fi

  echo "--- 2) 构建 YOLO-Seg 标签 ---"
  BUILD_OK=1
  for s in train val test; do
    EXTRA=""
    [ "$FIRST" -eq 1 ] && EXTRA="--no-real"
    if ! "$PY" preprocessing/build_dataset.py --split "$s" --methods "$m" \
        --frames-per-fake 3 --frames-per-real 11 \
        --threshold 0 --workers 8 $EXTRA; then
      BUILD_OK=0
    fi
  done
  FIRST=1

  echo "--- 3) 数量校验 ---"
  got=$(count_labels "$m")
  if [ "$BUILD_OK" -ne 1 ] || [ "$got" -lt $((EXPECTED - TOLERANCE)) ]; then
    echo "!!! $m 构建失败或标签数异常($got/$EXPECTED),保留 mask 不删除"
    exit 1
  fi
  echo "    标签数 $got/$EXPECTED ✓"

  echo "--- 4) 抽样验证 IoU (删除前,此时 mask 还在) ---"
  if ! "$PY" preprocessing/verify_labels.py --splits train,val,test \
      --methods "$m" --n "$IOU_SAMPLE" --montage "_debug/iou_${m}.jpg" \
      --montage-rows 6; then
    echo "!!! $m 的 IoU 验证未通过,保留 mask 不删除,请人工检查"
    exit 1
  fi

  echo "--- 5) 删除 $m 的 mask 视频,释放空间 ---"
  rm -rf "data/raw/manipulated_sequences/$m/masks"
  df -h /d | tail -1
  echo
done

echo "########## mask 全部处理完成 ##########"
echo "生成 metadata.csv ..."
"$PY" preprocessing/make_metadata.py || echo "(metadata 生成失败,可稍后单独重跑)"
echo
df -h /d | tail -1
