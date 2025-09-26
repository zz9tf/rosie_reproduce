#! /bin/bash

eval "$(conda shell.bash hook)"

conda activate patho-sam-cu113
conda env list

# 输入与输出路径
INPUT_IMG="/home/zheng/zheng/rosie_reproduce/cell_extraction/qupath/extract_png/tma_cores_pngs/TumorCenter_HE_block1_1-1_circular.png"
OUT_DIR="/home/zheng/zheng/rosie_reproduce/cell_extraction/patho"
OUT_TIF="$OUT_DIR/TumorCenter_HE_block1_1-1_circular_segmentation.tif"

# 运行 PathoSAM 自动分割，直接生成实例标签到 patho/ 目录
patho_sam.automatic_segmentation \
  -i "$INPUT_IMG" \
  -o "$OUT_TIF" \
  -m vit_b_histopathology \
  --output_choice instances \
  --tile_shape 512 512 \
  --halo 64 64 \
  -d cuda