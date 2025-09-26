#!/usr/bin/env python3
"""
TMA图像数据框生成器

从TMA图像中按照stripe和range来提取中心点，并计算每个channel的RGB平均值。
生成包含patient_id, image_id, center_x, center_y, image_path和8个marker的RGB均值的数据框。
"""

import os
import re
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
from tqdm import tqdm
from fastparquet import write as fp_write
from collections import defaultdict

def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件并转换为numpy数组。
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        图像的numpy数组，形状为 (H, W, C)
    """
    if image_path.lower().endswith(('.tiff', '.tif')):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    
    if image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    
    return image

def extract_center_points(image_shape: Tuple[int, int], 
                         stripe_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    按照stripe和kernel提取中心点坐标。
    
    Args:
        image_shape: 图像尺寸 (height, width)
        stripe_size: stripe的步长（每次移动的距离）
        kernel_size: kernel的大小（每个中心点周围提取的区域大小）
        
    Returns:
        centers_ys: (Ny,) 行坐标
        centers_xs: (Nx,) 列坐标
    """
    H, W = image_shape
    centers_ys = np.arange(0, H, stripe_size)
    centers_xs = np.arange(0, W, stripe_size)
    centers_x = np.repeat(centers_xs, centers_ys.size)  # (Nx*Ny,) 每个 x 重复 Ny 次
    centers_y = np.tile(centers_ys, centers_xs.size)
    return centers_y, centers_x

def box_means_rgb(image, centers_x, centers_y, K):
    """
    image: (H, W, 3) uint8/float
    centers_x, centers_y: (N,) 中心坐标（像素索引，整型或可转整）
    K: 窗口边长，按半开区间 [x0, x0+K)，边界自动裁剪
    返回: (N, 3) 每个中心的 RGB 均值（只用图内有效像素）
    """
    H, W, C = image.shape
    assert C == 3

    # 建立积分图：先转 float，沿 y、x 做累计，再在顶部与左侧各加一圈 0
    img = image.astype(np.float64)
    ii = img.cumsum(axis=0).cumsum(axis=1)                 # (H, W, 3)
    ii = np.pad(ii, ((1,0),(1,0),(0,0)), mode='constant')  # (H+1, W+1, 3)

    # 以半开区间 [x0, x0+K), [y0, y0+K) 定义窗口；边界裁剪
    r = K // 2
    cx = np.asarray(centers_x, dtype=np.int64)
    cy = np.asarray(centers_y, dtype=np.int64)

    x0 = cx - r
    y0 = cy - r
    x1 = x0 + K
    y1 = y0 + K

    # 裁剪到 [0, W-1]/[0, H-1]
    x0 = np.clip(x0, 0, W-1)
    y0 = np.clip(y0, 0, H-1)
    x1 = np.clip(x1, 0, W-1)
    y1 = np.clip(y1, 0, H-1)

    # 转成积分图坐标（多了前导 0 行/列，所以 +1）
    x0p = x0; x1p = x1 + 1
    y0p = y0; y1p = y1 + 1

    # 利用积分图求每个窗口的 RGB 和：S = A - B - C + D
    # 形状自动广播到 (N, 3)
    S = (ii[y1p, x1p] - ii[y0p, x1p] - ii[y1p, x0p] + ii[y0p, x0p])  # (N, 3)

    # 有效像素数（边界裁剪后）
    areas = (x1 - x0) * (y1 - y0) # (N,)
    # 防零（理论上中心在图内就不会为 0）
    areas = np.maximum(areas, 1)

    means = S / areas[:, None] # (N, 3)
    return means

def get_patient_ids(tma_dir: str) -> List[str]:
    """
    获取所有patient ID列表。
    
    Args:
        tma_dir: TMA目录路径
        
    Returns:
        patient ID列表
    """
    markers = ['HE', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'MHC1', 'PDL1']
    patient_ids = set()
    
    for marker in tqdm(markers, desc="扫描Markers", unit="marker"):
        marker_dir = os.path.join(tma_dir, f"tma_tumorcenter_{marker}")
        if os.path.exists(marker_dir):
            # 获取该marker下的所有图像文件
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
                for image_file in Path(marker_dir).glob(ext):
                    # 从文件名提取patient信息
                    patient_id = f"{image_file.stem.split('_')[-1]}"
                    patient_ids.add(patient_id)
    
    return sorted(list(patient_ids))

def parse_image_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    从文件名解析 patient_id, block, core_x, core_y。

    兼容多种命名：优先用正则，失败则基于下划线/连字符的启发式。
    期望匹配：block<d>、x<d>、y<d>、patient<id>

    Returns:
        (patient_id, block, core_x, core_y)（解析失败则为 None）
    """
    stem = Path(filename).stem
    # 正则优先
    pattern = re.compile(r"block(\w+)_.*?x(\d+).*?y(\d+).*?(patient\w+)", re.IGNORECASE)
    m = pattern.search(stem)
    if m:
        block = m.group(1)
        core_x = int(m.group(2))
        core_y = int(m.group(3))
        patient_id = m.group(4)
        return patient_id, block, core_x, core_y

    # 退化到启发式
    parts = re.split(r"[_\-]+", stem)
    block: Optional[str] = None
    core_x: Optional[int] = None
    core_y: Optional[int] = None
    patient_id: Optional[str] = None
    for p in parts:
        lp = p.lower()
        if lp.startswith('block') and block is None:
            block = p[len('block'):]
        elif (lp.startswith('x') or lp.startswith('col')) and core_x is None:
            num = re.sub(r"\D", "", p)
            if num:
                core_x = int(num)
        elif (lp.startswith('y') or lp.startswith('row')) and core_y is None:
            num = re.sub(r"\D", "", p)
            if num:
                core_y = int(num)
        elif lp.startswith('patient') and patient_id is None:
            patient_id = p
    return patient_id, block, core_x, core_y

def build_images_df(tma_dir: str, markers: List[str]) -> pd.DataFrame:
    """
    扫描 tma_dir 下各 marker 目录，收集图像及其解析后的元数据，构建 DataFrame。

    Returns:
        DataFrame[marker, path, filename, patient_id, block, core_x, core_y]
    """
    records: List[Dict[str, object]] = []
    for marker in markers:
        marker_dir = os.path.join(tma_dir, f"tma_tumorcenter_{marker}")
        if not os.path.isdir(marker_dir):
            continue
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.tif']:
            for image_file in Path(marker_dir).glob(ext):
                patient_id, block, core_x, core_y = parse_image_filename(image_file.name)
                records.append({
                    'marker': marker,
                    'path': str(image_file),
                    'filename': image_file.name,
                    'patient_id': patient_id,
                    'block': block,
                    'core_x': core_x,
                    'core_y': core_y,
                })
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    # 仅保留解析成功的
    df = df.dropna(subset=['patient_id', 'block', 'core_x', 'core_y'])
    if df.empty:
        return df
    # 规范类型
    df['core_x'] = df['core_x'].astype(int)
    df['core_y'] = df['core_y'].astype(int)
    return df

def process_single_image(image_path: str,
                         marker: str,
                         stripe_size: int,
                         kernel_size: int,
                         base_df: pd.DataFrame) -> None:
    """
    处理单张图像：计算所有中心点的 RGB 均值，并批量写入 base_df 对应 marker 列。

    要求 base_df 的 index 为 (center_x, center_y)。
    """
    image = load_image(image_path)
    height, width = image.shape[:2]
    centers_ys, centers_xs = extract_center_points((height, width), stripe_size)
    rgb_means = box_means_rgb(image, centers_xs, centers_ys, kernel_size)

    targets = set(zip(centers_xs, centers_ys))
    mask_idx = base_df.index.isin(targets)

    missing = targets.difference(base_df.index)
    assert len(missing) == 0, f"Missing {len(missing)} center points, such as: {list(missing)[:10]}"

    base_df.loc[mask_idx, f'{marker}_R'] = rgb_means[:, 0]
    base_df.loc[mask_idx, f'{marker}_G'] = rgb_means[:, 1]
    base_df.loc[mask_idx, f'{marker}_B'] = rgb_means[:, 2]
    base_df.loc[mask_idx, f'{marker}_image_path'] = f"{marker}/{Path(image_path).name}"

def append_to_parquet(df: pd.DataFrame, output_file: str) -> None:
    """
    将 df 追加写入到单一 Parquet 文件（fastparquet）。
    若文件不存在则创建；存在则使用 append 追加，避免将历史数据读回内存。
    """
    if not os.path.exists(output_file):
        fp_write(output_file, df, compression='SNAPPY', file_scheme='simple')
    else:
        fp_write(output_file, df, compression='SNAPPY', file_scheme='simple', append=True)

def create_patient_skeleton(patient_id: str, 
                           centers_y: np.ndarray, 
                           centers_x: np.ndarray, 
                           markers: List[str]) -> pd.DataFrame:
    """
    创建patient的基础数据结构（骨架），包含所有坐标和marker列。
    使用(center_xs, center_ys)作为index以提高查找效率。
    
    Args:
        patient_id: 患者ID
        centers_ys: 中心点行坐标
        centers_xs: 中心点列坐标
        markers: marker列表
        
    Returns:
        基础数据框，以(center_xs, center_ys)为index
    """
    n = centers_x.size
    assert centers_y.size == n, "centers_x 和 centers_y 长度必须一致"
    # 创建基础数据
    data = {
        'patient_id': np.full(n, patient_id, dtype=object),
        'center_x': centers_x,
        'center_y': centers_y,
    }
    
    for m in markers:
        data[f'{m}_R'] = 0.0
        data[f'{m}_G'] = 0.0
        data[f'{m}_B'] = 0.0
        data[f'{m}_image_path'] = ''
        
    # 创建DataFrame并使用(center_x, center_y)作为index
    df = pd.DataFrame(data)
    df.set_index(['center_x', 'center_y'], drop=False, inplace=True)
    return df
  
def process_tma_directory(tma_dir: str, 
                         output_file: str,
                         stripe_size: int = 8,
                         kernel_size: int = 8) -> None:
    """
    按 (patient_id, block, core_x, core_y) 分组处理同一 core 的多 marker 图片，
    统一填充到同一 DataFrame，再拼接保存为 parquet。
    """
    markers = ['HE', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'MHC1', 'PDL1']

    print(f"🔍 Processing TMA directory: {tma_dir}")
    print(f"📏 Stripe size: {stripe_size}, Kernel size: {kernel_size}")

    images_df = build_images_df(tma_dir, markers)
    if images_df.empty:
        print("❌ 未找到可用的图像文件")
        return

    groups = images_df.groupby(['patient_id', 'block', 'core_x', 'core_y'], dropna=False)
    print(f"📦 将处理 {groups.ngroups} 个 (patient, block, x, y) 组")

    # 流式写入：每个 group 处理后直接追加写入，避免累计内存
    total_written = 0
    for (patient_id, block, g_core_x, g_core_y), gdf in tqdm(groups, total=groups.ngroups, desc="Groups", unit="group"):
        # 用组内第一张图建立骨架（取其尺寸生成中心点网格）
        first_row = gdf.iloc[0]
        first_image = load_image(first_row['path'])
        H, W = first_image.shape[:2]
        centers_ys, centers_xs = extract_center_points((H, W), stripe_size)

        base_df = create_patient_skeleton(patient_id, centers_ys, centers_xs, markers)
        base_df['block'] = block
        base_df['core_group_x'] = int(g_core_x)
        base_df['core_group_y'] = int(g_core_y)

        # 处理该组内的各 marker 图片
        for row in gdf.sort_values(['marker', 'filename']).itertuples(index=False):
            process_single_image(row.path, row.marker, stripe_size, kernel_size, base_df)
            tqdm.write(f"    [{row.marker}] {Path(row.path).name} done")

        # 设置主 image_path：优先 HE
        if 'HE_image_path' in base_df.columns and base_df['HE_image_path'].str.len().max() > 0:
            base_df['image_path'] = base_df['HE_image_path']
        else:
            avail = [m for m in markers if f'{m}_image_path' in base_df.columns and base_df[f'{m}_image_path'].str.len().max() > 0]
            base_df['image_path'] = base_df[f'{avail[0]}_image_path'] if avail else ''

        out_df = base_df.reset_index(drop=True)
        append_to_parquet(out_df, output_file)
        total_written += len(out_df)
        tqdm.write(f"  ✅ wrote group ({patient_id}, {block}, {g_core_x}, {g_core_y}) rows={len(out_df)} total={total_written}")

    if total_written == 0:
        print("❌ 没有提取到任何数据")
        return

    print(f"\n✅ 数据框已保存到: {output_file}")
    print(f"📊 累计写入 {total_written} 个样本")

# python dataframe_generator.py --tma-dir '/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores' --output 'tma_data.parquet'
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TMA图像数据框生成器')
    parser.add_argument('--tma-dir', type=str, 
                       default='/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores',
                       help='TMA目录路径')
    parser.add_argument('--output', type=str, 
                       default='tma_data.parquet',
                       help='输出文件路径')
    parser.add_argument('--stripe-size', type=int, default=8,
                       help='Stripe大小')
    parser.add_argument('--kernel-size', type=int, default=8,
                       help='Kernel大小')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.tma_dir):
        print(f"❌ TMA目录不存在: {args.tma_dir}")
        return
    
    process_tma_directory(
        tma_dir=args.tma_dir,
        output_file=args.output,
        stripe_size=args.stripe_size,
        kernel_size=args.kernel_size
    )

if __name__ == '__main__':
    main()
