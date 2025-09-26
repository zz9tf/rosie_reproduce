#!/usr/bin/env python3
"""
从Zarr格式直接提取TMA图像position的DataFrame生成器

直接从zarr数组中按照stripe和range来提取中心点，并计算每个channel的RGB平均值。
生成包含patient_id, image_id, center_x, center_y, image_path和8个marker的RGB均值的数据框。

优势：
- 内存效率高，支持分块读取
- 避免重复的图像加载
- 可以并行处理多个marker
- 支持大文件处理
"""

import os
import re
import numpy as np
import pandas as pd
import zarr
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
from tqdm import tqdm
from fastparquet import write as fp_write
import json
import gc
import psutil

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_zarr_metadata(zarr_path: str) -> Dict:
    """
    加载zarr目录的元数据信息
    
    Args:
        zarr_path: zarr目录路径
        
    Returns:
        包含marker信息的字典
    """
    metadata_file = os.path.join(zarr_path, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 如果没有metadata.json，扫描zarr目录
        metadata = {'markers': [], 'marker_info': {}}
        for item in os.listdir(zarr_path):
            item_path = os.path.join(zarr_path, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 尝试打开zarr数组
                try:
                    array = zarr.open(item_path, mode='r')
                    if hasattr(array, 'shape') and len(array.shape) == 4:  # (num_images, H, W, C)
                        metadata['markers'].append(item)
                        metadata['marker_info'][item] = {
                            'num_images': array.shape[0],
                            'image_shape': array.shape[1:4]
                        }
                except:
                    continue
        return metadata

def extract_center_points(image_shape: Tuple[int, int], 
                         stripe_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    按照stripe和kernel提取中心点坐标。
    
    Args:
        image_shape: 图像尺寸 (height, width)
        stripe_size: stripe的步长（每次移动的距离）
        
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

def box_means_rgb_from_zarr(zarr_array: zarr.Array, 
                           image_idx: int,
                           centers_x: np.ndarray, 
                           centers_y: np.ndarray, 
                           K: int) -> np.ndarray:
    """
    从zarr数组中提取指定图像的RGB均值
    
    Args:
        zarr_array: zarr数组对象
        image_idx: 图像索引
        centers_x, centers_y: (N,) 中心坐标
        K: 窗口边长
        
    Returns:
        (N, 3) 每个中心的 RGB 均值
    """
    # 读取单张图像
    image = zarr_array[image_idx]  # (H, W, 3)
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
    
    # 确保RGB值在合理范围内
    means = np.clip(means, 0, 255)
    
    return means

def parse_image_filename_from_zarr(filename: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """
    从zarr中的文件名解析 patient_id, block, core_x, core_y。
    
    Args:
        filename: 图像文件名
        
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

def build_zarr_images_df(zarr_path: str, markers: List[str]) -> pd.DataFrame:
    """
    扫描 zarr 目录下各 marker，收集图像及其解析后的元数据，构建 DataFrame。
    
    Args:
        zarr_path: zarr目录路径
        markers: marker列表
        
    Returns:
        DataFrame[marker, zarr_path, image_idx, filename, patient_id, block, core_x, core_y]
    """
    records: List[Dict[str, object]] = []
    
    # 加载元数据
    metadata = load_zarr_metadata(zarr_path)
    
    for marker in markers:
        marker_zarr_path = os.path.join(zarr_path, marker)
        if not os.path.exists(marker_zarr_path):
            print(f"⚠️ 警告: 未找到marker {marker} 的zarr数据")
            continue
            
        try:
            # 打开zarr数组
            zarr_array = zarr.open(marker_zarr_path, mode='r')
            num_images = zarr_array.shape[0]
            
            # 从元数据获取文件名列表
            if marker in metadata.get('marker_info', {}):
                filenames = metadata['marker_info'][marker].get('files', [])
            else:
                # 如果没有文件名信息，生成默认名称
                filenames = [f"{marker}_image_{i:04d}.png" for i in range(num_images)]
            
            # 确保文件名数量匹配
            if len(filenames) < num_images:
                filenames.extend([f"{marker}_image_{i:04d}.png" for i in range(len(filenames), num_images)])
            
            for image_idx in range(num_images):
                filename = filenames[image_idx] if image_idx < len(filenames) else f"{marker}_image_{image_idx:04d}.png"
                patient_id, block, core_x, core_y = parse_image_filename_from_zarr(filename)
                
                records.append({
                    'marker': marker,
                    'zarr_path': marker_zarr_path,
                    'image_idx': image_idx,
                    'filename': filename,
                    'patient_id': patient_id,
                    'block': block,
                    'core_x': core_x,
                    'core_y': core_y,
                })
                
        except Exception as e:
            print(f"❌ 处理marker {marker} 时出错: {e}")
            continue
    
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
    df['image_idx'] = df['image_idx'].astype(int)
    
    return df

def process_single_zarr_image(zarr_path: str,
                             image_idx: int,
                             marker: str,
                             stripe_size: int,
                             kernel_size: int,
                             base_df: pd.DataFrame) -> None:
    """
    处理zarr中的单张图像：计算所有中心点的 RGB 均值，并批量写入 base_df 对应 marker 列。
    
    Args:
        zarr_path: zarr数组路径
        image_idx: 图像索引
        marker: marker名称
        stripe_size: stripe大小
        kernel_size: kernel大小
        base_df: 基础数据框，要求index为(center_x, center_y)
    """
    try:
        # 打开zarr数组
        zarr_array = zarr.open(zarr_path, mode='r')
        
        # 获取图像尺寸
        height, width = zarr_array.shape[1], zarr_array.shape[2]
        centers_ys, centers_xs = extract_center_points((height, width), stripe_size)
        
        # 计算RGB均值
        rgb_means = box_means_rgb_from_zarr(zarr_array, image_idx, centers_xs, centers_ys, kernel_size)
        
        # 更新base_df
        targets = set(zip(centers_xs, centers_ys))
        mask_idx = base_df.index.isin(targets)
        
        missing = targets.difference(base_df.index)
        if len(missing) > 0:
            print(f"⚠️ 警告: 缺少 {len(missing)} 个中心点，例如: {list(missing)[:5]}")
        
        base_df.loc[mask_idx, f'{marker}_R'] = rgb_means[:, 0]
        base_df.loc[mask_idx, f'{marker}_G'] = rgb_means[:, 1]
        base_df.loc[mask_idx, f'{marker}_B'] = rgb_means[:, 2]
        base_df.loc[mask_idx, f'{marker}_image_path'] = f"{marker}/image_{image_idx:04d}.png"
        # 🔍 添加追溯信息
        base_df.loc[mask_idx, f'{marker}_zarr_path'] = zarr_path
        base_df.loc[mask_idx, f'{marker}_image_idx'] = image_idx
        
    except Exception as e:
        print(f"❌ 处理 {marker} 图像 {image_idx} 时出错: {e}")

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
        # 🔍 添加追溯信息字段
        data[f'{m}_zarr_path'] = ''
        data[f'{m}_image_idx'] = -1  # -1表示未设置
        
    # 创建DataFrame并使用(center_x, center_y)作为index
    df = pd.DataFrame(data)
    df.set_index(['center_x', 'center_y'], drop=False, inplace=True)
    return df

def append_to_parquet(df: pd.DataFrame, output_file: str) -> None:
    """
    将 df 追加写入到单一 Parquet 文件（fastparquet）。
    若文件不存在则创建；存在则使用 append 追加，避免将历史数据读回内存。
    """
    if not os.path.exists(output_file):
        fp_write(output_file, df, compression='SNAPPY', file_scheme='simple')
    else:
        fp_write(output_file, df, compression='SNAPPY', file_scheme='simple', append=True)

def traceback_to_zarr_location(df_row: pd.Series, marker: str) -> Dict[str, any]:
    """
    从DataFrame的一行数据追溯回zarr中的具体位置
    
    Args:
        df_row: DataFrame中的一行数据
        marker: 要追溯的marker名称
        
    Returns:
        包含zarr位置信息的字典
    """
    zarr_path = df_row.get(f'{marker}_zarr_path', '')
    image_idx = df_row.get(f'{marker}_image_idx', -1)
    
    if not zarr_path or image_idx == -1:
        return {
            'zarr_path': '',
            'image_idx': -1,
            'center_x': df_row.get('center_x', -1),
            'center_y': df_row.get('center_y', -1),
            'patient_id': df_row.get('patient_id', ''),
            'block': df_row.get('block', ''),
            'core_x': df_row.get('core_group_x', -1),
            'core_y': df_row.get('core_group_y', -1),
            'error': f'Marker {marker} 的追溯信息不完整'
        }
    
    return {
        'zarr_path': zarr_path,
        'image_idx': int(image_idx),
        'center_x': int(df_row.get('center_x', -1)),
        'center_y': int(df_row.get('center_y', -1)),
        'patient_id': df_row.get('patient_id', ''),
        'block': df_row.get('block', ''),
        'core_x': int(df_row.get('core_group_x', -1)),
        'core_y': int(df_row.get('core_group_y', -1))
    }

def extract_patch_from_zarr(traceback_info: Dict[str, any], 
                           patch_size: int = 32) -> Optional[np.ndarray]:
    """
    根据追溯信息从zarr中提取指定位置的图像块
    
    Args:
        traceback_info: 追溯信息字典
        patch_size: 提取的图像块大小
        
    Returns:
        提取的图像块 (patch_size, patch_size, 3) 或 None
    """
    zarr_path = traceback_info.get('zarr_path', '')
    image_idx = traceback_info.get('image_idx', -1)
    center_x = traceback_info.get('center_x', -1)
    center_y = traceback_info.get('center_y', -1)
    
    if not zarr_path or image_idx == -1 or center_x == -1 or center_y == -1:
        print(f"❌ 追溯信息不完整: {traceback_info}")
        return None
    
    try:
        # 打开zarr数组
        zarr_array = zarr.open(zarr_path, mode='r')
        
        # 读取完整图像
        image = zarr_array[image_idx]  # (H, W, 3)
        H, W, C = image.shape
        
        # 计算patch边界
        half_size = patch_size // 2
        x0 = max(0, center_x - half_size)
        y0 = max(0, center_y - half_size)
        x1 = min(W, center_x + half_size)
        y1 = min(H, center_y + half_size)
        
        # 提取patch
        patch = image[y0:y1, x0:x1, :]
        
        # 如果patch小于指定大小，进行padding
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            padded_patch = np.zeros((patch_size, patch_size, C), dtype=patch.dtype)
            padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded_patch
        
        return patch
        
    except Exception as e:
        print(f"❌ 从zarr提取patch时出错: {e}")
        return None

def process_zarr_directory(zarr_path: str, 
                          output_file: str,
                          stripe_size: int = 8,
                          kernel_size: int = 8,
                          max_patients: Optional[int] = None) -> None:
    """
    从zarr目录按 (patient_id, block, core_x, core_y) 分组处理同一 core 的多 marker 图片，
    统一填充到同一 DataFrame，再拼接保存为 parquet。
    
    Args:
        zarr_path: zarr目录路径
        output_file: 输出parquet文件路径
        stripe_size: stripe大小
        kernel_size: kernel大小
        max_patients: 最大处理病人数量，None表示处理所有病人
    """
    markers = ['HE', 'CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'MHC1', 'PDL1']

    print(f"🔍 Processing Zarr directory: {zarr_path}")
    print(f"📏 Stripe size: {stripe_size}, Kernel size: {kernel_size}")
    if max_patients is not None:
        print(f"👥 限制处理病人数量: {max_patients}")
    print(f"💾 初始内存使用: {get_memory_usage():.1f} MB")

    # 构建图像数据框
    images_df = build_zarr_images_df(zarr_path, markers)
    if images_df.empty:
        print("❌ 未找到可用的zarr图像数据")
        return

    # 按组分组
    groups = images_df.groupby(['patient_id', 'block', 'core_x', 'core_y'], dropna=False)
    
    # 如果设置了最大病人数量，则限制处理的病人
    if max_patients is not None:
        # 获取所有唯一的病人ID
        unique_patients = images_df['patient_id'].unique()
        if len(unique_patients) > max_patients:
            # 选择前max_patients个病人
            selected_patients = unique_patients[:max_patients]
            # 过滤数据框，只保留选中的病人
            images_df = images_df[images_df['patient_id'].isin(selected_patients)]
            # 重新分组
            groups = images_df.groupby(['patient_id', 'block', 'core_x', 'core_y'], dropna=False)
            print(f"👥 从 {len(unique_patients)} 个病人中选择前 {max_patients} 个病人")
            print(f"📦 将处理 {groups.ngroups} 个 (patient, block, x, y) 组")
        else:
            print(f"📦 将处理 {groups.ngroups} 个 (patient, block, x, y) 组 (所有 {len(unique_patients)} 个病人)")
    else:
        print(f"📦 将处理 {groups.ngroups} 个 (patient, block, x, y) 组")

    # 流式写入：每个 group 处理后直接追加写入，避免累计内存
    total_written = 0
    for (patient_id, block, g_core_x, g_core_y), gdf in tqdm(groups, total=groups.ngroups, desc="Groups", unit="group"):
        # 用组内第一张图建立骨架（取其尺寸生成中心点网格）
        first_row = gdf.iloc[0]
        
        # 从zarr获取图像尺寸
        try:
            zarr_array = zarr.open(first_row['zarr_path'], mode='r')
            H, W = zarr_array.shape[1], zarr_array.shape[2]
        except Exception as e:
            print(f"❌ 无法打开zarr数组 {first_row['zarr_path']}: {e}")
            continue
            
        centers_ys, centers_xs = extract_center_points((H, W), stripe_size)
        base_df = create_patient_skeleton(patient_id, centers_ys, centers_xs, markers)
        base_df['block'] = block
        base_df['core_group_x'] = int(g_core_x)
        base_df['core_group_y'] = int(g_core_y)

        # 处理该组内的各 marker 图片
        for row in gdf.sort_values(['marker', 'image_idx']).itertuples(index=False):
            process_single_zarr_image(
                row.zarr_path, 
                row.image_idx, 
                row.marker, 
                stripe_size, 
                kernel_size, 
                base_df
            )
            tqdm.write(f"    [{row.marker}] image_{row.image_idx:04d} done")

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
        
        # 定期垃圾回收
        if total_written % 1000 == 0:
            gc.collect()

    if total_written == 0:
        print("❌ 没有提取到任何数据")
        return

    print(f"\n✅ 数据框已保存到: {output_file}")
    print(f"📊 累计写入 {total_written} 个样本")
    print(f"💾 最终内存使用: {get_memory_usage():.1f} MB")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从Zarr格式提取TMA图像position的DataFrame生成器')
    parser.add_argument('--zarr-dir', type=str, 
                       default='/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores_Zarr',
                       help='Zarr目录路径')
    parser.add_argument('--output', type=str,
                       default='tma_data_from_zarr.parquet',
                       help='输出文件路径')
    parser.add_argument('--stripe-size', type=int, default=8,
                       help='Stripe大小')
    parser.add_argument('--kernel-size', type=int, default=8,
                       help='Kernel大小')
    parser.add_argument('--max-patients', type=int, default=None,
                       help='最大处理病人数量，None表示处理所有病人')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_dir):
        print(f"❌ Zarr目录不存在: {args.zarr_dir}")
        return
    
    process_zarr_directory(
        zarr_path=args.zarr_dir,
        output_file=args.output,
        stripe_size=args.stripe_size,
        kernel_size=args.kernel_size,
        max_patients=args.max_patients
    )

if __name__ == '__main__':
    main()
