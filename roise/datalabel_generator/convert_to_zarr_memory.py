#!/usr/bin/env python3
"""
内存安全的 Zarr 转换脚本

这个脚本使用单张图像处理的方式，避免内存爆炸问题。
每次只处理一张图像，立即写入 Zarr，然后释放内存。
"""

import os
import numpy as np
import zarr
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Optional
import argparse
from tqdm import tqdm
import json
import gc
import psutil

def get_memory_usage():
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def load_image_safe(image_path: str) -> Optional[np.ndarray]:
    """安全加载图像，如果失败返回 None"""
    try:
        if image_path.lower().endswith(('.tiff', '.tif')):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
        
        if image is None:
            return None
        
        return image
    except Exception as e:
        print(f"⚠️ 无法加载图像 {image_path}: {e}")
        return None

def get_image_info_safe(image_path: str) -> Optional[Tuple[int, int, int]]:
    """安全获取图像信息"""
    image = load_image_safe(image_path)
    if image is not None:
        return image.shape
    return None

def scan_image_directory(image_dir: str) -> Dict[str, List[str]]:
    """扫描图像目录，获取所有图像文件"""
    image_files = {}
    
    if not os.path.exists(image_dir):
        print(f"❌ 目录不存在: {image_dir}")
        return image_files
    
    for item in os.listdir(image_dir):
        item_path = os.path.join(image_dir, item)
        if os.path.isdir(item_path):
            if item.startswith('tma_tumorcenter_'):
                marker_name = item.replace('tma_tumorcenter_', '')
            else:
                marker_name = item
            
            image_list = []
            for file in os.listdir(item_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
                    image_list.append(os.path.join(item_path, file))
            
            if image_list:
                image_files[marker_name] = sorted(image_list)
                print(f"📁 {marker_name}: {len(image_list)} 个图像文件")
    
    return image_files

def convert_marker_to_zarr(marker_name: str, file_list: List[str], 
                               zarr_path: str, chunk_size: Tuple[int, int, int] = (64, 64, 3)) -> None:
    """
    内存安全地转换单个生物标记物的图像为 Zarr 格式
    
    策略：
    1. 单张图像处理，避免批量加载
    2. 立即写入 Zarr，然后释放内存
    3. 定期强制垃圾回收
    4. 监控内存使用情况
    """
    print(f"\n📸 开始转换 {marker_name} 标记物...")
    print(f"💾 初始内存使用: {get_memory_usage():.1f} MB")
    
    # 获取图像尺寸（只加载第一张图像）
    first_image = load_image_safe(file_list[0])
    if first_image is None:
        print(f"❌ 无法加载第一张图像，跳过 {marker_name}")
        return
    
    height, width, channels = first_image.shape
    num_images = len(file_list)
    
    print(f"   - 图像数量: {num_images}")
    print(f"   - 图像尺寸: {height} x {width} x {channels}")
    print(f"   - 单张图像内存: {height * width * channels / 1024 / 1024:.1f} MB")
    
    # 立即释放第一张图像
    del first_image
    gc.collect()
    
    # 创建 Zarr 数组
    array_path = os.path.join(zarr_path, marker_name)
    array = zarr.open(
        array_path,
        mode='w',
        shape=(num_images, height, width, channels),
        chunks=(1, chunk_size[0], chunk_size[1], chunk_size[2]),
        dtype=np.uint8
    )
    
    # 设置数组属性
    array.attrs['description'] = f'{marker_name} biomarker images'
    array.attrs['num_images'] = num_images
    array.attrs['image_shape'] = (height, width, channels)
    array.attrs['dtype'] = 'uint8'
    
    # 单张图像处理
    failed_count = 0
    for i, file_path in enumerate(tqdm(file_list, desc=f"转换 {marker_name}")):
        # 加载单张图像
        image = load_image_safe(file_path)
        
        if image is not None:
            # 立即写入 Zarr
            array[i] = image
            
            # 立即释放内存
            del image
        else:
            # 创建零填充图像作为占位符
            array[i] = np.zeros((height, width, channels), dtype=np.uint8)
            failed_count += 1
        
        # 每处理10张图像强制垃圾回收
        if i % 10 == 0:
            gc.collect()
        
        # 每处理100张图像报告内存使用情况
        if i % 100 == 0 and i > 0:
            memory_usage = get_memory_usage()
            print(f"   💾 处理 {i}/{num_images} 张图像，内存使用: {memory_usage:.1f} MB")
    
    # 最终内存清理
    gc.collect()
    final_memory = get_memory_usage()
    print(f"   💾 转换完成，最终内存使用: {final_memory:.1f} MB")
    
    if failed_count > 0:
        print(f"⚠️ {marker_name} 转换完成，但有 {failed_count} 个图像加载失败")
    else:
        print(f"✅ {marker_name} 转换完成: {num_images} 个图像")

def create_metadata(image_files: Dict[str, List[str]], zarr_path: str) -> None:
    """创建元数据文件"""
    metadata = {
        'num_markers': len(image_files),
        'markers': list(image_files.keys()),
        'total_images': sum(len(files) for files in image_files.values()),
        'marker_info': {}
    }
    
    for marker_name, file_list in image_files.items():
        metadata['marker_info'][marker_name] = {
            'num_images': len(file_list),
            'files': [os.path.basename(f) for f in file_list]
        }
    
    # 保存元数据
    metadata_file = os.path.join(zarr_path, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📋 元数据已保存: {metadata_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='内存安全的 Zarr 转换脚本')
    parser.add_argument('--input-dir', type=str, 
                       default='/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores',
                       help='输入图像目录路径')
    parser.add_argument('--output-dir', type=str,
                       default='/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_Cores_Zarr',
                       help='输出 Zarr 目录路径')
    parser.add_argument('--chunk-height', type=int, default=512,
                       help='分块高度')
    parser.add_argument('--chunk-width', type=int, default=512,
                       help='分块宽度')
    parser.add_argument('--chunk-channels', type=int, default=3,
                       help='分块通道数')
    parser.add_argument('--markers', nargs='+', 
                       help='指定要转换的生物标记物，例如: --markers HE CD3 CD8 CD56 CD68 CD163 MHC1 PDL1')
    parser.add_argument('--dry-run', action='store_true',
                       help='只扫描文件，不进行转换')
    parser.add_argument('--max-images', type=int,
                       help='限制每个标记物转换的最大图像数量（用于测试）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔄 内存安全的 Zarr 转换脚本")
    print("=" * 60)
    print(f"💾 系统总内存: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"💾 可用内存: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    
    # 扫描图像目录
    print(f"\n📂 扫描输入目录: {args.input_dir}")
    image_files = scan_image_directory(args.input_dir)
    
    if not image_files:
        print("❌ 没有找到图像文件")
        return
    
    # 如果指定了特定的标记物，只处理这些标记物
    if args.markers:
        filtered_files = {}
        for marker in args.markers:
            if marker in image_files:
                filtered_files[marker] = image_files[marker]
            else:
                print(f"⚠️ 警告: 未找到标记物 {marker}")
        image_files = filtered_files
    
    # 如果指定了最大图像数量，限制每个标记物的图像数量
    if args.max_images:
        limited_files = {}
        for marker_name, file_list in image_files.items():
            limited_files[marker_name] = file_list[:args.max_images]
        image_files = limited_files
        print(f"🔢 限制每个标记物最多转换 {args.max_images} 张图像")
    
    if not image_files:
        print("❌ 没有找到要转换的图像文件")
        return
    
    print(f"\n📊 转换计划:")
    total_images = sum(len(files) for files in image_files.values())
    print(f"   - 生物标记物数量: {len(image_files)}")
    print(f"   - 总图像数量: {total_images}")
    print(f"   - 标记物列表: {list(image_files.keys())}")
    
    # 估算内存需求
    if image_files:
        first_marker = list(image_files.keys())[0]
        first_file = image_files[first_marker][0]
        image_info = get_image_info_safe(first_file)
        if image_info:
            height, width, channels = image_info
            single_image_mb = height * width * channels / 1024 / 1024
            print(f"   - 单张图像内存需求: {single_image_mb:.1f} MB")
            print(f"   - 预计峰值内存: {single_image_mb * 1.5:.1f} MB (包含处理开销)")
    
    if args.dry_run:
        print("\n🔍 干运行模式，不进行实际转换")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换每个生物标记物
    chunk_size = (args.chunk_height, args.chunk_width, args.chunk_channels)
    
    for marker_name, file_list in image_files.items():
        convert_marker_to_zarr(marker_name, file_list, args.output_dir, chunk_size)
    
    # 创建元数据
    create_metadata(image_files, args.output_dir)
    
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"📊 转换统计:")
    print(f"   - 生物标记物: {list(image_files.keys())}")
    print(f"   - 总图像数: {total_images}")
    print(f"   - 分块大小: {chunk_size}")
    print(f"💾 最终内存使用: {get_memory_usage():.1f} MB")
    print("=" * 60)

if __name__ == '__main__':
    main()
