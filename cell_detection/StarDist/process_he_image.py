#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HE图像StarDist处理脚本
===================

🔬 专门处理大型HE染色图像的StarDist细胞分割工具

功能特点:
- 大图像分块处理
- HE染色优化预处理
- GPU内存管理
- 结果拼接和可视化

作者: AI Assistant
日期: 2025-09-23
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import time
from typing import Tuple, List, Dict
import logging

# 图像处理
from skimage import io, measure, exposure
from skimage.util import img_as_ubyte
from skimage.transform import resize
import tifffile

# StarDist
from stardist.models import StarDist2D
from stardist import fill_label_holes, random_label_cmap
from stardist.plot import render_label

# 数据处理
import pandas as pd
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HEImageProcessor:
    """
    HE图像处理器
    
    专门处理大型HE染色图像的StarDist分割
    """
    
    def __init__(self, 
                 model_name: str = '2D_versatile_he',
                 tile_size: int = 1024,
                 overlap: int = 128,
                 prob_thresh: float = 0.5,
                 nms_thresh: float = 0.6):
        """
        初始化HE图像处理器
        
        Args:
            model_name: StarDist模型名称
            tile_size: 分块大小
            overlap: 重叠区域大小
            prob_thresh: 概率阈值
            nms_thresh: NMS阈值
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        
        # 初始化StarDist模型
        try:
            self.model = StarDist2D.from_pretrained(model_name)
            logger.info(f"✅ 加载HE模型: {model_name}")
        except:
            # 回退到通用模型
            logger.warning(f"⚠️ HE模型加载失败，使用通用模型")
            self.model = StarDist2D.from_pretrained('2D_versatile_fluo')
            logger.info(f"✅ 加载通用模型: 2D_versatile_fluo")
        
        logger.info(f"🧬 HE图像处理器初始化完成")
        logger.info(f"📏 分块大小: {tile_size}x{tile_size}, 重叠: {overlap}")
    
    def preprocess_he_image(self, image: np.ndarray) -> np.ndarray:
        """
        HE染色图像预处理 - 保持3通道RGB
        
        Args:
            image: HE染色图像
            
        Returns:
            预处理后的3通道图像
        """
        # 转换为float32并标准化到[0,1]
        if image.dtype != np.float32:
            processed = image.astype(np.float32) / 255.0
        else:
            processed = image.copy()
            if processed.max() > 1:
                processed = processed / 255.0
        
        # 对比度增强 - 对每个通道分别处理
        if processed.ndim == 3:
            for i in range(processed.shape[2]):
                processed[:, :, i] = exposure.equalize_adapthist(
                    processed[:, :, i], clip_limit=0.02
                )
            logger.info("🎨 保持3通道RGB，分别增强对比度")
        else:
            processed = exposure.equalize_adapthist(processed, clip_limit=0.02)
            logger.info("⚫ 单通道图像对比度增强")
        
        # 轻微高斯滤波减少噪声
        processed = cv2.GaussianBlur(processed, (3, 3), 0.5)
        
        return processed
    
    def create_tiles(self, image: np.ndarray) -> List[Dict]:
        """
        创建图像分块
        
        Args:
            image: 输入图像
            
        Returns:
            分块信息列表
        """
        height, width = image.shape[:2]
        tiles = []
        
        # 计算分块数量
        n_tiles_y = (height - self.overlap) // (self.tile_size - self.overlap) + 1
        n_tiles_x = (width - self.overlap) // (self.tile_size - self.overlap) + 1
        
        logger.info(f"📐 图像尺寸: {width}x{height}")
        logger.info(f"🔢 分块数量: {n_tiles_x}x{n_tiles_y} = {n_tiles_x * n_tiles_y}")
        
        for i in range(n_tiles_y):
            for j in range(n_tiles_x):
                # 计算分块位置
                y_start = i * (self.tile_size - self.overlap)
                x_start = j * (self.tile_size - self.overlap)
                
                y_end = min(y_start + self.tile_size, height)
                x_end = min(x_start + self.tile_size, width)
                
                # 调整起始位置确保分块大小一致
                if y_end - y_start < self.tile_size:
                    y_start = max(0, y_end - self.tile_size)
                if x_end - x_start < self.tile_size:
                    x_start = max(0, x_end - self.tile_size)
                
                tiles.append({
                    'id': f"tile_{i}_{j}",
                    'y_start': y_start,
                    'y_end': y_end,
                    'x_start': x_start,
                    'x_end': x_end,
                    'tile_y': i,
                    'tile_x': j
                })
        
        return tiles
    
    def process_single_tile(self, image_tile: np.ndarray, tile_info: Dict) -> Tuple[np.ndarray, Dict]:
        """
        处理单个分块
        
        Args:
            image_tile: 图像分块
            tile_info: 分块信息
            
        Returns:
            (标签图, 分割信息)
        """
        try:
            # 预处理
            processed_tile = self.preprocess_he_image(image_tile)
            
            # StarDist分割 - 3通道RGB图像
            if processed_tile.ndim == 3:
                # 3通道图像使用YXC axes
                labels, details = self.model.predict_instances(
                    processed_tile,
                    axes='YXC',
                    prob_thresh=self.prob_thresh,
                    nms_thresh=self.nms_thresh
                )
            else:
                # 2通道图像使用YX axes
                labels, details = self.model.predict_instances(
                    processed_tile,
                    axes='YX',
                    prob_thresh=self.prob_thresh,
                    nms_thresh=self.nms_thresh
                )
            
            # 填充孔洞
            labels = fill_label_holes(labels)
            
            num_cells = len(np.unique(labels)) - 1
            
            seg_info = {
                'tile_id': tile_info['id'],
                'num_cells': num_cells,
                'tile_shape': image_tile.shape,
                'details': details
            }
            
            return labels, seg_info
            
        except Exception as e:
            logger.error(f"❌ 分块处理失败 {tile_info['id']}: {e}")
            return np.zeros(image_tile.shape[:2], dtype=np.uint16), {
                'tile_id': tile_info['id'],
                'num_cells': 0,
                'error': str(e)
            }
    
    def stitch_labels(self, tile_results: List[Dict], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        拼接分块标签结果
        
        Args:
            tile_results: 分块结果列表
            image_shape: 原图像形状
            
        Returns:
            拼接后的标签图
        """
        height, width = image_shape[:2]
        final_labels = np.zeros((height, width), dtype=np.uint16)
        label_counter = 1
        
        logger.info("🧩 开始拼接分块结果...")
        
        for result in tile_results:
            tile_info = result['tile_info']
            labels = result['labels']
            
            if labels is None or result['seg_info']['num_cells'] == 0:
                continue
            
            # 获取分块位置
            y_start = tile_info['y_start']
            y_end = tile_info['y_end']
            x_start = tile_info['x_start']
            x_end = tile_info['x_end']
            
            # 处理重叠区域 - 简单策略：只保留分块中心区域
            overlap_y = self.overlap // 2
            overlap_x = self.overlap // 2
            
            # 计算有效区域
            if tile_info['tile_y'] > 0:  # 不是第一行
                y_start_eff = y_start + overlap_y
            else:
                y_start_eff = y_start
                
            if tile_info['tile_x'] > 0:  # 不是第一列
                x_start_eff = x_start + overlap_x
            else:
                x_start_eff = x_start
            
            y_end_eff = min(y_end - overlap_y, height)
            x_end_eff = min(x_end - overlap_x, width)
            
            # 提取有效区域的标签
            tile_y_start = y_start_eff - y_start
            tile_x_start = x_start_eff - x_start
            tile_y_end = tile_y_start + (y_end_eff - y_start_eff)
            tile_x_end = tile_x_start + (x_end_eff - x_start_eff)
            
            tile_labels_crop = labels[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
            
            # 重新标记标签避免冲突
            unique_labels = np.unique(tile_labels_crop)
            for old_label in unique_labels:
                if old_label > 0:  # 跳过背景
                    mask = tile_labels_crop == old_label
                    final_labels[y_start_eff:y_end_eff, x_start_eff:x_end_eff][mask] = label_counter
                    label_counter += 1
        
        total_cells = len(np.unique(final_labels)) - 1
        logger.info(f"✅ 拼接完成 - 总计检测到 {total_cells} 个细胞")
        
        return final_labels
    
    def process_large_image(self, 
                          image_path: str,
                          output_dir: str,
                          save_tiles: bool = False) -> Dict:
        """
        处理大型图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            save_tiles: 是否保存分块结果
            
        Returns:
            处理结果
        """
        start_time = time.time()
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_tiles:
            (output_path / 'tiles').mkdir(exist_ok=True)
        
        logger.info(f"🔍 开始处理大型HE图像: {Path(image_path).name}")
        
        # 读取图像 - 分批读取以节省内存
        try:
            # 先获取图像信息
            with tifffile.TiffFile(image_path) as tif:
                image_shape = tif.pages[0].shape
                logger.info(f"📏 图像尺寸: {image_shape}")
        except:
            # 如果不是TIFF，使用OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_shape = image.shape
        
        # 创建分块
        if 'image' not in locals():
            image = io.imread(image_path)
            
        tiles = self.create_tiles(image)
        
        # 处理每个分块
        tile_results = []
        total_cells = 0
        
        logger.info(f"🔄 开始处理 {len(tiles)} 个分块...")
        
        for i, tile_info in enumerate(tiles):
            logger.info(f"📷 处理分块 ({i+1}/{len(tiles)}): {tile_info['id']}")
            
            # 提取分块
            y_start, y_end = tile_info['y_start'], tile_info['y_end']
            x_start, x_end = tile_info['x_start'], tile_info['x_end']
            
            image_tile = image[y_start:y_end, x_start:x_end]
            
            # 处理分块
            labels, seg_info = self.process_single_tile(image_tile, tile_info)
            
            total_cells += seg_info['num_cells']
            
            # 保存分块结果
            if save_tiles and seg_info['num_cells'] > 0:
                tile_output_dir = output_path / 'tiles' / tile_info['id']
                tile_output_dir.mkdir(exist_ok=True)
                
                # 保存标签图
                tifffile.imwrite(
                    tile_output_dir / f"{tile_info['id']}_labels.tif",
                    labels.astype(np.uint16)
                )
                
                # 保存可视化
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(image_tile)
                axes[0].set_title('original tile')
                axes[0].axis('off')
                
                axes[1].imshow(render_label(labels, img=image_tile, alpha=0.6))
                axes[1].set_title(f'segmentation result ({seg_info["num_cells"]} cells)')
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(tile_output_dir / f"{tile_info['id']}_result.png", dpi=150)
                plt.close()
            
            tile_results.append({
                'tile_info': tile_info,
                'labels': labels,
                'seg_info': seg_info
            })
        
        # 拼接结果
        logger.info("🧩 拼接分块结果...")
        final_labels = self.stitch_labels(tile_results, image_shape)
        
        # 保存最终结果
        result_name = Path(image_path).stem
        
        # 保存标签图
        tifffile.imwrite(
            output_path / f"{result_name}_labels.tif",
            final_labels.astype(np.uint16)
        )
        
        # 分析细胞属性
        logger.info("📊 分析细胞属性...")
        props = measure.regionprops(final_labels)
        
        cell_data = []
        for prop in props:
            cell_data.append({
                'cell_id': prop.label,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'centroid_y': prop.centroid[0],
                'centroid_x': prop.centroid[1],
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'eccentricity': prop.eccentricity,
                'solidity': prop.solidity,
                'equivalent_diameter': prop.equivalent_diameter
            })
        
        cell_stats = pd.DataFrame(cell_data)
        if len(cell_stats) > 0:
            cell_stats['aspect_ratio'] = cell_stats['major_axis_length'] / cell_stats['minor_axis_length']
            cell_stats['roundness'] = 4 * np.pi * cell_stats['area'] / (cell_stats['perimeter'] ** 2)
        
        # 保存统计数据
        cell_stats.to_csv(output_path / f"{result_name}_cell_stats.csv", index=False)
        
        # 创建缩略图可视化
        logger.info("🎨 生成可视化结果...")
        
        # 缩小图像用于可视化
        scale_factor = min(2048 / image_shape[1], 2048 / image_shape[0])
        if scale_factor < 1:
            viz_height = int(image_shape[0] * scale_factor)
            viz_width = int(image_shape[1] * scale_factor)
            
            image_small = resize(image, (viz_height, viz_width), preserve_range=True, anti_aliasing=True)
            labels_small = resize(final_labels, (viz_height, viz_width), order=0, preserve_range=True, anti_aliasing=False)
            
            image_small = img_as_ubyte(image_small)
            labels_small = labels_small.astype(np.uint16)
        else:
            image_small = image
            labels_small = final_labels
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # 可视化前确保图像在有效范围
        if image_small.dtype.kind == 'f':
            image_small_viz = np.clip(image_small, 0.0, 1.0)
        else:
            image_small_viz = image_small

        axes[0].imshow(image_small_viz)
        axes[0].set_title('🔬 原始HE图像', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(render_label(labels_small, img=image_small_viz, alpha=0.5))
        axes[1].set_title('🎯 细胞分割结果', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(labels_small, cmap=random_label_cmap())
        axes[2].set_title('🏷️ 细胞标签', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        final_cell_count = len(np.unique(final_labels)) - 1
        processing_time = time.time() - start_time
        
        plt.suptitle(
            f'HE图像StarDist分割结果 - 检测到 {final_cell_count} 个细胞核\n'
            f'处理时间: {processing_time:.1f}秒, 图像尺寸: {image_shape[1]}x{image_shape[0]}',
            fontsize=16, fontweight='bold'
        )
        
        plt.tight_layout()
        plt.savefig(output_path / f"{result_name}_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 生成处理报告
        report = {
            'input_image': str(image_path),
            'image_shape': image_shape,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'num_tiles': len(tiles),
            'total_cells_detected': final_cell_count,
            'processing_time_seconds': processing_time,
            'cell_density_per_mm2': 'N/A',  # 需要像素尺寸信息
            'tile_results': [r['seg_info'] for r in tile_results],
            'output_files': {
                'labels': f"{result_name}_labels.tif",
                'statistics': f"{result_name}_cell_stats.csv",
                'visualization': f"{result_name}_visualization.png"
            }
        }
        
        # 保存报告
        with open(output_path / f"{result_name}_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("🎉 大型HE图像处理完成!")
        logger.info(f"📊 总计检测细胞: {final_cell_count}")
        logger.info(f"⏱️ 处理时间: {processing_time:.1f}秒")
        logger.info(f"📁 结果保存至: {output_path}")
        
        return report


def main():
    """
    主函数 - 处理HE图像
    """
    print("🔬 HE图像StarDist处理工具")
    print("=" * 50)
    
    # HE图像路径
    he_image_path = "/home/zheng/zheng/rosie_reproduce/cell_extraction/qupath/extract_png/tma_cores_pngs/TumorCenter_HE_block1_1-1_circular.png"
    output_dir = "/home/zheng/zheng/rosie_reproduce/cell_extraction/StarDist/he_results"
    
    if not os.path.exists(he_image_path):
        print(f"❌ 图像文件不存在: {he_image_path}")
        return
    
    # 初始化处理器
    processor = HEImageProcessor(
        tile_size=512,  # 较小的分块适应GPU内存
        overlap=64,     # 减少重叠以提高速度
        prob_thresh=0.5,
        nms_thresh=0.4
    )
    
    # 处理图像
    try:
        result = processor.process_large_image(
            he_image_path,
            output_dir,
            save_tiles=True  # 保存分块结果用于调试
        )
        
        print("✅ HE图像处理成功完成!")
        print(f"📊 检测到 {result['total_cells_detected']} 个细胞核")
        print(f"⏱️ 处理时间: {result['processing_time_seconds']:.1f} 秒")
        
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
