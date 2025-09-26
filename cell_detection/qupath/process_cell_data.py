#!/usr/bin/env python3
"""
细胞数据处理脚本
从QuPath检测结果中提取细胞的位置、大小和RGB数据
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from PIL import Image
import os

def parse_qupath_results(results_file):
    """解析QuPath结果文件"""
    # 这个函数需要根据实际的QuPath输出格式来调整
    # 目前假设结果保存为JSON格式
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except:
        print(f"无法解析结果文件: {results_file}")
        return None

def extract_cell_rgb_data(cell_details):
    """提取细胞RGB数据"""
    cell_data = []
    
    for cell in cell_details:
        if cell.get('rgbMatrix') is not None:
            rgb_matrix = np.array(cell['rgbMatrix'])  # shape: (height, width, 3)
            mask = np.array(cell['mask'])  # shape: (height, width)
            
            # 只提取细胞区域的像素（mask=1的区域）
            cell_pixels = rgb_matrix[mask == 1]  # shape: (n_pixels, 3)
            
            cell_info = {
                'id': cell['id'],
                'centroid': cell['centroid'],
                'boundingBox': cell['boundingBox'],
                'area': cell['area'],
                'areaPixels': cell['areaPixels'],
                'rgbMatrix': rgb_matrix,
                'mask': mask,
                'cellPixels': cell_pixels,  # 只包含细胞区域的RGB值
                'meanRGB': np.mean(cell_pixels, axis=0) if len(cell_pixels) > 0 else [0, 0, 0],
                'stdRGB': np.std(cell_pixels, axis=0) if len(cell_pixels) > 0 else [0, 0, 0],
                'measurements': cell.get('measurements', {})
            }
            
            cell_data.append(cell_info)
    
    return cell_data

def save_cell_images(cell_data, output_dir, method_name):
    """保存每个细胞的图像"""
    cell_images_dir = Path(output_dir) / f'{method_name}_cell_images'
    cell_images_dir.mkdir(parents=True, exist_ok=True)
    
    for cell in cell_data:
        cell_id = cell['id']
        rgb_matrix = cell['rgbMatrix']
        mask = cell['mask']
        
        # 创建带掩码的细胞图像
        masked_image = rgb_matrix.copy()
        masked_image[mask == 0] = [255, 255, 255]  # 背景设为白色
        
        # 保存原始细胞图像
        img_original = Image.fromarray(rgb_matrix.astype(np.uint8))
        img_original.save(cell_images_dir / f'cell_{cell_id:03d}_original.png')
        
        # 保存带掩码的细胞图像
        img_masked = Image.fromarray(masked_image.astype(np.uint8))
        img_masked.save(cell_images_dir / f'cell_{cell_id:03d}_masked.png')
        
        # 保存掩码
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(cell_images_dir / f'cell_{cell_id:03d}_mask.png')
    
    print(f"保存了 {len(cell_data)} 个细胞图像到: {cell_images_dir}")

def analyze_rgb_distribution(cell_data, output_dir, method_name):
    """分析细胞RGB分布"""
    if not cell_data:
        print(f"没有{method_name}细胞数据可分析")
        return
    
    # 提取所有细胞的RGB统计
    mean_rgbs = np.array([cell['meanRGB'] for cell in cell_data])
    std_rgbs = np.array([cell['stdRGB'] for cell in cell_data])
    areas = np.array([cell['area'] for cell in cell_data])
    
    # 创建分析图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{method_name} 细胞RGB分析', fontsize=16, fontweight='bold')
    
    # RGB均值分布
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        axes[0, i].hist(mean_rgbs[:, i], bins=30, alpha=0.7, color=color)
        axes[0, i].set_title(f'{color.upper()}通道均值分布')
        axes[0, i].set_xlabel('像素值')
        axes[0, i].set_ylabel('频数')
        axes[0, i].grid(True, alpha=0.3)
    
    # RGB均值散点图
    axes[1, 0].scatter(mean_rgbs[:, 0], mean_rgbs[:, 1], alpha=0.6, c=areas, cmap='viridis')
    axes[1, 0].set_xlabel('红色均值')
    axes[1, 0].set_ylabel('绿色均值')
    axes[1, 0].set_title('RG散点图（颜色表示面积）')
    
    axes[1, 1].scatter(mean_rgbs[:, 1], mean_rgbs[:, 2], alpha=0.6, c=areas, cmap='viridis')
    axes[1, 1].set_xlabel('绿色均值')
    axes[1, 1].set_ylabel('蓝色均值')
    axes[1, 1].set_title('GB散点图（颜色表示面积）')
    
    axes[1, 2].scatter(areas, np.mean(mean_rgbs, axis=1), alpha=0.6)
    axes[1, 2].set_xlabel('细胞面积 (μm²)')
    axes[1, 2].set_ylabel('RGB均值')
    axes[1, 2].set_title('面积 vs RGB均值')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / f'{method_name}_rgb_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"RGB分析图表已保存到: {output_path}")
    
    plt.show()
    
    # 保存统计数据
    stats_df = pd.DataFrame({
        'cell_id': [cell['id'] for cell in cell_data],
        'area': areas,
        'centroid_x': [cell['centroid']['x'] for cell in cell_data],
        'centroid_y': [cell['centroid']['y'] for cell in cell_data],
        'mean_r': mean_rgbs[:, 0],
        'mean_g': mean_rgbs[:, 1],
        'mean_b': mean_rgbs[:, 2],
        'std_r': std_rgbs[:, 0],
        'std_g': std_rgbs[:, 1],
        'std_b': std_rgbs[:, 2]
    })
    
    stats_file = Path(output_dir) / f'{method_name}_cell_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"细胞统计数据已保存到: {stats_file}")
    
    return stats_df

def compare_methods(watershed_data, stardist_data, output_dir):
    """对比两种方法的结果"""
    if not watershed_data or not stardist_data:
        print("缺少对比数据")
        return
    
    # 提取统计信息
    w_areas = np.array([cell['area'] for cell in watershed_data])
    s_areas = np.array([cell['area'] for cell in stardist_data])
    
    w_rgb = np.array([cell['meanRGB'] for cell in watershed_data])
    s_rgb = np.array([cell['meanRGB'] for cell in stardist_data])
    
    # 创建对比图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Watershed vs StarDist 对比分析', fontsize=16, fontweight='bold')
    
    # 细胞数量对比
    axes[0, 0].bar(['Watershed', 'StarDist'], [len(watershed_data), len(stardist_data)], 
                   color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0, 0].set_title('检测细胞数量对比')
    axes[0, 0].set_ylabel('细胞数量')
    
    # 面积分布对比
    axes[0, 1].hist([w_areas, s_areas], bins=20, alpha=0.7, 
                    label=['Watershed', 'StarDist'], color=['skyblue', 'lightcoral'])
    axes[0, 1].set_title('细胞面积分布对比')
    axes[0, 1].set_xlabel('面积 (μm²)')
    axes[0, 1].set_ylabel('频数')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RGB均值对比
    w_rgb_mean = np.mean(w_rgb, axis=1)
    s_rgb_mean = np.mean(s_rgb, axis=1)
    
    axes[1, 0].hist([w_rgb_mean, s_rgb_mean], bins=20, alpha=0.7,
                    label=['Watershed', 'StarDist'], color=['skyblue', 'lightcoral'])
    axes[1, 0].set_title('RGB均值分布对比')
    axes[1, 0].set_xlabel('RGB均值')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 面积vs RGB散点图对比
    axes[1, 1].scatter(w_areas, w_rgb_mean, alpha=0.6, label='Watershed', color='skyblue')
    axes[1, 1].scatter(s_areas, s_rgb_mean, alpha=0.6, label='StarDist', color='lightcoral')
    axes[1, 1].set_xlabel('细胞面积 (μm²)')
    axes[1, 1].set_ylabel('RGB均值')
    axes[1, 1].set_title('面积 vs RGB均值对比')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存对比图表
    output_path = Path(output_dir) / 'method_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"方法对比图表已保存到: {output_path}")
    
    plt.show()

def main():
    """主函数"""
    # 设置路径
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'cell_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    print("🔍 开始处理细胞检测数据...")
    
    # 注意：这里需要根据实际的QuPath输出格式来调整
    # 目前假设数据已经以某种方式导出为可读格式
    
    print("📊 由于QuPath结果格式的复杂性，请按以下步骤手动导出数据：")
    print("1. 在QuPath中运行检测脚本")
    print("2. 检测结果会包含每个细胞的详细信息：")
    print("   - 位置信息：centroid (x, y), boundingBox")
    print("   - 大小信息：area, areaPixels")
    print("   - RGB数据：rgbMatrix (height x width x 3)")
    print("   - 掩码数据：mask (height x width)")
    print("   - 测量数据：measurements")
    
    print("\n📝 数据结构示例：")
    example_cell = {
        'id': 1,
        'centroid': {'x': 100.5, 'y': 200.3},
        'boundingBox': {
            'minX': 90, 'minY': 190,
            'maxX': 110, 'maxY': 210,
            'width': 20, 'height': 20
        },
        'area': 45.6,
        'areaPixels': 123,
        'rgbMatrix': "[[[[255,128,64],...]]...]",  # height x width x 3 数组
        'mask': "[[[1,1,0],...]]",  # height x width 掩码
        'measurements': {
            'Area': 45.6,
            'Perimeter': 24.1,
            'Circularity': 0.85
        }
    }
    
    print(json.dumps(example_cell, indent=2, ensure_ascii=False))
    
    print(f"\n📁 分析结果将保存到: {output_dir}")
    print("✅ 脚本准备就绪，等待QuPath数据导出...")

if __name__ == "__main__":
    main()
