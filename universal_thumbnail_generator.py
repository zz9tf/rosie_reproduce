#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用SVS缩略图生成器 🖼️
为任意SVS文件生成完整缩略图
"""

import os
import sys
from PIL import Image

try:
    import openslide
    from openslide import OpenSlide
except ImportError:
    print("❌ 错误: 缺少openslide-python库")
    print("请运行: pip install openslide-python")
    sys.exit(1)


def generate_complete_thumbnail(svs_path, output_size=(1024, 1024), output_path=None):
    """
    生成完整的SVS缩略图 - 显示整个block的样貌
    
    Args:
        svs_path (str): SVS文件路径
        output_size (tuple): 输出尺寸
        output_path (str): 输出路径
        
    Returns:
        str: 生成的缩略图路径
    """
    try:
        print(f"📂 正在加载: {os.path.basename(svs_path)}")
        
        # 打开SVS文件
        slide = OpenSlide(svs_path)
        
        print(f"📊 文件信息:")
        print(f"   - 厂商: {slide.properties.get('openslide.vendor', 'Unknown')}")
        print(f"   - 层级数: {slide.level_count}")
        print(f"   - 基础尺寸: {slide.dimensions}")
        
        # 选择最佳层级 - 考虑内存限制
        best_level = find_best_thumbnail_level(slide, output_size)
        level_dimensions = slide.level_dimensions[best_level]
        
        print(f"🎯 使用层级 {best_level}: {level_dimensions}")
        print(f"   下采样倍数: {slide.level_downsamples[best_level]:.2f}x")
        
        # 读取整个层级的图像
        print(f"🔍 读取完整图像...")
        image = slide.read_region((0, 0), best_level, level_dimensions)
        
        # 转换为RGB
        if image.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # 使用alpha通道作为mask
            image = background
        
        # 调整到目标尺寸，保持宽高比
        image.thumbnail(output_size, Image.Resampling.LANCZOS)
        
        # 生成输出文件名
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(svs_path))[0]
            output_path = f"{base_name}_complete_thumbnail_{output_size[0]}x{output_size[1]}.jpg"
        
        # 保存
        image.save(output_path, 'JPEG', quality=90, optimize=True)
        
        print(f"✅ 完整缩略图已保存: {output_path}")
        print(f"📐 最终尺寸: {image.size}")
        
        # 关闭slide
        slide.close()
        
        return output_path
        
    except Exception as e:
        print(f"❌ 生成缩略图失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def find_best_thumbnail_level(slide, target_size):
    """
    找到生成缩略图的最佳层级 - 考虑内存限制
    
    Args:
        slide: OpenSlide对象
        target_size: 目标尺寸
        
    Returns:
        int: 最佳层级索引
    """
    # 内存限制：最大500MB
    max_memory_mb = 500
    max_pixels = (max_memory_mb * 1024 * 1024) // 4  # 4 bytes per pixel (RGBA)
    
    best_level = slide.level_count - 1  # 默认选择最小的层级
    
    for i in range(slide.level_count):
        width, height = slide.level_dimensions[i]
        pixels = width * height
        
        # 检查内存限制
        if pixels <= max_pixels:
            best_level = i
            break
    
    return best_level


def analyze_svs_file(svs_path):
    """
    分析SVS文件的详细信息
    
    Args:
        svs_path (str): SVS文件路径
    """
    try:
        print(f"\n🔬 SVS文件详细信息分析:")
        print("="*60)
        
        slide = OpenSlide(svs_path)
        
        # 基本信息
        file_size_mb = os.path.getsize(svs_path) / (1024 * 1024)
        print(f"📄 基本信息:")
        print(f"   文件路径: {svs_path}")
        print(f"   文件大小: {file_size_mb:.2f} MB")
        print(f"   厂商: {slide.properties.get('openslide.vendor', 'Unknown')}")
        print(f"   层级数: {slide.level_count}")
        print(f"   基础尺寸: {slide.dimensions}")
        
        # 层级详细信息
        print(f"\n🏔️ 层级详细信息:")
        for i in range(slide.level_count):
            width, height = slide.level_dimensions[i]
            downsample = slide.level_downsamples[i]
            pixels = width * height
            memory_mb = (pixels * 3) / (1024 * 1024)  # RGB, 3 bytes per pixel
            
            print(f"   层级 {i}: {width}x{height} (下采样: {downsample:.2f}x, 内存: {memory_mb:.1f}MB)")
        
        # 像素校准
        mpp_x = slide.properties.get('openslide.mpp-x')
        mpp_y = slide.properties.get('openslide.mpp-y')
        if mpp_x and mpp_y:
            print(f"\n🔬 像素校准:")
            print(f"   X轴像素大小: {mpp_x} μm/pixel")
            print(f"   Y轴像素大小: {mpp_y} μm/pixel")
            
            # 计算实际物理尺寸
            width_um = slide.dimensions[0] * float(mpp_x)
            height_um = slide.dimensions[1] * float(mpp_y)
            print(f"   实际物理尺寸: {width_um/1000:.2f} x {height_um/1000:.2f} mm")
        
        # 关联图像
        print(f"\n🖼️ 关联图像:")
        macro_width = slide.properties.get('openslide.associated.macro.width')
        macro_height = slide.properties.get('openslide.associated.macro.height')
        if macro_width and macro_height:
            print(f"   宏图像: {macro_width}x{macro_height}")
        
        thumbnail_width = slide.properties.get('openslide.associated.thumbnail.width')
        thumbnail_height = slide.properties.get('openslide.associated.thumbnail.height')
        if thumbnail_width and thumbnail_height:
            print(f"   缩略图: {thumbnail_width}x{thumbnail_height}")
        
        # 元数据统计
        print(f"\n📊 元数据统计:")
        print(f"   总元数据字段数: {len(slide.properties)}")
        
        # 按前缀分类
        prefixes = {}
        for key in slide.properties.keys():
            prefix = key.split('.')[0] if '.' in key else 'other'
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        for prefix, count in sorted(prefixes.items()):
            print(f"   {prefix}: {count} 个字段")
        
        slide.close()
        
    except Exception as e:
        print(f"❌ 分析失败: {str(e)}")


def generate_multiple_thumbnails(svs_path, sizes=[(512, 512), (1024, 1024), (2048, 2048)]):
    """
    生成多种尺寸的完整缩略图
    
    Args:
        svs_path (str): SVS文件路径
        sizes (list): 尺寸列表
        
    Returns:
        list: 生成的缩略图路径列表
    """
    print("🖼️  通用SVS缩略图生成器")
    print("="*60)
    
    results = []
    
    for size in sizes:
        print(f"\n📷 生成 {size[0]}x{size[1]} 完整缩略图...")
        
        # 生成输出文件名
        base_name = os.path.splitext(os.path.basename(svs_path))[0]
        output_name = f"{base_name}_complete_{size[0]}x{size[1]}.jpg"
        
        result = generate_complete_thumbnail(svs_path, size, output_name)
        
        if result:
            print(f"   ✅ 成功: {result}")
            results.append(result)
        else:
            print(f"   ❌ 失败")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="通用SVS缩略图生成器")
    parser.add_argument("svs_path", help="SVS文件路径")
    parser.add_argument("--sizes", nargs='+', type=int, default=[512, 1024, 2048], 
                       help="缩略图尺寸列表 (默认: 512 1024 2048)")
    parser.add_argument("--analyze", action='store_true', help="分析文件详细信息")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.svs_path):
        print(f"❌ 文件不存在: {args.svs_path}")
        return
    
    # 分析文件信息
    if args.analyze:
        analyze_svs_file(args.svs_path)
    
    # 生成缩略图
    sizes = [(size, size) for size in args.sizes]
    results = generate_multiple_thumbnails(args.svs_path, sizes)
    
    print(f"\n🎉 缩略图生成完成!")
    print(f"📁 生成的文件:")
    for result in results:
        print(f"   - {result}")


if __name__ == "__main__":
    main()
