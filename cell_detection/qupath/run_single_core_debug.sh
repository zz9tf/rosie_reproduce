#!/bin/bash

# 🔬 单核心Watershed细胞检测调试脚本

# 设置路径
TMA_IMAGE="/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter/HE/TumorCenter_HE_block1.svs"
DEBUG_SCRIPT="/home/zheng/zheng/rosie_reproduce/cell_extraction/qupath/detect_cell/single_core_watershed.groovy"
OUTPUT_DIR="/home/zheng/zheng/rosie_reproduce/cell_extraction/qupath/detect_cell/single_core_debug"

echo "=========================================="
echo "🔬 单核心Watershed细胞检测调试"
echo "=========================================="
echo "📁 TMA图像: $(basename "$TMA_IMAGE")"
echo "🔧 调试脚本: $(basename "$DEBUG_SCRIPT")"
echo "📊 输出目录: $OUTPUT_DIR"
echo "🎯 目标: 提取单个核心+详细调试细胞信息"
echo "=========================================="

# 检查文件是否存在
if [ ! -f "$TMA_IMAGE" ]; then
    echo "❌ 错误: TMA图像文件不存在: $TMA_IMAGE"
    exit 1
fi

if [ ! -f "$DEBUG_SCRIPT" ]; then
    echo "❌ 错误: 调试脚本不存在: $DEBUG_SCRIPT"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 运行QuPath脚本
echo "🚀 开始运行单核心调试..."
echo "⏰ 开始时间: $(date)"

qupath script -i "$TMA_IMAGE" "$DEBUG_SCRIPT"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "✅ QuPath脚本执行成功"
else
    echo "❌ QuPath脚本执行失败"
    exit 1
fi

echo "⏰ 结束时间: $(date)"
echo "=========================================="

# 检查输出结果
echo "📊 检查输出结果..."

# 查找生成的文件
CORE_DIR=$(find "$OUTPUT_DIR" -name "core_*" -type d | head -1)
if [ -d "$CORE_DIR" ]; then
    echo "✅ 找到核心目录: $(basename "$CORE_DIR")"
    
    # 统计文件
    FULL_RES_IMAGE=$(find "$CORE_DIR" -name "*_full_resolution.png" | head -1)
    ORIGINAL_IMAGES=$(find "$CORE_DIR" -name "*_original.png" | wc -l)
    MASKED_IMAGES=$(find "$CORE_DIR" -name "*_masked.png" | wc -l)
    DEBUG_REPORT=$(find "$CORE_DIR" -name "*_debug_report.json" | head -1)
    
    if [ -f "$FULL_RES_IMAGE" ]; then
        echo "   🖼️  核心全分辨率图像: $(basename "$FULL_RES_IMAGE")"
        echo "       文件大小: $(du -h "$FULL_RES_IMAGE" | cut -f1)"
    fi
    
    echo "   🔬 细胞图像统计:"
    echo "       原始图像: $ORIGINAL_IMAGES 个"
    echo "       掩码图像: $MASKED_IMAGES 个"
    
    if [ -f "$DEBUG_REPORT" ]; then
        echo "   📋 调试报告: $(basename "$DEBUG_REPORT")"
        echo "       文件大小: $(du -h "$DEBUG_REPORT" | cut -f1)"
        
        # 提取关键信息
        if command -v jq >/dev/null 2>&1; then
            echo "   📊 检测统计:"
            echo "       总细胞数: $(jq -r '.total_cells_detected' "$DEBUG_REPORT" 2>/dev/null || echo "N/A")"
            echo "       调试细胞数: $(jq -r '.cells_debugged' "$DEBUG_REPORT" 2>/dev/null || echo "N/A")"
            echo "       像素校准: $(jq -r '.pixel_calibration_um_per_pixel' "$DEBUG_REPORT" 2>/dev/null || echo "N/A") μm/pixel"
        fi
    fi
    
    echo ""
    echo "📁 目录内容预览:"
    ls -la "$CORE_DIR" | head -10
    
else
    echo "⚠️  警告: 没有找到核心输出目录"
fi

echo "=========================================="
echo "🎉 单核心调试完成！"
echo "📁 详细结果查看: $OUTPUT_DIR"
echo "🐛 调试信息已保存到JSON报告中"
echo "=========================================="

