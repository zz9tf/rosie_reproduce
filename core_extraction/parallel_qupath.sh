#!/usr/bin/env bash
set -euo pipefail

# 配置参数
NUM_CONTAINERS=7  # 固定启动7个容器，每个运行不同的标记脚本
SCRIPT_DIR="/home/zheng/zheng/rosie_reproduce/core_extraction"

# 定义7个不同的标记脚本和对应的输出目录
MARKERS=(
    "CD3:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/CD3"
    "CD8:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/CD8"
    "CD56:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/CD56"
    "CD68:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/CD68"
    "CD163:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/CD163"
    "MHC1:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/MHC1"
    "PDL1:/home/zheng/zheng/mini2/hancock_data/TMA/TMA_TumorCenter_1024/PDL1"
)

# 清理可能存在的旧容器
echo "🧹 清理可能存在的旧 QuPath 容器..."
docker ps -a --filter "name=qupath-parallel-" -q | xargs -r docker rm -f

# 启动多个 QuPath 容器的函数
start_qupath_container() {
    local container_id=$1
    local marker_info=$2
    local marker_name=$(echo "$marker_info" | cut -d':' -f1)
    local output_dir=$(echo "$marker_info" | cut -d':' -f2)
    local container_name="qupath-parallel-${marker_name,,}"
    local base_port=$((8080 + container_id))
    
    echo "🚀 启动容器 ${container_name} 处理 ${marker_name} 标记..."
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    # 运行对应的标记脚本（强制使用 bash 入口，避免 qupath 作为默认入口拦截参数）
    docker run -d --rm \
        -v "/home/zheng:/home/zheng" \
        -v "/mnt/SSD1/zheng:/mnt/SSD1/zheng" \
        -v "/mnt/SSD3/public:/mnt/SSD3/public" \
        -v "/smile/mini_share/mini2/zheng:/smile/mini_share/mini2/zheng" \
        --user "$(id -u):$(id -g)" \
        --workdir "/home/zheng/zheng/rosie_reproduce/core_extraction" \
        --name "$container_name" \
        -p "${base_port}:8080" \
        --entrypoint /bin/bash \
        qupath-v6 \
        -lc "set -euo pipefail; chmod +x /home/zheng/zheng/rosie_reproduce/core_extraction/tma_qupath_${marker_name}.sh; /home/zheng/zheng/rosie_reproduce/core_extraction/tma_qupath_${marker_name}.sh"
}

# 并行启动容器
echo "🎯 准备启动 ${NUM_CONTAINERS} 个 QuPath 容器，分别处理不同的标记..."
pids=()

for i in $(seq 1 $NUM_CONTAINERS); do
    # 获取对应的标记信息
    marker_index=$((i-1))
    marker_info="${MARKERS[$marker_index]}"
    
    # 在后台启动容器
    start_qupath_container $i "$marker_info" &
    pids+=($!)
    
    # 稍微延迟避免同时启动太多容器
    sleep 2
done

echo "⏳ 等待所有容器启动完成..."
wait "${pids[@]}"

echo "✅ 所有 QuPath 容器已启动！"
echo "📊 容器状态："
docker ps --filter "name=qupath-parallel-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔍 监控容器日志的命令："
for marker_info in "${MARKERS[@]}"; do
    marker_name=$(echo "$marker_info" | cut -d':' -f1)
    echo "  docker logs -f qupath-parallel-${marker_name,,}"
done

echo ""
echo "🛑 停止所有容器的命令："
echo "  docker ps --filter 'name=qupath-parallel-' -q | xargs docker stop"
