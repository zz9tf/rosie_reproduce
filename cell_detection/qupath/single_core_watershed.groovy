/**
 * 🔬 单核心高精度Watershed细胞检测和调试
 * 专注于提取一个TMA核心并详细调试细胞信息
 */

import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.regions.RegionRequest
import javax.imageio.ImageIO
import java.awt.image.BufferedImage

// 获取当前图像信息
def imageData = getCurrentImageData()
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

print "=========================================="
print "🔬 单核心Watershed细胞检测调试: " + name
print "=========================================="

// 步骤1: TMA去阵列化
print "📋 执行TMA去阵列化..."
runPlugin('qupath.imagej.detect.dearray.TMADearrayerPluginIJ', 
    '{"coreDiameterMM":1.9,"labelsHorizontal":"1-6","labelsVertical":"1-12","labelOrder":"Row first","densityThreshold":5,"boundsScale":105}')

relabelTMAGrid("1-6", "1-12", true)
print "✅ TMA去阵列化完成"

// 步骤2: 获取第一个TMA核心
def cores = getCurrentHierarchy().getTMAGrid().getTMACoreList().findAll()
print "📊 找到 " + cores.size() + " 个TMA核心"

if (cores.isEmpty()) {
    print "❌ 错误: 没有找到TMA核心"
    return
}

def targetCore = cores[0]  // 使用第一个核心
def coreId = targetCore.getDisplayedName() ?: "Core_1"
print "🎯 选择核心进行处理: ${coreId}"

// 步骤3: 设置输出目录
def outputBaseDir = '/home/zheng/zheng/rosie_reproduce/cell_extraction/qupath/single_core_debug'
def coreOutputDir = new File(outputBaseDir, "core_${coreId}")
coreOutputDir.mkdirs()

print "📁 输出目录: ${coreOutputDir.absolutePath}"

// 步骤4: 提取核心的最高精度图像
print "\n--- 🖼️  提取核心图像 ---"
def server = getCurrentImageData().getServer()
def cal = server.getPixelCalibration()

// 获取核心的边界框
def coreROI = targetCore.getROI()
def coreMinX = coreROI.getBoundsX() as int
def coreMinY = coreROI.getBoundsY() as int  
def coreWidth = coreROI.getBoundsWidth() as int
def coreHeight = coreROI.getBoundsHeight() as int

print "📐 核心边界框: x=${coreMinX}, y=${coreMinY}, width=${coreWidth}, height=${coreHeight}"
print "📏 像素校准: ${cal.getPixelWidthMicrons()} μm/pixel"

// 创建最高精度的区域请求
def rgbDownsample = 1.0  // 最高精度
def coreRegionRequest = RegionRequest.createInstance(server.getPath(), rgbDownsample, coreMinX, coreMinY, coreWidth, coreHeight)

try {
    def coreImage = server.readBufferedImage(coreRegionRequest)
    def coreImageFile = new File(coreOutputDir, "${coreId}_full_resolution.png")
    ImageIO.write(coreImage, "png", coreImageFile)
    print "✅ 核心图像已保存: ${coreImageFile.absolutePath}"
    print "🖼️  图像尺寸: ${coreImage.getWidth()} x ${coreImage.getHeight()}"
} catch (Exception e) {
    print "❌ 保存核心图像失败: ${e.getMessage()}"
    return
}

// 步骤5: 清除之前的检测结果并选择核心
print "\n--- 🧹 清理和选择 ---"
def detectionsToRemove = getDetectionObjects()
if (!detectionsToRemove.isEmpty()) {
    removeObjects(detectionsToRemove, true)
    print "🗑️  清除了 ${detectionsToRemove.size()} 个之前的检测对象"
}

selectObjects(targetCore)
print "✅ 已选择目标核心: ${coreId}"

// 步骤6: Watershed细胞检测
print "\n--- 🔬 Watershed细胞检测 ---"
print "⚙️  检测参数:"
print "   - 检测图像: Hematoxylin OD"
print "   - 像素尺寸: 0.2 μm"
print "   - 背景半径: 8.0 μm"
print "   - 最小面积: 10.0 μm²"
print "   - 最大面积: 400.0 μm²"

runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
    '{"detectionImage":"Hematoxylin OD",' +
    '"requestedPixelSizeMicrons":0.2,' +
    '"backgroundRadiusMicrons":8.0,' +
    '"backgroundByReconstruction":true,' +
    '"medianRadiusMicrons":0.0,' +
    '"sigmaMicrons":1.5,' +
    '"minAreaMicrons":10.0,' +
    '"maxAreaMicrons":400.0,' +
    '"threshold":0.1,' +
    '"maxBackground":2.0,' +
    '"watershedPostProcess":true,' +
    '"cellExpansionMicrons":5.0,' +
    '"includeNuclei":true,' +
    '"smoothBoundaries":true,' +
    '"makeMeasurements":true}')

// 步骤7: 获取检测结果并调试
print "\n--- 🔍 检测结果调试 ---"
def detectionObjects = getDetectionObjects()
def totalCells = detectionObjects.size()
print "🎯 检测到 ${totalCells} 个细胞"

if (totalCells == 0) {
    print "⚠️  警告: 没有检测到细胞，请检查参数设置"
    return
}

// 步骤8: 详细调试前5个细胞
print "\n--- 🐛 详细调试细胞信息 ---"
def debugCells = detectionObjects.take(Math.min(1000, totalCells))
def cellResults = []

debugCells.eachWithIndex { detection, cellIndex ->
    if (cellIndex <= 950) {
        return
    }
    print "\n🔬 调试细胞 ${cellIndex + 1}/${debugCells.size()}:"
    
    try {
        // 基本ROI信息
        def roi = detection.getROI()
        print "   📐 ROI类型: ${roi.getClass().getSimpleName()}"
        print "   📍 中心点: (${roi.getCentroidX()}, ${roi.getCentroidY()})"
        print "   📏 边界框: x=${roi.getBoundsX()}, y=${roi.getBoundsY()}, w=${roi.getBoundsWidth()}, h=${roi.getBoundsHeight()}"
        print "   📊 面积(像素): ${roi.getArea()}"
        print "   📊 面积(μm²): ${roi.getScaledArea(cal)}"
        
        // 测量列表调试
        def measurementList = detection.getMeasurementList()
        print "   📋 测量列表类型: ${measurementList.getClass().getSimpleName()}"
        
        def measurementNames = measurementList.getMeasurementNames()
        print "   📝 可用测量项 (${measurementNames.size()}个):"
        measurementNames.each { measurementName ->
            print "      - ${measurementName}"
        }
        
        // 使用正确的API获取测量值
        def measurements = [:]
        measurementNames.each { measurementName ->
            try {
                // 正确的API: 使用字符串名称而不是索引
                def value = measurementList.get(measurementName)
                if (value != null && !Double.isNaN(value)) {
                    measurements[measurementName] = value
                    print "      ✅ ${measurementName} = ${value}"
                } else {
                    print "      ⚠️  ${measurementName} = null/NaN"
                }
            } catch (Exception e) {
                print "      ❌ ${measurementName}: ${e.getMessage()}"
                // 添加默认值
                if (measurementName.contains("Area")) {
                    measurements[measurementName] = roi.getScaledArea(cal)
                } else if (measurementName.contains("Centroid") && measurementName.contains("X")) {
                    measurements[measurementName] = roi.getCentroidX()
                } else if (measurementName.contains("Centroid") && measurementName.contains("Y")) {
                    measurements[measurementName] = roi.getCentroidY()
                }
            }
        }
        
        // 保存细胞图像
        def cellId = String.format("cell_%03d", cellIndex + 1)
        def minX = roi.getBoundsX() as int
        def minY = roi.getBoundsY() as int
        def width = roi.getBoundsWidth() as int
        def height = roi.getBoundsHeight() as int
        
        // 创建细胞区域请求
        def cellRegionRequest = RegionRequest.createInstance(server.getPath(), rgbDownsample, minX, minY, width, height)
        def cellImage = server.readBufferedImage(cellRegionRequest)
        
        // 保存原始细胞图像
        def originalFile = new File(coreOutputDir, "${cellId}_original.png")
        ImageIO.write(cellImage, "png", originalFile)
        
        // 创建带掩码的图像
        def maskedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        def graphics = maskedImage.createGraphics()
        graphics.drawImage(cellImage, 0, 0, null)
        graphics.dispose()
        
        // 应用掩码
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                def globalX = minX + x
                def globalY = minY + y
                if (!roi.contains(globalX, globalY)) {
                    maskedImage.setRGB(x, y, 0x00000000) // 透明
                }
            }
        }
        
        def maskedFile = new File(coreOutputDir, "${cellId}_masked.png")
        ImageIO.write(maskedImage, "png", maskedFile)
        
        // 收集细胞信息
        def cellInfo = [
            core_id: coreId,
            cell_id: cellId,
            cell_index: cellIndex + 1,
            centroid: [x: roi.getCentroidX(), y: roi.getCentroidY()],
            bounding_box: [x: minX, y: minY, width: width, height: height],
            area_um2: roi.getScaledArea(cal),
            area_pixels: roi.getArea(),
            original_image: originalFile.absolutePath,
            masked_image: maskedFile.absolutePath,
            measurements: measurements
        ]
        
        cellResults.add(cellInfo)
        print "   ✅ 细胞 ${cellIndex + 1} 处理完成"
        
    } catch (Exception e) {
        print "   ❌ 处理细胞 ${cellIndex + 1} 时出错: ${e.getMessage()}"
        e.printStackTrace()
    }
}

// 步骤9: 生成调试报告
print "\n--- 📊 生成调试报告 ---"

// 保存详细的调试JSON
def debugReportFile = new File(coreOutputDir, "${coreId}_debug_report.json")
def jsonContent = new StringBuilder()
jsonContent.append("{\n")
jsonContent.append("  \"core_id\": \"${coreId}\",\n")
jsonContent.append("  \"image_name\": \"${name}\",\n")
jsonContent.append("  \"processing_time\": \"${new Date()}\",\n")
jsonContent.append("  \"total_cells_detected\": ${totalCells},\n")
jsonContent.append("  \"cells_debugged\": ${cellResults.size()},\n")
jsonContent.append("  \"pixel_calibration_um_per_pixel\": ${cal.getPixelWidthMicrons()},\n")
jsonContent.append("  \"core_bounds\": {\n")
jsonContent.append("    \"x\": ${coreMinX},\n")
jsonContent.append("    \"y\": ${coreMinY},\n")
jsonContent.append("    \"width\": ${coreWidth},\n")
jsonContent.append("    \"height\": ${coreHeight}\n")
jsonContent.append("  },\n")
jsonContent.append("  \"cells\": [\n")

cellResults.eachWithIndex { cellInfo, index ->
    jsonContent.append("    {\n")
    jsonContent.append("      \"cell_id\": \"${cellInfo.cell_id}\",\n")
    jsonContent.append("      \"centroid\": {\"x\": ${cellInfo.centroid.x}, \"y\": ${cellInfo.centroid.y}},\n")
    jsonContent.append("      \"area_um2\": ${cellInfo.area_um2},\n")
    jsonContent.append("      \"area_pixels\": ${cellInfo.area_pixels},\n")
    jsonContent.append("      \"original_image\": \"${cellInfo.original_image}\",\n")
    jsonContent.append("      \"masked_image\": \"${cellInfo.masked_image}\",\n")
    jsonContent.append("      \"measurements\": {\n")
    
    def measurementEntries = []
    cellInfo.measurements.each { key, value ->
        measurementEntries.add("        \"${key}\": ${value}")
    }
    jsonContent.append(measurementEntries.join(",\n"))
    jsonContent.append("\n      }\n")
    
    if (index < cellResults.size() - 1) {
        jsonContent.append("    },\n")
    } else {
        jsonContent.append("    }\n")
    }
}

jsonContent.append("  ]\n")
jsonContent.append("}\n")

debugReportFile.text = jsonContent.toString()

print "✅ 调试报告已保存: ${debugReportFile.absolutePath}"
print "📊 处理统计:"
print "   - 总检测细胞数: ${totalCells}"
print "   - 调试细胞数: ${cellResults.size()}"
print "   - 成功保存图像: ${cellResults.size() * 2} 个文件"

print "\n=========================================="
print "🎉 单核心Watershed调试完成！"
print "📁 所有结果保存在: ${coreOutputDir.absolutePath}"
print "=========================================="
