/**
 * TMA所有细胞检测脚本
 * 在TMA核心区域内检测所有细胞（不仅仅是阳性细胞）
 * 基于现有的TMA工作流程进行改进
 */

// 确保TMA已经去阵列化
if (!isTMADearrayed()) {
    print "正在执行TMA去阵列化..."
    runPlugin('qupath.imagej.detect.dearray.TMADearrayerPluginIJ', 
        '{"coreDiameterMM":1.9,"labelsHorizontal":"1-6","labelsVertical":"1-12","labelOrder":"Row first","densityThreshold":5,"boundsScale":105}')
}

// 重新标记TMA网格
relabelTMAGrid("1-6", "1-12", true)

// 选择TMA核心并检测组织区域
print "检测TMA核心中的组织区域..."
selectTMACores()
createAnnotationsFromPixelClassifier("tissueDetection", 0.0, 0.0)

// 等待组织检测完成
Thread.sleep(1000)

// 在组织区域内检测所有细胞
print "在组织区域内检测所有细胞..."
selectAnnotations()

// 使用通用细胞检测（适用于H&E染色）
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
    '{"detectionImage":"Hematoxylin OD",' +
    '"requestedPixelSizeMicrons":0.5,' +
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

// 获取检测结果统计
def detectionObjects = getDetectionObjects()
def totalCells = detectionObjects.size()
print "总共检测到 " + totalCells + " 个细胞"

// 按TMA核心统计细胞数量
def hierarchy = getCurrentHierarchy()
def cores = hierarchy.getTMAGrid().getTMACoreList().findAll()
def coreStats = [:]

cores.each { core ->
    def coreDetections = core.getChildObjects().findAll { it.isDetection() }
    def coreId = core.getDisplayedName() ?: "Unknown"
    coreStats[coreId] = coreDetections.size()
    
    if (coreDetections.size() > 0) {
        print "TMA核心 ${coreId}: ${coreDetections.size()} 个细胞"
    }
}

// 可选：基于形态学特征对细胞进行分类
print "正在对细胞进行形态学分类..."

def pathClassSmall = getPathClass("Small Cell")
def pathClassMedium = getPathClass("Medium Cell") 
def pathClassLarge = getPathClass("Large Cell")

// 计算面积分位数用于分类
def areas = detectionObjects.collect { 
    it.getROI().getScaledArea(getCurrentImageData().getServer().getPixelCalibration()) 
}
areas.sort()

def q33 = areas[(int)(areas.size() * 0.33)]
def q66 = areas[(int)(areas.size() * 0.66)]

// 分类细胞
def smallCount = 0, mediumCount = 0, largeCount = 0

detectionObjects.each { detection ->
    def area = detection.getROI().getScaledArea(getCurrentImageData().getServer().getPixelCalibration())
    if (area < q33) {
        detection.setPathClass(pathClassSmall)
        smallCount++
    } else if (area < q66) {
        detection.setPathClass(pathClassMedium)
        mediumCount++
    } else {
        detection.setPathClass(pathClassLarge)
        largeCount++
    }
}

print "细胞分类结果:"
print "- 小细胞: " + smallCount + " 个 (< ${String.format('%.1f', q33)} μm²)"
print "- 中等细胞: " + mediumCount + " 个 (${String.format('%.1f', q33)}-${String.format('%.1f', q66)} μm²)"
print "- 大细胞: " + largeCount + " 个 (> ${String.format('%.1f', q66)} μm²)"

// 导出测量结果
def tmaName = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'cell_measurements')
mkdirs(dirOutput)

def filename = buildFilePath(dirOutput, tmaName.replaceAll("\\s", "") + '_all_cells.csv')
saveTMAMeasurements(filename)

print "所有细胞检测完成！结果已保存到: " + filename
