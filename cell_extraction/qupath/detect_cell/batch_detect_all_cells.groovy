/**
 * 批量检测所有细胞脚本
 * 适用于整个项目的批处理
 * 可以选择不同的检测方法
 */

// 检测方法选择
def USE_STARDIST = false  // 设置为true使用StarDist，false使用传统方法
def DETECTION_METHOD = "watershed"  // "watershed" 或 "positive" 或 "stardist"

// 获取项目中的所有图像
def project = getProject()
def imageList = project.getImageList()

print "开始批量处理 " + imageList.size() + " 个图像..."

// 创建输出目录
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'batch_cell_detection_results')
mkdirs(dirOutput)

// 处理每个图像
imageList.eachWithIndex { entry, index ->
    try {
        print "\n=== 处理图像 ${index + 1}/${imageList.size()}: ${entry.getImageName()} ==="
        
        // 打开图像
        def imageData = entry.readImageData()
        setImageData(imageData)
        
        def imageName = GeneralTools.getNameWithoutExtension(entry.getImageName())
        
        // 清除现有检测结果
        clearDetections()
        
        // 根据选择的方法进行检测
        switch (DETECTION_METHOD) {
            case "watershed":
                detectCellsWatershed()
                break
            case "positive":
                detectCellsPositive()
                break
            case "stardist":
                if (USE_STARDIST) {
                    detectCellsStarDist()
                } else {
                    print "StarDist未启用，使用Watershed方法"
                    detectCellsWatershed()
                }
                break
            default:
                detectCellsWatershed()
        }
        
        // 获取检测结果
        def detections = getDetectionObjects()
        print "检测到 " + detections.size() + " 个细胞"
        
        // 保存结果
        def resultFile = buildFilePath(dirOutput, imageName + '_cell_detection_results.txt')
        def results = []
        results.add("图像: " + imageName)
        results.add("检测方法: " + DETECTION_METHOD)
        results.add("检测到的细胞总数: " + detections.size())
        results.add("处理时间: " + new Date().toString())
        results.add("")
        
        // 如果是TMA，按核心统计
        if (isTMADearrayed()) {
            def hierarchy = getCurrentHierarchy()
            def cores = hierarchy.getTMAGrid().getTMACoreList().findAll()
            results.add("TMA核心统计:")
            cores.each { core ->
                def coreDetections = core.getChildObjects().findAll { it.isDetection() }
                def coreId = core.getDisplayedName() ?: "Unknown"
                results.add("  ${coreId}: ${coreDetections.size()} 个细胞")
            }
        }
        
        // 写入结果文件
        new File(resultFile).text = results.join('\n')
        
        // 保存项目数据
        entry.saveImageData(imageData)
        
        print "图像 ${imageName} 处理完成"
        
    } catch (Exception e) {
        print "处理图像 ${entry.getImageName()} 时出错: " + e.getMessage()
        e.printStackTrace()
    }
}

print "\n批量处理完成！结果保存在: " + dirOutput

// ============ 检测方法函数 ============

/**
 * Watershed细胞检测方法
 */
def detectCellsWatershed() {
    // 选择检测区域
    selectDetectionRegion()
    
    // 执行Watershed细胞检测
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
        '{"detectionImage":"Hematoxylin OD",' +
        '"requestedPixelSizeMicrons":0.5,' +
        '"backgroundRadiusMicrons":8.0,' +
        '"medianRadiusMicrons":0.0,' +
        '"sigmaMicrons":1.5,' +
        '"minAreaMicrons":10.0,' +
        '"maxAreaMicrons":400.0,' +
        '"threshold":0.1,' +
        '"watershedPostProcess":true,' +
        '"cellExpansionMicrons":5.0,' +
        '"includeNuclei":true,' +
        '"smoothBoundaries":true,' +
        '"makeMeasurements":true}')
}

/**
 * 阳性细胞检测方法
 */
def detectCellsPositive() {
    // 选择检测区域
    selectDetectionRegion()
    
    // 执行阳性细胞检测
    runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', 
        '{"detectionImageBrightfield":"Optical density sum",' +
        '"requestedPixelSizeMicrons":0.5,' +
        '"backgroundRadiusMicrons":8.0,' +
        '"backgroundByReconstruction":true,' +
        '"medianRadiusMicrons":0.0,' +
        '"sigmaMicrons":1.5,' +
        '"minAreaMicrons":10.0,' +
        '"maxAreaMicrons":400.0,' +
        '"threshold":0.01,' +
        '"maxBackground":2.0,' +
        '"watershedPostProcess":true,' +
        '"excludeDAB":false,' +
        '"cellExpansionMicrons":5.0,' +
        '"includeNuclei":true,' +
        '"smoothBoundaries":true,' +
        '"makeMeasurements":true}')
}

/**
 * StarDist检测方法
 */
def detectCellsStarDist() {
    // 注意：需要安装StarDist扩展
    print "StarDist检测需要安装相应的扩展插件"
    // 这里需要根据实际的StarDist API进行实现
}

/**
 * 选择检测区域
 */
def selectDetectionRegion() {
    def annotations = getAnnotationObjects()
    if (annotations.isEmpty()) {
        // 如果是TMA，选择TMA核心
        if (isTMADearrayed()) {
            selectTMACores()
            createAnnotationsFromPixelClassifier("tissueDetection", 0.0, 0.0)
            Thread.sleep(500)  // 等待组织检测完成
            selectAnnotations()
        } else {
            // 创建全图注释
            def imageServer = getCurrentImageData().getServer()
            def roi = ROIs.createRectangleROI(0, 0, imageServer.getWidth(), imageServer.getHeight(), getCurrentViewer().getImagePlane())
            def annotation = PathObjects.createAnnotationObject(roi)
            addObject(annotation)
            selectObjects(annotation)
        }
    } else {
        selectAnnotations()
    }
}
