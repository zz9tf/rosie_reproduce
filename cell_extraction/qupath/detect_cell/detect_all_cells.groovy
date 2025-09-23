/**
 * 通用细胞检测脚本 - 适用于H&E染色
 * 检测所有细胞（包括细胞核和细胞边界）
 */

// 获取当前图像信息
def imageData = getCurrentImageData()
def imageServer = imageData.getServer()
def name = GeneralTools.getNameWithoutExtension(imageServer.getMetadata().getName())

print "开始检测细胞: " + name

// 选择要检测的区域（如果有注释区域则选择注释，否则检测整个图像）
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    print "未找到注释区域，将检测整个图像"
    // 创建全图注释
    def roi = ROIs.createRectangleROI(0, 0, imageServer.getWidth(), imageServer.getHeight(), getCurrentViewer().getImagePlane())
    def annotation = PathObjects.createAnnotationObject(roi)
    addObject(annotation)
    selectObjects(annotation)
} else {
    print "在 " + annotations.size() + " 个注释区域内检测细胞"
    selectAnnotations()
}

// 执行细胞检测
// 参数说明：
// - detectionImage: 用于检测的图像通道
// - requestedPixelSizeMicrons: 请求的像素尺寸（微米）
// - backgroundRadiusMicrons: 背景半径
// - medianRadiusMicrons: 中值滤波半径
// - sigmaMicrons: 高斯模糊sigma值
// - minAreaMicrons: 最小细胞面积
// - maxAreaMicrons: 最大细胞面积
// - threshold: 检测阈值
// - watershedPostProcess: 是否使用分水岭后处理
// - cellExpansionMicrons: 细胞边界扩展距离
// - includeNuclei: 是否包含细胞核
// - smoothBoundaries: 是否平滑边界

runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', 
    '{"detectionImage":"Hematoxylin OD",' +
    '"requestedPixelSizeMicrons":0.5,' +
    '"backgroundRadiusMicrons":8.0,' +
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
print "检测到 " + detectionObjects.size() + " 个细胞对象"

// 可选：为不同类型的细胞设置分类
// 例如基于大小或形状特征
def smallCells = detectionObjects.findAll { it.getROI().getScaledArea(imageData.getServer().getPixelCalibration()) < 50 }
def largeCells = detectionObjects.findAll { it.getROI().getScaledArea(imageData.getServer().getPixelCalibration()) >= 50 }

print "小细胞: " + smallCells.size() + " 个"
print "大细胞: " + largeCells.size() + " 个"

print "细胞检测完成！"
