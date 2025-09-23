/**
 * StarDist深度学习细胞核检测脚本
 * 使用预训练的深度学习模型检测细胞核
 * 对重叠和聚集的细胞有更好的分离效果
 */

import qupath.ext.stardist.StarDist2D

// 获取当前图像信息
def imageData = getCurrentImageData()
def imageServer = imageData.getServer()
def name = GeneralTools.getNameWithoutExtension(imageServer.getMetadata().getName())

print "使用StarDist检测细胞核: " + name

// 选择检测区域
def annotations = getAnnotationObjects()
if (annotations.isEmpty()) {
    print "未找到注释区域，将检测整个图像"
    def roi = ROIs.createRectangleROI(0, 0, imageServer.getWidth(), imageServer.getHeight(), getCurrentViewer().getImagePlane())
    def annotation = PathObjects.createAnnotationObject(roi)
    addObject(annotation)
    selectObjects(annotation)
} else {
    print "在 " + annotations.size() + " 个注释区域内检测细胞"
    selectAnnotations()
}

// StarDist参数配置
def pathModel = '/path/to/stardist/model'  // 需要指定StarDist模型路径

// 使用预训练的H&E模型
def stardist = StarDist2D.builder(pathModel)
    .threshold(0.5)              // 检测阈值
    .channels('DAPI')            // 对于H&E图像，使用Hematoxylin通道
    .normalizePercentiles(1, 99) // 归一化百分位数
    .pixelSize(0.5)              // 像素尺寸（微米）
    .cellExpansion(5.0)          // 细胞扩展距离
    .cellConstrainScale(1.5)     // 细胞约束比例
    .measureShape()              // 测量形状特征
    .measureIntensity()          // 测量强度特征
    .build()

// 执行检测
stardist.detectObjects(imageData, getSelectedObjects())

// 获取检测结果
def detectionObjects = getDetectionObjects()
print "StarDist检测到 " + detectionObjects.size() + " 个细胞核"

// 可选：基于检测结果进行细胞分类
def nucleiArea = detectionObjects.collect { 
    it.getROI().getScaledArea(imageData.getServer().getPixelCalibration()) 
}

def avgArea = nucleiArea.sum() / nucleiArea.size()
print "平均细胞核面积: " + String.format("%.2f", avgArea) + " μm²"

// 根据面积大小分类
def pathClassSmall = getPathClass("Small Cell")
def pathClassLarge = getPathClass("Large Cell")

detectionObjects.each { detection ->
    def area = detection.getROI().getScaledArea(imageData.getServer().getPixelCalibration())
    if (area < avgArea) {
        detection.setPathClass(pathClassSmall)
    } else {
        detection.setPathClass(pathClassLarge)
    }
}

print "StarDist细胞检测完成！"
