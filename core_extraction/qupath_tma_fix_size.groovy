// QuPath TMA核心提取脚本 (圆形核心)
// 使用TMA核心的原始圆形ROI进行提取
// 支持命令行参数: input_path, output_path, target_size

import javax.imageio.ImageIO
import qupath.lib.regions.RegionRequest
import java.awt.geom.AffineTransform
import java.awt.image.AffineTransformOp

// 获取命令行参数
def args = getQuPath().getScriptArgs()
def inputPath = args.length > 0 ? args[0] : null
def outputPath = args.length > 1 ? args[1] : '/home/zheng/zheng/rosie_reproduce/core_extraction/tma_cores_pngs'
def targetSize = args.length > 2 ? Double.parseDouble(args[2]) : 1024.0

// 验证输入参数
if (inputPath == null) {
    print "Usage: qupath script -i <input_image> <script_path> <output_path> <target_size>"
    print "Example: qupath script -i /path/to/image.svs /path/to/script.groovy /path/to/output 512"
    return
}

// 设置输出目录
def dirOutput = outputPath
new File(dirOutput).mkdirs()

print "Script parameters:"
print "   Input: " + inputPath
print "   Output: " + dirOutput
print "   Target size: " + targetSize + " pixels"

// 获取当前图像数据
def imageData = getCurrentImageData()
def server = getCurrentServer()
def path = server.getPath()
def imageWidth = server.getWidth()
def imageHeight = server.getHeight()

print "Starting TMA circular core extraction..."
print "Output directory: " + dirOutput

// 步骤1: 检查TMA是否已经去阵列化
def hierarchy = getCurrentHierarchy()
def isTMADearrayed = hierarchy.getTMAGrid() != null

if (!isTMADearrayed) {
    print "Running TMA dearraying..."
    
    // 运行TMA去阵列化插件
    runPlugin('qupath.imagej.detect.dearray.TMADearrayerPluginIJ', 
        '{"coreDiameterMM":1.9,"labelsHorizontal":"1-6","labelsVertical":"1-12","labelOrder":"Row first","densityThreshold":5,"boundsScale":105}')
    
    // 重新获取层次结构
    hierarchy = getCurrentHierarchy()
    isTMADearrayed = hierarchy.getTMAGrid() != null
    
    if (isTMADearrayed) {
        print "TMA dearraying completed"
    } else {
        print "TMA dearraying failed"
        return
    }
} else {
    print "TMA already dearrayed"
}

// 步骤2: 检查TMA网格尺寸
def grid = hierarchy.getTMAGrid()
def cols = grid.getGridWidth()
def rows = grid.getGridHeight()

print "TMA grid size: " + cols + " x " + rows

// 步骤3: 获取TMA核心列表
def cores = hierarchy.getTMAGrid().getTMACoreList().findAll()
print "Found " + cores.size() + " TMA cores"

// 步骤4: 设置提取参数
double targetImageSize = targetSize  // 使用命令行参数指定的目标图片尺寸
double pixelSize = server.getPixelCalibration().getAveragedPixelSize()
double physicalCoreSize = 1.0  // 物理核心尺寸（毫米）

// 计算下采样因子：目标图片尺寸 / 原始图片尺寸
double originalImageSize = physicalCoreSize * 1000 / pixelSize  // 原始图片尺寸（像素）
double downsample = originalImageSize / targetImageSize

print "Extraction parameters:"
print "   Original pixel size: " + String.format("%.3f", pixelSize) + " μm/pixel"
print "   Physical core size: " + physicalCoreSize + " mm"
print "   Original image size: " + String.format("%.0f", originalImageSize) + " pixels"
print "   Target image size: " + targetImageSize + " pixels"
print "   Downsample factor: " + String.format("%.2f", downsample)

// 步骤5: 提取每个核心 (使用原始圆形ROI)
def caseID = "TumorCenter_HE_block1"
def coreNum = 0

for (core in cores) {
    try {
        // 获取核心信息
        def coreROI = core.getROI()
        def centroidX = coreROI.getCentroidX()
        def centroidY = coreROI.getCentroidY()
        
        // 获取核心标签
        def coreLabel = core.getName()
        if (coreLabel == null || coreLabel.isEmpty()) {
            coreLabel = "core_" + (coreNum + 1)
        }
        
        print "Extracting circular core: " + coreLabel + " (centroid: " + String.format("%.1f", centroidX) + ", " + String.format("%.1f", centroidY) + ")"
        
        // 获取核心的边界框
        def x = coreROI.getBoundsX()
        def y = coreROI.getBoundsY()
        def width = coreROI.getBoundsWidth()
        def height = coreROI.getBoundsHeight()
        
        print "   Core bounds: x=" + x + ", y=" + y + ", w=" + width + ", h=" + height
        
        // 使用原始ROI提取
        def regionRequest = RegionRequest.createInstance(path, downsample, coreROI)
        def tileImage = server.readRegion(regionRequest)

        // 对提取图像进行180°旋转（上下左右都翻转）
        def w = tileImage.getWidth()
        def h = tileImage.getHeight()
        def transform = AffineTransform.getRotateInstance(Math.PI, w/2.0, h/2.0)
        def op = new AffineTransformOp(transform, AffineTransformOp.TYPE_BILINEAR)
        def rotated = new java.awt.image.BufferedImage(w, h, tileImage.getType())
        op.filter(tileImage, rotated)

        // 保存瓦片：解析行列后按180°镜像重命名
        Integer coreRow = null
        Integer coreCol = null
        try {
            coreRow = core.getRow()
            coreCol = core.getColumn()
        } catch (Throwable t) {}
        if (coreRow == null || coreCol == null) {
            def m = (coreLabel =~ /(\d+)[-_.](\d+)/)
            if (m.find()) {
                coreRow = Integer.parseInt(m.group(1))
                coreCol = Integer.parseInt(m.group(2))
            }
        }
        def outputFilename
        if (coreRow != null && coreCol != null) {
            def mirroredRow = rows - coreRow + 1
            def mirroredCol = cols - coreCol + 1
            outputFilename = String.format("%s_%d-%d_circular.png", caseID, (int)mirroredRow, (int)mirroredCol)
        } else {
            outputFilename = caseID + "_" + coreLabel + "_circular.png"
        }
        def outputFile = new File(dirOutput, outputFilename)
        ImageIO.write(rotated, "png", outputFile)
        
        print "   Saved: " + outputFilename + " (" + tileImage.getWidth() + "x" + tileImage.getHeight() + ")"
        
        coreNum++
        
    } catch (Exception e) {
        print "   Failed to extract core: " + e.getMessage()
    }
}

print "TMA circular core extraction completed!"
print "Output directory: " + dirOutput
print "Successfully extracted: " + coreNum + " cores"
