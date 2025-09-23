# Cell Extraction Research Project 🔬

This repository focuses on analyzing cell extraction technology from pathological images, with a particular emphasis on SVS file analysis and QuPath integration.

## 🎯 Project Overview

This project provides comprehensive tools for analyzing pathological images, specifically SVS (Aperio) files, with detailed metadata extraction, location analysis, and thumbnail generation capabilities.

## 📁 Project Structure

### Core Analysis Tools
- **`svs_analyzer.py`** - Comprehensive SVS file analyzer with metadata extraction and visualization
- **`svs_location_analyzer.py`** - Specialized location and spatial information analyzer
- **`simple_thumbnail.py`** - Lightweight thumbnail generator for SVS files

### Generated Reports
- **`svs_analysis_report.json`** - Complete metadata analysis report
- **`svs_location_report.json`** - Detailed location and spatial information
- **`SVS_Location_Analysis_Summary.md`** - Human-readable analysis summary

### Visualizations
- **`svs_analysis_chart.png`** - Multi-panel analysis visualization
- **`cd3_block1_simple_256x256.jpg`** - Sample thumbnail

## 🔧 Key Features

### SVS File Analysis
- **Complete metadata extraction** - All 33 metadata fields analyzed
- **Multi-level resolution analysis** - 3 resolution levels with memory usage estimates
- **Pixel calibration** - Precise micrometer-per-pixel measurements
- **Location information** - Detailed spatial and regional analysis

### QuPath Integration
- **QuPath v0.4.3** installed and configured
- **Groovy script support** for automated analysis
- **TMA analysis capabilities** for tissue microarray processing

### Image Processing
- **Thumbnail generation** - Multiple sizes (256x256, 512x512, 1024x1024)
- **Memory-optimized processing** - Handles large SVS files efficiently
- **Multi-format support** - SVS, TIFF, and other pathological image formats

## 📊 Analysis Results

### CD3 Immunohistochemistry Analysis
- **File**: TumorCenter_CD3_block1.svs
- **Size**: 2.68 GB
- **Resolution**: 0.121295 μm/pixel
- **Magnification**: 82.44x
- **Effective region**: 177664 × 268800 pixels
- **Scan date**: September 13, 2022

### Key Findings
- ✅ Complete location information extracted
- ✅ Multi-level resolution structure analyzed
- ✅ Pixel calibration verified
- ❌ No direct TMA metadata (requires external mapping)

## 🚀 Quick Start

### Prerequisites
```bash
conda create -n cell-extraction python=3.9
conda activate cell-extraction
conda install -c conda-forge openslide-python pillow numpy matplotlib
```

### Basic Usage
```bash
# Analyze SVS file
python svs_analyzer.py /path/to/file.svs

# Generate location report
python svs_location_analyzer.py /path/to/file.svs

# Create thumbnails
python simple_thumbnail.py
```

### QuPath Integration
```bash
# Run QuPath analysis
qupath script -i /path/to/file.svs /path/to/script.groovy

# Check QuPath version
qupath --version
```

## 📈 Research Applications

### Pathological Image Analysis
- **Cell detection and counting** - CD3+ T cell analysis
- **Tissue segmentation** - Automated region identification
- **Quantitative analysis** - Density and distribution metrics

### TMA (Tissue Microarray) Processing
- **Batch analysis** - Process multiple cores simultaneously
- **Position mapping** - Coordinate TMA cores with clinical data
- **Quality control** - Automated tissue detection and validation

## 🔬 Technical Specifications

### Supported Formats
- **SVS** (Aperio) - Primary focus
- **TIFF/BigTIFF** - High-resolution images
- **OME-TIFF** - Multi-dimensional images
- **Other formats** - Via Bio-Formats

### Performance
- **Memory efficient** - Handles 2.68GB files
- **Multi-threaded** - Parallel processing support
- **Caching** - Optimized tile access

## 📚 Documentation

- **`SVS_Location_Analysis_Summary.md`** - Detailed analysis methodology
- **`requirements.txt`** - Python dependencies
- **Generated reports** - JSON format for programmatic access

## 🤝 Contributing

This project is part of ongoing research in pathological image analysis. Contributions and improvements are welcome!

## 📄 License

See LICENSE file for details.
