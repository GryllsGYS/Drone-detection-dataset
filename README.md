# Drone-detection-dataset (YOLO格式转换版)

这是[原始Drone-detection-dataset](https://github.com/franziska-sn/Drone-detection-dataset)的**视频标签**改进版本，添加了Python支持，可将数据集转换为YOLO模型训练格式。

## 数据集概述

数据集包含红外(IR)和可见光，可用于训练和评估无人机检测传感器和系统。

**视频标签**: 飞机(Airplane)、鸟(Bird)、无人机(Drone)和直升机(Helicopter)

数据集包含650个视频(365个红外和285个可见光)。如果从所有视频中提取所有图像，数据集总共有203328张带标注的图像。

## 改进内容

本仓库对原始数据集做了以下改进：

1. **添加Python代码**：实现了数据集向YOLO格式的转换
2. **修改MATLAB代码**：优化了原始处理流程，便于与Python代码配合使用
3. **完整处理流程**：通过依次运行MATLAB和Python脚本，可将原始数据集转换为YOLO训练所需格式

## 使用说明

### 1. 使用MATLAB处理原始数据

在main_process.m中修改采样系数(默认为采样所有帧)
运行main_process.m提取视频帧和标注：

```matlab
run main_process.m
```
### 2. 使用Python脚本转换为YOLO格式
然后运行Python脚本将MATLAB生成的数据转换为YOLO格式：
```python
python convert.py
```
### 生成的YOLO数据集结构
转换完成后，将生成YOLO格式的数据集，包括：
* 图像文件
* 对应的标签文件(.txt)
* 训练/验证集划分

---

**PS**：在连续采样到第225个视频时会报错，故分成了两个步骤来采样所有的视频