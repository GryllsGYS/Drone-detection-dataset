import os
import scipy.io
import numpy as np
import shutil
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml


def convert_matlab_to_yolo():
    """将MATLAB生成的数据集转换为YOLO格式"""

    print("开始转换MATLAB数据集到YOLO格式...")

    # 设置路径
    matlab_data_path = os.path.join('Data', 'Training_data_IR.mat')
    img_src_dir = os.path.join('Data', 'Training_data_IR')

    # 创建YOLO格式的数据集目录
    dataset_dir = 'dataset_yolo'
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 创建必要的目录结构
    os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)

    # 加载MATLAB数据
    print(f"加载MATLAB数据: {matlab_data_path}")
    mat_data = scipy.io.loadmat(matlab_data_path, simplify_cells=True)

    # 获取trainingData表格
    # 注意：这里假设trainingData是一个字段，具体结构需要根据加载后的数据调整
    try:
        training_data = mat_data.get('trainingData', None)
        if training_data is None:
            # 尝试不同的键名
            for key in mat_data.keys():
                if not key.startswith('__'):  # 忽略scipy.io内部变量
                    training_data = mat_data[key]
                    break

        # 定义类别映射
        classes = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}

        # 获取所有图像文件路径
        image_paths = []
        for root, _, files in os.walk(img_src_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))

        # 分割训练集和验证集
        train_paths, val_paths = train_test_split(
            image_paths, test_size=0.2, random_state=42)

        # 处理训练集
        process_dataset(train_paths, training_data,
                        class_to_id, 'train', dataset_dir)

        # 处理验证集
        process_dataset(val_paths, training_data,
                        class_to_id, 'val', dataset_dir)

        # 创建数据集配置文件
        create_dataset_yaml(classes, dataset_dir)

        print(f"转换完成！数据集保存在 {dataset_dir} 目录")

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        print("请检查.mat文件结构并调整代码")


def process_dataset(image_paths, training_data, class_to_id, dataset_type, dataset_dir):
    """处理数据集（训练集或验证集）"""
    print(f"处理{dataset_type}集...")

    for img_path in tqdm(image_paths):
        # 获取基本文件名
        img_filename = os.path.basename(img_path)
        img_name, _ = os.path.splitext(img_filename)

        # 复制图像到目标目录
        dst_img_path = os.path.join(
            dataset_dir, 'images', dataset_type, img_filename)
        shutil.copy2(img_path, dst_img_path)

        # 为此图像创建YOLO格式标签
        create_yolo_label(img_path, img_name, training_data,
                          class_to_id, dataset_type, dataset_dir)


def create_yolo_label(img_path, img_name, training_data, class_to_id, dataset_type, dataset_dir):
    """为单个图像创建YOLO格式的标签文件"""
    # 读取图像获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    img_height, img_width = img.shape[:2]

    # 查找此图像的标注数据
    bboxes = []
    for i, row in enumerate(training_data):
        # 这里需要根据实际mat文件结构调整
        # 假设row包含图像路径和边界框信息
        row_img_path = row['imageFilename'] if isinstance(
            row, dict) else row[0]

        if img_name in row_img_path:
            # 对每个类别检查边界框
            for cls_name in class_to_id.keys():
                bbox_key = f'{cls_name}Bboxes'
                if bbox_key in row or (isinstance(row, (list, tuple)) and len(row) > class_to_id[cls_name] + 1):
                    cls_bboxes = row.get(bbox_key, []) if isinstance(
                        row, dict) else row[class_to_id[cls_name] + 1]
                    if len(cls_bboxes) > 0:
                        # MATLAB中可能有多个边界框
                        for bbox in cls_bboxes:
                            if len(bbox) == 4:  # [x, y, width, height]
                                x, y, w, h = bbox

                                # 转换为YOLO格式：中心点坐标和宽高（归一化到0-1）
                                x_center = (x + w/2) / img_width
                                y_center = (y + h/2) / img_height
                                width = w / img_width
                                height = h / img_height

                                # 类别ID + 坐标
                                cls_id = class_to_id[cls_name]
                                bboxes.append(
                                    f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 写入标签文件
    label_path = os.path.join(dataset_dir, 'labels',
                              dataset_type, f"{img_name}.txt")
    with open(label_path, 'w') as f:
        f.write('\n'.join(bboxes))


def create_dataset_yaml(classes, dataset_dir):
    """创建数据集的YAML配置文件"""
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')

    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)


if __name__ == "__main__":
    convert_matlab_to_yolo()
