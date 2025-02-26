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

    # 设置路径 - 更新为正确的文件路径
    matlab_data_path = os.path.join('Data', 'Training_data_V.mat')
    img_src_dir = os.path.join('Data', 'Training_data_V')

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
    try:
        training_data = mat_data.get('trainingData', None)
        if training_data is None:
            # 尝试不同的键名
            for key in mat_data.keys():
                if not key.startswith('__'):  # 忽略scipy.io内部变量
                    training_data = mat_data[key]
                    print(f"找到数据键: {key}")
                    break

        if training_data is None:
            raise ValueError("无法在.mat文件中找到训练数据")

        print(f"找到{len(training_data)}条训练数据记录")

        # 定义类别映射 - 按照MATLAB中的顺序
        classes = ['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']
        class_to_id = {cls: idx for idx, cls in enumerate(classes)}

        # 获取所有图像文件路径
        image_paths = []
        for root, _, files in os.walk(img_src_dir):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_paths.append(os.path.join(root, file))

        print(f"找到{len(image_paths)}个图像文件")

        # 分割训练集和验证集
        train_paths, val_paths = train_test_split(
            image_paths, test_size=0.2, random_state=42)

        print(f"训练集: {len(train_paths)}个图像, 验证集: {len(val_paths)}个图像")

        # 处理训练集
        process_dataset(train_paths, training_data,
                        classes, 'train', dataset_dir)

        # 处理验证集
        process_dataset(val_paths, training_data,
                        classes, 'val', dataset_dir)

        # 创建数据集配置文件
        create_dataset_yaml(classes, dataset_dir)

        print(f"转换完成！数据集保存在 {dataset_dir} 目录")

    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        print("请检查.mat文件结构并调整代码")


def process_dataset(image_paths, training_data, classes, dataset_type, dataset_dir):
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
        create_yolo_label(img_path, img_name, img_filename, training_data,
                          classes, dataset_type, dataset_dir)


def create_yolo_label(img_path, img_name, img_filename, training_data, classes, dataset_type, dataset_dir):
    """为单个图像创建YOLO格式的标签文件"""
    # 读取图像获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return

    img_height, img_width = img.shape[:2]

    # 创建标签列表
    bboxes = []

    # 在训练数据中查找匹配的图像记录
    for record in training_data:
        # 第1列是图像路径
        record_img_path = record[0] if isinstance(
            record, (list, np.ndarray)) else record.get('imageFilename', '')

        # 检查是否与当前图像匹配
        if img_name in record_img_path or os.path.basename(record_img_path) == img_filename:
            # 对每个类别（从第2列到第5列）
            for idx, class_name in enumerate(classes):
                # 获取此类别的边界框 (从idx+1索引获取，因为第1列是图像路径)
                bbox_col_idx = idx + 1

                # 获取边界框数据
                if isinstance(record, (list, np.ndarray)) and len(record) > bbox_col_idx:
                    class_bboxes = record[bbox_col_idx]
                elif isinstance(record, dict) and class_name + 'Bboxes' in record:
                    class_bboxes = record[class_name + 'Bboxes']
                else:
                    continue

                # 确保是可迭代的
                if not isinstance(class_bboxes, (list, np.ndarray)) or len(class_bboxes) == 0:
                    continue

                # 如果是单个边界框，确保它是二维的
                if not isinstance(class_bboxes[0], (list, np.ndarray)):
                    class_bboxes = [class_bboxes]

                # 处理每个边界框
                for bbox in class_bboxes:
                    if len(bbox) == 4:  # [x, y, width, height]
                        x, y, w, h = bbox

                        # 排除无效框
                        if w <= 0 or h <= 0:
                            continue

                        # 转换为YOLO格式：中心点坐标和宽高（归一化到0-1）
                        x_center = (x + w/2) / img_width
                        y_center = (y + h/2) / img_height
                        width = w / img_width
                        height = h / img_height

                        # 限制在[0,1]范围内
                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # 类别ID + 坐标
                        bboxes.append(
                            f"{idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

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
