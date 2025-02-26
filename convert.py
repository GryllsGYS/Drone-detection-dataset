import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
import random
import shutil


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    将 [x, y, width, height] 格式的边界框转换为 YOLO 格式 [center_x, center_y, width, height]（归一化到0-1）
    """
    if not isinstance(bbox, (list, np.ndarray)) or len(bbox) != 4:
        return None

    x, y, width, height = bbox

    # 计算中心点坐标
    center_x = (x + width / 2) / img_width
    center_y = (y + height / 2) / img_height

    # 归一化宽度和高度
    norm_width = width / img_width
    norm_height = height / img_height

    # 确保值在0-1范围内
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    norm_width = max(0, min(1, norm_width))
    norm_height = max(0, min(1, norm_height))

    return [center_x, center_y, norm_width, norm_height]


def main():
    # 定义类别
    classes = {
        'AIRPLANE': 0,
        'BIRD': 1,
        'DRONE': 2,
        'HELICOPTER': 3
    }

    # 加载 .mat 文件
    print("正在加载 MATLAB 数据文件...")
    mat_file = 'Training_data_V_array.mat'
    try:
        data = loadmat(mat_file)
        training_data = data['trainingDataArray']
    except Exception as e:
        print(f"加载 .mat 文件出错: {e}")
        return

    # 创建 YOLO 格式的目录结构
    yolo_dir = "yolo_dataset"
    os.makedirs(os.path.join(yolo_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_dir, "labels", "val"), exist_ok=True)

    # 获取训练数据
    print("正在处理训练数据...")

    # 使用索引访问表格数据，根据实际 mat 文件结构可能需要调整
    image_files = []
    total_images = training_data.shape[0]

    # 随机选择20%的数据作为验证集
    val_indices = set(random.sample(
        range(total_images), int(total_images * 0.2)))

    # 处理每张图片及其标注
    for i in range(total_images):
        try:
            # 获取图像文件路径
            image_path = str(training_data[i, 0][0])
            if image_path.startswith("'") and image_path.endswith("'"):
                image_path = image_path[1:-1]

            # 获取文件名
            img_filename = os.path.basename(image_path)
            img_name = os.path.splitext(img_filename)[0]

            # 判断是训练集还是验证集
            is_val = i in val_indices
            subset = "val" if is_val else "train"

            # 打开图像获取尺寸
            try:
                with Image.open(os.path.join("Training_data_V", img_filename)) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"无法打开图像 {img_filename}: {e}")
                continue

            # 创建 YOLO 格式的标签文件
            yolo_labels = []

            # 处理每个类别的边界框
            for class_idx, class_name in enumerate(['AIRPLANE', 'BIRD', 'DRONE', 'HELICOPTER']):
                bbox_data = training_data[i, class_idx + 1]

                # 边界框可能为空或包含多个框
                if bbox_data.size > 0:
                    # 处理单个边界框
                    if bbox_data.ndim == 1:
                        bbox = bbox_data.tolist()
                        yolo_bbox = convert_bbox_to_yolo(
                            bbox, img_width, img_height)
                        if yolo_bbox:
                            yolo_labels.append(
                                f"{classes[class_name]} {' '.join(map(str, yolo_bbox))}")

                    # 处理多个边界框
                    elif bbox_data.ndim == 2:
                        for j in range(bbox_data.shape[0]):
                            bbox = bbox_data[j].tolist()
                            yolo_bbox = convert_bbox_to_yolo(
                                bbox, img_width, img_height)
                            if yolo_bbox:
                                yolo_labels.append(
                                    f"{classes[class_name]} {' '.join(map(str, yolo_bbox))}")

            # 复制图像到目标目录
            src_img_path = os.path.join("Training_data_V", img_filename)
            dst_img_path = os.path.join(
                yolo_dir, "images", subset, img_filename)
            shutil.copy2(src_img_path, dst_img_path)

            # 写入 YOLO 格式的标签文件
            label_path = os.path.join(
                yolo_dir, "labels", subset, f"{img_name}.txt")
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

            # 添加到图像列表
            image_files.append(os.path.join("images", subset, img_filename))

            if (i + 1) % 100 == 0:
                print(f"已处理 {i+1}/{total_images} 张图像")

        except Exception as e:
            print(f"处理图像 {i} 时出错: {e}")
            continue

    # 创建训练和验证集文件列表
    train_list = [f for f in image_files if "/train/" in f]
    val_list = [f for f in image_files if "/val/" in f]

    with open(os.path.join(yolo_dir, "train.txt"), 'w') as f:
        f.write('\n'.join(train_list))

    with open(os.path.join(yolo_dir, "val.txt"), 'w') as f:
        f.write('\n'.join(val_list))

    # 创建数据集配置文件
    yaml_content = f"""# YOLO 数据集配置
path: {os.path.abspath(yolo_dir)}  # 数据集根路径
train: train.txt  # 训练图像相对路径
val: val.txt  # 验证图像相对路径

# 类别
nc: {len(classes)}  # 类别数量
names: {list(classes.keys())}  # 类别名称
"""

    with open(os.path.join(yolo_dir, "dataset.yaml"), 'w') as f:
        f.write(yaml_content)

    print(f"转换完成! 数据集已保存到 {yolo_dir} 目录")
    print(f"共处理 {len(train_list)} 张训练图像和 {len(val_list)} 张验证图像")


if __name__ == "__main__":
    main()
