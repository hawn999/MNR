import os
import argparse
import numpy as np
import cv2
import torch
import math

# 动态导入项目中的数据集类
from data import RAVEN, PGM, Analogy, CLEVR_MATRIX


def save_stitched_image(image_data, file_path, is_color=False):
    """
    将一个样本的所有面板拼接成一张大图并保存。
    image_data: (num_panels, H, W) for grayscale or (num_panels, H, W, 3) for color
    """
    # 如果是PyTorch张量，转换为Numpy数组
    if isinstance(image_data, torch.Tensor):
        image_data = image_data.numpy()

    # 如果是彩色图片 (N, C, H, W)，转换为 (N, H, W, C) 以便拼接
    if is_color and image_data.shape[1] in [1, 3]:
        image_data = np.transpose(image_data, (0, 2, 3, 1))

    num_panels, height, width = image_data.shape[0], image_data.shape[1], image_data.shape[2]

    # 计算网格大小，例如16个面板为4x4，9个为3x3
    grid_size = int(math.sqrt(num_panels))
    if grid_size * grid_size != num_panels:
        print(f"警告: 面板数量 {num_panels} 不是完美的平方数，拼接可能不正确。")
        return

    # 创建一个空白画布来容纳拼接后的图像
    if is_color:
        canvas = np.zeros((height * grid_size, width * grid_size, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((height * grid_size, width * grid_size), dtype=np.uint8)

    # 将每个小图填充到画布上
    for i, panel in enumerate(image_data):
        row = i // grid_size
        col = i % grid_size

        # 将可能存在的浮点数像素值（如0-1范围）转换为8位无符号整数（0-255范围）
        if panel.max() <= 1.0 and panel.dtype != np.uint8:
            panel = (panel * 255).astype(np.uint8)

        canvas[row * height:(row + 1) * height, col * width:(col + 1) * width] = panel

    # 保存拼接后的图片
    cv2.imwrite(file_path, canvas)
    print(f"已保存拼接图片至: {file_path}")


def main():
    parser = argparse.ArgumentParser(description='根据序号保存数据集的原始图片')
    parser.add_argument('--dataset-dir', default='/home/scxhc1/nvme_data/resized_datasets_raven',
                        help='path to dataset')
    parser.add_argument('--dataset-name', default='RAVEN',
                        help='dataset name')
    parser.add_argument('--data-split', default='test', help='数据分区 (train, val, or test)')
    parser.add_argument('--indices', default="90", type=str, help='需要提取的样本序号，用逗号分隔 (e.g., "10,25,103")')
    parser.add_argument('--output-dir', default='./incorrect_images', help='保存图片的输出文件夹')
    parser.add_argument('--subset', default='None', type=str,
                        help='subset selection for dataset')
    args = parser.parse_args()

    # 1. 解析参数
    full_dataset_path = os.path.join(args.dataset_dir, args.dataset_name)
    indices_to_save = [int(i) for i in args.indices.split(',')]

    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 根据数据集名称选择并实例化Dataset
    print(f"正在加载数据集: {args.dataset_name}...")
    dataset = None
    is_color = False

    # [cite_start]这个逻辑块与 main.py 中的 get_data_loader 类似 [cite: 60]
    if args.dataset_name in ('RAVEN', 'RAVEN-FAIR', 'I-RAVEN'):
        dataset = RAVEN(full_dataset_path, data_split=args.data_split, image_size=160, transform=None)
    elif 'PGM' in args.dataset_name:
        dataset = PGM(full_dataset_path, data_split=args.data_split, image_size=160, transform=None)
    elif 'Analogy' in args.dataset_name:
        dataset = Analogy(full_dataset_path, data_split=args.data_split, image_size=160, transform=None)
    elif 'CLEVR-Matrix' in args.dataset_name:
        dataset = CLEVR_MATRIX(full_dataset_path, data_split=args.data_split, image_size=240, transform=None)
        is_color = True  # CLEVR 是彩色图片 [cite: 4]
    else:
        raise ValueError(f"不支持的数据集: {args.dataset_name}")

    if not dataset:
        print("数据集加载失败！")
        return

    # 3. 循环处理每个指定的序号
    for index in indices_to_save:
        try:
            # 直接调用内部的 _get_data 或 get_data 方法来绕过transform，获取原始numpy数据
            if hasattr(dataset, '_get_data'):
                # [cite_start]适用于 RAVEN, PGM, Analogy [cite: 93, 100, 113]
                raw_image, _, data_file_name = dataset._get_data(index)
            elif hasattr(dataset, 'get_data'):
                # [cite_start]适用于 CLEVR_MATRIX [cite: 106]
                raw_image, _, data_file_name = dataset.get_data(index)
            else:
                # 作为备选方案，使用 __getitem__
                raw_image, _, _, _, data_file_name = dataset[index]

            print(f"正在处理序号: {index} | 文件: {data_file_name}")

            # 4. 拼接并保存图片
            output_filename = f"{args.dataset_name}_{args.data_split}_idx_{index}.png"
            output_filepath = os.path.join(args.output_dir, output_filename)
            save_stitched_image(raw_image, output_filepath, is_color)

        except IndexError:
            print(f"错误: 序号 {index} 超出数据集范围 (总大小: {len(dataset)})。")
        except Exception as e:
            print(f"处理序号 {index} 时发生错误: {e}")


if __name__ == '__main__':
    main()