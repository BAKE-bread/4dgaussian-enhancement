import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def find_cam_dir(file_path):
    """从文件路径向上查找第一个以 'cam' 开头的目录名，返回目录名或 None"""
    path = os.path.dirname(file_path)
    while path and path != os.path.dirname(path):  # 避免无限循环
        base = os.path.basename(path)
        if base.startswith('cam'):
            return base
        path = os.path.dirname(path)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu', weights_only=False))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # 收集所有待处理的图像文件路径
    filenames = []
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        # 递归遍历所有子目录，查找图像文件
        for root, dirs, files in os.walk(args.img_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    # 只保留属于某个 camXX 目录下的文件
                    if find_cam_dir(full_path) is not None:
                        filenames.append(full_path)
    
    if not filenames:
        print("No image files found in the given path (or none belong to a camXX directory).")
        exit(0)
    
    # 创建输出根目录
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f"Warning: Could not read image {filename}, skipping...")
            continue
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 获取所属的 cam 目录名
        cam_dir = find_cam_dir(filename)
        if cam_dir is None:
            print(f"Warning: Could not determine cam directory for {filename}, skipping...")
            continue
        
        # 创建输出目录: /output/camXX/depth_maps/
        output_subdir = os.path.join(args.outdir, cam_dir, 'depth_maps')
        os.makedirs(output_subdir, exist_ok=True)
        
        # 输出文件名（保留原始文件名，扩展名改为 .png）
        file_name = os.path.basename(filename)
        output_filename = os.path.splitext(file_name)[0] + '.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(output_path, combined_result)
    
    print(f"Processing completed. Results saved to {args.outdir}")