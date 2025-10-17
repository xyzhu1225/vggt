from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from visual_util import predictions_to_glb, predictions_to_ply

import torch
import os
import argparse
import shutil
import cv2
import numpy as np

def extract_frames_from_video(video_path, frame_interval_sec=1, tmp_dir="./tmp_imgs"):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * frame_interval_sec)
    image_paths = []

    count = 0
    frame_idx = 0
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        count += 1
        if count % frame_interval == 0:
            image_path = os.path.join(tmp_dir, f"{frame_idx:06d}.png")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            frame_idx += 1

    vs.release()
    return sorted(image_paths)


def gather_image_paths(input_path, tmp_dir="./tmp"):
    """
    支持:
    - 单图片
    - 多图片（逗号分隔）
    - 文件夹
    - 视频文件
    所有图片都会被复制/保存到 tmp_dir/images 下。
    """
    exts_img = [".png", ".jpg", ".jpeg"]
    exts_vid = [".mp4", ".mov", ".avi", ".mkv"]
    tmp_dir = os.path.join(tmp_dir, "images")
    # 清空临时目录
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    # 多图情况（逗号分隔）
    if "," in input_path:
        paths = [p.strip() for p in input_path.split(",") if os.path.exists(p.strip())]
        image_paths = []
        for i, src_path in enumerate(paths):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            shutil.copy2(src_path, dst_path)
            image_paths.append(dst_path)
        return sorted(image_paths)

    # 视频文件
    if any(input_path.lower().endswith(ext) for ext in exts_vid):
        print(f"[INFO] Detected video input: {input_path}")
        return extract_frames_from_video(input_path, tmp_dir=tmp_dir)

    # 文件夹
    if os.path.isdir(input_path):
        imgs = [
            os.path.join(input_path, f)
            for f in sorted(os.listdir(input_path))
            if any(f.lower().endswith(ext) for ext in exts_img)
        ]
        if not imgs:
            raise ValueError(f"No valid image found in directory: {input_path}")

        image_paths = []
        for i, src_path in enumerate(imgs):
            ext = os.path.splitext(src_path)[1]
            dst_path = os.path.join(tmp_dir, f"{i:06d}{ext}")
            shutil.copy2(src_path, dst_path)
            image_paths.append(dst_path)
        return sorted(image_paths)

    # 单图片
    if os.path.isfile(input_path) and any(input_path.lower().endswith(ext) for ext in exts_img):
        ext = os.path.splitext(input_path)[1]
        dst_path = os.path.join(tmp_dir, f"000000{ext}")
        shutil.copy2(input_path, dst_path)
        return [dst_path]

    raise ValueError(f"Invalid input path: {input_path}")

# ----------------------------------------
# Core: Run VGGT
# ----------------------------------------
def run_vggt(args):
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print(f"[INFO] Using device: {args.device}")
    print(f"[INFO] Loading model from {args.model_path}")
    model = VGGT.from_pretrained(args.model_path).to(args.device)

    # 解析输入
    image_paths = gather_image_paths(args.input_path)
    print(f"[INFO] Collected {len(image_paths)} images for inference.")

    # 载入 & 预处理
    images = load_and_preprocess_images(image_paths).to(args.device)

    # 推理
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        predictions = model(images)

    # 提取外参 & 内参
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    def tensor_to_numpy(x):
        """递归把 tensor 转为 numpy，如果是 list 就递归处理里面的元素"""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy().squeeze(0)
        elif isinstance(x, list):
            return [tensor_to_numpy(i) for i in x]
        elif isinstance(x, dict):
            return {k: tensor_to_numpy(v) for k, v in x.items()}
        else:
            return x  # 不处理其他类型
    
    # 转换 predictions
    predictions = tensor_to_numpy(predictions)
    predictions["images"] = images.cpu().numpy()
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    
    # 输出
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    prediction_save_path = os.path.join("./tmp", "predictions_vggt.npz")
    np.savez(prediction_save_path, **predictions)
    
    if args.output_type == "ply":
        out_path, pointcloud = predictions_to_ply(predictions, out_path=args.output_path, conf_thres=args.conf_thres)
        print(f"[INFO] Wrote PLY to: {out_path}")
    else:
        glbscene = predictions_to_glb(predictions, conf_thres=args.conf_thres, target_dir="./tmp")
        glbscene.export(file_obj=args.output_path)
        print(f"[INFO] Wrote GLB to: {args.output_path}")

    print("[INFO] Inference complete.")
    return predictions

# ----------------------------------------
# CLI Entry
# ----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input (video / image / multi-image comma-separated / folder)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save output .glb or .ply file")
    parser.add_argument("--output_type", type=str, default="glb", choices=["glb", "ply"],
                        help="Output file format")
    parser.add_argument("--model_path", type=str, default="/data2/zxy/workspace/models/VGGT-1B",
                        help="Path to pretrained VGGT model")
    parser.add_argument("--conf_thres", type=float, default=30.0,
                        help="Confidence threshold for point cloud filtering")
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="Compute device (cuda or cpu)")
    args = parser.parse_args()

    run_vggt(args)
