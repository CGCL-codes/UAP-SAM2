import copy
import json
import os
import random
import re
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from time import time
import sys
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import trange, tqdm
from typing import *

from attack_setting import SamForwarder, seed_everything
from dataset_YOUTUBE import Dataset_YOUTUBE, Dataset_YOUTUBE_IMAGE
from sam2.build_sam import build_sam2_video_predictor

Data = Union[np.ndarray, Tensor]
DATA_ROOT_VIDEO_YOUTUBE = Path("./data/YOUTUBE/train/JPEGImages")
DATA_ROOT_IMAGE_YOUTUBE = Path("./dataset/YOUTUBE/train/JPEGImages")

def load_model(args,device = "cuda:1"):
    if args.checkpoints == 'sam2-t':
        checkpoint = "./checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "configs/sam2/sam2_hiera_t.yaml"

    else:
        raise ValueError(f"Unsupported checkpoint type: {args.checkpoints}")
    sam2 = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    sam2.eval()
    sam_fwder = SamForwarder(sam2).to(device)
    sam_fwder.eval()
    predictor = sam2
    return sam_fwder, predictor
def get_iou(x: np.ndarray, y: np.ndarray) -> float:
  while x.ndim < y.ndim:
    x = np.expand_dims(x, 0)
  while y.ndim < x.ndim:
    y = np.expand_dims(y, 0)
  if x.shape == y.shape:
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0.0
  else:
    min_shape = tuple(min(a, b) for a, b in zip(x.shape, y.shape))
    x = x[tuple(slice(0, s) for s in min_shape)]
    y = y[tuple(slice(0, s) for s in min_shape)]
    denominator = np.sum(np.logical_or(x, y))
    return np.sum(np.logical_and(x, y)) / denominator if denominator != 0 else 0

def get_iou_auto(x:Union[Data, List[Data]], y:Data) -> float:
  if isinstance(x, list):
    iou = max([get_iou(m, y) for m in x])
  else:
    iou = get_iou(x, y)
  return iou

def get_frame_index(img_ID):
    base_name = os.path.splitext(os.path.basename(img_ID))[0]
    try:
        if base_name.isdigit() and len(base_name) == 5:
            return int(base_name)
        else:
            raise ValueError(f"Unexpected frame name format: {base_name}")
    except ValueError as e:
        print(f"Error processing {img_ID}: {e}")
        raise
def save_image_only(image, video_name, frame_idx, save_dir):

    video_save_dir = os.path.join(save_dir, video_name)
    os.makedirs(video_save_dir, exist_ok=True)
    file_prefix = f'{frame_idx:04d}'
    filename = f'{file_prefix}.jpg'
    save_path = os.path.join(video_save_dir, filename)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_to_save = image

    cv2.imwrite(save_path, image_to_save)

def choose_dataset(args = None):
    if args.train_dataset == 'YOUTUBE':
        video_dirs = [video_dir for video_dir in DATA_ROOT_VIDEO_YOUTUBE.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)
        video_sample_ids = {}
        start_frames = {}
        for video_dir in video_dirs:
            if video_dir.is_dir():
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "./data/YOUTUBE/train/Annotations"

        custom_dataset = Dataset_YOUTUBE(sample_ids, DATA_ROOT_VIDEO_YOUTUBE, json_path, args=args, start_frames=start_frames)

    elif args.train_dataset == 'youtube-image':
        video_dirs = [video_dir for video_dir in DATA_ROOT_IMAGE_YOUTUBE.iterdir() if video_dir.is_dir()]
        video_dirs.sort(key=lambda x: x.name, reverse=True)
        print(f"dataset num :{len(video_dirs)}")
        num_samples = len(video_dirs) if args.limit_img == -1 else min(max(args.limit_img, 0), len(video_dirs))
        video_dirs = random.sample(video_dirs, num_samples)
        video_sample_ids = {}
        start_frames = {}
        for video_dir in video_dirs:
            if video_dir.is_dir():
                frames = [f"{video_dir.name}/{fp.stem}" for fp in video_dir.iterdir() if fp.is_file()]
                frames.sort()
                if frames:
                    first_frame_name = os.path.basename(frames[0].split('/')[-1])
                    try:
                        start_frame_idx = int(first_frame_name.lstrip('0') or '0')
                        start_frames[video_dir.name] = start_frame_idx
                    except ValueError as e:
                        print(f"Error processing first frame of {video_dir.name}: {e}")
                        start_frames[video_dir.name] = None
                if args.limit_frames > 0 and len(frames) >= args.limit_frames:
                    step = max(1, len(frames) // args.limit_frames)
                    selected_frames = frames[::step][:args.limit_frames]
                else:
                    print(f"Warning: Video {video_dir.name} has less than {args.limit_frames} frames. All frames will be used.")
                    selected_frames = frames
                selected_frames = selected_frames[::-1]
                video_sample_ids[video_dir.name] = selected_frames
        sample_ids = [item for sublist in video_sample_ids.values() for item in sublist]
        json_path = "./img_dataset/YOUTUBE/train/Annotations"
        custom_dataset = Dataset_YOUTUBE_IMAGE(sample_ids, DATA_ROOT_IMAGE_YOUTUBE, json_path, args=args, start_frames=start_frames)
    return custom_dataset

def process_videos_test(video_root_dir, output_dir, mask_gt_dict, start_P_dict, predictor, category,
                   video_range=None, skipped_frames=None, args=None):
    seed_everything(seed=args.seed)
    total_iou = 0
    iou_count = 0

    if skipped_frames is None:
        skipped_frames = set()

    print(f"Processing {category} samples...")

    video_names = sorted(os.listdir(video_root_dir))
    if video_range is not None:
        start_idx, end_idx = video_range
        video_names = video_names[start_idx:end_idx]

    for video_name in video_names:
        video_dir = os.path.join(video_root_dir, video_name)
        if not os.path.isdir(video_dir):
            continue

        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(inference_state)

        if video_name in start_P_dict:
            if args.test_prompts == 'pt':
                points = start_P_dict[video_name]
                points = np.array(points, dtype=np.float32)
                labels = np.array([1], np.int32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            elif args.test_prompts == 'bx':
                box = start_P_dict[video_name]
                box = np.array(box, dtype=np.float32)
                ann_obj_id = 1
                ann_frame_idx = 0
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=ann_obj_id,
                    box=box
                )

        else:
            continue

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        vis_frame_stride = 1
        for out_frame_idx in tqdm(range(0, len(frame_names), vis_frame_stride)):
            fig, ax = plt.subplots(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")

            image_path = os.path.join(video_dir, frame_names[out_frame_idx])
            image = np.array(Image.open(image_path))
            ax.imshow(image)

            if (video_name, out_frame_idx) in mask_gt_dict:
                mask_gt = mask_gt_dict[(video_name, out_frame_idx)]
            else:
                print(f"Warning: No ground truth mask found for {video_name}, frame {out_frame_idx}")
                continue

            for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():

                if not np.any(mask_gt > 0):
                    continue

                iou_img = get_iou_auto(out_mask, mask_gt)

                if category == "clean" and iou_img < 0.3:
                    print(f"Skipping frame {out_frame_idx} of {video_name} due to low IoU ({iou_img:.4f})")
                    skipped_frames.add((video_name, out_frame_idx))
                    continue

                if category == "adversarial" and (video_name, out_frame_idx) in skipped_frames:
                    print(f"Skipping adversarial IoU for {video_name}, frame {out_frame_idx}, as clean IoU was < 0.3")
                    continue

                total_iou += iou_img
                iou_count += 1
                print(f"IoU for {category} {video_name}, frame {out_frame_idx}, object {out_obj_id}: {iou_img:.4f}")
                print(f"Current iou_count: {iou_count}")

            if args.save_img_with_mask:
                os.makedirs(output_dir, exist_ok=True)
                save_prefix = "clean" if category == "clean" else "adv"
                save_path = os.path.join(output_dir, f"{video_name}_{save_prefix}_frame_{out_frame_idx:04d}.jpg")
                plt.savefig(save_path)
                plt.close(fig)

    avg_iou = total_iou / iou_count if iou_count > 0 else 0

    return avg_iou, iou_count, skipped_frames

def collate_fn(batch):
    buffer_list, P_list, sample_id,gt ,point = zip(*batch)
    return buffer_list, P_list, sample_id,gt,point
def infonce_loss(adv_feature, original_feature, target_feature, temperature=0.1):

    adv_feature_flat = adv_feature.reshape(1, -1)
    original_feature_flat = original_feature.reshape(1, -1)
    target_feature_flat = target_feature.reshape(1, -1)
    similarity_adv_original = torch.matmul(adv_feature_flat, original_feature_flat.T) / temperature
    similarity_adv_target = torch.matmul(adv_feature_flat, target_feature_flat.T) / temperature

    loss = -similarity_adv_target + similarity_adv_original

    return loss.squeeze()

def get_fused_prototype(img_list, sam_fwder, device):
    features = []
    for img in img_list:

        feat = sam_fwder.get_image_feature(img.to(device))
        features.append(feat)
    return torch.mean(torch.cat(features, dim=0), dim=0, keepdim=True)

def get_video_to_indices(dataset):
    video_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        img_ID = dataset.get_img_id(idx)
        video_name = img_ID.split('/')[0]
        video_to_indices[video_name].append(idx)
    return video_to_indices