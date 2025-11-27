import csv
import json
import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
import sys
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sam2_util import get_frame_index, save_image_only, choose_dataset, collate_fn, \
    get_video_to_indices, load_model, process_videos_test
from attack_setting import SamForwarder, make_prompts, seed_everything

def run(args, custom_dataset):
    device = "cuda:1"
    sam_fwder, predictor = load_model(args, device=device)

    video_to_indices = get_video_to_indices(custom_dataset)
    total_miou_clean = 0.0
    total_miou_adv = 0.0
    iou_count = 0
    iou_count_adv = 0
    video_count = 0
    printed_videos = set()
    denorm = lambda x: sam_fwder.denorm_image(x)

    total_frame_count = 0
    mask_gt_dict_all = {}
    start_P_dict_all = {}
    video_result_paths = {}
    video_result_cleans = {}

    for video_name, indices in video_to_indices.items():
        print(f"\n{'=' * 40} Processing video: {video_name} {'=' * 40}")

        uap_id = f"{args.train_dataset}"
        uap_path = f"uap_file/{uap_id}.pth"
        if not os.path.exists(uap_path):
            print(f"UAP file not found for {video_name}, skipping...")
            continue
        uap = torch.load(uap_path, map_location=device)

        video_subset = Subset(custom_dataset, indices)
        video_loader = DataLoader(video_subset, batch_size=1, collate_fn=collate_fn)
        video_result_path = f"./adv/{args.train_dataset}/{video_name}"
        video_result_clean = f"./clean/{args.train_dataset}/{video_name}"


        mask_gt_dict = {}
        frame_counter = {}
        start_P_dict = {}
        for batch in tqdm(video_loader):
            total_frame_count += 1
            images, P_list, sample_ids, mask_gt_list, point = batch
            image = images[0]
            mask_gt = mask_gt_list[0]
            img_ID = sample_ids[0]
            video_name = img_ID.split('/')[0]

            frame_idx = get_frame_index(img_ID)
            print(f"Processing frame {frame_idx} from video {video_name}")

            if video_name not in printed_videos:
                start_frame_idx = custom_dataset.get_start_frame_idx(video_name)
                if start_frame_idx is not None:
                    print(f"Start frame index for video {video_name}: {start_frame_idx}")
                else:
                    print(f"Could not find start frame index for video {video_name}")
                printed_videos.add(video_name)

            if video_name not in frame_counter:
                frame_counter[video_name] = 0

            indexed_frame_idx = frame_counter[video_name]
            if frame_idx == start_frame_idx:
                start_point = point[0]
                print("Load new video and initialize memory........................")

            if video_name not in start_P_dict:
                start_P_dict[video_name] = start_point

            X = sam_fwder.transform_image(image).to(device)
            benign_img = denorm(X).to(device)
            adv_img = torch.clamp(benign_img + uap, 0, 1)

            adv_image = adv_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            adv_image = (adv_image * 255).astype('uint8')
            save_image_only(adv_image, video_name, frame_idx, video_result_path)

            im = benign_img.clone().detach().squeeze().permute(1, 2, 0).cpu().numpy()
            im = (im * 255).astype('uint8')
            save_image_only(im, video_name, frame_idx, video_result_clean)

            frame_counter[video_name] += 1
            mask_gt_dict[(video_name, indexed_frame_idx)] = mask_gt

        mask_gt_dict_all.update(mask_gt_dict)
        start_P_dict_all.update(start_P_dict)
        video_result_paths[video_name] = video_result_path
        video_result_cleans[video_name] = video_result_clean
        video_count += 1

    skipped_frames = []

    for video_name in video_result_paths.keys():
        video_output_adv = f"./adv/{args.train_dataset}"
        video_output_clean = f"./clean/{args.train_dataset}"

        miou_clean, iou_count_video, skipped_frames_video = process_videos_test(
            video_result_cleans[video_name],
            video_output_clean,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="clean",
            args=args)

        miou_adv, iou_count_adv_video, _ = process_videos_test(
            video_result_paths[video_name],
            video_output_adv,
            mask_gt_dict_all,
            start_P_dict_all,
            predictor,
            category="adversarial",
            skipped_frames=skipped_frames_video,
            args=args)

        total_miou_clean += miou_clean * iou_count_video
        total_miou_adv += miou_adv * iou_count_adv_video
        iou_count += iou_count_video
        iou_count_adv += iou_count_adv_video

    avg_miou_clean = total_miou_clean / iou_count if iou_count > 0 else 0
    avg_miou_adv = total_miou_adv / iou_count_adv if iou_count_adv > 0 else 0
    print(f'iou_count: {iou_count}')

    return video_count, avg_miou_clean, avg_miou_adv, iou_count, iou_count_adv, total_frame_count


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Your script description here")

    parser.add_argument('--limit_img', default=100, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--limit_frames', default=15, type=int, help='limit run image count, set -1 for all')
    parser.add_argument('--train_dataset', default='YOUTUBE')
    parser.add_argument('--test_dataset', default='YOUTUBE')
    parser.add_argument('--point', help='point coord formatted as h,w; e.g. 0.3,0.4 or 200,300')
    parser.add_argument('--train_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--test_prompts', choices=['bx', 'pt'], default='pt', help='type of prompts (box or point)')
    parser.add_argument('--checkpoints', default='sam2-t', help='model checkpoint')
    parser.add_argument('--fea_num', default=30, type=int)
    parser.add_argument('--weight_fea', default=0.000001, type=float)

    parser.add_argument('--seed', default=30, type=int, help='rand seed')
    parser.add_argument('--eps', default=10, type=int)
    parser.add_argument('--P_num', default=10, type=int)
    parser.add_argument('--prompts_num', default=256, type=int)

    parser.add_argument('--save_img_with_mask', action='store_true', help='save the image with mask')
    parser.add_argument('--save_mask', action='store_true', help='save the mask')

    parser.add_argument('--loss_fea', action='store_true')
    parser.add_argument('--loss_t', action='store_true')
    parser.add_argument('--loss_diff', action='store_true')

    return parser

def get_args(parser: ArgumentParser) -> Namespace:
    args = parser.parse_args()
    args.fps = -1
    args.debug = False
    return args
if __name__ == '__main__':

    parser = get_parser()
    args = get_args(parser)
    seed_everything(args.seed)
    custom_dataset = choose_dataset(args)
    video_test, miouimg, miouadv,frames_clean_test,frames_adv_test,frames_train= run(args, custom_dataset)
    print(f":: miouimg: {miouimg * 100:.2f} %, miouadv: {miouadv * 100:.2f} %, video_test: {video_test} , frame_clean_test:{frames_clean_test} , frames_adv_test:{frames_adv_test},frame_train:{frames_train}")

